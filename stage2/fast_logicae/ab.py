"""Why did the long run stop transferring? Two 3k-step hard-forward pretraining arms, side by side, then the same SST-2
fine-tune from each (reuses longrun.py's build / data / probe / finetune):

  A  pair options ON,  iid windows (uniform over the corpus, like the vetting run that transferred)
  B  pair options OFF, stream-order windows (article order, 160k-window chunks, like the long run)

Known references (same fine-tune): vetting hard64 (pairs off, iid) 0.805; long run (pairs on, stream order) 0.743 at
30k steps, 0.751 at 7.5k; scratch 0.772 (pairs off) / 0.779 (pairs on).
If A transfers and B does not, data order broke it; if B transfers and A does not, the pair options did.

  python3 ab.py      env: LR_OUT (/root/ab), LR_TOKEN, THREADS, AB_STEPS (3000), AB_SPLIT (2500), AB_WIN (192000)
"""
import os
import subprocess
import threading
import time

os.environ.setdefault("LR_OUT", "/root/ab")
import numpy as np  # noqa: E402

import longrun as L  # noqa: E402

STEPS = int(os.environ.get("AB_STEPS", 3000))
SPLIT = int(os.environ.get("AB_SPLIT", 2500))       # long run: chunk 0 for the first 2500 steps, then chunk 1
NWIN = int(os.environ.get("AB_WIN", 192000))         # iid windows (vetting: 400k file, ~192k drawn)
CHW = SPLIT * L.BATCH                                 # stream-order chunk size (160k at 2500 steps)
MAXPARA = int(os.environ.get("AB_MAXPARA", 0))       # smoke tests only


def data():
    """one streamed pass: iid windows (Bernoulli start positions, rate NWIN / corpus tokens) and the first two
    stream-order chunks (non-overlapping 64-token windows in article order, as longrun.producer)"""
    iid_p, s0_p, s1_p = (os.path.join(L.D, f) for f in ("iid.ids", "stream0.ids", "stream1.ids"))
    if all(os.path.exists(p) for p in (iid_p, s0_p, s1_p)):
        return iid_p, s0_p, s1_p
    import common as C
    import pretrain_data as P
    tok = C.tokenizer()
    rate, rng = NWIN / 111.5e6, np.random.default_rng(1)
    iid, stream, n = [], [], 0
    for para in L.stream_paragraphs(tok, P.SPLIT_FILES["train"]):
        for s in np.flatnonzero(rng.random(len(para)) < rate):
            w = np.zeros(L.SEQ, np.int64)
            seg = para[s:s + L.SEQ]
            w[:len(seg)] = seg
            iid.append(w)
        if len(stream) < 2 * CHW:
            for s in range(0, len(para), L.SEQ):
                seg = para[s:s + L.SEQ]
                if len(seg) >= 2 and len(stream) < 2 * CHW:
                    w = np.zeros(L.SEQ, np.int64)
                    w[:len(seg)] = seg
                    stream.append(w)
        n += 1
        if MAXPARA and n >= MAXPARA:
            break
    L.write_rows(iid_p, iid)
    L.write_rows(s0_p, stream[:CHW])
    L.write_rows(s1_p, stream[CHW:])
    L.say(f"data (streamed): {len(iid)} iid windows; stream-order chunks {len(stream[:CHW])} + {len(stream[CHW:])}")
    return iid_p, s0_p, s1_p


def pretrain(arm, pairs, files, threads):
    ck = os.path.join(L.OUT, f"{arm}.pt")
    log = open(os.path.join(L.OUT, f"{arm}.log"), "a")
    base = [L.FL, "pretrain", "--threads", str(threads), "--val", os.path.join(L.D, "val.ids"), "--seq", str(L.SEQ),
            "--batch", str(L.BATCH), "--mlm-targets", "512", "--eval-every", "500", "--save", ck]
    t0 = time.time()
    stops = [STEPS] if len(files) == 1 else [SPLIT, STEPS]
    for i, (f, stop) in enumerate(zip(files, stops)):
        cmd = base + ["--data", f, "--stop-after", str(stop)]
        cmd += (["--steps", str(STEPS), "--seed", "17"] + (L.PAIRS if pairs else [])) if i == 0 else ["--resume", ck]
        subprocess.run(cmd, stderr=log, check=True)
    txt = open(os.path.join(L.OUT, f"{arm}.log")).read()
    ce = [float(x) for x in __import__("re").findall(r'"val_mlm_ce":([0-9.]+)', txt)]
    rec = {"phase": "pretrain", "arm": arm, "pairs": pairs, "data": "iid" if len(files) == 1 else "stream-order",
           "steps": STEPS, "val_mlm_ce": ce[-1] if ce else None, "minutes": round((time.time() - t0) / 60, 1)}
    rec.update(L.probe(ck))
    L.result(rec)
    L.publish([(ck, f"{arm}.pt")])
    return ck


def main():
    os.makedirs(L.D, exist_ok=True)
    L.say(f"A/B: {STEPS} steps each; A = pairs + iid windows, B = no pairs + stream-order chunks; SST-2 fine-tune {L.FT_STEPS} steps")
    L.build()
    L.eval_data()
    iid_p, s0_p, s1_p = data()
    th = max(1, L.T // 2)
    cks = {}
    jobs = [threading.Thread(target=lambda: cks.__setitem__("A", pretrain("A_pairs_iid", True, [iid_p], th))),
            threading.Thread(target=lambda: cks.__setitem__("B", pretrain("B_nopairs_stream", False, [s0_p, s1_p], th)))]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    fts = [threading.Thread(target=L.finetune, args=(name, cks[k], "sst2", th))
           for k, name in (("A", "A_pairs_iid"), ("B", "B_nopairs_stream")) if k in cks]
    for j in fts:
        j.start()
    for j in fts:
        j.join()
    L.say("done")


if __name__ == "__main__":
    main()
