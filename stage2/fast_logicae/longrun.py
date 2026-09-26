"""The long LogicAE pretraining run: fastlogic.c, hard (straight-through) masked-word pretraining of the 4.4M model
with the pair options built in, 30,000 steps x 64 windows x 64 tokens = 123M tokens of WikiText-103, then adaptation
checks (SST-2, QNLI) against scratch.

  data      streamed: a producer thread reads the parquet over HTTP (HfFileSystem), tokenizes per record batch and
            cuts paragraphs into 64-token windows in stream order, written as chunk files of CHUNK_STEPS x 64 windows
            (at most 2 ahead of the trainer; used chunks are deleted). A second pass starts at a 32-token offset.
            The corpus is never held in memory or saved.
  training  one `fastlogic pretrain` per chunk: the first starts the run (--steps TOTAL), the rest --resume it
            (bit-exact), so the schedule is one 30k-step cosine. Rotating checkpoint: saved.pt (overwritten
            atomically after every chunk). saved_q1.pt at 25% (for a local adaptation test), saved_fastae.pt at the end.
  logging   perf.log every PERF_SEC s: step, s/step, tokens/s, ETA, last train CE, last validation CE; after every
            chunk: masked-word acc@1/@10 on 2,000 fixed targets and the share of dead trunk channels (results.jsonl).
  serving   run.log, perf.log, results.jsonl, saved.pt, saved_q1.pt, saved_fastae.pt at :8888/<LR_TOKEN>/.

  python3 longrun.py     env: THREADS, LR_OUT (/root/lr), LR_TOKEN, LR_EXPORT (/root/lr_exports), STEPS (30000),
                         CHUNK_STEPS (2500), PERF_SEC (300), FT_STEPS (1000)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.dirname(HERE)
sys.path.insert(0, S2)
OUT = os.environ.get("LR_OUT", "/root/lr")
os.environ.setdefault("GTM_STAGE2_DATA", os.path.join(OUT, "data"))
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
T = int(os.environ.get("THREADS", os.cpu_count() or 4))
TOKEN = os.environ.get("LR_TOKEN", "")
EXPORT = os.environ.get("LR_EXPORT", "/root/lr_exports")
STEPS = int(os.environ.get("STEPS", 30000))
CHUNK = int(os.environ.get("CHUNK_STEPS", 2500))
PERF_SEC = int(os.environ.get("PERF_SEC", 300))
FT_STEPS = int(os.environ.get("FT_STEPS", 1000))
BATCH, SEQ = 64, 64
PAIRS = ["--match", "1", "--global-every", "4", "--global-channels", "128", "--global-mean", "1"]
FL, DIAG, D = os.path.join(OUT, "fastlogic"), os.path.join(OUT, "diag"), os.path.join(OUT, "d")
CK = os.path.join(OUT, "saved.pt")
LOCK = threading.Lock()
STOP = threading.Event()


def say(msg, f="run.log"):
    line = f"LR [{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOCK:
        for path in ("/proc/1/fd/1", os.path.join(OUT, f)) + ((os.path.join(OUT, "run.log"),) if f != "run.log" else ()):
            try:
                with open(path, "a") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
    publish()


def publish(files=()):
    if not TOKEN:
        return
    d = os.path.join(EXPORT, TOKEN)
    os.makedirs(d, exist_ok=True)
    for f, name in [(os.path.join(OUT, n), n) for n in ("run.log", "perf.log", "results.jsonl")] + list(files):
        if os.path.isfile(f):
            shutil.copy(f, os.path.join(d, name + ".tmp"))
            os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))


def result(r):
    with LOCK:
        with open(os.path.join(OUT, "results.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
    say("RESULT " + json.dumps(r))


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build():
    cc = ["gcc", "-O3", "-march=native", "-std=c11", "-fopenmp", "-Wno-unknown-pragmas"]
    for cmd in (cc + [os.path.join(HERE, "fastlogic.c"), "-lm", "-o", FL],
                cc + ["-DDIAG_HARD", "-I" + HERE, os.path.join(S2, "logicae", "diag.c"), "-lm", "-o", DIAG]):
        r = sh(cmd)
        if r.returncode:
            raise SystemExit(f"build failed: {r.stderr[-1500:]}")
    say("built fastlogic (bit-level hard forward, fast output layer) and diag")


def stream_paragraphs(tok, files, batch=2000):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    import pretrain_data as P
    fs = HfFileSystem()
    for fn in files:
        with fs.open(f"datasets/{P.WIKI}/{fn}", "rb", block_size=8 << 20) as fh:
            for rb in pq.ParquetFile(fh).iter_batches(batch_size=batch, columns=["text"]):
                paras = [t.strip() for t in rb.column(0).to_pylist() if t and t.strip() and not t.strip().startswith("=")]
                for e in tok.encode_batch(paras, add_special_tokens=False) if paras else []:
                    if len(e.ids) >= 2:
                        yield np.asarray(e.ids, np.int64)


def write_rows(path, rows, label=-1):
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(f"{label} " + " ".join(map(str, r)) + "\n")
    os.replace(path + ".tmp", path)


def producer(nchunks):
    """chunk_<k>.ids: CHUNK x BATCH windows of SEQ tokens, streamed; at most 2 chunks ahead of the trainer"""
    import common as C
    import pretrain_data as P
    tok = C.tokenizer()
    per = CHUNK * BATCH
    k, buf, pas = 0, [], 0
    while k < nchunks and not STOP.is_set():
        off = 0 if pas == 0 else SEQ // 2
        for para in stream_paragraphs(tok, P.SPLIT_FILES["train"]):
            for s0 in range(off if len(para) > off + 1 else 0, len(para), SEQ):
                w = np.zeros(SEQ, np.int64)
                seg = para[s0:s0 + SEQ]
                w[:len(seg)] = seg
                if len(seg) >= 2:
                    buf.append(w)
                if len(buf) == per:
                    while not STOP.is_set() and len([f for f in os.listdir(D) if f.startswith("chunk_") and f.endswith(".ids")]) >= 2:
                        time.sleep(10)
                    write_rows(os.path.join(D, f"chunk_{k}.ids"), buf)
                    say(f"data: chunk {k} written ({per} windows, pass {pas + 1})")
                    buf, k = [], k + 1
                    if k >= nchunks or STOP.is_set():
                        return
        pas += 1


def eval_data():
    """validation windows, 2,000 fixed masked-word targets (32 tokens of context each side), SST-2 / QNLI splits"""
    os.makedirs(D, exist_ok=True)
    if os.path.exists(os.path.join(D, "eval.done")):
        return
    import common as C
    import pretrain_data as P
    import tasks as TK
    tok = C.tokenizer()
    val = list(stream_paragraphs(tok, P.SPLIT_FILES["validation"]))
    rng = np.random.default_rng(2)
    rows = []
    for p in val:
        for s0 in range(0, len(p), SEQ):
            w = np.zeros(SEQ, np.int64)
            seg = p[s0:s0 + SEQ]
            w[:len(seg)] = seg
            rows.append(w)
    pick = rng.choice(len(rows), 2000, replace=False)
    write_rows(os.path.join(D, "val.ids"), [rows[i] for i in pick])
    cand = [(i, j) for i, p in enumerate(val) if len(p) >= 65 for j in range(32, len(p) - 32)]
    sel = np.random.default_rng(5).choice(len(cand), 2000, replace=False)
    mrows, true = [], []
    for s in sel:
        i, j = cand[s]
        w = val[i][j - 32:j + 33].copy()
        true.append(int(w[32]))
        w[32] = 1
        mrows.append(w)
    write_rows(os.path.join(D, "wt_mask.ids"), mrows)
    np.savetxt(os.path.join(D, "wt_mask.true"), true, fmt="%d")
    for task in ("sst2", "qnli"):
        Dt = TK.load(task)
        for split, d in Dt.items():
            rs = [list(x) for x in d["a"]] if task == "sst2" else [list(q) + [102] + list(s) for q, s in zip(d["a"], d["b"])]
            with open(os.path.join(D, f"{task}_{split}.ids"), "w") as f:
                for y, r in zip(d["y"], rs):
                    f.write(f"{int(y)} " + " ".join(map(str, r)) + "\n")
        with open(os.path.join(D, f"{task}_dev.ids")) as f, open(os.path.join(D, f"{task}_dev1k.ids"), "w") as g:
            g.writelines(f.readlines()[:1000])
    open(os.path.join(D, "eval.done"), "w").close()
    say("eval data: 2,000 validation windows, 2,000 masked-word targets, sst2 / qnli splits")


def perf_monitor(logpath, t_start, done0):
    """every PERF_SEC: step, s/step, tokens/s, ETA, last train CE, last validation CE"""
    last = None
    while not STOP.wait(PERF_SEC):
        try:
            txt = open(logpath).read()
        except OSError:
            continue
        rows = [json.loads(l) for l in txt.splitlines() if l.startswith('{"step"')]
        if not rows:
            say("perf: waiting for the first evaluation", "perf.log")
            continue
        r = rows[-1]
        step = r["step"]
        # compute_seconds restarts at every resumed chunk: use wall time since the run started for rates
        sps = (time.time() - t_start) / max(step - done0, 1)
        rate = ""
        if last and step > last[0]:
            rate = f", last window {(time.time() - last[1]) / (step - last[0]):.2f} s/step"
        last = (step, time.time())
        say(f"perf: step {step}/{STEPS} ({100 * step / STEPS:.1f}%), {sps:.2f} s/step avg{rate}, "
            f"{BATCH * SEQ / sps:.0f} tokens/s, ETA {(STEPS - step) * sps / 3600:.1f} h, "
            f"train CE {r.get('ce', float('nan')):.3f}, val CE {r.get('val_mlm_ce', float('nan')):.3f}, "
            f"code temperature {r.get('code_temperature', float('nan')):.3f}", "perf.log")


def probe(ck):
    r = sh([DIAG, "mlm", ck, os.path.join(D, "wt_mask.ids"), os.path.join(D, "wt_mask.true"), "65", "--hard"],
           env=dict(os.environ, OMP_NUM_THREADS=str(T)))
    m = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
    tmp = ck + ".probe"
    r2 = sh([DIAG, "revive", ck, os.path.join(D, "val.ids"), str(SEQ), tmp, "--hard"], env=dict(os.environ, OMP_NUM_THREADS=str(T)))
    if os.path.exists(tmp):
        os.remove(tmp)
    dead = [int(x) for x in re.findall(r"dead (\d+)/1024", r2.stderr)]
    return {"acc1": m.get("acc1"), "acc10": m.get("acc10"), "ce_probe": m.get("ce"),
            "dead_share": round(sum(dead) / (1024 * len(dead)), 3) if dead else None}


def finetune(name, init, task, threads):
    seq = {"sst2": 64, "qnli": 96}[task]
    m = os.path.join(OUT, f"ft_{task}_{name}")
    start = PAIRS if init is None else ["--load", init]
    t0 = time.time()
    with open(m + ".log", "w") as log:
        subprocess.run([FL, "train", "--threads", str(threads), "--data", os.path.join(D, f"{task}_train.ids"),
                        "--val", os.path.join(D, f"{task}_dev1k.ids"), "--seq", str(seq), "--batch", "32",
                        "--steps", str(FT_STEPS), "--eval-every", "250", "--seed", "17", "--save", m + ".ltc"] + start,
                       stderr=log, check=True)
    acc = {}
    for split in ("dev", "test"):
        r = sh([FL, "eval", "--load", m + ".ltc", "--data", os.path.join(D, f"{task}_{split}.ids"), "--seq", str(seq),
                "--threads", str(threads)])
        acc[split] = float(re.findall(r'"hard_accuracy":([0-9.]+)', r.stdout + r.stderr)[-1])
    curve = [float(x) for x in re.findall(r'"val_hard_accuracy":([0-9.]+)', open(m + ".log").read())]
    result({"phase": "adapt", "task": task, "init": name, "steps": FT_STEPS, "dev": acc["dev"], "test": acc["test"],
            "dev1k_curve": curve, "minutes": round((time.time() - t0) / 60, 1)})
    os.remove(m + ".ltc")


def main():
    os.makedirs(D, exist_ok=True)
    say(f"long run: {STEPS} steps x {BATCH} x {SEQ} tokens = {STEPS * BATCH * SEQ / 1e6:.0f}M tokens, hard forward, "
        f"pair options {' '.join(PAIRS)}, {T} threads, chunks of {CHUNK} steps")
    build()
    eval_data()
    nchunks = STEPS // CHUNK
    log = os.path.join(OUT, "train.log")
    done = 0
    if os.path.exists(CK) and os.path.exists(log):
        done = max([int(x) for x in re.findall(r'"step":(\d+)', open(log).read())] or [0]) // CHUNK * CHUNK
        say(f"resuming at step {done}")
    prod = threading.Thread(target=producer, args=(nchunks,), daemon=True)
    # a restarted run re-streams from the start of the corpus: skip the chunks already trained
    for f in os.listdir(D):
        if f.startswith("chunk_"):
            os.remove(os.path.join(D, f))
    prod.start()
    t_start = time.time()
    threading.Thread(target=perf_monitor, args=(log, t_start, done), daemon=True).start()
    for k in range(nchunks):
        stop = (k + 1) * CHUNK
        chunk = os.path.join(D, f"chunk_{k}.ids")
        while not os.path.exists(chunk):
            time.sleep(5)
        if stop <= done:
            os.remove(chunk)
            continue
        base = [FL, "pretrain", "--threads", str(T), "--data", chunk, "--val", os.path.join(D, "val.ids"),
                "--seq", str(SEQ), "--batch", str(BATCH), "--mlm-targets", "512", "--eval-every", "250",
                "--stop-after", str(stop), "--save", CK]
        cmd = base + (["--steps", str(STEPS), "--seed", "17"] + PAIRS if not os.path.exists(CK) else ["--resume", CK])
        with open(log, "a") as lf:
            subprocess.run(cmd, stderr=lf, check=True)
        os.remove(chunk)
        txt = open(log).read()
        ce = re.findall(r'"step":%d,[^\n]*"val_mlm_ce":([0-9.]+)' % stop, txt)
        rec = {"phase": "pretrain", "step": stop, "of": STEPS, "tokens_M": round(stop * BATCH * SEQ / 1e6, 1),
               "val_mlm_ce": float(ce[-1]) if ce else None, "hours": round((time.time() - t_start) / 3600, 2)}
        rec.update(probe(CK))
        result(rec)
        publish([(CK, "saved.pt")])
        if stop >= STEPS // 4 and not os.path.exists(os.path.join(OUT, "saved_q1.pt")):
            shutil.copy(CK, os.path.join(OUT, "saved_q1.pt"))
            publish([(os.path.join(OUT, "saved_q1.pt"), "saved_q1.pt")])
            say("saved_q1.pt (25%) is downloadable")
    STOP.set()
    final = os.path.join(OUT, "saved_fastae.pt")
    shutil.copy(CK, final)
    sh([FL, "export", "--load", final, "--out", os.path.join(OUT, "saved_fastae.lth")])
    publish([(final, "saved_fastae.pt"), (os.path.join(OUT, "saved_fastae.lth"), "saved_fastae.lth")])
    say("saved_fastae.pt (final) is downloadable; adaptation checks next")
    jobs = [(t, n, i) for t in ("sst2", "qnli") for n, i in (("scratch", None), ("pretrained", final))]
    th = max(1, T // 2)
    for a in range(0, len(jobs), 2):
        ts = [threading.Thread(target=finetune, args=(n, i, t, th)) for t, n, i in jobs[a:a + 2]]
        for x in ts:
            x.start()
        for x in ts:
            x.join()
    say("done")


if __name__ == "__main__":
    main()
