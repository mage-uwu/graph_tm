"""Vet LogicAE pretraining: can we pretrain one ~4.4M model that adapts better than scratch?

Pretraining arms (masked-word, WikiText-103, 64-token windows, the 4.4M architecture of run 1:
16 blocks x 1024, 128-bit codes, kernel 5, cycle 4), run in parallel for PT_MIN minutes each:
  soft64     the current recipe (soft relaxation), at task length
  hard64     straight-through: forward = the hardened network (hard_patch.py --hard-forward)
  hardrev64  hard64 + every CHUNK steps, gates whose output channel is constant on held-out text are reset to
             the scratch pass-through init (diag revive: the dead channels that break transfer, TRIAGE.md)
After every CHUNK steps, per arm: validation masked-word CE, acc@1 on 2,000 fixed WikiText targets with
65 tokens of context (diag mlm, hard forward for the hard arms), and the share of dead trunk channels.

Adaptation (always straight-through, which TRIAGE.md showed is required): SST-2 (seq 64) and QNLI (seq 96),
FT_STEPS steps, batch 32, from scratch / run 1's pt.ltc (17-token soft) / each arm's checkpoint; final-step
hardened accuracy on the GLUE validation set (test) and on the 2,000 held-out train examples (dev).
"Reusable" = pretrained beats scratch on both tasks.

  python3 vet.py        env: THREADS, VET_OUT (/root/vet), VET_TOKEN, VET_EXPORT (/root/vet_exports),
                        PT_MIN (120), CHUNK (500), FT_STEPS (1000), ARMS (soft64,hard64,hardrev64), NWIN (400000)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
OUT = os.environ.get("VET_OUT", "/root/vet")
os.environ.setdefault("GTM_STAGE2_DATA", os.path.join(OUT, "data"))
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
T = int(os.environ.get("THREADS", os.cpu_count() or 4))
TOKEN = os.environ.get("VET_TOKEN", "")
EXPORT = os.environ.get("VET_EXPORT", "/root/vet_exports")
PT_MIN = float(os.environ.get("PT_MIN", 120))
CHUNK = int(os.environ.get("CHUNK", 500))
FT_STEPS = int(os.environ.get("FT_STEPS", 1000))
ARMS = os.environ.get("ARMS", "soft64,hard64,hardrev64").split(",")
NWIN = int(os.environ.get("NWIN", 400000))
ARCH = "--vocab-size 30522 --code-bits 128 --width 1024 --blocks 16 --kernel 5 --cycle 4".split()
PT17 = os.path.join(REPO, "models", "logicae", "pt.ltc")
LT, DIAG, D = os.path.join(OUT, "lt"), os.path.join(OUT, "diag"), os.path.join(OUT, "d")
LOCK = threading.Lock()


def say(msg):
    line = f"VET [{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOCK:
        for path in ("/proc/1/fd/1", os.path.join(OUT, "vet.log")):
            try:
                with open(path, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass
    publish()


def publish(files=()):
    if not TOKEN:
        return
    d = os.path.join(EXPORT, TOKEN)
    os.makedirs(d, exist_ok=True)
    for f, name in [(os.path.join(OUT, n), n) for n in ("vet.log", "results.jsonl")] + list(files):
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
    lb = os.path.join(OUT, "lb")
    if not os.path.isdir(lb):
        with tarfile.open(os.path.join(REPO, "logic-bert.tar.gz")) as t:
            t.extractall(lb)
    src = os.path.join(lb, "logic-bert", "src", "logic_text.c")
    b = os.path.join(OUT, "build")
    os.makedirs(b, exist_ok=True)
    for script, a, c in (("logicae/transfer_patch.py", src, "t.c"), ("logicae/hard_patch.py", "t.c", "h.c"),
                         ("fastlae/fasttrain_patch.py", "h.c", "logic_text.c")):
        r = sh([sys.executable, os.path.join(REPO, "stage2", script), os.path.join(b, a), os.path.join(b, c)])
        if r.returncode:
            raise SystemExit(f"{script}: {r.stdout}{r.stderr}")
    cc = ["gcc", "-O3", "-march=native", "-std=c11", "-fopenmp", "-Wno-unknown-pragmas"]
    for cmd in (cc + [os.path.join(b, "logic_text.c"), "-lm", "-o", LT],
                cc + ["-DDIAG_HARD", "-I" + b, os.path.join(HERE, "diag.c"), "-lm", "-o", DIAG]):
        r = sh(cmd)
        if r.returncode:
            raise SystemExit(f"build failed: {r.stderr[-2000:]}")
    say("engine built (transfer + hard + fasttrain patches, c11) and diag")


def write_ids(path, labels, rows):
    with open(path + ".tmp", "w") as f:
        for y, r in zip(labels, rows):
            f.write(f"{int(y)} " + " ".join(map(str, r)) + "\n")
    os.replace(path + ".tmp", path)


def windows(flat, lens, n, seed, L):
    """n random L-token windows starting inside a paragraph, PAD past the paragraph end"""
    rng = np.random.default_rng(seed)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    para = rng.choice(len(lens), size=n, p=lens / lens.sum())
    s = starts[para] + (rng.random(n) * lens[para]).astype(np.int64)
    p = s[:, None] + np.arange(L)[None, :]
    ok = p < (starts[para] + lens[para])[:, None]
    return np.where(ok, flat[np.clip(p, 0, len(flat) - 1)], 0).astype(np.int64)


def stream_paragraphs(tok, repo, files, batch=2000):
    """yield token-id arrays per paragraph (headings dropped, >= 2 tokens), streaming the parquet files"""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    for fn in files:
        with fs.open(f"datasets/{repo}/{fn}", "rb", block_size=8 << 20) as fh:
            for rb in pq.ParquetFile(fh).iter_batches(batch_size=batch, columns=["text"]):
                paras = [t.strip() for t in rb.column(0).to_pylist() if t and t.strip() and not t.strip().startswith("=")]
                for e in tok.encode_batch(paras, add_special_tokens=False) if paras else []:
                    if len(e.ids) >= 2:
                        yield np.asarray(e.ids, np.int64)


def data():
    os.makedirs(D, exist_ok=True)
    if os.path.exists(os.path.join(D, "done")):
        return
    import common as C
    import pretrain_data as P
    import tasks as TK
    tok = C.tokenizer()
    # streamed: parquet record batches over HTTP (HfFileSystem), tokenized per batch, windows sampled and written as
    # they go; the corpus is never held in memory or saved to disk (111.5M train tokens -> NWIN windows)
    rate = NWIN / 111.5e6
    rng = np.random.default_rng(1)
    got = 0
    with open(os.path.join(D, "pre_train.ids.tmp"), "w") as f:
        for para in stream_paragraphs(tok, P.WIKI, P.SPLIT_FILES["train"]):
            n = len(para)
            for s0 in np.flatnonzero(rng.random(n) < rate):
                w = np.zeros(64, np.int64)
                seg = para[s0:s0 + 64]
                w[:len(seg)] = seg
                f.write("-1 " + " ".join(map(str, w)) + "\n")
                got += 1
    os.replace(os.path.join(D, "pre_train.ids.tmp"), os.path.join(D, "pre_train.ids"))
    val = list(stream_paragraphs(tok, P.WIKI, P.SPLIT_FILES["validation"]))
    lens = np.array([len(a) for a in val], dtype=np.int64)
    zv = {"flat": np.concatenate(val), "lens": lens}  # 0.23M tokens: small enough to hold
    write_ids(os.path.join(D, "pre_val.ids"), -np.ones(2000), windows(zv["flat"], zv["lens"], 2000, 2, 64))
    # 2,000 fixed masked-word targets with 32 real tokens each side (the TRIAGE.md probe)
    starts = np.concatenate([[0], np.cumsum(zv["lens"])[:-1]])
    cand = [s + p for s, n in zip(starts, zv["lens"]) if n >= 65 for p in range(32, n - 32)]
    pick = np.random.default_rng(5).choice(len(cand), 2000, replace=False)
    rows, true = [], []
    for k in pick:
        c = cand[k]
        w = zv["flat"][c - 32:c + 33].copy()
        true.append(int(w[32]))
        w[32] = 1
        rows.append(w)
    write_ids(os.path.join(D, "wt_mask.ids"), -np.ones(len(rows)), rows)
    np.savetxt(os.path.join(D, "wt_mask.true"), true, fmt="%d")
    for task in ("sst2", "qnli"):
        Dt = TK.load(task)
        for split, d in Dt.items():
            rows = [list(x) for x in d["a"]] if task == "sst2" else [list(q) + [102] + list(s) for q, s in zip(d["a"], d["b"])]
            write_ids(os.path.join(D, f"{task}_{split}.ids"), d["y"], rows)
        with open(os.path.join(D, f"{task}_dev.ids")) as f, open(os.path.join(D, f"{task}_dev1k.ids"), "w") as g:
            g.writelines(f.readlines()[:1000])
    open(os.path.join(D, "done"), "w").close()
    say(f"data (streamed): {got} x 64-token pretraining windows, 2,000 masked-word targets, sst2 / qnli splits")


def probe(ck, hard):
    """masked-word acc@1 / CE on the fixed targets + dead trunk channels on held-out WikiText"""
    h = ["--hard"] if hard else []
    r = sh([DIAG, "mlm", ck, os.path.join(D, "wt_mask.ids"), os.path.join(D, "wt_mask.true"), "65"] + h,
           env=dict(os.environ, OMP_NUM_THREADS=str(max(1, T // 3))))
    m = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
    tmp = ck + ".probe"
    r2 = sh([DIAG, "revive", ck, os.path.join(D, "pre_val.ids"), "64", tmp] + h,
            env=dict(os.environ, OMP_NUM_THREADS=str(max(1, T // 3))))
    if os.path.exists(tmp):
        os.remove(tmp)
    dead = [int(x) for x in re.findall(r"dead (\d+)/1024", r2.stderr)]
    return {"acc1": m.get("acc1"), "acc10": m.get("acc10"), "ce_probe": m.get("ce"),
            "dead_share": round(sum(dead) / (1024 * len(dead)), 3) if dead else None,
            "dead_by_block": dead}


def pretrain(arm, threads):
    hard = arm.startswith("hard")
    ck = os.path.join(OUT, f"{arm}.ltc")
    base = [LT, "pretrain", "--threads", str(threads), "--data", os.path.join(D, "pre_train.ids"),
            "--val", os.path.join(D, "pre_val.ids"), "--format", "ids", "--seq", "64", "--batch", "64",
            "--mlm-targets", "512"] + (["--hard-forward"] if hard else [])
    log = open(os.path.join(OUT, f"{arm}.log"), "a")
    if not os.path.exists(ck):  # probe the step time, then size the run to PT_MIN minutes
        pr = os.path.join(OUT, f"{arm}_probe.ltc")
        subprocess.run(base + ARCH + ["--seed", "17", "--steps", "1000", "--stop-after", "3", "--eval-every", "3",
                                      "--save", pr], stderr=log, check=True)
        sps = float(re.findall(r'"compute_seconds":([0-9.]+)', open(os.path.join(OUT, f"{arm}.log")).read())[-1]) / 3
        steps = max(CHUNK, int(PT_MIN * 60 / sps) // CHUNK * CHUNK)
        os.remove(pr)
        json.dump({"steps": steps, "sps": sps}, open(ck + ".plan", "w"))
        say(f"{arm}: {sps:.2f} s/step at {threads} threads -> {steps} steps ({steps * 64} windows)")
    steps = json.load(open(ck + ".plan"))["steps"]
    first = base + ARCH + ["--seed", "17", "--steps", str(steps)]
    done = int(re.findall(r'"step":(\d+)', open(os.path.join(OUT, f"{arm}.log")).read() or '"step":0')[-1]) if os.path.exists(ck) else 0
    k = done // CHUNK
    while (k + 1) * CHUNK <= steps:
        stop = (k + 1) * CHUNK
        cmd = (first if k == 0 and not os.path.exists(ck) else base + ["--resume", ck]) + \
              ["--stop-after", str(stop), "--eval-every", str(CHUNK), "--save", ck]
        subprocess.run(cmd, stderr=log, check=True)
        txt = open(os.path.join(OUT, f"{arm}.log")).read()
        ce = re.findall(r'"step":%d,[^\n]*"val_mlm_ce":([0-9.]+)' % stop, txt)
        rec = {"phase": "pretrain", "arm": arm, "step": stop, "of": steps, "val_mlm_ce": float(ce[-1]) if ce else None}
        rec.update(probe(ck, hard))
        if arm == "hardrev64" and stop < steps:
            r = sh([DIAG, "revive", ck, os.path.join(D, "pre_val.ids"), "64", ck + ".rev", "--hard"],
                   env=dict(os.environ, OMP_NUM_THREADS=str(threads)))
            rec["revived"] = json.loads(r.stdout.strip().splitlines()[-1])["reset_channels"]
            os.replace(ck + ".rev", ck)
        result(rec)
        k += 1
    publish([(ck, f"{arm}.ltc")])
    return ck


def finetune(name, init, task, threads):
    seq = {"sst2": 64, "qnli": 96}[task]
    m = os.path.join(OUT, f"ft_{task}_{name}")
    if os.path.exists(m + ".done"):
        return
    start = ARCH if init is None else ["--load", init, "--keep-temperature"]
    t0 = time.time()
    with open(m + ".log", "w") as log:
        subprocess.run([LT, "train", "--threads", str(threads), "--data", os.path.join(D, f"{task}_train.ids"),
                        "--val", os.path.join(D, f"{task}_dev1k.ids"), "--format", "ids", "--seq", str(seq),
                        "--batch", "32", "--steps", str(FT_STEPS), "--eval-every", "250", "--seed", "17",
                        "--hard-forward", "--save", m + ".ltc"] + start, stderr=log, check=True)
    acc = {}
    for split in ("dev", "test"):
        r = sh([LT, "eval", "--load", m + ".ltc", "--data", os.path.join(D, f"{task}_{split}.ids"), "--format", "ids",
                "--seq", str(seq), "--threads", str(threads)])
        acc[split] = float(re.findall(r'"hard_accuracy":([0-9.]+)', r.stdout + r.stderr)[-1])
    curve = [float(x) for x in re.findall(r'"val_hard_accuracy":([0-9.]+)', open(m + ".log").read())]
    result({"phase": "adapt", "task": task, "init": name, "steps": FT_STEPS, "dev": acc["dev"], "test": acc["test"],
            "dev1k_curve": curve, "minutes": round((time.time() - t0) / 60, 1)})
    for s in (".ltc", ".ltc.best"):
        if os.path.exists(m + s):
            os.remove(m + s)
    open(m + ".done", "w").close()


def pool(jobs, width):
    """run callables `width` at a time"""
    it, lock = iter(jobs), threading.Lock()

    def worker():
        while True:
            with lock:
                job = next(it, None)
            if job is None:
                return
            try:
                job()
            except Exception as ex:  # one failed job must not stop the rest
                say(f"FAILED {type(ex).__name__}: {str(ex)[-300:]}")
    ts = [threading.Thread(target=worker) for _ in range(width)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main():
    os.makedirs(OUT, exist_ok=True)
    say(f"vet: arms {ARMS}, {PT_MIN} min pretraining each, adaptation {FT_STEPS} steps, {T} threads")
    build()
    data()
    tp = max(1, T // len(ARMS))
    cks = {}
    pool([(lambda a=a: cks.__setitem__(a, pretrain(a, tp))) for a in ARMS], len(ARMS))
    inits = [("scratch", None), ("pt17_soft", PT17)] + [(a, os.path.join(OUT, f"{a}.ltc")) for a in ARMS]
    jobs = [(lambda n=n, i=i, t=t: finetune(n, i, t, max(1, T // 3))) for t in ("sst2", "qnli") for n, i in inits]
    pool(jobs, 3)
    say("done")


if __name__ == "__main__":
    main()
