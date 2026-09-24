"""Why does LogicAE's pretraining not transfer? Diagnostics on the playground pod.

Run 1 (stage2/logicae/run.sh): fine-tuning from the pretrained model was WORSE than from scratch
(SST-2 test 0.758 vs 0.794, QNLI 0.525 vs 0.593). Hypotheses and the runs that separate them, all
with run 1's exact fine-tuning recipe (2000 steps, batch 32, eval every 250, seed 17, same splits),
so every result is directly comparable with run 1's four numbers:

  keep      --load pt.ltc --keep-temperature: codes stay at the checkpoint's temperature 0.2.
            Run 1 reset it to 1.0 (reset_optimizer) and re-annealed, so the pretrained gates first
            saw soft codes unlike anything in pretraining (cause 1).
  codes     --load-part codes --keep-temperature: pretrained token codes, fresh gates
  gates     --load-part gates --keep-temperature: pretrained gates (+ wiring), fresh codes
            (causes 2 vs 4: which part carries the benefit or the harm)
  long      continue pretraining pt.ltc on 129-token windows (--keep-temperature, LONG_STEPS),
            then fine-tune it with --keep-temperature (cause 3: 17-token windows vs 64 / 128)
  control   --load pt.ltc exactly as run 1 (must reproduce run 1's test accuracy: setup check)

Order (most informative first): sst2 keep, qnli keep, sst2 codes, sst2 gates, long pretrain, sst2
long, qnli long, sst2 control, qnli codes, qnli gates. Each result: 'TX RESULT {...}' in the
container log and results.jsonl (test accuracy with the final model, as run 1; best dev; the dev
curve). Heartbeat 'TX HB' every 60 s.

The pretrained model comes from the repo: this waits until models/logicae/pt.ltc exists on the
branch (committed after run 1's export, sha256-verified against models/logicae/MANIFEST.txt).
Disk (5 GB pod): per-run checkpoints are deleted after evaluation.

  python3 transfer.py   (from a clone of the branch; env W, BRANCH, THREADS, FT_STEPS, LONG_STEPS, ONLY)
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
W = os.environ.get("W", "/root/tx")
BRANCH = os.environ.get("BRANCH", "claude/lucid-ritchie-qtpddj")
T = int(os.environ.get("THREADS", min(16, os.cpu_count() or 1)))
FT_STEPS = int(os.environ.get("FT_STEPS", 2000))
LONG_STEPS = int(os.environ.get("LONG_STEPS", 1500))
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]
RUN1 = {"sst2": {"pretrained": 0.75802752, "scratch": 0.79357798}, "qnli": {"pretrained": 0.52462017, "scratch": 0.59271463}}
SEQ = {"sst2": 64, "qnli": 128}
LOG = os.path.join(W, "transfer.log")
PHASE = {"p": "setup", "log": None}


def say(msg):
    line = f"TX [{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for p in (LOG, "/proc/1/fd/1"):
        try:
            with open(p, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def heartbeat():
    while True:
        time.sleep(60)
        last = ""
        try:
            if PHASE["log"]:
                ls = [l for l in open(PHASE["log"], errors="replace") if l.startswith('{"step"')]
                last = ls[-1].strip()[:220] if ls else ""
        except OSError:
            pass
        du = shutil.disk_usage(W)
        say(f"HB {PHASE['p']} | disk free {du.free / 2**30:.1f}G | {last}")


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def build():
    src = os.path.join(W, "lb")
    if not os.path.exists(os.path.join(W, "ltx")):
        with tarfile.open(os.path.join(REPO, "logic-bert.tar.gz")) as t:
            t.extractall(src)
        orig = os.path.join(src, "logic-bert", "src", "logic_text.c")
        os.makedirs(os.path.join(W, "src"), exist_ok=True)
        sh([sys.executable, os.path.join(HERE, "transfer_patch.py"), orig, os.path.join(W, "src", "logic_text.c")])
        flags = ["-O3", "-march=native", "-std=c11", "-fopenmp", "-Wno-unknown-pragmas"]
        sh(["gcc", *flags, os.path.join(W, "src", "logic_text.c"), "-lm", "-o", os.path.join(W, "ltx")])
    r = subprocess.run([os.path.join(W, "ltx"), "selftest"], capture_output=True, text=True)
    say(f"build: {r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:]}")


def data():
    """run 1's exact task splits (tasks.py is seeded) + 129-token WikiText-103 windows"""
    sys.path.insert(0, os.path.join(REPO, "stage2"))
    os.environ.setdefault("GTM_STAGE2_DATA", os.path.join(W, "data"))
    import numpy as np
    import common as C
    import tasks as TK
    sys.path.insert(0, HERE)
    import prep as P
    for task in ("sst2", "qnli"):
        if os.path.exists(os.path.join(W, f"{task}_test.ids")):
            continue
        for split, d in TK.load(task).items():
            rows = [list(x) for x in d["a"]] if task == "sst2" else [list(q) + [P.SEP] + list(s) for q, s in zip(d["a"], d["b"])]
            P.write_ids(os.path.join(W, f"{task}_{split}.ids"), d["y"], rows)
        say(f"data: {task} splits written")
    if not os.path.exists(os.path.join(W, "long_train.ids")) and (not ONLY or any(x.startswith("long") for x in ONLY)):
        import pyarrow.parquet as pq
        import pretrain_data as PD
        tok = C.tokenizer()
        P.WIN = 64  # 129-token windows, PAD outside the paragraph (prep.py's sampler)
        for split, fname, npara, n, seed in (("train", PD.SPLIT_FILES["train"][0], 300_000, 100_000, 11),
                                             ("val", PD.SPLIT_FILES["validation"][0], 10**9, 2000, 12)):
            lines = pq.read_table(C.fetch(PD.WIKI, fname)).column("text").to_pylist()
            paras = [x.strip() for x in lines if x.strip() and not x.strip().startswith("=")][:npara]
            ids = [a for a in C.encode(tok, paras) if len(a) >= 2]
            flat, lens = np.concatenate(ids).astype(np.int64), np.array([len(a) for a in ids], np.int64)
            Wn = P.windows(flat, lens, n, seed)
            assert not ((Wn == 1) | (Wn == 2)).any(), "reserved ids in text"
            P.write_ids(os.path.join(W, f"long_{split}.ids"), -np.ones(len(Wn), np.int64), Wn)
        hf = os.path.join(C.DATA, "hf")  # the 5 GB disk: drop the WikiText parquet cache
        for d in os.listdir(hf) if os.path.isdir(hf) else []:
            if "wikitext" in d:
                shutil.rmtree(os.path.join(hf, d), ignore_errors=True)
        say("data: 100k + 2k 129-token windows written")


def pretrained():
    """wait for models/logicae/pt.ltc on the branch; sha256-check it against the manifest"""
    dst = os.path.join(W, "pt.ltc")
    while not os.path.exists(dst):
        subprocess.run(["git", "-C", REPO, "fetch", "-q", "origin", BRANCH], capture_output=True)
        have = subprocess.run(["git", "-C", REPO, "ls-tree", "--name-only", f"origin/{BRANCH}", "models/logicae/"],
                              capture_output=True, text=True).stdout.split()
        if "models/logicae/pt.ltc" in have and "models/logicae/MANIFEST.txt" in have:
            man = subprocess.run(["git", "-C", REPO, "show", f"origin/{BRANCH}:models/logicae/MANIFEST.txt"],
                                 capture_output=True, text=True).stdout
            want = next(l.split()[0] for l in man.splitlines() if l.split()[-1:] == ["pt.ltc"])
            with open(dst + ".tmp", "wb") as f:
                sh(["git", "-C", REPO, "show", f"origin/{BRANCH}:models/logicae/pt.ltc"], stdout=f)
            got = hashlib.sha256(open(dst + ".tmp", "rb").read()).hexdigest()
            if got != want:
                say(f"pt.ltc sha256 mismatch ({got[:16]} vs {want[:16]}); retrying in 5 min")
                os.remove(dst + ".tmp")
            else:
                os.replace(dst + ".tmp", dst)
                say(f"pt.ltc from the branch, sha256 {got[:16]}... verified")
                break
        else:
            PHASE["p"] = "waiting for models/logicae/pt.ltc on the branch"
        time.sleep(300)
    return dst


def lt(args, log):
    PHASE["log"] = log
    with open(log, "w") as f:
        subprocess.run([os.path.join(W, "ltx"), *args], stderr=f, check=True)


def finetune(name, task, extra):
    seq, n = SEQ[task], f"{task}_{name}"
    PHASE["p"] = f"fine-tune {n}"
    log = os.path.join(W, f"{n}.log")
    t0 = time.time()
    lt(["train", "--threads", str(T), "--data", f"{task}_train.ids", "--val", f"{task}_dev.ids", "--format", "ids",
        "--seq", str(seq), "--batch", "32", "--steps", str(FT_STEPS), "--eval-every", "250", "--seed", "17",
        "--save", f"{n}.ltc", "--export", f"{n}.lth", *extra], log)
    r = subprocess.run([os.path.join(W, "ltx"), "eval", "--load", f"{n}.lth", "--data", f"{task}_test.ids", "--format", "ids",
                        "--seq", str(seq), "--batch", "64"], capture_output=True, text=True, check=True)
    acc = float(re.search(r'"hard_accuracy":([0-9.]+)', r.stdout).group(1))
    curve = [(int(s), float(v)) for s, v in re.findall(r'"step":(\d+)[^\n]*"val_hard_accuracy":([0-9.]+)', open(log).read())]
    notes = [l.strip() for l in open(log) if l.startswith(("keep-temperature", "load-part"))]
    res = {"run": n, "task": task, "variant": name, "test": round(acc, 4), "dev_best": max(v for _, v in curve) if curve else None,
           "dev_curve": curve, "run1_pretrained": RUN1[task]["pretrained"], "run1_scratch": RUN1[task]["scratch"],
           "args": " ".join(extra), "notes": notes, "minutes": round((time.time() - t0) / 60, 1)}
    say("RESULT " + json.dumps(res))
    with open(os.path.join(W, "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    for s in (".ltc", ".ltc.best", ".lth.best"):  # the 5 GB disk: keep only the hardened final model
        if os.path.exists(n + s):
            os.remove(n + s)
    return res


def long_pretrain():
    if os.path.exists("pt_long.ltc"):
        return
    PHASE["p"] = f"continued pretraining on 129-token windows ({LONG_STEPS} steps)"
    t0 = time.time()
    lt(["pretrain", "--threads", str(T), "--data", "long_train.ids", "--val", "long_val.ids", "--format", "ids",
        "--load", "pt.ltc", "--keep-temperature", "--seq", "129", "--batch", "32", "--mlm-targets", "512",
        "--steps", str(LONG_STEPS), "--eval-every", str(max(1, LONG_STEPS // 6)), "--seed", "17", "--save", "pt_long.ltc"],
       os.path.join(W, "pt_long.log"))
    curve = re.findall(r'"step":(\d+)[^\n]*"val_mlm_ce":([0-9.]+)', open(os.path.join(W, "pt_long.log")).read())
    say("RESULT " + json.dumps({"run": "pt_long", "val_mlm_ce_129": [(int(s), float(v)) for s, v in curve],
                                "minutes": round((time.time() - t0) / 60, 1)}))
    if os.path.exists("pt_long.ltc.best"):
        os.remove("pt_long.ltc.best")


def main():
    os.makedirs(W, exist_ok=True)
    os.chdir(W)
    threading.Thread(target=heartbeat, daemon=True).start()
    say(f"start ({subprocess.run(['git', '-C', REPO, 'log', '--oneline', '-1'], capture_output=True, text=True).stdout.strip()}); "
        f"threads {T}, fine-tune {FT_STEPS} steps, continued pretraining {LONG_STEPS} steps")
    build()
    data()
    pretrained()
    plan = [("keep", "sst2"), ("keep", "qnli"), ("codes", "sst2"), ("gates", "sst2"), ("long", None), ("long", "sst2"),
            ("long", "qnli"), ("control", "sst2"), ("codes", "qnli"), ("gates", "qnli")]
    args = {"keep": ["--load", "pt.ltc", "--keep-temperature"],
            "codes": ["--load", "pt.ltc", "--load-part", "codes", "--keep-temperature"],
            "gates": ["--load", "pt.ltc", "--load-part", "gates", "--keep-temperature"],
            "long": ["--load", "pt_long.ltc", "--keep-temperature"],
            "control": ["--load", "pt.ltc"]}
    done = set()
    if os.path.exists("results.jsonl"):
        done = {json.loads(l)["run"] for l in open("results.jsonl")}
    for name, task in plan:
        key = f"{task}_{name}" if task else "pt_long"
        if ONLY and key not in ONLY and name not in ONLY:
            continue
        if key in done:
            continue
        try:
            long_pretrain() if task is None else finetune(name, task, args[name])
        except Exception as ex:  # one failed run must not stop the rest
            say(f"FAILED {key}: {type(ex).__name__}: {str(ex)[-300:]}")
    PHASE["p"] = "done"
    say("done")


if __name__ == "__main__":
    main()
