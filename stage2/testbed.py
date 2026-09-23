"""
Testbed: run queued pretraining experiments to p-collapse and report comparable curves.

p-collapse = 2,000,000 training windows: the first checkpoint of the first full run (T=50000,
dense codes, node types, +-1/2/4 edges) where it clearly failed: WikiText-103 validation per-bit
0.6887 -> 0.6779 (context-free prior 0.689) and acc@10 0.2377 -> 0.2237, 2026-09-23 21:14:59 UTC.
Every experiment trains to P_COLLAPSE windows and is evaluated at EVAL_AT on the same 10k
validation windows, so curves line up with that run.

Queue: stage2/experiments.tsv, one experiment per line (tab-separated; # comments):
    name <TAB> data env (K=V;K=V or -) <TAB> gtm config
The data env selects the graph construction (GTM_DISTS, GTM_LAYOUT, GTM_FLAT_OFFSETS,
GTM_HV_BITS); shards are cached per data env under DATA/tb/data/<key>/. Results go to stdout as
`TB {json}` lines and to DATA/tb/results.jsonl.

  python3 testbed.py next --threads N     claim and run the next pending experiment (exit 3: none)
  python3 testbed.py run NAME --threads N run one experiment by name
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import common as C

P_COLLAPSE = 2_000_000
SHARD = 250_000
EVAL_AT = (250_000, 500_000, 1_000_000, 1_500_000, 2_000_000)
TB = os.path.join(C.DATA, "tb")
if os.environ.get("TB_SMOKE"):  # tiny end-to-end check of the harness
    P_COLLAPSE, SHARD, EVAL_AT, TB = 10_000, 5_000, (5_000, 10_000), os.path.join(C.DATA, "tb_smoke")
QUEUE = os.path.join(C.HERE, "experiments.tsv")


def queue():
    out = []
    for line in open(QUEUE):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, denv, cfg = [x.strip() for x in line.split("\t")][:3]
        env = {} if denv in ("", "-") else dict(kv.split("=", 1) for kv in denv.split(";"))
        out.append({"name": name, "env": env, "cfg": cfg.split()})
    return out


def data_key(env):
    return "_".join(f"{k.replace('GTM_', '').lower()}-{v.replace(',', '.').replace(':', '.')}"
                    for k, v in sorted(env.items())) or "default"


def py(args, env):
    e = dict(os.environ, **env)
    r = subprocess.run([sys.executable] + args, cwd=C.HERE, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{args} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def ensure_data(env):
    d = os.path.join(TB, "data", data_key(env))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for i in range(1, P_COLLAPSE // SHARD + 1):
            p = os.path.join(d, f"s{i}.gtmd")
            if not os.path.exists(p):
                py(["pretrain_data.py", "shard", "--split", "train", "--n", str(SHARD), "--seed", str(i),
                    "--out", p + ".tmp"], env)
                os.replace(p + ".tmp", p)
        v = os.path.join(d, "val.gtmd")
        if not os.path.exists(v + ".npz"):
            py(["pretrain_data.py", "evalset", "--split", "validation", "--n", "10000", "--out", v], env)
    return d


def clause_stats(model):
    from gtmcore import load_model
    cfg, hv, w, ta, step = load_model(model)
    thr = 1 << (cfg.B - 1)
    groups = {"centre": np.arange(cfg.C) % 2 == 1, "context": np.arange(cfg.C) % 2 == 0} if cfg.NT == 2 \
        else {"all": np.ones(cfg.C, bool)}
    st = {}
    for l, t in enumerate(ta):
        inc = t >= thr
        n = inc.shape[1] // 2
        for g, m in groups.items():
            st[f"L{l}_{g}_pos"] = round(float(inc[m, :n].sum(1).mean()), 1)
            st[f"L{l}_{g}_neg"] = round(float(inc[m, n:].sum(1).mean()), 1)
    st["w_abs_median"] = float(np.median(np.abs(w)))
    st["w_abs_max"] = int(np.abs(w).max())
    return st


def run(exp, threads):
    name, env = exp["name"], exp["env"]
    wd = os.path.join(TB, "runs", name)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    gtm = os.path.join(wd, "gtm")
    shutil.copy(C.GTM, gtm)  # the sync loop may rebuild stage1/gtm while we run
    t0 = time.time()
    d = ensure_data(env)
    print(f"== [{time.strftime('%H:%M:%S')}] {name}: data {data_key(env)} ready ({time.time() - t0:.0f}s), "
          f"cfg {' '.join(exp['cfg'])}, engine {rev}", flush=True)
    model = os.path.join(wd, "m.gtmm")
    subprocess.run([gtm, "init", "--data", os.path.join(d, "s1.gtmd"), "--out", model] + exp["cfg"],
                   check=True, capture_output=True)
    done, train_s = 0, 0.0
    for i in range(1, P_COLLAPSE // SHARD + 1):
        t1 = time.time()
        subprocess.run([gtm, "train", "--data", os.path.join(d, f"s{i}.gtmd"), "--model", model, "--epochs", "1",
                        "--threads", str(threads), "--save", model], check=True, capture_output=True)
        dt = time.time() - t1
        train_s += dt
        done += SHARD
        if done in EVAL_AT:
            out = py(["eval_mlm.py", "--model", model, "--evalset", os.path.join(d, "val.gtmd"), "--threads",
                      str(threads), "--tag", name], dict(env, GTM_STAGE2_DATA=C.DATA))
            res = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][0][7:])
            res.update(exp=name, windows=done, ex_per_s=round(SHARD / dt), train_s=round(train_s),
                       data=data_key(env), cfg=" ".join(exp["cfg"]), engine=rev, **clause_stats(model))
            line = json.dumps(res)
            print("TB " + line, flush=True)
            print(f"   {name} @{done // 1000}k: acc@1 {res['acc@1']:.4f} acc@10 {res['acc@10']:.4f} "
                  f"mrr {res['mrr']:.4f} per-bit {res['per-bit']:.4f} {res['fill']} | {res['ex_per_s']} ex/s", flush=True)
            with open(os.path.join(TB, "results.jsonl"), "a") as f:
                f.write(line + "\n")


def claim(name):
    os.makedirs(os.path.join(TB, "claimed"), exist_ok=True)
    try:
        os.close(os.open(os.path.join(TB, "claimed", name), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["next", "run"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    q = queue()
    if a.cmd == "run":
        return run(next(e for e in q if e["name"] == a.name), a.threads)
    for e in q:
        if claim(e["name"]):
            try:
                run(e, a.threads)
                print(f"== [{time.strftime('%H:%M:%S')}] {e['name']}: done", flush=True)
            except Exception as ex:  # keep the worker alive; the failure is in the log
                print(f"== {e['name']}: FAILED {ex}", flush=True)
            return
    sys.exit(3)


if __name__ == "__main__":
    main()
