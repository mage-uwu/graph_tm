"""LogicAE scale-up: after run.sh's first run finishes, export it, run a short width / depth sweep,
then continue the best configuration as the long run, ranking every checkpoint on the 20k test.

  python3 scale2x.py            (on the pod; stdlib only; paste-once, it waits for the first run)

Order: wait for <RUN1>/phase == "done" -> make sure run 1 is exported (export.py, or wait for an
export another process is already producing) -> sweep: each config pretrains in <W2>/sweep_<name>
with run 1's full schedule (same --steps, eval interval, LR/temperature schedule, data, seed) but
stops at step SWEEP_STOP (2 evaluations, 1258), so its validation CE is directly comparable with
run 1's curve at that step; its checkpoint is ranked on the 20k test -> the config with the lowest
validation CE at SWEEP_STOP wins -> the winner is resumed (--resume is bit-exact: stop+resume gives
the same checkpoint as an uninterrupted run) in <W2>/long up to the full step count, every
checkpoint kept as ckpt_<step>.ltc and ranked -> final ranking of pt.ltc / pt.ltc.best -> export.py
serves <W2>/long plus the sweep logs for EXPORT_HOURS. Run 1's files are only read.

Sweep (run 1 = --code-bits 128 --width 1024 --blocks 16 --depth 3, 4.40M params):
  w2048      width 2048                 4.86M    b32    blocks 32              4.86M
  d4         tree depth 4               4.94M    c256   code bits 256          8.31M
  c256w2048  code bits 256, width 2048  8.77M
Memory budget from logic_text.c's allocations (model + optimizer, training and validation work
buffers, softmax buffer, data load; plus a concurrent mlmrank): at most ~2.0 GB training / ~3.0 GB
with ranking (c256w2048), vs ~1.25 / 1.7 GB for run 1, whose heartbeat showed 2.4-2.8 GB with
page cache. Checkpoints 50-100 MB.

Safety: a watchdog (every 2 s) stops only the trainer if its peak RSS passes RSS_GATE_GB (6, ~3x the
budget: the budget was wrong, skip that config), if our processes together pass MEM_ABORT_GB (20, a
third of the 60 GB limit), or if free disk drops below 3 GB; checkpoint copies stop below 8 GB free.
A sweep config whose measured speed puts the full run above MAX_LONG_H (16 h) is stopped at its first
evaluation and is not eligible. Status lines go to <W2>/logicae.log and the container log:
  LAE2X ...  ("SWEEP {...}" per config, "PICK ...", "CKPT {...}" per long-run checkpoint, HB every
  5 min, "RESULT {...}" at the end)
Env overrides (tests): RUN1, W2, LAE2X_SWEEP ("name:args;..."), LAE2X_STEPS, LAE2X_EVERY, SWEEP_STOP,
THREADS, RSS_GATE_GB, MEM_ABORT_GB, MAX_LONG_H, EXPORT_HOURS, RUN1_EXPORT_HOURS.
"""
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

RUN1 = os.environ.get("RUN1", "/root/lae")
W2 = os.environ.get("W2", "/root/lae2x")
HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_PY = os.path.join(HERE, "export.py") if os.path.exists(os.path.join(HERE, "export.py")) else "/root/lae_export.py"
COMMON = "--vocab-size 30522 --kernel 5 --cycle 4"
SWEEP = os.environ.get("LAE2X_SWEEP", ";".join([
    f"w2048:{COMMON} --code-bits 128 --width 2048 --blocks 16",
    f"b32:{COMMON} --code-bits 128 --width 1024 --blocks 32",
    f"d4:{COMMON} --code-bits 128 --width 1024 --blocks 16 --depth 4",
    f"c256:{COMMON} --code-bits 256 --width 1024 --blocks 16",
    f"c256w2048:{COMMON} --code-bits 256 --width 2048 --blocks 16"]))
T = int(os.environ.get("THREADS", os.cpu_count() or 1))
RSS_GATE_GB = float(os.environ.get("RSS_GATE_GB", 6))
MAX_LONG_H = float(os.environ.get("MAX_LONG_H", 16))
MEM_ABORT_GB = float(os.environ.get("MEM_ABORT_GB", 20))
EXPORT_HOURS = float(os.environ.get("EXPORT_HOURS", 6))
RUN1_EXPORT_HOURS = float(os.environ.get("RUN1_EXPORT_HOURS", 4))
DATA = ["pre_train.ids", "pre_val.ids", "test_win.ids", "test_true.txt", "candidates.txt"]
GB = 2 ** 30
LOG = None


def say(msg):
    line = f"LAE2X [{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for path in ([LOG] if LOG else []) + ["/proc/1/fd/1"]:
        try:
            with open(path, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


def rss(pid, key="VmRSS"):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for l in fh:
                if l.startswith(key + ":"):
                    return int(l.split()[1]) * 1024
    except OSError:
        pass
    return 0


def cgroup_mem():
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(open(p).read().split()[0])
        except (OSError, ValueError):
            pass
    return 0


def free_gb(path):
    return shutil.disk_usage(path).free / GB


def phase_done(d):
    try:
        return open(os.path.join(d, "phase")).read().strip() == "done"
    except OSError:
        return False


def run1_curve():
    """run 1's validation CE by step, from its log's `curve "step":N  "val_mlm_ce":X` lines"""
    out = {}
    try:
        for l in open(os.path.join(RUN1, "logicae.log"), errors="replace"):
            m = re.match(r'curve "step":(\d+)\s+"val_mlm_ce":([0-9.]+)', l)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    except OSError:
        pass
    return out


def run1_steps():
    try:
        for l in open(os.path.join(RUN1, "logicae.log"), errors="replace"):
            m = re.search(r'"pretrain_steps":(\d+)', l)
            if m:
                return int(m.group(1))
    except OSError:
        pass
    return 12583


def ensure_run1_export():
    """export run 1 once: if another export.py (the earlier one-liner) is waiting or serving, wait
    for its manifest instead of starting a second server on the same port"""
    def manifests():
        return glob.glob(os.path.join(RUN1, "export", "*", "MANIFEST.txt"))
    if manifests():
        say(f"run 1 already exported ({manifests()[0]})")
        return
    other = subprocess.run(["pgrep", "-f", r"export\.py.*--wait|lae_export\.py"], capture_output=True, text=True).stdout.split()
    other = [p for p in other if int(p) != os.getpid()]
    if other:
        say(f"another export process is exporting run 1 (pid {' '.join(other)}); waiting for its manifest")
        for _ in range(60):
            if manifests():
                say(f"run 1 exported ({manifests()[0]})")
                return
            time.sleep(10)
        say("no run 1 manifest after 10 min; exporting here")
    subprocess.run([sys.executable, EXPORT_PY, "--dir", RUN1, "--hours", str(RUN1_EXPORT_HOURS)], check=False)


def watch(proc, extra=(), limit=MEM_ABORT_GB):
    """True if we killed `proc` for memory or disk"""
    peak = rss(proc.pid, "VmHWM")
    if peak > RSS_GATE_GB * GB:
        say(f"ABORT memory gate: trainer peak RSS {peak / GB:.2f} GB > {RSS_GATE_GB} GB (budget ~2 GB); stopping it")
        proc.send_signal(signal.SIGTERM)
        return True
    used = rss(proc.pid) + sum(rss(p.pid) for p in extra if p.poll() is None)
    if used > limit * GB:
        say(f"ABORT memory: our processes use {used / GB:.1f} GB > {limit} GB; stopping the trainer")
        proc.send_signal(signal.SIGTERM)
        return True
    if free_gb(W2) < 3:
        say(f"ABORT disk: {free_gb(W2):.1f} GB free; stopping the trainer")
        proc.send_signal(signal.SIGTERM)
        return True
    return False


def rank_start(model, threads):
    return subprocess.Popen(["nice", "-n", "10", "./mlmrank", model, "test_win.ids", "test_true.txt", "candidates.txt",
                             "8", str(threads)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def rank_result(p):
    out, err = p.communicate()
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (err or out).strip()[-300:]}


def rank(model, threads):
    return rank_result(rank_start(model, threads))


def setup_dir(d):
    os.makedirs(d, exist_ok=True)
    for f in DATA:
        if not os.path.lexists(os.path.join(d, f)):
            os.symlink(os.path.join(RUN1, f), os.path.join(d, f))
    for b in ("lt", "mlmrank"):
        shutil.copy(os.path.join(RUN1, b), os.path.join(d, b))


def train(d, label, arch, steps, every, stop=None, resume=None, curve1=None, rank_ckpts=False, gate_sps=False):
    """one pretraining process in directory d under the watchdog; returns a summary dict"""
    os.chdir(d)
    P = ["--threads", str(T), "--data", "pre_train.ids", "--val", "pre_val.ids", "--format", "ids", *arch,
         "--seq", "17", "--batch", "256", "--mlm-targets", "512", "--steps", str(steps), "--eval-every", str(every),
         "--save", "pt.ltc"]
    P += ["--resume", resume] if resume else ["--seed", "17"]
    P += ["--stop-after", str(stop)] if stop else []
    err = open("current.log", "w")
    tr = subprocess.Popen(["./lt", "pretrain", *P], stderr=err)
    say(f"{label}: pretrain started (pid {tr.pid}){' resuming ' + resume if resume else ''}, "
        f"to step {stop or steps} of {steps}")
    seen, ranker, pending, killed, why, last_hb, maxrss, last = set(), None, [], False, "", time.time(), 0, None
    while True:
        alive = tr.poll() is None
        if alive and not killed:
            killed = watch(tr, [ranker[0]] if ranker else ())
            why = "watchdog" if killed else ""
            maxrss = max(maxrss, rss(tr.pid, "VmHWM"))
        for l in open("current.log", errors="replace"):
            if l.startswith('{"step"') and "val_mlm_ce" in l:
                e = json.loads(l)
                if e["step"] not in seen:
                    seen.add(e["step"])
                    last = e
                    sps = e["compute_seconds"] / max(1, e["step"] - (min(seen) - every if resume else 0))
                    if gate_sps and alive and not killed and steps * sps / 3600 > MAX_LONG_H:
                        say(f"{label}: {sps:.2f} s/step -> full run {steps * sps / 3600:.1f} h > {MAX_LONG_H} h; "
                            f"stopping (not eligible)")
                        tr.send_signal(signal.SIGTERM)
                        killed, why = True, "too slow"
                    if rank_ckpts:
                        ck = f"ckpt_{e['step']}.ltc"
                        if free_gb(d) > 8:
                            shutil.copy("pt.ltc", ck + ".tmp")
                            os.replace(ck + ".tmp", ck)
                            pending.append((e, ck))
                        else:
                            say(f"low disk ({free_gb(d):.1f} GB): step {e['step']} not snapshotted")
        if ranker is not None and ranker[0].poll() is not None:
            p, e, ck = ranker
            ranker = None
            say("CKPT " + json.dumps({"run": label, "step": e["step"], "val_mlm_ce": e["val_mlm_ce"],
                                      "run1_val_mlm_ce": (curve1 or {}).get(e["step"]), "test20k": rank_result(p),
                                      "wall_h": round(e["wall_seconds"] / 3600, 2)}))
        if ranker is None and pending:
            e, ck = pending.pop(0)
            ranker = (rank_start(ck, max(1, T // 2)), e, ck)
        if time.time() - last_hb > 300:
            last_hb = time.time()
            say(f"HB {label} {'training' if alive else 'finished'} | last eval step {max(seen) if seen else 0}/"
                f"{stop or steps} | trainer RSS {rss(tr.pid) / GB:.2f} GB (peak {maxrss / GB:.2f}) | "
                f"cgroup {cgroup_mem() / GB:.1f} GB | disk free {free_gb(d):.1f} GB")
        if not alive and not pending and ranker is None:
            break
        time.sleep(2)
    err.close()
    tail = open("current.log", errors="replace").read()[-300:] if tr.returncode not in (0, -15) else ""
    say(f"{label}: exited ({tr.returncode}){' - ' + why if why else ''}; peak trainer RSS {maxrss / GB:.2f} GB"
        f"{'; ' + tail if tail else ''}")
    first = min(seen) if seen else 0
    return {"ok": tr.returncode == 0 and not killed, "why": why or (f"exit {tr.returncode}" if tr.returncode else ""),
            "last": last, "peak_rss_gb": round(maxrss / GB, 2),
            "s_per_step": round(last["compute_seconds"] / max(1, last["step"] - (first - every if resume else 0)), 3)
            if last else None}


def main():
    global LOG
    say(f"waiting for run 1 ({RUN1}/phase == done)")
    while not phase_done(RUN1):
        time.sleep(60)
    ensure_run1_export()

    os.makedirs(W2, exist_ok=True)
    LOG = os.path.join(W2, "logicae.log")
    steps = int(os.environ.get("LAE2X_STEPS", run1_steps()))
    every = int(os.environ.get("LAE2X_EVERY", max(1, steps // 20)))
    stop = int(os.environ.get("SWEEP_STOP", 2 * every))
    curve1 = run1_curve()
    configs = [(c.split(":", 1)[0].strip(), c.split(":", 1)[1].split()) for c in SWEEP.split(";") if ":" in c]
    say(f"setup: {len(configs)} sweep configs to step {stop} of {steps} (eval every {every}), threads {T}; "
        f"run 1 val CE at {stop}: {curve1.get(stop)}; disk free {free_gb(W2):.1f} GB; cgroup {cgroup_mem() / GB:.1f} GB")

    results = {}
    for name, arch in configs:
        d = os.path.join(W2, f"sweep_{name}")
        setup_dir(d)
        r = train(d, f"sweep {name}", arch, steps, every, stop=stop, curve1=curve1, gate_sps=True)
        if r["ok"] and r["last"] and r["last"]["step"] == stop:
            r["test20k"] = rank("pt.ltc", T)
        params = re.search(r"params=(\d+)", open("current.log", errors="replace").read())
        r["params"] = int(params.group(1)) if params else None
        results[name] = r
        say("SWEEP " + json.dumps({"config": name, "arch": " ".join(arch), "params": r["params"], "ok": r["ok"],
                                   "why": r["why"], "val_mlm_ce": r["last"] and r["last"]["val_mlm_ce"],
                                   "step": r["last"] and r["last"]["step"], "run1_val_mlm_ce": curve1.get(stop),
                                   "s_per_step": r["s_per_step"],
                                   "full_run_h": r["s_per_step"] and round(steps * r["s_per_step"] / 3600, 1),
                                   "peak_rss_gb": r["peak_rss_gb"], "test20k": r.get("test20k")}))

    ok = {n: r for n, r in results.items() if r["ok"] and r["last"] and r["last"]["step"] == stop}
    if not ok:
        say("PICK none: no sweep config finished; no long run")
    else:
        win = min(ok, key=lambda n: ok[n]["last"]["val_mlm_ce"])
        b1 = curve1.get(stop)
        say("PICK " + win + ": val CE at step " + str(stop) + " " + ", ".join(
            f"{n} {r['last']['val_mlm_ce']:.4f}" for n, r in sorted(ok.items(), key=lambda x: x[1]["last"]["val_mlm_ce"]))
            + (f"; run 1 {b1:.4f} ({'beaten' if ok[win]['last']['val_mlm_ce'] < b1 else 'NOT beaten'})" if b1 else ""))
        d = os.path.join(W2, "long")
        setup_dir(d)
        for f in ("pt.ltc", "pt.ltc.best"):
            src = os.path.join(W2, f"sweep_{win}", f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, f))
        arch = dict(configs)[win]
        r = train(d, f"long {win}", arch, steps, every, resume="pt.ltc", curve1=curve1, rank_ckpts=True)
        for f in ("pt.ltc", "pt.ltc.best"):
            if os.path.exists(f):
                say("RESULT " + json.dumps({"model": f"logicae-{win}-{f}", "arch": " ".join(arch),
                                            "test20k": rank(f, T)}))
        for n in results:  # the sweep logs travel with the export
            src = os.path.join(W2, f"sweep_{n}", "current.log")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, f"sweep_{n}.log"))
        if EXPORT_HOURS > 0:
            shutil.copy(LOG, os.path.join(d, "logicae.log"))
            subprocess.run([sys.executable, EXPORT_PY, "--dir", d, "--hours", str(EXPORT_HOURS)], check=False)
    say("done")


if __name__ == "__main__":
    main()
