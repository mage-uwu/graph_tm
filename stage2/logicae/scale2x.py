"""LogicAE 2x-scale follow-on: after run.sh's first run finishes, export it, then pretrain a model
with twice the parameters on the same data and schedule, ranking every checkpoint on the 20k test.

  python3 scale2x.py            (on the pod; stdlib only; paste-once, it waits for the first run)

Order: wait for <RUN1>/phase == "done" -> make sure run 1 is exported (export.py, or wait for an
export another process is already producing) -> build a separate work dir <W2> (run 1's files are
only read) -> 4-step probe with a memory gate -> pretraining with the same steps / eval interval /
LR schedule as run 1 -> at each evaluation the checkpoint is kept as ckpt_<step>.ltc and ranked on
the 20k test windows (mlmrank), next to run 1's validation CE at the same step -> final ranking of
pt.ltc and pt.ltc.best -> export.py serves <W2> for EXPORT_HOURS.

2x = --code-bits 256 --width 2048 (run 1: 128 / 1024), blocks, kernel, cycle, batch, seq, MLM
targets and seed unchanged: 8.77M parameters vs 4.40M. Memory budget from logic_text.c's
allocations: training peak (model + optimizer, training and validation work buffers, softmax
buffer, data load) ~2.1 GB vs ~1.3 GB for run 1, whose heartbeat showed 2.4-2.8 GB including page
cache; mlmrank on a checkpoint ~0.9 GB; checkpoints ~100 MB each.

Safety: the probe aborts if the trainer's peak RSS exceeds PROBE_MAX_GB (6). During the run a
watchdog kills the trainer (and never anything else) if the RSS of our processes exceeds MEM_ABORT_GB
(20, a third of the 60 GB limit) or free disk drops below 3 GB; checkpoint copies stop below 8 GB
free. Nothing in <RUN1> is written. All status lines go to <W2>/logicae.log and the container log:
  LAE2X ...  (HB every 5 min, "LAE2X CKPT {...}" per checkpoint, "LAE2X RESULT {...}" at the end)
Env overrides (for tests): RUN1, W2, LAE2X_ARCH, LAE2X_STEPS, LAE2X_EVERY, THREADS, PROBE_MAX_GB,
MEM_ABORT_GB, EXPORT_HOURS, RUN1_EXPORT_HOURS.
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
ARCH = os.environ.get("LAE2X_ARCH", "--vocab-size 30522 --code-bits 256 --width 2048 --blocks 16 --kernel 5 --cycle 4").split()
T = int(os.environ.get("THREADS", os.cpu_count() or 1))
PROBE_MAX_GB = float(os.environ.get("PROBE_MAX_GB", 6))
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


def main():
    global LOG
    say(f"waiting for run 1 ({RUN1}/phase == done)")
    while not phase_done(RUN1):
        time.sleep(60)
    ensure_run1_export()

    os.makedirs(W2, exist_ok=True)
    LOG = os.path.join(W2, "logicae.log")
    os.chdir(W2)
    for f in DATA:
        if not os.path.exists(f):
            os.symlink(os.path.join(RUN1, f), f)
    for b in ("lt", "mlmrank"):
        shutil.copy(os.path.join(RUN1, b), b)
    steps = int(os.environ.get("LAE2X_STEPS", run1_steps()))
    every = int(os.environ.get("LAE2X_EVERY", max(1, steps // 20)))
    curve1 = run1_curve()
    P = ["--threads", str(T), "--data", "pre_train.ids", "--val", "pre_val.ids", "--format", "ids", *ARCH,
         "--seq", "17", "--batch", "256", "--mlm-targets", "512", "--seed", "17"]
    say(f"setup: {' '.join(ARCH)}; steps {steps}, eval every {every}, threads {T}; "
        f"disk free {free_gb(W2):.1f} GB; cgroup mem {cgroup_mem() / GB:.1f} GB")

    # probe: 4 steps + one validation pass, peak RSS gate
    say("probe (4 steps + validation)")
    with open("probe.log", "w") as err:
        p = subprocess.Popen(["./lt", "pretrain", *P, "--steps", str(steps), "--stop-after", "4", "--eval-every", "4",
                              "--save", "probe.ltc"], stderr=err)
        peak = 0
        while p.poll() is None:
            peak = max(peak, rss(p.pid, "VmHWM"))
            if peak > PROBE_MAX_GB * GB:
                p.kill()
                say(f"ABORT probe: peak RSS {peak / GB:.2f} GB > {PROBE_MAX_GB} GB budget; no 2x run")
                return
            time.sleep(0.2)
    m = re.findall(r'"compute_seconds":([0-9.]+)', open("probe.log").read())
    if p.returncode != 0 or not m:
        say(f"ABORT probe failed (exit {p.returncode}): {open('probe.log').read()[-400:]}")
        return
    sps = float(m[-1]) / 4
    params = re.search(r"params=(\d+)", open("probe.log").read())
    say(f"probe ok: {sps:.2f} s/step, peak RSS {peak / GB:.2f} GB, params {params.group(1) if params else '?'}; "
        f"ETA {steps * sps / 3600:.1f} h")
    for f in glob.glob("probe.ltc*"):
        os.remove(f)

    # main run
    err = open("current.log", "w")
    tr = subprocess.Popen(["./lt", "pretrain", *P, "--steps", str(steps), "--eval-every", str(every), "--save", "pt.ltc"],
                          stderr=err)
    say(f"pretrain started (pid {tr.pid})")
    seen, ranker, pending, killed, last_hb, maxrss = set(), None, [], False, 0, 0
    while True:
        alive = tr.poll() is None
        if alive and not killed:
            killed = watch(tr, [ranker[0]] if ranker else ())
            maxrss = max(maxrss, rss(tr.pid))
        for l in open("current.log", errors="replace"):
            if l.startswith('{"step"') and "val_mlm_ce" in l:
                d = json.loads(l)
                if d["step"] not in seen:
                    seen.add(d["step"])
                    ck = f"ckpt_{d['step']}.ltc"
                    if free_gb(W2) > 8:
                        shutil.copy("pt.ltc", ck + ".tmp")
                        os.replace(ck + ".tmp", ck)
                        pending.append((d, ck))
                    else:
                        say(f"low disk ({free_gb(W2):.1f} GB): step {d['step']} not snapshotted")
        if ranker is not None and ranker[0].poll() is not None:
            p, d, ck = ranker
            ranker = None
            say("CKPT " + json.dumps({"step": d["step"], "val_mlm_ce": d["val_mlm_ce"],
                                      "run1_val_mlm_ce": curve1.get(d["step"]), "test20k": rank_result(p),
                                      "wall_h": round(d["wall_seconds"] / 3600, 2)}))
        if ranker is None and pending:
            d, ck = pending.pop(0)
            ranker = (rank_start(ck, max(1, T // 2)), d, ck)
        if time.time() - last_hb > 300:
            last_hb = time.time()
            say(f"HB {'training' if alive else 'finished'} | last eval step {max(seen) if seen else 0}/{steps} | "
                f"trainer RSS {rss(tr.pid) / GB:.2f} GB (max {maxrss / GB:.2f}) | cgroup {cgroup_mem() / GB:.1f} GB | "
                f"disk free {free_gb(W2):.1f} GB")
        if not alive and not pending and ranker is None:
            break
        time.sleep(2)
    err.close()
    say(f"pretrain exited ({tr.returncode}){' after watchdog abort' if killed else ''}; max trainer RSS {maxrss / GB:.2f} GB")
    for f in ("pt.ltc", "pt.ltc.best"):
        if os.path.exists(f):
            say("RESULT " + json.dumps({"model": f"logicae2x-{f}", "test20k": rank(f, T)}))
    if EXPORT_HOURS > 0:
        subprocess.run([sys.executable, EXPORT_PY, "--dir", W2, "--hours", str(EXPORT_HOURS)], check=False)
    say("done")


if __name__ == "__main__":
    main()
