"""
Testbed: run queued pretraining experiments to p-collapse and report comparable curves.

p-collapse = 2,000,000 training windows: the first checkpoint of the first full run (T=50000,
dense codes, node types, +-1/2/4 edges) where it clearly failed: WikiText-103 validation per-bit
0.6887 -> 0.6779 (context-free prior 0.689) and acc@10 0.2377 -> 0.2237, 2026-09-23 21:14:59 UTC.
Every experiment trains to P_COLLAPSE windows and is evaluated at EVAL_AT on the same 10k
validation windows, so curves line up with that run.

Options in the config (not passed to the engine): @ri=K residual init; @rank=K [@margin=M]
ranking output feedback (engine --rank, negatives from neg_table(), M defaults to T); @windows=N train N windows
instead of P_COLLAPSE (validation every 1M after 2M); @full=1 then run the proper run's downstream
half (20k-window test, SST-2 / QNLI / CoNLL adapters, bert-tiny baselines unless @baselines=0);
@best=1 runs it on the best validated checkpoint; @centre=word adds word-visible adapter features.

Queue: stage2/experiments.tsv, one experiment per line (tab-separated; # comments):
    name <TAB> data env (K=V;K=V or -) <TAB> gtm config
The data env selects the graph construction (GTM_DISTS, GTM_LAYOUT, GTM_FLAT_OFFSETS,
GTM_HV_BITS); shards are cached per data env under DATA/tb/data/<key>/. Results go to stdout as
`TB {json}` lines and to DATA/tb/results.jsonl.

  python3 testbed.py next --threads N     claim and run the next pending experiment (exit 3: none)
  python3 testbed.py run NAME --threads N run one experiment by name
"""
import argparse
import hashlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
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


HEARTBEAT_S = float(os.environ.get("TB_HEARTBEAT_S", 30))
STATE = {}  # what the running experiment is doing; the heartbeat thread prints it


def phase(**kw):
    STATE.update(kw, since=time.time())


def heartbeat():
    """every HEARTBEAT_S seconds one HB line on stdout (the pod log) and in TB/heartbeat.txt:
    experiment, phase, progress, last throughput, time in phase, disk / memory / load. A run
    that dies stops producing them; a stuck one shows its phase time growing."""
    while True:
        time.sleep(HEARTBEAT_S)
        if not STATE:
            continue
        du = shutil.disk_usage(TB)
        mem = next((int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")), 0)
        st = dict(STATE)
        line = (f"HB [{time.strftime('%H:%M:%S')}] {st.get('exp')} {st.get('phase')} "
                f"| {st.get('done', 0) / 1e6:.2f}M/{st.get('total', 0) / 1e6:.2f}M windows"
                f"{' | last ' + str(st['ex_s']) + ' ex/s' if st.get('ex_s') else ''}"
                f"{' | val acc@10 ' + format(st['acc10'], '.4f') if st.get('acc10') is not None else ''}"
                f" | in phase {time.time() - st['since']:.0f}s | up {(time.time() - st['t0']) / 60:.1f}m"
                f" | disk free {du.free / 2**30:.1f}G | mem avail {mem / 2**20:.1f}G | load {os.getloadavg()[0]:.1f}")
        print(line, flush=True)
        try:
            with open(os.path.join(TB, "heartbeat.txt"), "w") as f:
                f.write(line + "\n")
        except OSError:
            pass


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


def ensure_data(env, test=False):
    d = os.path.join(TB, "data", data_key(env))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for i in range(1, P_COLLAPSE // SHARD + 1):
            p = os.path.join(d, f"s{i}.gtmd")
            if not os.path.exists(p):
                phase(phase=f"data: cached shard {i}/{P_COLLAPSE // SHARD}")
                py(["pretrain_data.py", "shard", "--split", "train", "--n", str(SHARD), "--seed", str(i),
                    "--out", p + ".tmp"], env)
                os.replace(p + ".tmp", p)
        v = os.path.join(d, "val.gtmd")
        if not os.path.exists(v + ".npz"):
            phase(phase="data: validation set")
            py(["pretrain_data.py", "evalset", "--split", "validation", "--n", "10000", "--out", v], env)
        t = os.path.join(d, "test20k.gtmd")
        if test and not os.path.exists(t + ".npz"):
            phase(phase="data: 20k test set")
            py(["pretrain_data.py", "evalset", "--split", "test", "--n", "20000", "--out", t], env)
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


def node_diag(model, val, n_graphs=100):
    """oracle pass over validation graphs: message fill at the centre vs context nodes per layer,
    senders per context node, and the fraction of centre clauses still true after each layer"""
    from gtm_oracle import OracleGTM
    from gtmcore import Dataset
    m = OracleGTM.load(model)
    c = m.cfg
    if c.NT != 2 or c.D < 2:
        return {}
    ds = Dataset.load(val)
    odd = np.arange(c.C) % 2 == 1
    send = (m.ta[0][:, :c.H] >= m.half).any(1) if c.senders else np.ones(c.C, bool)
    acc = {}
    for g in range(min(n_graphs, ds.n_graphs)):
        n0, n1 = ds.node_offset[g], ds.node_offset[g + 1]
        types = ds.node_type[n0:n1]
        cen, ctx = np.nonzero(types == 1)[0][0], types == 0
        typeok = types[None, :] == (np.arange(c.C) % 2)[:, None]
        X = ds.X[n0:n1]
        out = typeok & ~(((m.ta[0] >= m.half).astype(np.int32) @ (~X).astype(np.int32).T) > 0)
        vals = {"senders_ctx": (out[:, ctx] & send[:, None]).sum(0).mean(), "centre_true_L0": out[odd, cen].mean()}
        for d in range(1, c.D):
            Xm = m._messages(ds, n0, n1 - n0, out)
            vals[f"fill{d}_centre"] = Xm[cen, :c.MS].mean()
            vals[f"fill{d}_ctx"] = Xm[ctx, :c.MS].mean()
            out = out & typeok & ~(((m.ta[d] >= m.half).astype(np.int32) @ (~Xm).astype(np.int32).T) > 0)
            vals[f"centre_true_L{d}"] = out[odd, cen].mean()
        for k, v in vals.items():
            acc.setdefault(k, []).append(float(v))
    return {k: round(float(np.mean(v)), 3) for k, v in acc.items()}


def residual_init(model, k, layers="msg"):
    """Tsetlin-native residual initialization: start the automata of message layers (d >= 1) k
    states below the include threshold instead of one, so every clause starts as its layer-0
    part with pass-through deeper layers, and a message literal is included only after ~k net
    reinforcements. Pure initial state: the engine and its semantics are unchanged."""
    from gtmcore import load_model, save_model
    cfg, hv, w, ta, step = load_model(model)
    half = 1 << (cfg.B - 1)
    for d in range(1, cfg.D):
        ta[d][:] = max(0, half - k)
    save_model(model, cfg, hv, w, ta, step)


def neg_table(n=100_000, seed=0):
    """negative code table for --rank: target codes of n tokens drawn from the real candidates
    with probability ~ (train frequency + 1)^0.75 (word2vec's negative-sampling law)"""
    p = os.path.join(C.DATA, f"negtable_{n}_{seed}.u8")
    if not os.path.exists(p):
        import eval_mlm as E
        codes, cand, freq, _ = E.context()
        w = (freq[cand].astype(np.float64) + 1) ** 0.75
        ids = np.random.default_rng(seed).choice(cand, size=n, p=w / w.sum())
        codes[ids].astype(np.uint8).tofile(p + ".tmp")
        os.replace(p + ".tmp", p)
    return p


def run(exp, threads):
    name, env = exp["name"], exp["env"]
    wd = os.path.join(TB, "runs", name)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    open(os.path.join(wd, "RUNNING"), "w").close()  # gc keeps this run's data
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    gtm = os.path.join(wd, "gtm")
    shutil.copy(C.GTM, gtm)  # the sync loop may rebuild stage1/gtm while we run
    t0 = time.time()
    opts = dict(a[1:].split("=", 1) for a in exp["cfg"] if a.startswith("@"))
    windows = int(float(opts.get("windows", P_COLLAPSE)))
    STATE.clear()
    STATE.update(exp=name, total=windows, done=0, t0=time.time())
    phase(phase="setup")
    eval_at = set(EVAL_AT) | set(range(P_COLLAPSE, windows + 1, 1_000_000))
    if "full" in opts or env.get("GTM_CODES") == "dist":
        deps()  # adapters / bert baselines, and the learned codes' SVD (scikit-learn)
    d = ensure_data(env, test="full" in opts)
    print(f"== [{time.strftime('%H:%M:%S')}] {name}: data {data_key(env)} ready ({time.time() - t0:.0f}s), "
          f"cfg {' '.join(exp['cfg'])}, engine {rev}", flush=True)
    model = os.path.join(wd, "m.gtmm")
    cfg = [a for a in exp["cfg"] if not a.startswith("@")]
    subprocess.run([gtm, "init", "--data", os.path.join(d, "s1.gtmd"), "--out", model] + cfg,
                   check=True, capture_output=True)
    if "ri" in opts:
        residual_init(model, int(opts["ri"]))
    rank = ["--rank", opts["rank"], "--rank-margin", opts.get("margin", "0"), "--neg-table", neg_table()] \
        if "rank" in opts else []
    done, train_s, best_acc = 0, 0.0, -1.0
    ncached, nshards = P_COLLAPSE // SHARD, windows // SHARD
    pending = {}  # shard index -> (Popen, path): the next streamed shard is built while one trains

    def stream(j):
        if ncached < j <= nshards and j not in pending:
            out = os.path.join(wd, f"stream{j}.gtmd")
            pending[j] = (subprocess.Popen([sys.executable, "pretrain_data.py", "shard", "--split", "train",
                                            "--n", str(SHARD), "--seed", str(j), "--out", out], cwd=C.HERE,
                                           env=dict(os.environ, **env), stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, text=True), out)

    for i in range(1, nshards + 1):
        shard = os.path.join(d, f"s{i}.gtmd")
        stream(i)  # beyond the cached shards (~590 MB each): streamed, prefetched one ahead
        if i in pending:
            proc, shard = pending.pop(i)
            phase(phase=f"data: stream shard {i}/{nshards}")
            if proc.wait() != 0:
                raise RuntimeError(f"shard {i} failed:\n{proc.stderr.read()}")
        stream(i + 1)
        phase(phase=f"train shard {i}/{nshards}")
        t1 = time.time()
        subprocess.run([gtm, "train", "--data", shard, "--model", model, "--epochs", "1",
                        "--threads", str(threads), "--save", model] + rank, check=True, capture_output=True)
        dt = time.time() - t1
        if shard.startswith(wd):
            os.remove(shard)
        train_s += dt
        done += SHARD
        STATE.update(done=done, ex_s=round(SHARD / dt))
        if done in eval_at:
            phase(phase=f"validation @{done // 1000}k")
            out = py(["eval_mlm.py", "--model", model, "--evalset", os.path.join(d, "val.gtmd"), "--threads",
                      str(threads), "--tag", name], dict(env, GTM_STAGE2_DATA=C.DATA))
            res = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][0][7:])
            res.update(exp=name, windows=done, ex_per_s=round(SHARD / dt), train_s=round(train_s),
                       data=data_key(env), cfg=" ".join(exp["cfg"]), engine=rev, **clause_stats(model),
                       **node_diag(model, os.path.join(d, "val.gtmd")))
            line = json.dumps(res)
            print("TB " + line, flush=True)
            print(f"   {name} @{done // 1000}k: acc@1 {res['acc@1']:.4f} acc@10 {res['acc@10']:.4f} "
                  f"mrr {res['mrr']:.4f} per-bit {res['per-bit']:.4f} {res['fill']} | centre fill "
                  f"{[res.get(f'fill{k}_centre') for k in range(1, 8) if f'fill{k}_centre' in res]} | {res['ex_per_s']} ex/s",
                  flush=True)
            with open(os.path.join(TB, "results.jsonl"), "a") as f:
                f.write(line + "\n")
            STATE["acc10"] = res["acc@10"]
            if res["acc@10"] > best_acc:  # the downstream half uses the best validated model (@best=1)
                best_acc = res["acc@10"]
                shutil.copy(model, os.path.join(wd, "best.gtmm"))
            if windows > P_COLLAPSE:  # long runs: keep the last two checkpoints to resume from
                shutil.copy(model, os.path.join(wd, f"ckpt_{done // 1000}k.gtmm"))
                for old in sorted((f for f in os.listdir(wd) if f.startswith("ckpt_")),
                                  key=lambda f: int(f[5:-6]))[:-2]:
                    os.remove(os.path.join(wd, old))
    if "full" in opts:
        if opts.get("best") == "1" and os.path.exists(os.path.join(wd, "best.gtmm")):
            model = os.path.join(wd, "best.gtmm")
            print(f"== {name}: downstream uses the best validated model (acc@10 {best_acc:.4f})", flush=True)
        full(name, env, model, d, threads, opts.get("baselines", "1") == "1", opts.get("centre", "mask"))


def full(name, env, model, d, threads, baselines=True, centre="mask"):
    """the proper run's downstream half, on the trained model: masked-token test (20k WikiText-103
    test windows, same windows bert-tiny is scored on), then frozen-feature adapters on SST-2 /
    QNLI / CoNLL-2003 next to the lexical baseline and bert-tiny frozen / fine-tuned"""
    e = dict(env, GTM_STAGE2_DATA=C.DATA)
    t = os.path.join(d, "test20k.gtmd")
    steps = [["eval_mlm.py", "--evalset", t, "--baselines"], ["bert_baselines.py", "mlm", "--evalset", t]] * baselines
    steps += [["eval_mlm.py", "--model", model, "--evalset", t, "--threads", str(threads), "--tag", name + "-test"]]
    for task in ("sst2", "qnli", "conll"):
        for c in (["mask", "word"] if centre == "word" else ["mask"]):  # word models: both views
            steps += [["adapters.py", "--task", task, "--features", f, "--model", model, "--tag", f"{name}-{c}",
                       "--threads", str(threads), "--centre", c] for f in ("gtm", "gtm+bow")]
        steps += [["adapters.py", "--task", task, "--features", "bow", "--tag", name + "-lexical", "--threads", str(threads)],
                  ["bert_baselines.py", "task", "--task", task, "--mode", "frozen"],
                  ["bert_baselines.py", "task", "--task", task, "--mode", "finetune"]] * baselines
    if os.environ.get("TB_SMOKE"):
        steps = [s + ["--limit", "300"] if s[0] == "adapters.py" else s for s in steps]
        e["BERT_LIMIT"] = "300"
    for k, s in enumerate(steps):
        print(f"== [{time.strftime('%H:%M:%S')}] {name}: {' '.join(s[:3])}", flush=True)
        phase(phase=f"downstream {k + 1}/{len(steps)}: {' '.join(a for a in s[:5] if not a.startswith('/'))}")
        try:
            out = py(s, e)
        except RuntimeError as ex:  # one failed measurement must not lose the rest
            print(f"== {name}: step FAILED {str(ex)[-2000:]}", flush=True)
            continue
        for l in out.splitlines():
            if l.startswith("RESULT "):
                print(f"FULL {name} {l}", flush=True)
                with open(os.path.join(TB, "full.jsonl"), "a") as f:
                    f.write(json.dumps(dict(json.loads(l[7:]), exp=name)) + "\n")


def deps():
    """the downstream half needs torch / transformers / scikit-learn (the testbed image has none)"""
    try:
        import sklearn, torch, transformers  # noqa: F401
    except ImportError:
        pip = [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages"]
        subprocess.run(pip + ["--index-url", "https://download.pytorch.org/whl/cpu", "torch"], check=True)
        subprocess.run(pip + ["transformers", "scikit-learn"], check=True)


def gc(q):
    """free disk (8 cached shards are ~4.7 GB per data env): drop cached data of every data env
    that no pending or running experiment uses, and shards beyond the cached range"""
    keep = set()
    for e in q:
        if not os.path.exists(os.path.join(TB, "claimed", e["name"])) or \
                os.path.exists(os.path.join(TB, "runs", e["name"], "RUNNING")):
            keep.add(data_key(e["env"]))
    root = os.path.join(TB, "data")
    for k in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, k)
        if k not in keep:
            shutil.rmtree(d, ignore_errors=True)
            print(f"== gc: removed data {k}", flush=True)
            continue
        for f in os.listdir(d):
            if f.startswith("s") and f[1:].split(".")[0].isdigit() and int(f[1:].split(".")[0]) > P_COLLAPSE // SHARD:
                os.remove(os.path.join(d, f))


def export(port=8888, hours=3):
    """EXPORT_* queue entries: hard-link every trained model (runs/*/*.gtmm), the learned input
    code embeddings (dist_emb_*.npy) and the result logs into one directory under a random token,
    write a sha256 manifest, and serve it read-only over HTTP on `port` for `hours` (detached, it
    outlives this worker) so the files can be copied off the pod"""
    tok = os.urandom(12).hex()
    out = os.path.join(TB, "export", tok)
    os.makedirs(out)
    src = [os.path.join(TB, "runs", r, f) for r in sorted(os.listdir(os.path.join(TB, "runs")))
           for f in sorted(os.listdir(os.path.join(TB, "runs", r))) if f.endswith(".gtmm")]
    src += [os.path.join(C.DATA, f) for f in sorted(os.listdir(C.DATA)) if f.startswith("dist_emb_")]
    src += [os.path.join(TB, f) for f in ("full.jsonl", "results.jsonl") if os.path.exists(os.path.join(TB, f))]
    lines = []
    for f in src:
        name = os.path.relpath(f, TB).replace(os.sep, "__") if f.startswith(TB) else os.path.basename(f)
        dst = os.path.join(out, name)
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy(f, dst)
        h = hashlib.sha256()
        with open(dst, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        lines.append(f"{h.hexdigest()}  {os.path.getsize(dst)}  {name}")
    with open(os.path.join(out, "MANIFEST.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    subprocess.Popen(["timeout", str(hours * 3600), sys.executable, os.path.abspath(__file__), "serve", out,
                      "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    print(f"== [{time.strftime('%H:%M:%S')}] EXPORT serving {len(lines)} files on :{port} for {hours} h "
          f"(token {tok})", flush=True)
    for l in lines:
        print(f"EXPORT {l}", flush=True)


def serve(d, port):
    """read-only file server for export(): only /<token>/<file> with the export's own token,
    no directory listings"""
    import http.server
    tok = os.path.basename(d.rstrip("/"))

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=d, **k)

        def send_head(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            if len(parts) != 2 or parts[0] != tok or not os.path.isfile(os.path.join(d, parts[1])):
                self.send_error(404)
                return None
            self.path = "/" + parts[1]
            return super().send_head()

        def log_message(self, *a):
            pass

    http.server.ThreadingHTTPServer(("", port), H).serve_forever()


def claim(name):
    os.makedirs(os.path.join(TB, "claimed"), exist_ok=True)
    try:
        os.close(os.open(os.path.join(TB, "claimed", name), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["next", "run", "serve"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--port", type=int, default=8888)
    a = ap.parse_args()
    if a.cmd == "serve":
        return serve(a.name, a.port)
    q = queue()
    if a.cmd == "run":
        threading.Thread(target=heartbeat, daemon=True).start()
        return run(next(e for e in q if e["name"] == a.name), a.threads)
    os.makedirs(TB, exist_ok=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    with open(os.path.join(TB, "gc.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        gc(q)
        mine = next((e for e in q if claim(e["name"])), None)
        if mine:
            os.makedirs(os.path.join(TB, "runs", mine["name"]), exist_ok=True)
            open(os.path.join(TB, "runs", mine["name"], "RUNNING"), "w").close()
    if mine is None:
        sys.exit(3)
    try:
        export() if mine["name"].startswith("EXPORT") else run(mine, a.threads)
        print(f"== [{time.strftime('%H:%M:%S')}] {mine['name']}: done", flush=True)
    except Exception as ex:  # keep the worker alive; the failure is in the log
        print(f"== {mine['name']}: FAILED {str(ex)[-3000:]}", flush=True)
    finally:
        try:
            os.remove(os.path.join(TB, "runs", mine["name"], "RUNNING"))
        except OSError:
            pass


if __name__ == "__main__":
    main()
