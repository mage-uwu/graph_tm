"""Jev-style decisions, adapted end-to-end: pretrained LogicAE (DDLGN) vs bert-tiny (same 4.4M params).

Every Jev / Laya question type is answered through ONE objective both models share (Laya's
option-scoring form, which also fits logic-bert's binary head):
  choice / score (K options)  input = state [SEP] option text  ->  "is this the answer?" (binary);
                              the K option scores of a state form its K-way logits
  noul                        input = state                    ->  P(true) (binary; logits [0, s])
Training rows per state: the gold option + one seeded wrong option (choice / score), or the state
itself (noul). Truncation is shared: the state is cut to SEQ - 1 - len(option) tokens so both models
read identical tokens. Calibration: one temperature per question fitted on dev (Laya's scheme);
test metrics from shootout.metrics (acc, ECE, NLL, Brier, acc@50% coverage, MAE for score).

Arms (each task):
  bert_pair     bert-tiny fine-tuned on the shared objective ([CLS] state [SEP] option [SEP],
                [CLS]+mean -> 1 logit, BCE), epoch picked on dev NLL
  bert_head     bert-tiny fine-tuned with a classic K-way head (shootout.berttiny_ft), reference
  lae_pt_keep   LogicAE run 1 pretrained (pt.ltc) -> adapted, code temperature kept at 0.2
                (transfer_patch --keep-temperature: no reset to soft codes)
  lae_pt        LogicAE pretrained -> adapted as run 1 did (temperature reset to 1, 1 -> 0.2)
  lae_scratch   LogicAE from scratch, same steps (the control pretraining must beat)
LogicAE arms train JEV_STEPS steps (batch 32, eval every 250) and are scored with the dev-best
hardened model (.lth.best; the deployable bit-parallel path). Latency: one decision = one state
with all K options, CPU, 1 thread, steady state.

Results go to $JEV_OUT/results.jsonl + summary.md as they finish, and (with JEV_TOKEN) are copied
into $JEV_EXPORT/<token>/ which export.py --serve publishes read-only on :8888.

  python3 jevft.py          env: JEV_TASKS, JEV_ARMS, JEV_STEPS, THREADS, JEV_OUT, PT (pt.ltc),
                                 LB_TGZ (logic-bert.tar.gz), JEV_TOKEN, JEV_EXPORT
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("OMP_WAIT_POLICY", "passive")  # spinning OpenMP threads cost 5-10x on shared 8-core/16-thread pods (stage2/fastlae)
os.environ.setdefault("THREADS", str(min(16, os.cpu_count() or 1)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np  # noqa: E402

import s1tasks as S  # noqa: E402
import shootout as SH  # noqa: E402

T = int(os.environ["THREADS"])
TASKS = os.environ.get("JEV_TASKS", "emotion,agnews,sst5,sst2,jailbreak").split(",")
ARMS = os.environ.get("JEV_ARMS", "bert_pair,bert_head,lae_pt_keep,lae_pt,lae_scratch").split(",")
STEPS = int(os.environ.get("JEV_STEPS", 2000))
OUT = os.environ.get("JEV_OUT", "/root/jev")
PT = os.environ.get("PT", os.path.join(REPO, "models", "logicae", "pt.ltc"))
LB_TGZ = os.environ.get("LB_TGZ", os.path.join(REPO, "logic-bert.tar.gz"))
TOKEN = os.environ.get("JEV_TOKEN", "")
EXPORT = os.environ.get("JEV_EXPORT", "/root/jev_exports")
SEQ = {"sst2": 48, "sst5": 48, "emotion": 48, "agnews": 64, "jailbreak": 64, "banking77": 48, "spam": 48, "qnli": 96}
SEP = SH.SEP_ID
LT = os.path.join(OUT, "ltx")


def say(msg):
    line = f"JEV [{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for path in ("/proc/1/fd/1", os.path.join(OUT, "jev.log")):
        try:
            with open(path, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    publish()


def publish(files=()):
    """copy the running record (and any given (path, name) files) into the served export directory"""
    if not TOKEN:
        return
    d = os.path.join(EXPORT, TOKEN)
    os.makedirs(d, exist_ok=True)
    for f, name in [(os.path.join(OUT, n), n) for n in ("jev.log", "results.jsonl", "summary.md")] + list(files):
        if os.path.isfile(f):
            shutil.copy(f, os.path.join(d, name + ".tmp"))
            os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))


# ------------------------------------------------------------------ the shared objective
def option_ids(spec):
    tok = SH.C.tokenizer()
    return [np.asarray(e.ids, np.int64) for e in tok.encode_batch(list(spec["options"]), add_special_tokens=False)]


def state(d, i, cap):
    s = list(d["ids"][0][i]) if len(d["ids"]) == 1 else list(d["ids"][0][i]) + [SEP] + list(d["ids"][1][i])
    return s[:max(1, cap)]


def rows(D, split, opts, seq, mode):
    """mode 'train': gold + one seeded wrong option per state; 'all': every option (scoring).
    -> (states, options or None, labels, K per state)"""
    d, spec = D[split], D["spec"]
    y = np.asarray(d["y"])
    if spec["type"] == "noul":
        return [state(d, i, seq) for i in range(len(y))], None, y.astype(np.int64), 2
    K = len(opts)
    rng = np.random.default_rng(23 if split == "train" else 29)
    st, op, lab = [], [], []
    for i in range(len(y)):
        cand = [int(y[i]), int((y[i] + rng.integers(1, K)) % K)] if mode == "train" else range(K)
        for k in cand:
            st.append(state(d, i, seq - 1 - len(opts[k])))
            op.append(list(opts[k]))
            lab.append(int(k == y[i]))
    return st, op, np.asarray(lab, np.int64), K


def lae_seqs(st, op):
    return [s + [SEP] + o for s, o in zip(st, op)] if op is not None else st


def to_logits(score, K, noul):
    return np.stack([np.zeros_like(score), score], 1) if noul else score.reshape(-1, K)


# ------------------------------------------------------------------ LogicAE arms
def build():
    if os.path.exists(LT):
        return
    src = os.path.join(OUT, "lb")
    with tarfile.open(LB_TGZ) as t:
        t.extractall(src)
    patched = os.path.join(OUT, "logic_text.c")
    subprocess.run([sys.executable, os.path.join(REPO, "stage2", "logicae", "transfer_patch.py"),
                    os.path.join(src, "logic-bert", "src", "logic_text.c"), patched], check=True)
    # fasttrain (stage2/fastlae): parallel backward pass, checkpoints byte-identical to the unpatched engine
    subprocess.run([sys.executable, os.path.join(REPO, "stage2", "fastlae", "fasttrain_patch.py"), patched, patched], check=True)
    subprocess.run(["gcc", "-O3", "-march=native", "-std=c11", "-fopenmp", "-Wno-unknown-pragmas", patched, "-lm", "-o", LT],
                   check=True)
    r = subprocess.run([LT, "selftest"], capture_output=True, text=True)
    say(f"build: {(r.stdout.strip().splitlines() or [r.stderr[-200:]])[-1]}")


def lae_votes(lth, path, seq):
    r = subprocess.run([LT, "predict", "--load", lth, "--data", path, "--format", "ids", "--seq", str(seq), "--batch", "64",
                        "--threads", str(T)], capture_output=True, text=True, check=True)
    v = np.array([json.loads(l)["votes"] for l in r.stdout.splitlines() if l.startswith("{")], np.float64)
    return v[:, 1] - v[:, 0]


def lae_arm(task, arm, D, opts, seq, tdir):
    noul = D["spec"]["type"] == "noul"
    files = {}
    for split, mode in (("train", "train"), ("dev", "train"), ("dev", "all"), ("test", "all")):
        st, op, lab, K = rows(D, split, opts, seq, mode)
        files[(split, mode)] = (os.path.join(tdir, f"{split}_{mode}.ids"), lab, K)
        if not os.path.exists(files[(split, mode)][0]):
            SH.ids_file(files[(split, mode)][0], lae_seqs(st, op), lab, seq)
    extra = {"lae_pt_keep": ["--load", PT, "--keep-temperature"], "lae_pt": ["--load", PT],
             "lae_scratch": SH.LAE_ARCH}[arm]
    m = os.path.join(tdir, arm)
    t0 = time.time()
    with open(m + ".log", "w") as log:
        subprocess.run([LT, "train", "--threads", str(T), "--data", files[("train", "train")][0], "--val",
                        files[("dev", "train")][0], "--format", "ids", "--seq", str(seq), "--batch", "32", "--steps", str(STEPS),
                        "--eval-every", "250", "--seed", "17", "--save", m + ".ltc", "--export", m + ".lth", *extra],
                       stderr=log, check=True)
    train_min = (time.time() - t0) / 60
    lth = m + ".lth.best" if os.path.exists(m + ".lth.best") else m + ".lth"
    zd = to_logits(lae_votes(lth, files[("dev", "all")][0], seq), files[("dev", "all")][2], noul)
    zt = to_logits(lae_votes(lth, files[("test", "all")][0], seq), files[("test", "all")][2], noul)
    txt = open(m + ".log").read()
    curve = [(int(s), float(v)) for s, v in re.findall(r'"step":(\d+)[^\n]*"val_hard_accuracy":([0-9.]+)', txt)]
    notes = [l.strip() for l in txt.splitlines() if l.startswith(("keep-temperature", "load-part"))]
    K = files[("test", "all")][2]
    e = os.path.join(tdir, "_lat.ids")  # one decision: one state x K options, batch K
    st, op, _, _ = rows(D, "test", opts, seq, "all")
    SH.ids_file(e, lae_seqs(st, op)[:K], None, seq)
    r = subprocess.run([LT, "bench", "--load", lth, "--data", e, "--format", "ids", "--seq", str(seq), "--batch",
                        str(1 if noul else K), "--repeats", "50", "--warmup", "5", "--threads", "1"],
                       capture_output=True, text=True, check=True)
    lat = json.loads(r.stdout.strip().splitlines()[-1])["median_batch_ms"]
    for s in (".ltc", ".ltc.best"):  # keep only the hardened models
        if os.path.exists(m + s):
            os.remove(m + s)
    return zd, zt, {"train_minutes": round(train_min, 1), "model": os.path.basename(lth), "dev_curve": curve,
                    "notes": notes, "latency_ms_1thread": round(lat, 3)}


# ------------------------------------------------------------------ bert arms
def bert_pair(task, D, opts, seq):
    import torch
    from transformers import BertModel
    torch.manual_seed(17)
    torch.set_num_threads(T)
    noul = D["spec"]["type"] == "noul"
    enc = BertModel.from_pretrained(SH.BERT_REPO, cache_dir=os.path.join(SH.C.DATA, "hf"))
    head = torch.nn.Linear(2 * enc.config.hidden_size, 1)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)

    def pack(st, op):  # shootout.bert_inputs' pair form: [CLS] state [SEP] option [SEP]
        return {"ids": [st] if op is None else [st, op]}

    tr_st, tr_op, tr_y, _ = rows(D, "train", opts, seq, "train")
    dv = rows(D, "dev", opts, seq, "all")
    te = rows(D, "test", opts, seq, "all")
    ntr = len(tr_y)
    epochs = 3 if len(D["train"]["y"]) > 5000 else 8
    bs, steps = 32, epochs * ((ntr + 31) // 32)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))

    def fwd(d, idx):
        ids, am, tt = SH.bert_inputs(d, idx)
        h = enc(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state
        mean = (h * am[..., None]).sum(1) / am.sum(1, keepdim=True)
        return head(torch.cat([h[:, 0], mean], 1))[:, 0]

    def score(r):
        enc.eval()
        d = pack(r[0], r[1])
        with torch.no_grad():
            s = np.concatenate([fwd(d, range(i, min(len(r[2]), i + 256))).numpy() for i in range(0, len(r[2]), 256)])
        return to_logits(s.astype(np.float64), r[3], noul)
    dtr, ytr = pack(tr_st, tr_op), torch.as_tensor(tr_y, dtype=torch.float32)
    ydev = np.asarray(D["dev"]["y"])
    rng, best, t0 = np.random.default_rng(17), None, time.time()
    for ep in range(epochs):
        enc.train()
        perm = rng.permutation(ntr)
        for i in range(0, ntr, bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(fwd(dtr, b), ytr[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        zd = score(dv)
        nll = -np.log(np.clip(SH.softmax(zd)[np.arange(len(ydev)), ydev], 1e-12, 1)).mean()
        if best is None or nll < best[0]:
            best = (nll, ep, zd, score(te))
            state_dict = ({k: v.clone() for k, v in enc.state_dict().items()}, {k: v.clone() for k, v in head.state_dict().items()})
    train_min = (time.time() - t0) / 60
    enc.load_state_dict(state_dict[0])
    head.load_state_dict(state_dict[1])
    enc.eval()
    torch.set_num_threads(1)
    K = te[3]
    d1 = pack(te[0][:K] if not noul else te[0][:1], te[1][:K] if not noul else None)
    n1 = 1 if noul else K
    with torch.no_grad():
        for _ in range(5):
            fwd(d1, range(n1))
        ts = []
        for _ in range(30):
            t1 = time.perf_counter()
            fwd(d1, range(n1))
            ts.append(time.perf_counter() - t1)
    torch.set_num_threads(T)
    return best[2], best[3], {"epoch": best[1] + 1, "epochs": epochs, "train_minutes": round(train_min, 1),
                              "latency_ms_1thread": round(1000 * float(np.median(ts)), 3)}


def bert_head(task, D, opts, seq):
    """shootout's classic K-way fine-tune, on the same truncated states"""
    cap = seq - 1 - max(len(o) for o in opts) if D["spec"]["type"] != "noul" else seq
    Dt = {"spec": D["spec"]}
    for split in ("train", "dev", "test"):
        d = D[split]
        Dt[split] = {"y": d["y"], "ids": [[np.asarray(state(d, i, cap), np.int32) for i in range(len(d["y"]))]]}
    t0 = time.time()
    zd, zt, info = SH.berttiny_ft(task, Dt)
    return zd, zt, dict(info, train_minutes=round((time.time() - t0) / 60, 1))


# ------------------------------------------------------------------ main
def summary():
    res = [json.loads(l) for l in open(os.path.join(OUT, "results.jsonl"))]
    out = ["# Jev-style decisions: pretrained LogicAE vs bert-tiny, adapted on one shared objective", "",
           f"LogicAE arms: {STEPS} steps, batch 32. Test metrics after one dev-fitted temperature per question. "
           "Latency: one decision (state x all options), CPU, 1 thread.", ""]
    for task in dict.fromkeys(r["task"] for r in res):
        rs = sorted([r for r in res if r["task"] == task], key=lambda r: -r["test"]["acc"])
        r0 = rs[0]
        out += [f"## {task} ({r0['type']}, K={r0['K']}, test n={r0['test']['n']})", "",
                "| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |", "|---|---|---|---|---|---|---|---|---|"]
        for j, r in enumerate(rs, 1):
            m = r["test"]
            out.append(f"| {j} | {r['arm']} | {m['acc']:.4f} | {m['ece']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} | "
                       f"{m['acc_at_50cov']:.4f} | {m.get('mae', '-')} | {r['info'].get('latency_ms_1thread', '-')} |")
        out.append("")
    arms = list(dict.fromkeys(r["arm"] for r in res))
    tasks = list(dict.fromkeys(r["task"] for r in res))
    grid = {(r["task"], r["arm"]): r["test"]["acc"] for r in res}
    full = [a for a in arms if all((t, a) in grid for t in tasks)]
    out += ["## Mean test accuracy over the tasks every arm finished", "", "| # | arm | mean acc |", "|---|---|---|"]
    for j, a in enumerate(sorted(full, key=lambda a: -np.mean([grid[(t, a)] for t in tasks])), 1):
        out.append(f"| {j} | {a} | {np.mean([grid[(t, a)] for t in tasks]):.4f} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    say(f"start: tasks {TASKS}, arms {ARMS}, LogicAE {STEPS} steps, threads {T}")
    if any(a.startswith("lae") for a in ARMS):
        build()
    done = set()
    rpath = os.path.join(OUT, "results.jsonl")
    if os.path.exists(rpath):
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(rpath)}
    for task in TASKS:
        D = S.load(task)
        spec, seq = D["spec"], SEQ[task]
        opts = option_ids(spec)
        say(f"{task}: {spec['type']} K={len(spec['options'])} train {len(D['train']['y'])} dev {len(D['dev']['y'])} "
            f"test {len(D['test']['y'])} seq {seq}")
        tdir = os.path.join(OUT, task)
        os.makedirs(tdir, exist_ok=True)
        for arm in ARMS:
            if (task, arm) in done:
                continue
            say(f"{task} {arm}: start")
            try:
                if arm.startswith("lae"):
                    zd, zt, info = lae_arm(task, arm, D, opts, seq, tdir)
                else:
                    zd, zt, info = {"bert_pair": bert_pair, "bert_head": bert_head}[arm](task, D, opts, seq)
            except Exception as ex:  # one failed arm must not stop the grid
                say(f"{task} {arm}: FAILED {type(ex).__name__}: {str(ex)[-300:]}")
                continue
            t = SH.fit_temperature(zd, np.asarray(D["dev"]["y"]))
            res = {"task": task, "type": spec["type"], "K": len(spec["options"]), "arm": arm, "seq": seq,
                   "test": SH.metrics(zt, np.asarray(D["test"]["y"]), t, spec["type"]),
                   "dev_acc": round(float((zd.argmax(1) == np.asarray(D["dev"]["y"])).mean()), 4), "info": info}
            with open(rpath, "a") as f:
                f.write(json.dumps(res) + "\n")
            summary()
            lth = [os.path.join(tdir, arm + s) for s in (".lth", ".lth.best") if os.path.exists(os.path.join(tdir, arm + s))]
            say(f"RESULT {json.dumps(res)}")
            publish([(lth[-1], f"{task}_{arm}.lth")] if lth else [])
    say("done")


if __name__ == "__main__":
    main()
