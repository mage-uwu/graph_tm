"""System-1 shootout: Laya / Jev-style typed decisions from BERT, GraphTM and DDLGN (LogicAE)
encoders, scored for accuracy, calibration and CPU forward-pass latency.

Request shape (Jev / Laya): a state + typed questions {id: {type: choice|score|noul,
instructions, criteria}} -> typed answers with calibrated probabilities (see s1tasks.py).

Backends (each answers every question through the same harness):
  bow             hashed unigram + bigram bag, logistic head per question (the lexical floor)
  berttiny        bert-tiny (4.4M, frozen): [CLS] + mean final-layer features, logistic heads
  berttiny_ft     bert-tiny fine-tuned end-to-end, one network per question (epoch picked on dev)
  graphtm         pretrained GraphTM N5 (frozen): clause bits OR-pooled over per-token window
                  graphs (stage2/adapters.py), logistic heads
  logicae         pretrained LogicAE / DDLGN (frozen, hardened .lth): last-block gate bits
                  OR-pooled over positions (lfeat.c), logistic heads
  logicae_ft      DDLGN trained end-to-end from scratch, one network per question; logic-bert's
                  head is binary, so noul / 2-way questions only (run 1's sst2 / qnli models are
                  reused when present)
  laya            Laya (ModernBERT-large, 421M) zero-shot as shipped, temperatures refit on dev:
                  the open System-1 reference (question and options inside the sequence)
Frozen-feature backends share one encoder pass per state: a request with several questions
costs one encoder forward + one tiny head per question. Pairs (qnli) use [f(q), f(s), f(q) AND
f(s)] for the pooled-bit encoders and bow; bert reads the pair jointly ([CLS] q [SEP] s).

Calibration: one temperature per question (= per (type, option count) bucket, Laya's scheme),
fitted on dev by NLL. Metrics on test: accuracy, ECE (15 bins, top-label), NLL, Brier,
accuracy at 50% coverage (the most confident half), and for score questions the MAE of the
expected level.

Latency (CPU, this machine): per request, steady-state compute for the encoder + heads,
tokenization measured separately and added; p50 / p95 over LAT_N distinct states, at 1 thread
and at THREADS. Requests: "1q" (the agnews question) and "7q" (all seven single-text questions
on the same state). C engines are timed with their own repeat loops (lt bench, gtm score
--reps), which exclude process start and model load (reported separately).

  python3 shootout.py [--tasks a,b] [--backends a,b]      env: THREADS, S1_OUT, GTM_MODEL,
      LAE_DIR (run 1 dir: pt.ltc, lt, *_dev.ids, *_scratch.lth), LB (logic-bert src), LAYA_DIR
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("GTM_CODES", "dist")  # the N5 GraphTM's input codes and graph shape
os.environ.setdefault("GTM_DISTS", "1,2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import s1tasks as S  # noqa: E402

THREADS = int(os.environ.get("THREADS", os.cpu_count() or 1))
OUT = os.environ.get("S1_OUT", os.path.join(C.DATA, "system1", "results"))
LAE = os.environ.get("LAE_DIR", "/root/lae")
LB = os.environ.get("LB", "/root/logic-bert/src")
GTM_MODEL = os.environ.get("GTM_MODEL", os.path.join(C.ROOT, "models", "graphtm", "runs__N5_R4b_dist_bert_word__best.gtmm"))
LAYA_DIR = os.environ.get("LAYA_DIR", os.path.join(C.DATA, "system1", "laya"))
LAT_N = int(os.environ.get("LAT_N", 20))
LAYA_TEST, LAYA_DEV = int(os.environ.get("S1_LAYA_TEST", 300)), int(os.environ.get("S1_LAYA_DEV", 200))
SINGLE = ["sst2", "sst5", "emotion", "agnews", "banking77", "spam", "jailbreak"]  # one-text questions
ALL = SINGLE[:5] + ["qnli"] + SINGLE[5:]
BERT_REPO = C.TOKENIZER_REPO
CLS_ID, SEP_ID, PAD_ID = 101, 102, 0
LOG = None


def say(msg):
    line = f"S1 [{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for p in ([LOG] if LOG else []) + ["/proc/1/fd/1"]:
        try:
            with open(p, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ------------------------------------------------------------------ heads, calibration, metrics
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logreg(Xtr, ytr, Xdv, ydv, K):
    """multinomial logistic head, L2 strength picked on dev NLL -> (dev logits, predict fn)"""
    from sklearn.linear_model import LogisticRegression
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(C=c, max_iter=3000, tol=1e-4).fit(Xtr, ytr)
        z = logits_of(m, Xdv, K)
        nll = -np.log(np.clip(softmax(z)[np.arange(len(ydv)), ydv], 1e-12, 1)).mean()
        if best is None or nll < best[0]:
            best = (nll, c, m)
    return best[2], best[1]


def logits_of(m, X, K):
    """(n, K) logits in label order even when a class is missing from train"""
    z = m.decision_function(X)
    if z.ndim == 1:
        z = np.stack([np.zeros_like(z), z], 1)
    out = np.full((z.shape[0], K), -30.0)
    out[:, m.classes_] = z
    return out


def fit_temperature(z, y):
    ts = np.exp(np.linspace(np.log(0.01), np.log(1e5), 481))
    nll = [-np.log(np.clip(softmax(z / t)[np.arange(len(y)), y], 1e-12, 1)).mean() for t in ts]
    return float(ts[int(np.argmin(nll))])


def metrics(z, y, t, qtype):
    p = softmax(z / t)
    n, K = p.shape
    pred, conf = p.argmax(1), p.max(1)
    ok = pred == y
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(ok[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(15) if (bins == b).any())
    top = np.argsort(-conf)[: max(1, n // 2)]
    oh = np.eye(K)[y]
    m = {"n": int(n), "acc": round(float(ok.mean()), 4), "ece": round(float(ece), 4),
         "nll": round(float(-np.log(np.clip(p[np.arange(n), y], 1e-12, 1)).mean()), 4),
         "brier": round(float(((p - oh) ** 2).sum(1).mean()), 4), "acc_at_50cov": round(float(ok[top].mean()), 4),
         "temperature": round(t, 4)}
    if qtype == "score":
        m["mae"] = round(float(np.abs((p * np.arange(K)).sum(1) - y).mean()), 4)
    return m


# ------------------------------------------------------------------ features
def pair(d):
    return len(d["ids"]) == 2


def bow_feats(d, bins=1 << 14):
    """hashed unigrams + bigrams (binary); pairs: [bag(q), bag(s), bag(q) AND bag(s)]"""
    from scipy import sparse

    def bag(seqs):
        rows, cols = [], []
        for i, s in enumerate(seqs):
            s = np.asarray(s, np.int64)
            f = np.concatenate([s % bins, (s[:-1] * 1000003 + s[1:]) % bins]) if len(s) > 1 else s % bins
            f = np.unique(f)
            rows.append(np.full(len(f), i))
            cols.append(f)
        return sparse.csr_matrix((np.ones(sum(len(c) for c in cols), np.float32), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(len(seqs), bins))
    if not pair(d):
        return bag(d["ids"][0])
    q, s = bag(d["ids"][0]), bag(d["ids"][1])
    return sparse.hstack([q, s, q.multiply(s)]).tocsr()


def bits_pairs(fq, fs):
    from scipy import sparse
    return sparse.csr_matrix(np.hstack([fq, fs, fq & fs]).astype(np.float32))


_BERT = {}


def bert():
    if "m" not in _BERT:
        import torch
        from transformers import BertModel
        torch.set_num_threads(THREADS)
        _BERT["m"] = BertModel.from_pretrained(BERT_REPO, cache_dir=os.path.join(C.DATA, "hf")).eval()
    return _BERT["m"]


def bert_inputs(d, idx):
    import torch
    seqs = []
    for i in idx:
        if pair(d):
            q, s = d["ids"][0][i], d["ids"][1][i]
            seqs.append(([CLS_ID] + list(q) + [SEP_ID] + list(s) + [SEP_ID], len(q) + 2))
        else:
            seqs.append(([CLS_ID] + list(d["ids"][0][i]) + [SEP_ID], None))
    L = max(len(s) for s, _ in seqs)
    ids = torch.zeros((len(seqs), L), dtype=torch.long)
    tt = torch.zeros_like(ids)
    for j, (s, qlen) in enumerate(seqs):
        ids[j, :len(s)] = torch.as_tensor(s)
        if qlen:
            tt[j, qlen:len(s)] = 1
    return ids, (ids != PAD_ID).long(), tt


def berttiny_feats(d):
    import torch
    m, out = bert(), []
    n = len(d["y"])
    with torch.no_grad():
        for i in range(0, n, 256):
            ids, am, tt = bert_inputs(d, range(i, min(n, i + 256)))
            h = m(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state
            mean = (h * am[..., None]).sum(1) / am.sum(1, keepdim=True)
            out.append(torch.cat([h[:, 0], mean], 1).numpy())
    return np.concatenate(out)


def graphtm_feats(d, tag):
    import adapters as AD
    f = [AD.pooled(GTM_MODEL, [np.asarray(s, np.int64) for s in col], THREADS, f"s1_{tag}_{j}") for j, col in enumerate(d["ids"])]
    from scipy import sparse
    return bits_pairs(*f) if len(f) == 2 else sparse.csr_matrix(f[0].astype(np.float32))


def ids_file(path, seqs, y=None, seq=128):
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f"{-1 if y is None else int(y[i])} " + " ".join(str(int(x)) for x in list(s)[:seq]) + "\n")


def lae_hard_model():
    """run 1's pretrained LogicAE, hardened (pt.ltc -> pt.lth)"""
    lth = os.path.join(OUT, "logicae_pt.lth")
    if not os.path.exists(lth):
        subprocess.run([os.path.join(LAE, "lt"), "export", "--load", os.path.join(LAE, "pt.ltc"), "--out", lth], check=True,
                       capture_output=True)
    return lth


def lfeat_bin():
    b = os.path.join(OUT, "lfeat")
    if not os.path.exists(b):
        subprocess.run(["gcc", "-O3", "-march=native", "-std=c11", "-fopenmp", "-Wno-unknown-pragmas", f"-I{LB}",
                        os.path.join(HERE, "lfeat.c"), "-lm", "-o", b], check=True)
    return b


def logicae_feats(d, tag):
    lth, out = lae_hard_model(), []
    for j, col in enumerate(d["ids"]):
        p, fb = os.path.join(OUT, f"_lf_{tag}_{j}.ids"), os.path.join(OUT, f"_lf_{tag}_{j}.bin")
        ids_file(p, col)
        subprocess.run([lfeat_bin(), lth, p, "128", str(THREADS), fb], check=True, capture_output=True)
        W = (os.path.getsize(fb) // len(col)) * 8
        out.append(np.unpackbits(np.fromfile(fb, np.uint8).reshape(len(col), -1), axis=1, bitorder="little")[:, :W].astype(bool))
        os.remove(p)
        os.remove(fb)
    from scipy import sparse
    return bits_pairs(*out) if len(out) == 2 else sparse.csr_matrix(out[0].astype(np.float32))


FEATS = {"bow": lambda d, tag: bow_feats(d), "berttiny": lambda d, tag: berttiny_feats(d),
         "graphtm": graphtm_feats, "logicae": logicae_feats}


# ------------------------------------------------------------------ end-to-end backends
def berttiny_ft(task, D):
    """fine-tune bert-tiny per question; epoch picked on dev NLL -> dev / test logits"""
    import torch
    from transformers import BertModel
    torch.manual_seed(17)
    K = len(D["spec"]["options"])
    enc = BertModel.from_pretrained(BERT_REPO, cache_dir=os.path.join(C.DATA, "hf"))
    head = torch.nn.Linear(2 * enc.config.hidden_size, K)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    ntr = len(D["train"]["y"])
    epochs = 3 if ntr > 5000 else 8
    bs = 32

    def fwd(d, idx):
        ids, am, tt = bert_inputs(d, idx)
        h = enc(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state
        mean = (h * am[..., None]).sum(1) / am.sum(1, keepdim=True)
        return head(torch.cat([h[:, 0], mean], 1))

    def predict(d):
        enc.eval()
        with torch.no_grad():
            return np.concatenate([fwd(d, range(i, min(len(d["y"]), i + 256))).numpy() for i in range(0, len(d["y"]), 256)])
    ytr = torch.as_tensor(D["train"]["y"])
    rng = np.random.default_rng(17)
    best, steps = None, epochs * ((ntr + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    for ep in range(epochs):
        enc.train()
        perm = rng.permutation(ntr)
        for i in range(0, ntr, bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(fwd(D["train"], b), ytr[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        zd = predict(D["dev"])
        nll = -np.log(np.clip(softmax(zd)[np.arange(len(D["dev"]["y"])), D["dev"]["y"]], 1e-12, 1)).mean()
        if best is None or nll < best[0]:
            best = (nll, ep, zd, predict(D["test"]), {k: v.clone() for k, v in enc.state_dict().items()},
                    {k: v.clone() for k, v in head.state_dict().items()})
    enc.load_state_dict(best[4])
    head.load_state_dict(best[5])
    enc.eval()
    _FT_MODELS[task] = (enc, head)
    return best[2], best[3], {"epoch": best[1] + 1, "epochs": epochs}


_FT_MODELS = {}
LAE_ARCH = os.environ.get("LAE_ARCH", "--vocab-size 30522 --code-bits 128 --width 1024 --blocks 16 --kernel 5 --cycle 4").split()


def lt_votes(lth, ids_path, seq):
    r = subprocess.run([os.path.join(LAE, "lt"), "predict", "--load", lth, "--data", ids_path, "--format", "ids", "--seq",
                        str(seq), "--batch", "64", "--threads", str(THREADS)], capture_output=True, text=True, check=True)
    v = np.array([json.loads(l)["votes"] for l in r.stdout.splitlines() if l.startswith("{")], np.float64)
    return np.stack([np.zeros(len(v)), v[:, 1] - v[:, 0]], 1)  # logit difference; the temperature scales it


def lae_rows(d):
    return [list(q) + [SEP_ID] + list(s) for q, s in zip(*d["ids"])] if pair(d) else [list(s) for s in d["ids"][0]]


def logicae_ft(task, D):
    """binary questions only. Reuses run 1's <task>_scratch.lth (calibrated on run 1's held-out dev);
    otherwise trains from scratch on this task's split."""
    if len(D["spec"]["options"]) != 2:
        return None
    seq = 128 if pair(D["test"]) or task == "jailbreak" else 64
    tdir = os.path.join(OUT, f"lae_{task}")
    os.makedirs(tdir, exist_ok=True)
    ids_file(os.path.join(tdir, "test.ids"), lae_rows(D["test"]), D["test"]["y"], seq)
    run1 = os.path.join(LAE, f"{task}_scratch.lth")
    if os.path.exists(run1) and os.path.exists(os.path.join(LAE, f"{task}_dev.ids")):
        lth, dev_ids, info = run1, os.path.join(LAE, f"{task}_dev.ids"), {"model": f"run 1 {task}_scratch (2000 steps)"}
        ydev = np.array([int(l.split()[0]) for l in open(dev_ids)])
    else:
        steps = 1000 if len(D["train"]["y"]) > 2000 else 500
        for split in ("train", "dev"):
            ids_file(os.path.join(tdir, f"{split}.ids"), lae_rows(D[split]), D[split]["y"], seq)
        lth, dev_ids, ydev = os.path.join(tdir, "m.lth"), os.path.join(tdir, "dev.ids"), D["dev"]["y"]
        if not os.path.exists(lth):
            say(f"logicae_ft {task}: training {steps} steps from scratch (seq {seq})")
            with open(os.path.join(tdir, "train.log"), "w") as log:
                subprocess.run([os.path.join(LAE, "lt"), "train", "--threads", str(THREADS), "--data", os.path.join(tdir, "train.ids"),
                                "--val", dev_ids, "--format", "ids", *LAE_ARCH, "--seq", str(seq), "--batch", "32", "--steps",
                                str(steps), "--eval-every", "250", "--seed", "17", "--save", os.path.join(tdir, "m.ltc"),
                                "--export", lth], stderr=log, check=True)
        info = {"model": f"from scratch, {steps} steps"}
    _LAE_FT[task] = (lth, seq)
    return lt_votes(lth, dev_ids, seq), lt_votes(lth, os.path.join(tdir, "test.ids"), seq), dict(info, ydev=ydev)


_LAE_FT = {}


# ------------------------------------------------------------------ laya (reference)
def laya_agent():
    if "a" in _BERT:
        return _BERT["a"]
    import torch
    torch.set_num_threads(THREADS)
    from huggingface_hub import snapshot_download
    d = snapshot_download("convaiinnovations/laya", cache_dir=os.path.join(C.DATA, "hf"),
                          allow_patterns=["*.py", "rl_agent_config.json", "encoder/*", "tokenizer/*", "model.safetensors"])
    sys.path.insert(0, d)
    from rl_agent_api import RLAgent
    _BERT["a"] = RLAgent(d, device="cpu")
    return _BERT["a"]


def laya_question(spec):
    if spec["type"] == "choice":
        return {"type": "choice", "instructions": spec["q"], "criteria": list(spec["options"])}
    if spec["type"] == "score":
        return {"type": "score", "instructions": spec["q"], "criteria": list(spec["options"])}
    return {"type": "noul", "instructions": spec["q"]}


def laya_state(d, i):
    return f"Question: {d['text'][0][i]}\nSentence: {d['text'][1][i]}" if pair(d) else d["text"][0][i]


def laya_logits(spec, d, idx):
    """raw option logits (before Laya's shipped temperatures) for one question per state"""
    import torch
    ag = laya_agent()  # puts the downloaded Laya code on sys.path
    from rl_common import QTYPES, build_sequence, collate_items
    q = ag._to_internal(laya_question(spec))
    out = []
    for i in idx:
        seq, markers = build_sequence(ag.tok, laya_state(d, i), q, ag.cfg["max_len"], ag.cfg["head_max_len"])
        item = {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": [0.0] * len(markers), "label": -1,
                "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"}
        b = collate_items([[item]], ag.tok.pad_token_id)
        with torch.no_grad():
            z, _ = ag.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        z = z.float().numpy()[0, :len(spec["options"])]
        out.append(np.pad(z, (0, len(spec["options"]) - len(z)), constant_values=-1e4))
    return np.array(out)


# ------------------------------------------------------------------ latency
def timeit(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def lat_summary(ms):
    ms = np.asarray(ms)
    return {"p50_ms": round(float(np.percentile(ms, 50)), 3), "p95_ms": round(float(np.percentile(ms, 95)), 3)}


def latency(backend, heads, states, threads):
    """per-request compute latency: '1q' = the sst2 question, 'Nq' = every single-text question the
    backend answers, asked of the same state (frozen encoders: one encoder pass + N heads)"""
    tok = C.tokenizer()
    res = {}
    tok_ms = np.array([1000 * timeit(lambda s=s: tok.encode(s, add_special_tokens=False), 20) for s in states])
    enc = [np.asarray(tok.encode(s, add_special_tokens=False).ids[:S.SINGLE_MAX], np.int64) for s in states]
    avail = [q for q in SINGLE if q in heads]
    if backend == "laya":  # 421M parameters: seconds per request at 1 thread
        states, enc, tok_ms = states[:8], enc[:8], tok_ms[:8]
    for label, qs in (("1q", ["sst2"]), (f"{len(avail)}q", avail)):
        if not qs or not all(q in heads for q in qs) or label in res:
            continue
        ms = []
        for s, e in zip(states, enc):
            d = {"ids": [[e]], "y": np.zeros(1, int), "text": [[s]]}
            if backend == "bow":
                t = timeit(lambda: [heads[q][0].decision_function(f) for f in [bow_feats(d)] for q in qs], 20)
            elif backend == "berttiny":
                import torch
                torch.set_num_threads(threads)
                t = timeit(lambda: [heads[q][0].decision_function(f) for f in [berttiny_feats(d)] for q in qs], 20)
            elif backend == "berttiny_ft":
                import torch
                torch.set_num_threads(threads)

                def run():
                    with torch.no_grad():
                        for q in qs:  # one fine-tuned network per question
                            enc_m, head = _FT_MODELS[q]
                            ids, am, tt = bert_inputs(d, [0])
                            h = enc_m(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state
                            head(torch.cat([h[:, 0], (h * am[..., None]).sum(1) / am.sum(1, keepdim=True)], 1))
                t = timeit(run, 20)
            elif backend in ("graphtm", "logicae"):
                x = {q: np.zeros((1, heads[q][0].coef_.shape[1]), np.float32) for q in qs}
                th = timeit(lambda: [heads[q][0].decision_function(x[q]) for q in qs], 20)
                t = th + (graphtm_request_s(e, threads) if backend == "graphtm" else lt_bench_s(lae_hard_model(), e, 128, threads))
            elif backend == "logicae_ft":
                t = sum(lt_bench_s(_LAE_FT[q][0], e, _LAE_FT[q][1], threads) for q in qs)
            elif backend == "laya":
                import torch
                torch.set_num_threads(threads)
                ag = laya_agent()
                questions = {q: laya_question(S.load(q)["spec"]) for q in qs}
                t = timeit(lambda: ag.system_one(s, questions), 3)
            ms.append(1000 * t)
        ms = np.array(ms) + (tok_ms if backend in ("graphtm", "logicae", "logicae_ft") else 0)
        res[label] = dict(lat_summary(ms), questions=qs)
    return res


def graphtm_request_s(e, threads):
    """window graphs for one state (one per token) -> engine best-of-reps seconds + graph build"""
    import adapters as AD
    t = np.load(os.path.join(C.DATA, "teacher.npz"))
    mask_id = int(np.nonzero(t["vocab"] == "[MASK]")[0][0])
    if "rows" not in _BERT:
        _BERT["rows"] = C.symbol_rows(len(t["codes"]), mask_id=mask_id)
    rows = _BERT["rows"]
    hdr = open(GTM_MODEL, "rb").read(16)
    n_out = int(np.frombuffer(hdr, "<u4", 1, 12)[0])
    cen = np.stack([np.zeros(len(e), np.int64), np.arange(len(e))], 1)
    p = os.path.join(OUT, "_lat.gtmd")

    def build():
        npg, epn, edges, X, nt, _ = C.window_graphs([e], cen, mask_id if AD.CENTRE == "mask" else None, rows)
        C.write_gtmd(p, C.H, C.N_NODE_TYPES, C.N_EDGE_TYPES, n_out, 1, npg, epn, edges, X, np.zeros((len(cen), n_out), np.int32), nt)
    tb = timeit(build, 3)
    out = C.gtm("score", "--model", GTM_MODEL, "--data", p, "--threads", threads, "--reps", 30)
    return tb + float(out.split("best ")[1].split("s")[0])


def lt_bench_s(lth, e, seq, threads):
    p = os.path.join(OUT, "_lat.ids")
    ids_file(p, [e], seq=seq)
    r = subprocess.run([os.path.join(LAE, "lt"), "bench", "--load", lth, "--data", p, "--format", "ids", "--seq", str(seq),
                        "--batch", "1", "--repeats", "50", "--warmup", "5", "--threads", str(threads)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])["median_batch_ms"] / 1000


# ------------------------------------------------------------------ main
def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(ALL))
    ap.add_argument("--backends", default="bow,berttiny,graphtm,logicae,berttiny_ft,logicae_ft,laya")
    ap.add_argument("--no-latency", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    LOG = os.path.join(OUT, "shootout.log")
    tasks, backends = a.tasks.split(","), a.backends.split(",")
    say(f"tasks {tasks}; backends {backends}; threads {THREADS}; out {OUT}")
    D = {}
    for t in tasks:
        D[t] = S.load(t)
        say(f"task {t}: {D[t]['spec']['type']} x{len(D[t]['spec']['options'])}, "
            f"train {len(D[t]['train']['y'])} dev {len(D[t]['dev']['y'])} test {len(D[t]['test']['y'])}")
    res = open(os.path.join(OUT, "results.jsonl"), "a")
    heads = {b: {} for b in backends}
    for b in backends:
        for t in tasks:
            spec, K = D[t]["spec"], len(D[t]["spec"]["options"])
            t0 = time.time()
            try:
                if b in FEATS:
                    F = {s: FEATS[b](D[t][s], f"{b}_{t}_{s}") for s in ("train", "dev", "test")}
                    tf = time.time() - t0
                    m, c = logreg(F["train"], D[t]["train"]["y"], F["dev"], D[t]["dev"]["y"], K)
                    heads[b][t] = (m, c)
                    zd, zt, info = logits_of(m, F["dev"], K), logits_of(m, F["test"], K), {"C": c, "features": F["train"].shape[1],
                                                                                          "feature_s": round(tf, 1)}
                    ydev = D[t]["dev"]["y"]
                elif b == "berttiny_ft":
                    zd, zt, info = berttiny_ft(t, D[t])
                    ydev = D[t]["dev"]["y"]
                    heads[b][t] = True
                elif b == "logicae_ft":
                    r = logicae_ft(t, D[t])
                    if r is None:
                        continue
                    zd, zt, info = r
                    ydev = info.pop("ydev")
                    heads[b][t] = True
                elif b == "laya":
                    nd, nt = min(LAYA_DEV, len(D[t]["dev"]["y"])), min(LAYA_TEST, len(D[t]["test"]["y"]))
                    zd, zt = laya_logits(spec, D[t]["dev"], range(nd)), laya_logits(spec, D[t]["test"], range(nt))
                    ydev = D[t]["dev"]["y"][:nd]
                    info = {"zero_shot": True, "test_subset": nt}
                    heads[b][t] = True
                else:
                    continue
                yt = D[t]["test"]["y"][: len(zt)]
                T = fit_temperature(zd, ydev)
                r = {"backend": b, "task": t, "family": spec["family"], "type": spec["type"], "K": K, "test": metrics(zt, yt, T, spec["type"]),
                     "uncalibrated": metrics(zt, yt, 1.0, spec["type"]), "info": info, "seconds": round(time.time() - t0, 1)}
                say("RESULT " + json.dumps(r))
                res.write(json.dumps(r) + "\n")
                res.flush()
            except Exception as ex:  # one failure must not sink the shootout
                say(f"FAILED {b} {t}: {type(ex).__name__}: {str(ex)[-500:]}")
    if not a.no_latency:
        tt = S.load("agnews")["test"]
        states = [tt["text"][0][i] for i in range(LAT_N)]
        for b in backends:
            for th in sorted({1, THREADS}):
                try:
                    lat = latency(b, heads[b], states, th)
                    if lat:
                        r = {"backend": b, "latency": lat, "threads": th}
                        say("LATENCY " + json.dumps(r))
                        res.write(json.dumps(r) + "\n")
                        res.flush()
                except Exception as ex:
                    say(f"FAILED latency {b} threads={th}: {type(ex).__name__}: {str(ex)[-500:]}")
    res.close()
    summary(os.path.join(OUT, "results.jsonl"), os.path.join(OUT, "summary.md"))
    say("done")


def summary(jsonl, md):
    rows = [json.loads(l) for l in open(jsonl)]
    R = {(r["backend"], r["task"]): r for r in rows if "task" in r}
    L = {(r["backend"], r["threads"]): r["latency"] for r in rows if "latency" in r}
    bs = list(dict.fromkeys(r["backend"] for r in rows))
    ts = [t for t in ALL if any((b, t) in R for b in bs)]
    out = ["| task (type, K) | " + " | ".join(bs) + " |", "|---|" + "---|" * len(bs)]
    for t in ts:
        k = next(R[(b, t)] for b in bs if (b, t) in R)
        out.append(f"| {t} ({k['type']}, {k['K']}) | " + " | ".join(
            f"{R[(b, t)]['test']['acc']:.3f} / {R[(b, t)]['test']['ece']:.3f}" if (b, t) in R else "-" for b in bs) + " |")
    out += ["", "accuracy / ECE (calibrated) per cell. Mean over the tasks each backend answers:", "",
            "| backend | tasks | mean acc | mean ECE | 1q p50 ms (1 thr) | Nq p50 ms (1 thr) | 1q p50 ms (all thr) | Nq p50 ms (all thr) |",
            "|---|---|---|---|---|---|---|---|"]
    ths = sorted({th for (_, th) in L})
    for b in bs:
        rr = [R[(b, t)]["test"] for t in ts if (b, t) in R]
        if not rr:
            continue
        def lat_of(th, multi):
            d = L.get((b, th), {})
            ks = [k for k in d if (k != "1q") == multi]
            return f"{d[ks[0]]['p50_ms']} ({ks[0]})" if multi and ks else (d[ks[0]]["p50_ms"] if ks else "-")
        lat = [lat_of(th, m) for th in (ths[:1] + ths[-1:] if ths else []) for m in (False, True)]
        lat = [lat[0], lat[1], lat[2], lat[3]] if len(lat) == 4 else ["-"] * 4
        out.append(f"| {b} | {len(rr)} | {np.mean([x['acc'] for x in rr]):.3f} | {np.mean([x['ece'] for x in rr]):.3f} | "
                   + " | ".join(str(x) for x in lat) + " |")
    open(md, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
