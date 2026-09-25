"""LogicAE on Laya's own benchmark: LocalLLaMA/typed-decisions (the task Laya / Jev are measured on).

400 test cases / 2,000 typed decisions over four workflows (agent-trace observability, customer
service, invoice processing, security incidents). Each case is a JSON state plus 5 questions
(choice / score / noul) whose gold answers are a teacher's probability distributions. The question
set is fixed per workflow: 20 questions in all, 1,200 training cases (300 per workflow).

Laya's published recipe (github.com/NandhaKishorM/laya, notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb):
ModernBERT-large (421M) + marker head, fine-tuned 4 epochs on the train split with soft cross-entropy
against the teacher distributions plus an RLCD policy-gradient term (proper scoring rule reward),
then one temperature per (question type, option count) on a held-out calibration slice. Metric:
accuracy = argmax vs the gold label, over all 2,000 decisions. Published: Laya fine-tuned 0.766,
TypeSafe Jev 1.13.0 0.727, teacher self-agreement 0.735, majority class 0.461, Laya zero-shot 0.362.

Here, the same splits and metric, with the question fixed per model (every question has one schema):
  majority     per-question majority label of the training split
  bow          hashed unigram+bigram logistic regression per question (argmax labels)
  berttiny     bert-tiny (4.4M) fine-tuned per question, K-way head, soft cross-entropy against the
               teacher distribution (Laya's supervised term; no RL term), epoch picked on calibration
  lae          LogicAE one-vs-rest per question: one yes/no model per option (one for noul), trained
               from scratch on soft labels (each training case repeated R times with its label drawn
               from the teacher probability, seeded), hardened; K-way logits = vote differences
  lae_gm       the same with the global view v2 (match bit + majority-pooled channels)
State text: the JSON flattened to "key value ; key value" (185 bert tokens on average vs 264 as
JSON), truncated at SEQ. 10% of the training cases are held out for calibration (temperature per
(type, K), pooled over questions, as Laya does) and epoch selection.

  python3 laya_td.py      env: TD_ARMS, TD_STEPS (LogicAE steps/model), TD_ARCH, TD_SEQ, TD_REP, THREADS, JEV_OUT, JEV_TOKEN
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import jevft as J  # noqa: E402  (say / publish / build / lae_votes / OUT / T; the engine build)
SH = J.SH

ARMS = os.environ.get("TD_ARMS", "majority,bow,berttiny,lae_gm,lae").split(",")
STEPS = int(os.environ.get("TD_STEPS", 400))
ARCH = os.environ.get("TD_ARCH", "--vocab-size 30522 --code-bits 128 --width 512 --blocks 8 --kernel 5 --cycle 4").split()
GLOBAL = ["--match", "1", "--global-every", "4", "--global-channels", "64", "--global-mean", "1"]
SEQ = int(os.environ.get("TD_SEQ", 256))
REP = int(os.environ.get("TD_REP", 8))
DS = "https://huggingface.co/api/datasets/LocalLLaMA/typed-decisions/parquet/all/{}/0.parquet"
PUBLISHED = {"Laya fine-tuned (421M, GPU; published)": 0.766, "teacher self-agreement (published)": 0.735,
             "TypeSafe Jev 1.13.0 (published)": 0.727, "majority class (published)": 0.461,
             "Laya zero-shot (published)": 0.362}
TDIR = os.path.join(J.OUT, "td")


def flat(x, pre=""):
    if isinstance(x, dict):
        return " ; ".join(flat(v, (pre + " " + str(k)).strip()) for k, v in x.items())
    if isinstance(x, list):
        return (pre + " : " if pre else "") + " , ".join(flat(v) for v in x)
    return (pre + " " if pre else "") + str(x).replace("_", " ")


def options(q):
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def load():
    import pyarrow.parquet as pq
    os.makedirs(TDIR, exist_ok=True)
    out = {}
    for split in ("train", "test"):
        p = os.path.join(TDIR, f"{split}.parquet")
        if not os.path.exists(p):
            with urllib.request.urlopen(DS.format(split), timeout=120) as r, open(p + ".tmp", "wb") as f:
                f.write(r.read())
            os.replace(p + ".tmp", p)
        out[split] = pq.read_table(p).to_pylist()
    tok = SH.C.tokenizer()
    rng = np.random.default_rng(20260925)
    train = out["train"]
    calib = set(rng.permutation(len(train))[: len(train) // 10].tolist())
    Q = {}  # (workflow, qid) -> {"type","options", split -> {"ids","y","p","score"}}
    for split, rows in (("train", train), ("test", out["test"])):
        ids = [np.asarray(e.ids[:SEQ], np.int32) for e in
               tok.encode_batch([flat(json.loads(r["state"])) for r in rows], add_special_tokens=False)]
        for i, r in enumerate(rows):
            part = split if split == "test" else ("calib" if i in calib else "train")
            qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
            for qid, q in qs.items():
                key = (r["workflow"], qid)
                e = Q.setdefault(key, {"type": q["type"], "options": options(q)})
                opts, g = e["options"], gold[qid]
                pr = np.array([g["probabilities"].get(o, 0.0) for o in opts], np.float64)
                pr = pr / pr.sum() if pr.sum() > 0 else np.full(len(opts), 1 / len(opts))
                lab = str(g["label"]).lower() if q["type"] == "noul" else str(g["label"])
                d = e.setdefault(part, {"ids": [], "y": [], "p": [], "score": []})
                d["ids"].append(ids[i])
                d["y"].append(opts.index(lab))
                d["p"].append(pr)
                d["score"].append(float(g.get("score", 0.0)))
    for e in Q.values():
        for d in (e[s] for s in ("train", "calib", "test")):
            d["y"], d["p"] = np.array(d["y"]), np.stack(d["p"])
    return Q


# ------------------------------------------------------------------ arms: (Q, key) -> calib logits, test logits
def arm_majority(e):
    c = np.bincount(e["train"]["y"], minlength=len(e["options"])).astype(float) + 1e-3
    z = np.log(c / c.sum())
    return np.tile(z, (len(e["calib"]["y"]), 1)), np.tile(z, (len(e["test"]["y"]), 1)), {}


def arm_bow(e):
    from sklearn.linear_model import LogisticRegression
    X = {s: SH.bow_feats({"ids": [e[s]["ids"]]}) for s in ("train", "calib", "test")}
    K, best = len(e["options"]), None
    for c in (0.1, 1.0, 10.0):
        m = LogisticRegression(C=c, max_iter=3000).fit(X["train"], e["train"]["y"])
        z = SH.logits_of(m, X["calib"], K)
        nll = -np.log(np.clip(SH.softmax(z)[np.arange(len(z)), e["calib"]["y"]], 1e-12, 1)).mean()
        if best is None or nll < best[0]:
            best = (nll, m)
    return SH.logits_of(best[1], X["calib"], K), SH.logits_of(best[1], X["test"], K), {}


def arm_berttiny(e, epochs=10, bs=16):
    import torch
    from transformers import BertModel
    torch.manual_seed(17)
    torch.set_num_threads(J.T)
    K = len(e["options"])
    enc = BertModel.from_pretrained(SH.BERT_REPO, cache_dir=os.path.join(SH.C.DATA, "hf"))
    head = torch.nn.Linear(2 * enc.config.hidden_size, K)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
    D = {s: {"ids": [e[s]["ids"]]} for s in ("train", "calib", "test")}
    P = torch.as_tensor(e["train"]["p"], dtype=torch.float32)
    n = len(e["train"]["y"])
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))

    def fwd(d, idx):
        ids, am, tt = SH.bert_inputs(d, idx)
        h = enc(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state
        mean = (h * am[..., None]).sum(1) / am.sum(1, keepdim=True)
        return head(torch.cat([h[:, 0], mean], 1))

    def predict(s):
        enc.eval()
        with torch.no_grad():
            m = len(e[s]["y"])
            return np.concatenate([fwd(D[s], range(i, min(m, i + 64))).numpy() for i in range(0, m, 64)])
    rng, best = np.random.default_rng(17), None
    for ep in range(epochs):
        enc.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = -(P[b] * torch.log_softmax(fwd(D["train"], b), -1)).sum(-1).mean()  # soft CE (Laya's supervised term)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        zc = predict("calib")
        ce = -(e["calib"]["p"] * np.log(np.clip(SH.softmax(zc), 1e-12, 1))).sum(-1).mean()
        if best is None or ce < best[0]:
            best = (ce, ep, zc, predict("test"))
    return best[2], best[3], {"epoch": best[1] + 1}


def arm_lae(e, key, extra):
    tag = f"{key[0]}__{key[1]}__{'gm' if extra else 'plain'}"
    K = len(e["options"])
    ks = [None] if e["type"] == "noul" else list(range(K))
    files = {}
    for s in ("calib", "test"):
        files[s] = os.path.join(TDIR, f"{key[0]}__{key[1]}_{s}.ids")
        if not os.path.exists(files[s]):
            SH.ids_file(files[s], e[s]["ids"], e[s]["y"] * 0, SEQ)
    rng = np.random.default_rng(abs(hash(tag)) % (2 ** 32))
    zc, zt, lats = [], [], []
    for k in ks:
        m = os.path.join(TDIR, f"{tag}_{'bin' if k is None else k}")
        pk = e["train"]["p"][:, 1 if k is None else k]
        rows, ys = [], []
        for i in range(len(pk)):  # soft labels: R copies per case, label ~ Bernoulli(teacher probability)
            for _ in range(REP):
                rows.append(e["train"]["ids"][i])
                ys.append(int(rng.random() < pk[i]))
        SH.ids_file(m + "_train.ids", rows, np.array(ys), SEQ)
        with open(m + ".log", "w") as log:
            subprocess.run([J.LT, "train", "--threads", str(J.T), "--data", m + "_train.ids", "--format", "ids", "--seq", str(SEQ),
                            "--batch", "32", "--steps", str(STEPS), "--eval-every", str(STEPS), "--seed", "17",
                            "--save", m + ".ltc", "--export", m + ".lth", *ARCH, *extra], stderr=log, check=True)
        J.publish([(m + ".lth", os.path.basename(m) + ".lth")])  # downloadable as soon as it exists (pod disks are not durable)
        zc.append(J.lae_votes(m + ".lth", files["calib"], SEQ))
        zt.append(J.lae_votes(m + ".lth", files["test"], SEQ))
        r = subprocess.run([J.FL, "bench", m + ".lth", files["test"], str(SEQ), "--batch", "1", "--lanes", "1", "--threads", "1",
                            "--repeats", "20"], capture_output=True, text=True)
        try:
            lats.append(json.loads(r.stdout.strip().splitlines()[-1])["fast_ms"])
        except (IndexError, ValueError, KeyError):
            lats.append(float("nan"))
        for s in (".ltc",):
            if os.path.exists(m + s):
                os.remove(m + s)
    if ks == [None]:
        zc, zt = J.to_logits(zc[0], 2, True), J.to_logits(zt[0], 2, True)
    else:
        zc, zt = np.stack(zc, 1), np.stack(zt, 1)
    return zc, zt, {"models": len(ks), "latency_ms_fastlae": float(np.sum(lats))}


# ------------------------------------------------------------------ evaluation (Laya's metric)
def evaluate(Q, Z):
    """Z: key -> (calib logits, test logits). Temperature per (type, K) pooled over questions on calib."""
    buckets = {}
    for key, (zc, zt) in Z.items():
        e = Q[key]
        buckets.setdefault((e["type"], len(e["options"])), []).append(key)
    rows = []
    for b, keys in buckets.items():
        zc = np.concatenate([Z[k][0] for k in keys])
        yc = np.concatenate([Q[k]["calib"]["y"] for k in keys])
        t = SH.fit_temperature(zc, yc)
        for k in keys:
            e, p = Q[k], SH.softmax(Z[k][1] / t)
            y, g = e["test"]["y"], e["test"]["p"]
            for i in range(len(y)):
                r = {"workflow": k[0], "question": k[1], "type": e["type"], "correct": int(p[i].argmax() == y[i]),
                     "conf": float(p[i].max()), "soft": float((p[i] * g[i]).sum()), "brier": float(((p[i] - g[i]) ** 2).sum())}
                if e["type"] == "score":
                    r["mae"] = abs(float((p[i] * np.arange(len(p[i]))).sum()) - e["test"]["score"][i])
                rows.append(r)
    acc = np.mean([r["correct"] for r in rows])
    conf, cor = np.array([r["conf"] for r in rows]), np.array([r["correct"] for r in rows])
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(cor[bins == i].mean() - conf[bins == i].mean()) * (bins == i).mean() for i in range(15) if (bins == i).any())
    out = {"decisions": len(rows), "accuracy": round(float(acc), 4), "soft_acc": round(float(np.mean([r["soft"] for r in rows])), 4),
           "brier": round(float(np.mean([r["brier"] for r in rows])), 4), "ece": round(float(ece), 4),
           "score_mae": round(float(np.mean([r["mae"] for r in rows if "mae" in r])), 4)}
    for f in ("workflow", "type"):
        out["by_" + f] = {v: round(float(np.mean([r["correct"] for r in rows if r[f] == v])), 4) for v in sorted({r[f] for r in rows})}
    return out


def summary(res):
    lines = ["# LocalLLaMA/typed-decisions (Laya's benchmark): 2,000 test decisions", "",
             "| system | accuracy | soft acc | Brier | ECE | score MAE | ms / case (CPU, 1 thread) |", "|---|---|---|---|---|---|---|"]
    allr = [(r["arm"], r) for r in res] + [(k, {"accuracy": v}) for k, v in PUBLISHED.items()]
    for name, r in sorted(allr, key=lambda x: -x[1]["accuracy"]):
        lines.append(f"| {name} | {r['accuracy']:.3f} | {r.get('soft_acc', '-')} | {r.get('brier', '-')} | {r.get('ece', '-')} | "
                     f"{r.get('score_mae', '-')} | {r.get('latency_ms_case', '-')} |")
    lines += ["", "| system | " + " | ".join(sorted(res[0]["by_workflow"])) + " | " + " | ".join(sorted(res[0]["by_type"])) + " |",
              "|---|" + "---|" * (len(res[0]["by_workflow"]) + len(res[0]["by_type"]))]
    for r in res:
        lines.append(f"| {r['arm']} | " + " | ".join(str(r["by_workflow"][w]) for w in sorted(r["by_workflow"])) + " | "
                     + " | ".join(str(r["by_type"][t]) for t in sorted(r["by_type"])) + " |")
    with open(os.path.join(J.OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(TDIR, exist_ok=True)
    J.say(f"typed-decisions: arms {ARMS}, LogicAE {STEPS} steps/model {' '.join(ARCH)}, seq {SEQ}, soft-label copies {REP}, threads {J.T}")
    if any(a.startswith("lae") for a in ARMS):
        J.build()
    Q = load()
    J.say(f"data: {len(Q)} questions, {sum(len(e['train']['y']) for e in Q.values())} train / "
          f"{sum(len(e['calib']['y']) for e in Q.values())} calib / {sum(len(e['test']['y']) for e in Q.values())} test decisions")
    rpath = os.path.join(J.OUT, "results.jsonl")
    done = {json.loads(l)["arm"] for l in open(rpath)} if os.path.exists(rpath) else set()
    res = [json.loads(l) for l in open(rpath)] if os.path.exists(rpath) else []
    for arm in ARMS:
        if arm in done:
            continue
        J.say(f"{arm}: start")
        t0, Z, info = time.time(), {}, {}
        for j, (key, e) in enumerate(sorted(Q.items())):
            if arm == "majority":
                zc, zt, inf = arm_majority(e)
            elif arm == "bow":
                zc, zt, inf = arm_bow(e)
            elif arm == "berttiny":
                zc, zt, inf = arm_berttiny(e)
            else:
                zc, zt, inf = arm_lae(e, key, GLOBAL if arm == "lae_gm" else [])
            Z[key] = (zc, zt)
            info[f"{key[0]}/{key[1]}"] = inf
            J.say(f"{arm}: {j + 1}/{len(Q)} {key[0]}/{key[1]} ({(time.time() - t0) / 60:.1f} min)")
        r = dict(arm=arm, **evaluate(Q, Z), minutes=round((time.time() - t0) / 60, 1))
        lat = [v.get("latency_ms_fastlae") for v in info.values() if v.get("latency_ms_fastlae") is not None]
        if lat:
            r["latency_ms_case"] = round(float(np.sum(lat)) / 4, 3)  # all models of one workflow's 5 questions, mean over workflows
        res.append(r)
        with open(rpath, "a") as f:
            f.write(json.dumps(r) + "\n")
        summary(res)
        J.say("RESULT " + json.dumps({k: v for k, v in r.items() if not k.startswith("by_")}))
    J.say("done")


if __name__ == "__main__":
    main()
