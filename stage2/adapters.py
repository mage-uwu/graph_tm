"""
Downstream adapters on a FROZEN pretrained GraphTM.

Features (binary):
  gtm   clause-output bits of the pretrained model (`gtm score --bits`).
          sentence tasks: one window graph per token with that token masked (the model's
          contextual prediction at every position), OR-pooled over the sentence.
          qnli: [pool(question), pool(sentence), pool(question) AND pool(sentence)].
          conll: the graph that masks the word's first piece (context only, no identity).
  bow   hashed token ids: bag of the sentence (sst2), [bag(q), bag(s), bag(q) AND bag(s)] (qnli),
        ids at i-1, i, i+1 (conll). The lexical baseline.
  gtm+bow  both.
Heads:
  logreg  logistic regression, L2 strength picked on dev
  tm      depth-1 GraphTM on one node per example (a plain coalesced TM), s and epoch picked on dev
Columns that are constant on train are dropped; the TM head keeps at most TM_MAX_F features
(highest train variance) because the engine takes at most 8192 input bits per layer.

usage: python3 adapters.py --task sst2|qnli|conll --features gtm|bow|gtm+bow [--model M] [--heads logreg,tm]
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

import common as C
import tasks as TK

CHUNK = 250000
BOW_BINS = {"sst2": 4096, "qnli": 2048, "conll": 2048}
TM_MAX_F = 8190
TM_CLAUSES, TM_EPOCHS = 2000, 6
# (T, s) grid picked on dev. T must be large relative to the number of clauses firing per
# example, or weight updates overshoot the vote margin every step (see stage2/README.md).
TM_GRID = ((2000, 5.0), (8000, 5.0), (8000, 15.0))
LOGREG_C = (0.01, 0.1, 1.0)


# ---------------------------------------------------------------- features
def gtm_bits(model, seqs, centres, threads, tag):
    """packed clause bits (N, Cw) for window graphs with the centre token masked"""
    t = np.load(os.path.join(C.DATA, "teacher.npz"))
    mask_id = int(np.nonzero(t["vocab"] == "[MASK]")[0][0])
    rows = C.symbol_rows(len(t["codes"]), mask_id=mask_id)
    hdr = open(model, "rb").read(16)
    n_clauses, n_out = int(np.frombuffer(hdr, "<u4", 2, 8)[0]), int(np.frombuffer(hdr, "<u4", 1, 12)[0])
    out = []
    for i in range(0, len(centres), CHUNK):
        c = centres[i:i + CHUNK]
        npg, epn, edges, X, nt, _ = C.window_graphs(seqs, c, mask_id, rows)
        p = os.path.join(C.DATA, f"_feat_{tag}.gtmd")
        C.write_gtmd(p, C.H, C.N_NODE_TYPES, C.N_EDGE_TYPES, n_out, 1, npg, epn, edges, X,
                     np.zeros((len(c), n_out), np.int32), nt)
        b = C.score_bits(model, p, len(c), n_clauses, threads, tag)
        os.remove(p)
        out.append(np.packbits(b, axis=1, bitorder="little"))
    return np.concatenate(out), n_clauses


def pooled(model, seqs, threads, tag):
    """OR over the per-position masked graphs of each sequence -> (n_seqs, C) bool"""
    lens = np.array([len(s) for s in seqs])
    assert (lens > 0).all()
    centres = np.stack([np.repeat(np.arange(len(seqs)), lens), np.concatenate([np.arange(l) for l in lens])], 1)
    packed, nc = gtm_bits(model, seqs, centres, threads, tag)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    pool = np.bitwise_or.reduceat(packed, starts, axis=0)
    return np.unpackbits(pool, axis=1, bitorder="little")[:, :nc].astype(bool)


def bag(seqs, bins):
    f = np.zeros((len(seqs), bins), dtype=bool)
    for i, s in enumerate(seqs):
        f[i, s % bins] = True
    return f


def token_rows(d):
    """(sentence, word-start position) for every word of a conll split"""
    return np.concatenate([np.stack([np.full(len(st), i), st], 1) for i, st in enumerate(d["starts"])])


def features(task, kind, split, d, model, threads):
    parts = []
    if task == "sst2":
        if "gtm" in kind:
            parts.append(pooled(model, d["a"], threads, f"{task}{split}"))
        if "bow" in kind:
            parts.append(bag(d["a"], BOW_BINS[task]))
    elif task == "qnli":
        if "gtm" in kind:
            qa, sa = pooled(model, d["a"], threads, f"{task}{split}q"), pooled(model, d["b"], threads, f"{task}{split}s")
            parts += [qa, sa, qa & sa]
        if "bow" in kind:
            qa, sa = bag(d["a"], BOW_BINS[task]), bag(d["b"], BOW_BINS[task])
            parts += [qa, sa, qa & sa]
    elif task == "conll":
        cent = token_rows(d)
        if "gtm" in kind:
            packed, nc = gtm_bits(model, d["a"], cent, threads, f"{task}{split}")
            parts.append(np.unpackbits(packed, axis=1, bitorder="little")[:, :nc].astype(bool))
        if "bow" in kind:
            B = BOW_BINS[task]
            for off in (-1, 0, 1):
                f = np.zeros((len(cent), B), dtype=bool)
                for r, (si, p) in enumerate(cent):
                    q = p + off
                    if 0 <= q < len(d["a"][si]):
                        f[r, d["a"][si][q] % B] = True
                parts.append(f)
    return np.concatenate(parts, 1)


def labels(task, d):
    return np.concatenate(d["tags"]) if task == "conll" else d["y"]


# ---------------------------------------------------------------- metrics
def metric(task, d, pred):
    if task != "conll":
        return {"acc": float((pred == d["y"]).mean())}
    lens = [len(t) for t in d["tags"]]
    ps = np.split(pred, np.cumsum(lens)[:-1])
    return {"entity_f1": TK.entity_f1(d["tags"], ps), "token_acc": float((pred == np.concatenate(d["tags"])).mean())}


def key_metric(m):
    return m.get("entity_f1", m.get("acc"))


# ---------------------------------------------------------------- heads
def head_logreg(F, Y, D, task):
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    Xs = {s: csr_matrix(F[s].astype(np.float32)) for s in F}
    best = None
    for c in LOGREG_C:
        clf = LogisticRegression(C=c, max_iter=400).fit(Xs["train"], Y["train"])
        m = metric(task, D["dev"], clf.predict(Xs["dev"]))
        if best is None or key_metric(m) > key_metric(best[1]):
            best = (c, m, clf)
    c, mdev, clf = best
    return {"dev": mdev, "test": metric(task, D["test"], clf.predict(Xs["test"])), "C": c}


def tm_dataset(path, F, y, n_classes):
    n, f = F.shape
    Y = np.zeros((n, n_classes), np.int32)
    Y[np.arange(n), y] = 1
    C.write_gtmd(path, f, 1, 1, n_classes, 0, np.ones(n), np.zeros(n), np.zeros((0, 2)), C.literal_rows(F), Y)


def tm_predict(model, path, n, n_classes, threads):
    s, _ = C.score_sums(model, path, n, n_classes, threads, "tmhead")
    return s.argmax(1)


def head_tm(F, Y, D, task, threads, tag):
    nc = TK.N_CLASSES[task]
    paths = {s: os.path.join(C.DATA, f"_tm_{tag}_{s}.gtmd") for s in F}
    for s in F:
        tm_dataset(paths[s], F[s], Y[s], nc)
    mp, bp = os.path.join(C.DATA, f"_tm_{tag}.gtmm"), os.path.join(C.DATA, f"_tm_{tag}_best.gtmm")
    best, curve = None, []
    for T_val, s_val in TM_GRID:
        C.gtm("init", "--data", paths["train"], "--out", mp, "--clauses", TM_CLAUSES, "--T", T_val, "--s", s_val,
              "--depth", 1)
        for ep in range(TM_EPOCHS):
            t0 = time.time()
            C.gtm("train", "--data", paths["train"], "--model", mp, "--epochs", 1, "--threads", threads, "--save", mp)
            dt = time.time() - t0
            m = metric(task, D["dev"], tm_predict(mp, paths["dev"], len(Y["dev"]), nc, threads))
            curve.append((T_val, s_val, ep, round(key_metric(m), 4), round(len(Y["train"]) / dt)))
            if best is None or key_metric(m) > key_metric(best[2]):
                best = ((T_val, s_val), ep, m)
                shutil.copy(mp, bp)
    test = metric(task, D["test"], tm_predict(bp, paths["test"], len(Y["test"]), nc, threads))
    for p in list(paths.values()) + [mp, bp]:
        os.remove(p)
    return {"dev": best[2], "test": test, "T_s": best[0], "epochs": best[1] + 1, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TK.N_CLASSES))
    ap.add_argument("--features", required=True, choices=["gtm", "bow", "gtm+bow"])
    ap.add_argument("--model")
    ap.add_argument("--tag", default="")
    ap.add_argument("--heads", default="logreg,tm")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N train/dev/test sequences")
    a = ap.parse_args()
    D = TK.load(a.task)
    if a.limit:
        D = {s: {k: v[:a.limit] for k, v in d.items()} for s, d in D.items()}
    t0 = time.time()
    F = {s: features(a.task, a.features, s, D[s], a.model, a.threads) for s in ("train", "dev", "test")}
    Y = {s: labels(a.task, D[s]) for s in F}
    keep = F["train"].any(0) & ~F["train"].all(0)
    F = {s: v[:, keep] for s, v in F.items()}
    info = {"task": a.task, "features": a.features, "tag": a.tag, "n_features": int(keep.sum()),
            "density": float(F["train"].mean()), "feature_s": round(time.time() - t0, 1)}
    print(f"[{a.task}/{a.features}/{a.tag}] {info['n_features']} features, density {info['density']:.3f}, "
          f"built in {info['feature_s']}s", flush=True)
    for h in a.heads.split(","):
        t0 = time.time()
        if h == "logreg":
            r = head_logreg(F, Y, D, a.task)
        else:
            Ft = F
            if F["train"].shape[1] > TM_MAX_F:
                top = np.argsort(-F["train"].var(0))[:TM_MAX_F]
                Ft = {s: v[:, top] for s, v in F.items()}
            r = head_tm(Ft, Y, D, a.task, a.threads, f"{a.task}_{a.features}")
        r.update(info, head=h, head_s=round(time.time() - t0, 1))
        print(f"  head {h}: dev {r['dev']}  TEST {r['test']}  ({r['head_s']}s)", flush=True)
        print("RESULT " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
