"""
Downstream tasks, one of each classic BERT task shape, tokenized with the pretraining vocabulary.

  sst2   single sentence, 2 classes (GLUE SST-2)
  qnli   sentence pair, 2 classes (GLUE QNLI: does the sentence answer the question?)
  conll  token tagging, 9 BIO tags (CoNLL-2003 NER), scored by entity-level F1

Splits: GLUE test labels are hidden, so the official validation set is our test set and a
seeded slice of train (DEV_N examples) is the dev set for head/epoch selection. CoNLL has
real train/validation/test.

load(name) -> dict split -> dict of arrays; cached in DATA/task_<name>.pkl
"""
import os
import pickle

import numpy as np

import common as C

DEV_N = 2000
Q_MAX, S_MAX, SENT_MAX = 48, 96, 64
CONLL_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
N_CLASSES = {"sst2": 2, "qnli": 2, "conll": len(CONLL_TAGS)}


def _glue(name, cols):
    import pyarrow.parquet as pq
    out = {}
    for split in ("train", "validation"):
        t = pq.read_table(C.fetch("nyu-mll/glue", f"{name}/{split}-00000-of-00001.parquet"))
        out[split] = {c: t.column(c).to_pylist() for c in cols + ["label"]}
    return out


def _dev_split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    return perm[DEV_N:], perm[:DEV_N]


def _take(d, idx):
    return {k: [v[i] for i in idx] if isinstance(v, list) else v[idx] for k, v in d.items()}


def load(name):
    path = os.path.join(C.DATA, f"task_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    tok = C.tokenizer()
    if name in ("sst2", "qnli"):
        cols = ["sentence"] if name == "sst2" else ["question", "sentence"]
        raw = _glue(name, cols)
        out = {}
        for split, d in raw.items():
            e = {"y": np.array(d["label"], dtype=np.int64)}
            if name == "sst2":
                e["a"] = [x[:SENT_MAX] for x in C.encode(tok, d["sentence"])]
            else:
                e["a"] = [x[:Q_MAX] for x in C.encode(tok, d["question"])]
                e["b"] = [x[:S_MAX] for x in C.encode(tok, d["sentence"])]
            out[split] = e
        tr, dv = _dev_split(len(out["train"]["y"]))
        res = {"train": _take(out["train"], tr), "dev": _take(out["train"], dv), "test": out["validation"]}
    elif name == "conll":
        import pyarrow.parquet as pq
        res = {}
        for split, hf in (("train", "train"), ("dev", "validation"), ("test", "test")):
            t = pq.read_table(C.fetch("eriktks/conll2003", f"conll2003/{hf}/0000.parquet", revision="refs/convert/parquet"))
            words, tags = t.column("tokens").to_pylist(), t.column("ner_tags").to_pylist()
            seqs, starts, ys, keep = [], [], [], []
            for w, y in zip(words, tags):
                if not w:
                    continue
                pieces = [tok.encode(x, add_special_tokens=False).ids or [tok.token_to_id("[UNK]")] for x in w]
                st = np.cumsum([0] + [len(p) for p in pieces[:-1]])
                seqs.append(np.array([i for p in pieces for i in p], dtype=np.int32))
                starts.append(st.astype(np.int64))
                ys.append(np.array(y, dtype=np.int64))
            res[split] = {"a": seqs, "starts": starts, "tags": ys}
    else:
        raise ValueError(name)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def entity_spans(tags):
    """conlleval-style spans (type, start, end) from BIO tag ids; I-X after O/other type opens a span"""
    spans, cur = [], None
    for i, t in enumerate(list(tags) + [0]):
        name = CONLL_TAGS[t]
        if name == "O" or name.startswith("B-") or cur is None or name[2:] != cur[0]:
            if cur:
                spans.append((cur[0], cur[1], i))
            cur = (name[2:], i) if name != "O" else None
    return set(spans)


def entity_f1(true_seqs, pred_seqs):
    tp = fp = fn = 0
    for t, p in zip(true_seqs, pred_seqs):
        a, b = entity_spans(t), entity_spans(p)
        tp += len(a & b)
        fp += len(b - a)
        fn += len(a - b)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-12, prec + rec)
