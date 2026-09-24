"""System-1 shootout tasks: Laya / Jev-style typed questions over public datasets.

Each task is one typed question asked of every state (Laya's request shape):
  choice  pick one of K named options          -> probabilities over the options
  score   ordinal level 0..K-1                 -> probabilities over the levels (+ expected score)
  noul    calibrated P(statement is true)      -> probabilities [false, true]

  task       family (Laya's names)          type    K   source
  sst2       sentiment and rating           choice   2  nyu-mll/glue sst2 (test = validation)
  sst5       sentiment and rating           score    5  SetFit/sst5
  emotion    emotion and tone               choice   6  dair-ai/emotion
  agnews     topic classification           choice   4  fancyzhx/ag_news
  banking77  intent and routing             choice  77  legacy-datasets/banking77
  qnli       inference and fact checking    noul     2  nyu-mll/glue qnli (question, sentence)
  spam       moderation and safety          noul     2  ucirvine/sms_spam (seeded 70/10/20 split)
  jailbreak  moderation and safety          noul     2  jackhhao/jailbreak-classification

Splits: train capped at TRAIN_CAP (seeded subset), dev = DEV_N held out of train (head / epoch /
temperature selection), test = the official test (or validation) split capped at TEST_CAP (seeded).
Tokens: bert-base-uncased WordPiece (the vocabulary bert-tiny, the GraphTM and LogicAE share), no
special tokens; singles cut to 126, pairs to 48 + 78 (fits bert-tiny's 128 with [CLS]/[SEP]s).
"""
import json
import os
import pickle
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as C  # noqa: E402

TRAIN_CAP = int(os.environ.get("S1_TRAIN_CAP", 20000))
TEST_CAP = int(os.environ.get("S1_TEST_CAP", 2000))
DEV_N = int(os.environ.get("S1_DEV_N", 1000))
SINGLE_MAX, Q_MAX, S_MAX = 126, 48, 78
CACHE = os.path.join(C.DATA, "system1")

TASKS = {
    "sst2": dict(family="sentiment and rating", type="choice", ds="nyu-mll/glue", cfg="sst2", text="sentence",
                 test="validation", q="What is the sentiment of this review?", options=["negative", "positive"]),
    "sst5": dict(family="sentiment and rating", type="score", ds="SetFit/sst5", cfg="default", text="text", test="test",
                 q="How positive is this review?",
                 options=["very negative", "negative", "neutral", "positive", "very positive"]),
    "emotion": dict(family="emotion and tone", type="choice", ds="dair-ai/emotion", cfg="split", text="text",
                    test="test", q="Which emotion does the author express?",
                    options=["sadness", "joy", "love", "anger", "fear", "surprise"]),
    "agnews": dict(family="topic classification", type="choice", ds="fancyzhx/ag_news", cfg="default", text="text",
                   test="test", q="What is the topic of this news article?",
                   options=["world", "sports", "business", "science and technology"]),
    "banking77": dict(family="intent and routing", type="choice", ds="legacy-datasets/banking77", cfg="default",
                      text="text", test="test", q="Which banking request is the customer making?", options=None),
    "qnli": dict(family="inference and fact checking", type="noul", ds="nyu-mll/glue", cfg="qnli",
                 text=("question", "sentence"), test="validation", flip=True,
                 q="Does the sentence contain the answer to the question?"),
    "spam": dict(family="moderation and safety", type="noul", ds="ucirvine/sms_spam", cfg="plain_text", text="sms",
                 test=None, q="Is this message spam?"),
    "jailbreak": dict(family="moderation and safety", type="noul", ds="jackhhao/jailbreak-classification",
                      cfg="default", text="prompt", label="type", label_map={"benign": 0, "jailbreak": 1},
                      test="test", q="Is this prompt a jailbreak attempt?"),
}
# glue qnli: 0 = entailment (answers), 1 = not_entailment; flip so that noul "true" (1) = answers


def _get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(path + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(path + ".tmp", path)
    return path


def _parquet(ds, cfg, split):
    import pyarrow.parquet as pq
    urls = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=60))[cfg][split]
    tabs = [pq.read_table(_get(u, os.path.join(CACHE, "hf", ds.replace("/", "__"), cfg, split, f"{i}.parquet")))
            for i, u in enumerate(urls)]
    import pyarrow as pa
    return pa.concat_tables(tabs)


def _label_names(ds, cfg, col="label"):
    info = json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/info?dataset={ds}", timeout=60))
    return info["dataset_info"][cfg]["features"][col]["names"]


def _labels(spec, t):
    col = spec.get("label", "label")
    y = t.column(col).to_pylist()
    if "label_map" in spec:
        y = [spec["label_map"][v] for v in y]
    y = np.asarray(y, np.int64)
    return 1 - y if spec.get("flip") else y


def _texts(spec, t):
    if isinstance(spec["text"], tuple):
        return [t.column(c).to_pylist() for c in spec["text"]]
    return [t.column(spec["text"]).to_pylist()]


def _subset(n, cap, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return np.sort(idx[:cap]) if n > cap else np.arange(n)


def load(name):
    """{'spec': question spec, 'train'|'dev'|'test': {'text': [cols], 'ids': [cols of int32 arrays], 'y'}}"""
    path = os.path.join(CACHE, f"s1_{name}_{TRAIN_CAP}_{DEV_N}_{TEST_CAP}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    spec = dict(TASKS[name])
    if spec["type"] == "noul":
        spec["options"] = ["false", "true"]
    if spec["options"] is None:
        spec["options"] = [s.replace("_", " ").rstrip("?") for s in _label_names(spec["ds"], spec["cfg"])]
    tr = _parquet(spec["ds"], spec["cfg"], "train")
    Xtr, ytr = _texts(spec, tr), _labels(spec, tr)
    if spec["test"] is None:  # single split: seeded 70 / 10 / 20 (dev carved from the 80)
        perm = np.random.default_rng(1).permutation(len(ytr))
        te_idx, tr_idx = perm[: len(ytr) // 5], perm[len(ytr) // 5:]
        Xte, yte = [[c[i] for i in te_idx] for c in Xtr], ytr[te_idx]
        Xtr, ytr = [[c[i] for i in tr_idx] for c in Xtr], ytr[tr_idx]
    else:
        te = _parquet(spec["ds"], spec["cfg"], spec["test"])
        Xte, yte = _texts(spec, te), _labels(spec, te)
    dev_n = min(DEV_N, len(ytr) // 5)
    perm = np.random.default_rng(0).permutation(len(ytr))
    dv, trn = perm[:dev_n], perm[dev_n:][:TRAIN_CAP]
    ts = _subset(len(yte), TEST_CAP, 2)
    tok = C.tokenizer()
    res = {"spec": spec}
    for split, X, y, idx in (("train", Xtr, ytr, trn), ("dev", Xtr, ytr, dv), ("test", Xte, yte, ts)):
        cols = [[str(c[i] or "") for i in idx] for c in X]
        caps = [SINGLE_MAX] if len(cols) == 1 else [Q_MAX, S_MAX]
        ids = [[a[:cap] for a in C.encode(tok, col)] for col, cap in zip(cols, caps)]
        for col in ids:  # empty after tokenization -> one [UNK] so every backend sees a token
            for j, a in enumerate(col):
                if len(a) == 0:
                    col[j] = np.array([100], np.int32)
        res[split] = {"text": cols, "ids": ids, "y": y[idx]}
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


if __name__ == "__main__":
    for n in (sys.argv[1:] or TASKS):
        d = load(n)
        s = d["spec"]
        print(n, s["type"], len(s["options"]), {k: len(d[k]["y"]) for k in ("train", "dev", "test")},
              "classes", np.bincount(d["train"]["y"], minlength=len(s["options"])).tolist()[:10], flush=True)
