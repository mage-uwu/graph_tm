"""
Pretraining data: WikiText-103 -> masked window graphs with teacher-LSH targets.

  prep                       tokenize train/validation/test (one paragraph per line, headings
                             dropped) and build the teacher codes. Writes DATA/wt103_*.npz and
                             DATA/teacher.npz.
  shard  --split S --n N --seed K --out P
                             N masked windows sampled uniformly over the split's tokens
  evalset --split S --n N --out P
                             the same, plus P.npz with the true ids and window tokens (for the
                             retrieval metrics and the bert-tiny baseline)

Targets: bert-base-uncased input word embeddings (768-d), centred on the real-token mean,
signed against N_BITS fixed Gaussian hyperplanes (seed LSH_SEED). Every output is a balanced
binary target, which is what a coalesced TM handles natively; similar tokens get similar codes.
"""
import argparse
import os
import time

import numpy as np

import common as C

N_BITS = 256
LSH_SEED = 11
TEACHER_REPO = "bert-base-uncased"
WIKI = "Salesforce/wikitext"
SPLIT_FILES = {
    "train": ["wikitext-103-raw-v1/train-00000-of-00002.parquet", "wikitext-103-raw-v1/train-00001-of-00002.parquet"],
    "validation": ["wikitext-103-raw-v1/validation-00000-of-00001.parquet"],
    "test": ["wikitext-103-raw-v1/test-00000-of-00001.parquet"],
}


def special_ids(vocab):
    """ids that are never retrieval candidates: [PAD] [UNK] [CLS] [SEP] [MASK] [unused*]"""
    return np.array([i for i, t in enumerate(vocab) if t.startswith("[")], dtype=np.int64)


def prep():
    import pyarrow.parquet as pq
    os.makedirs(C.DATA, exist_ok=True)
    tok = C.tokenizer()
    for split, files in SPLIT_FILES.items():
        out = os.path.join(C.DATA, f"wt103_{split}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        lines = []
        for f in files:
            lines += pq.read_table(C.fetch(WIKI, f)).column("text").to_pylist()
        paras = [s.strip() for s in lines if s.strip() and not s.strip().startswith("=")]
        ids = [a for a in C.encode(tok, paras) if len(a) >= 2]
        lens = np.array([len(a) for a in ids], dtype=np.int64)
        np.savez(out, flat=np.concatenate(ids), lens=lens)
        print(f"{split}: {len(ids)} paragraphs, {lens.sum()} tokens ({time.time() - t0:.0f}s)", flush=True)

    out = os.path.join(C.DATA, "teacher.npz")
    if not os.path.exists(out):
        from safetensors import safe_open
        path = C.fetch(TEACHER_REPO, "model.safetensors", repo_type="model")
        with safe_open(path, "np") as f:
            key = [k for k in f.keys() if k.endswith("word_embeddings.weight")][0]
            E = f.get_tensor(key).astype(np.float64)
        vocab = tok.get_vocab()
        inv = [None] * len(vocab)
        for t, i in vocab.items():
            inv[i] = t
        real = np.setdiff1d(np.arange(len(inv)), special_ids(inv))
        R = np.random.default_rng(LSH_SEED).standard_normal((E.shape[1], N_BITS))
        codes = ((E - E[real].mean(0)) @ R > 0).astype(np.uint8)
        np.savez(out, codes=codes, candidates=real, vocab=np.array(inv))
        print(f"teacher: {TEACHER_REPO} {E.shape} -> {N_BITS}-bit codes, {len(real)} candidate tokens, "
              f"bit balance {codes[real].mean():.3f}", flush=True)


def load_split(split):
    z = np.load(os.path.join(C.DATA, f"wt103_{split}.npz"))
    flat, lens = z["flat"], z["lens"]
    return np.split(flat, np.cumsum(lens)[:-1]), lens


def sample_centres(lens, n, seed):
    """n (paragraph, position) pairs, uniform over all tokens"""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, lens.sum(), size=n)
    offs = np.cumsum(lens)
    para = np.searchsorted(offs, g, side="right")
    pos = g - (offs[para] - lens[para])
    return np.stack([para, pos], 1)


def build(split, n, seed, out, codes, mask_id, rows, seqs=None, lens=None):
    if seqs is None:
        seqs, lens = load_split(split)
    cent = sample_centres(lens, n, seed)
    if C.LAYOUT == "flat":
        npg, epn, edges, X, nt, wt, h = C.flat_graphs(seqs, cent, C.symbol_bits(len(codes)), pad_id=0)
        true = wt[:, C.WIN]
        C.write_gtmd(out, h, 1, 1, N_BITS, 1, npg, epn, edges, X, codes[true], nt)
        return true, wt
    npg, epn, edges, X, nt, wt = C.window_graphs(seqs, cent, mask_id, rows)
    true = wt[:, C.WIN]
    C.write_gtmd(out, C.H, C.N_NODE_TYPES, C.N_EDGE_TYPES, N_BITS, 1, npg, epn, edges, X, codes[true], nt)
    return true, wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "shard", "evalset"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "prep":
        return prep()
    t = np.load(os.path.join(C.DATA, "teacher.npz"))
    codes = t["codes"].astype(np.int32)
    mask_id = int(np.nonzero(t["vocab"] == "[MASK]")[0][0])
    rows = C.symbol_rows(len(codes), mask_id=mask_id)
    t0 = time.time()
    true, wt = build(a.split, a.n, a.seed + (0 if a.split == "train" else 10**6), a.out, codes, mask_id, rows)
    if a.cmd == "evalset":
        np.savez(a.out + ".npz", true=true, windows=wt)
    print(f"{a.cmd} {a.split}: {a.n} graphs -> {a.out} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
