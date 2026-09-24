"""
Token-code ceiling test: how much of the masked-token task can ANY model get from +-K context
when tokens are given as 128-bit codes of a given kind? Measured with the linear readout that
the GraphTM tracks (ridge from the +-1/+-2 neighbours' codes to the 256 teacher bits, ranked
over the 29,522 real tokens exactly as eval_mlm.py does). If a code kind lifts this ceiling,
a TM on that code has room to follow; if not, no model on that code can.

Code kinds (all 128 bits per token, each bit on for ~half of all token occurrences):
  random   the current input: fixed random 64-of-128 codes (common.symbol_bits)
  dist     learned from the corpus, gradient free: position-aware co-occurrence counts with the
           4096 most frequent tokens at offsets -2,-1,+1,+2, PPMI, 128-d SVD, sign at the
           frequency-weighted median. No teacher information.
  dist+id  64 dist bits + 64 random bits (similarity + identity)
  berttiny sign bits of bert-tiny's own learned input embeddings (reference: what a learned
           gradient-trained embedding gives at this width; not usable as a GraphTM input)

usage: python3 code_ceiling.py [--train 500000] [--cooc-tokens 40000000] [--offsets -2,-1,1,2]
"""
import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

import common as C
import eval_mlm as E

WIN = 8


def balanced_sign(emb, freq):
    """bits = emb > frequency-weighted median per dimension (each bit ~50% of occurrences)"""
    w = freq.astype(np.float64) + 1e-3
    bits = np.zeros(emb.shape, bool)
    for j in range(emb.shape[1]):
        o = np.argsort(emb[:, j])
        cw = np.cumsum(w[o])
        thr = emb[o[np.searchsorted(cw, cw[-1] / 2)], j]
        bits[:, j] = emb[:, j] > thr
    return bits


def dist_embedding(flat, V, n_tok, K=4096, offs=(-2, -1, 1, 2), dim=128, seed=0):
    p = os.path.join(C.DATA, f"dist_emb_{n_tok}_{K}_{dim}.npy")
    if os.path.exists(p):
        return np.load(p)
    x = flat[:n_tok].astype(np.int64)
    freq = np.bincount(x, minlength=V)
    ctx = np.full(V, -1, np.int64)
    ctx[np.argsort(-freq, kind="stable")[:K]] = np.arange(K)
    M = sp.csr_matrix((V, K * len(offs)), dtype=np.float32)
    step = 10_000_000
    for a in range(0, n_tok, step):
        b = min(n_tok, a + step)
        for k, o in enumerate(offs):
            lo, hi = max(a, -o), min(b, n_tok - o)
            w, c = x[lo:hi], ctx[x[lo + o:hi + o]]
            m = c >= 0
            M = M + sp.csr_matrix((np.ones(m.sum(), np.float32), (w[m], c[m] + k * K)), shape=M.shape)
    M = M.tocoo()
    tot = M.data.sum()
    rs = np.asarray(M.sum(1)).ravel()
    cs = np.asarray(M.sum(0)).ravel() ** 0.75
    cs /= cs.sum()
    pmi = np.log(M.data / tot) - np.log(rs[M.row] / tot) - np.log(cs[M.col])
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=M.shape)
    U, S, _ = randomized_svd(P, dim, random_state=seed, n_iter=5)
    emb = (U * np.sqrt(S)).astype(np.float32)
    np.save(p, emb)
    return emb


def berttiny_embedding():
    from transformers import BertModel
    m = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", cache_dir=os.path.join(C.DATA, "hf"))
    return m.embeddings.word_embeddings.weight.detach().numpy()


def train_windows(flat, lens, n, seed, offs):
    rng = np.random.default_rng(seed)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    g = rng.integers(0, len(flat), size=n)
    para = np.searchsorted(np.cumsum(lens), g, side="right")
    lo, hi = starts[para], starts[para] + lens[para]
    ids = np.stack([np.where((g + o >= lo) & (g + o < hi), flat[np.clip(g + o, 0, len(flat) - 1)], -1) for o in offs], 1)
    return ids, flat[g]


def features(ids, bits, pad_bits):
    """(n, len(offs)) token ids (-1 = outside) -> (n, len(offs)*H + 1) +-1 floats with bias"""
    b = np.where(ids[..., None] >= 0, bits[np.maximum(ids, 0)], pad_bits)
    f = b.reshape(len(ids), -1).astype(np.float32) * 2 - 1
    return np.hstack([f, np.ones((len(f), 1), np.float32)])


def ceiling(name, bits, flat, lens, val, codes, cand, freq_rank, a):
    offs = [int(o) for o in a.offsets.split(",")]
    pad = np.zeros(bits.shape[1], bool)
    A = np.zeros((len(offs) * bits.shape[1] + 1,) * 2, np.float64)
    B = np.zeros((A.shape[0], codes.shape[1]), np.float64)
    for s in range(0, a.train, 100_000):
        ids, y = train_windows(flat, lens, min(100_000, a.train - s), 1000 + s, offs)
        X = features(ids, bits, pad)
        A += X.T @ X
        B += X.T @ (codes[y].astype(np.float32) * 2 - 1)
    W = np.linalg.solve(A + a.l2 * np.eye(len(A)), B).astype(np.float32)
    Xv = features(val["windows"][:, [WIN + o for o in offs]], bits, pad)
    pred = Xv @ W
    true = val["true"]
    cb = codes[cand].astype(np.float32) * 2 - 1
    r = E.ranks_from_scores(lambda i, j: pred[i:j] @ cb.T, true, cand, len(true))
    res = E.summarize(r, true, freq_rank, name, {"per-bit": float(((pred > 0) == codes[true].astype(bool)).mean())})
    E.show(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500_000)
    ap.add_argument("--cooc-tokens", type=int, default=40_000_000)
    ap.add_argument("--offsets", default="-2,-1,1,2")
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--kinds", default="random,dist,dist+id,berttiny")
    a = ap.parse_args()
    z = np.load(os.path.join(C.DATA, "wt103_train.npz"))
    flat, lens = z["flat"], z["lens"]
    codes, cand, freq, freq_rank = E.context()
    V = len(codes)
    val = np.load(os.path.join(C.DATA, "val10k.gtmd.npz"))
    rand = C.symbol_bits(V)
    for kind in a.kinds.split(","):
        if kind == "random":
            bits = rand
        elif kind == "dist":
            bits = balanced_sign(dist_embedding(flat, V, a.cooc_tokens), freq)
        elif kind == "dist+id":
            bits = np.hstack([balanced_sign(dist_embedding(flat, V, a.cooc_tokens), freq)[:, :64], rand[:, :64]])
        elif kind == "berttiny":
            bits = balanced_sign(berttiny_embedding(), freq)
        ceiling(kind, bits, flat, lens, val, codes, cand, freq_rank, a)


if __name__ == "__main__":
    main()
