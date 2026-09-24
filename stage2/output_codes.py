"""
Output (target) code test: which code should the GraphTM predict for the hidden token?

For each output code kind, two numbers on the 10k validation windows:
  cap    bert-tiny's full MLM distribution P pushed through the decoder: votes = P @ (2B - 1),
         score_t = votes . D_t. How much a model that knew everything bert-tiny knows could show.
  ridge  linear readout from the +-2 neighbours' learned input codes (GTM_CODES=dist) to the
         output bits, same decoder. What a GraphTM (which tracks this readout) can reach.
B are the target bits the TM is trained on; D is the decoder matrix used for ranking: the same bits
(+-1), or the real-valued embedding the bits were cut from ("cont": tied like BERT's output layer).

Kinds: teacher256 (current: LSH of bert-base input embeddings), random256, dist128 (the input
codes themselves, fully tied), dist256 (256-d co-occurrence SVD, balanced sign bits).

usage: python3 output_codes.py [--train 500000]
"""
import argparse
import os

import numpy as np

import code_ceiling as CC
import common as C
import eval_mlm as E


def bert_probs(val):
    p = os.path.join(C.DATA, "bert_tiny_val10k_probs.npy")
    if os.path.exists(p):
        return np.load(p)
    import torch
    from transformers import BertForMaskedLM
    import bert_baselines as BB
    m = BertForMaskedLM.from_pretrained(BB.REPO, cache_dir=os.path.join(C.DATA, "hf")).eval()
    _, cand, _, _ = E.context()
    seqs, mpos = [], []
    for w in val["windows"]:
        v = w[w >= 0].tolist()
        c = int((w[:C.WIN] >= 0).sum())
        v[c] = BB.MASK
        seqs.append([BB.CLS] + v + [BB.SEP])
        mpos.append(c + 1)
    S = np.zeros((len(seqs), len(cand)), np.float32)
    with torch.no_grad():
        for i, ids, att in BB.batches(seqs, 256):
            lg = m(input_ids=ids, attention_mask=att).logits
            S[i:i + len(ids)] = lg[torch.arange(len(ids)), torch.as_tensor(mpos[i:i + len(ids)])][:, torch.as_tensor(cand)].numpy()
    P = np.exp(S - S.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    np.save(p, P.astype(np.float16))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500_000)
    ap.add_argument("--l2", type=float, default=10.0)
    a = ap.parse_args()
    z = np.load(os.path.join(C.DATA, "wt103_train.npz"))
    flat, lens = z["flat"], z["lens"]
    teacher, cand, freq, freq_rank = E.context()
    V = len(teacher)
    val = np.load(os.path.join(C.DATA, "val10k.gtmd.npz"))
    true = val["true"]
    P = bert_probs(val).astype(np.float32)
    e128 = CC.dist_embedding(flat, V, 40_000_000)
    e256 = CC.dist_embedding(flat, V, 40_000_000, dim=256)
    inp = CC.balanced_sign(e128, freq)
    rng = np.random.default_rng(7)
    kinds = {  # name: (target bits, decoder matrix)
        "teacher256": (teacher.astype(bool), None),
        "random256": (rng.random((V, 256)) < 0.5, None),
        "dist128": (inp, None),
        "dist128-cont": (inp, e128),
        "dist256": (CC.balanced_sign(e256, freq), None),
        "dist256-cont": (CC.balanced_sign(e256, freq), e256),
    }
    offs = [-2, -1, 1, 2]
    pad = np.zeros(inp.shape[1], bool)
    A = np.zeros((len(offs) * 128 + 1,) * 2)
    ys, Xs = [], []
    for s in range(0, a.train, 100_000):
        ids, y = CC.train_windows(flat, lens, min(100_000, a.train - s), 1000 + s, offs)
        X = CC.features(ids, inp, pad)
        A += X.T @ X
        Xs.append(X)
        ys.append(y)
    Ai = np.linalg.inv(A + a.l2 * np.eye(len(A)))
    Xv = CC.features(val["windows"][:, [CC.WIN + o for o in offs]], inp, pad)
    for name, (bits, dec) in kinds.items():
        B = bits.astype(np.float32) * 2 - 1
        D = (B if dec is None else dec)[cand].astype(np.float32)
        votes = P @ B[cand]
        r = E.ranks_from_scores(lambda i, j: votes[i:j] @ D.T, true, cand, len(true))
        cap = E.summarize(r, true, freq_rank, name)
        W = Ai @ sum(X.T @ B[y] for X, y in zip(Xs, ys))
        pred = (Xv @ W).astype(np.float32)
        r = E.ranks_from_scores(lambda i, j: pred[i:j] @ D.T, true, cand, len(true))
        rid = E.summarize(r, true, freq_rank, name)
        print(f"{name:14s} cap acc@1 {cap['acc@1']:.4f} acc@10 {cap['acc@10']:.4f} | "
              f"ridge acc@1 {rid['acc@1']:.4f} acc@10 {rid['acc@10']:.4f} mrr {rid['mrr']:.4f} "
              f"rare@10 {rid['acc@10_rare']:.4f}", flush=True)


if __name__ == "__main__":
    main()
