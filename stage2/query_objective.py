"""
Is the output cap the code, or the objective? Linear models on the same input (+-2 neighbours'
learned codes, 513 features), ranked by score_t = q . D_t over the 29,522 real tokens:
  ridge-bits     q regresses the target bits independently (what the TM's per-bit training does)
  softmax-codes  q trained with softmax cross-entropy through the SAME fixed codes (rank the true
                 token above all others; BERT's output objective) -- teacher256 and dist256
  full-softmax   unconstrained linear softmax (513 x 29,522): the linear ceiling with no code
usage: python3 query_objective.py [--train 500000] [--epochs 2]
"""
import argparse
import os

import numpy as np
import torch

import code_ceiling as CC
import common as C
import eval_mlm as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(0)
    z = np.load(os.path.join(C.DATA, "wt103_train.npz"))
    flat, lens = z["flat"], z["lens"]
    teacher, cand, freq, freq_rank = E.context()
    V = len(teacher)
    val = np.load(os.path.join(C.DATA, "val10k.gtmd.npz"))
    e128 = CC.dist_embedding(flat, V, 40_000_000)
    inp = CC.balanced_sign(e128, freq)
    d256 = CC.balanced_sign(CC.dist_embedding(flat, V, 40_000_000, dim=256), freq)
    offs = [-2, -1, 1, 2]
    pad = np.zeros(128, bool)
    ids, y = CC.train_windows(flat, lens, a.train, 1000, offs)
    X = torch.from_numpy(CC.features(ids, inp, pad))
    pos = np.full(V, -1)
    pos[cand] = np.arange(len(cand))
    Y = torch.from_numpy(pos[y])
    keep = Y >= 0
    X, Y = X[keep], Y[keep]
    Xv = torch.from_numpy(CC.features(val["windows"][:, [CC.WIN + o for o in offs]], inp, pad))
    true = val["true"]
    for name, codes in [("softmax-teacher256", teacher), ("softmax-dist256", d256), ("full-softmax", None)]:
        if codes is None:
            D = None
            W = torch.zeros(X.shape[1], len(cand), requires_grad=True)
        else:
            D = torch.from_numpy(codes[cand].astype(np.float32) * 2 - 1)
            W = torch.zeros(X.shape[1], D.shape[1], requires_grad=True)
        s = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([W, s], lr=3e-3)
        for ep in range(a.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 2048):
                b = perm[i:i + 2048]
                q = X[b] @ W
                logits = q if D is None else (q @ D.T) * s.exp()
                loss = torch.nn.functional.cross_entropy(logits, Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            q = Xv @ W
            sc = (lambda i, j: q[i:j].numpy()) if D is None else (lambda i, j: (q[i:j] @ D.T).numpy())
        r = E.ranks_from_scores(sc, true, cand, len(true))
        res = E.summarize(r, true, freq_rank, name)
        print(f"{name:20s} acc@1 {res['acc@1']:.4f} acc@10 {res['acc@10']:.4f} mrr {res['mrr']:.4f} "
              f"rare@10 {res['acc@10_rare']:.4f}", flush=True)


if __name__ == "__main__":
    main()
