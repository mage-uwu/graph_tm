"""Does the residual after layer 0 survive being snapped to a finite alphabet? (gate for a looped / table-per-pass model)

Frozen attention-only BitNet (float). Layer 0 exact. The residual after layer 0, x1, is product-quantized:
direction u = x1 / rms(x1) split into M subspaces of 128 / M dims, each snapped to the nearest of C k-means codes
(fit on training windows); rms(x1) kept exact (one scalar per token). Then layer 1 + head run exactly on it.
  strict      x1 := rms(x1) * decode(codes) everywhere (the codes are the state, as in a looped model)
  attn_only   layer 1 reads the decoded x1, the residual stream keeps the exact x1
Gain = share of the float BitNet's gain over unigram; first --eval-windows validation windows.

  python3 pq_residual.py [--ms 1,2,4,8,16] [--codes 256]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import common as K
from inspect_attn import load_attn
from lg1b import L1B

S = os.environ.get("SCRATCH", "/tmp")


def kmeans(Z, C, iters=20, seed=0):
    g = torch.Generator().manual_seed(seed); cen = Z[torch.randperm(len(Z), generator=g)[:C]].clone()
    for _ in range(iters):
        a = torch.cdist(Z, cen).argmin(1)
        s = torch.zeros_like(cen).index_add_(0, a, Z); n = torch.bincount(a, minlength=C).float()[:, None]
        cen = torch.where(n > 0, s / n.clamp(min=1), cen)
    return cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,2,4,8,16"); ap.add_argument("--codes", type=int, default=256)
    ap.add_argument("--fit-windows", type=int, default=2048); ap.add_argument("--eval-windows", type=int, default=1024)
    a1 = ap.parse_args(); torch.manual_seed(0)
    a = argparse.Namespace(layer=1, window=16, trees=8, levels=3, kbits=7, frac=8, rom_scale=16, seed=0, oracle_scale=0, route="table")
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); ntr = len(raw) * 9 // 10
    vb = np.frombuffer(raw[ntr:], np.uint8).astype(np.int64); nw = (len(vb) - 1) // T
    VX = torch.tensor(vb[:nw * T].reshape(nw, T)[:a1.eval_windows]); VY = torch.tensor(vb[1:nw * T + 1].reshape(nw, T)[:a1.eval_windows])
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[VY.numpy()].mean())
    b = np.frombuffer(raw[:ntr], np.uint8).astype(np.int64); rng = np.random.default_rng(0)
    fitX = b[rng.integers(0, len(b) - T, a1.fit_windows)[:, None] + np.arange(T)]
    net = L1B(m, a, fitX[:256])

    def x_after0(X):
        with torch.no_grad(): return net.teacher_l1(X)[0]

    def rest(x1in, x1res):
        """layer 1 exact on x1in (attention input), residual from x1res, then the head"""
        B, T_ = x1in.shape[:2]; q, k, v = net.qkv(x1in, 1)
        p = torch.softmax((q @ k.transpose(-1, -2) / net.hd ** .5).masked_fill(~net.causal, float("-inf")), -1)
        x = x1res + net.bitlin((p @ v).transpose(1, 2).reshape(B, T_, net.d), net.Wo[1])
        return (x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * net.gf) @ net.head

    t0 = time.time()
    with torch.no_grad():
        Z = torch.cat([x_after0(torch.tensor(fitX[i:i + 64])) for i in range(0, len(fitX), 64)]).reshape(-1, net.d)
        Z = Z / torch.sqrt((Z * Z).mean(-1, keepdim=True) + 1e-6)
        Zfit = Z[torch.randperm(len(Z), generator=torch.Generator().manual_seed(1))[:60000]]
        V1 = [x_after0(VX[i:i + 64]) for i in range(0, len(VX), 64)]

    def score(fn):
        ce = hit = 0.0
        with torch.no_grad():
            for j, i in enumerate(range(0, len(VX), 64)):
                lp = torch.log_softmax(fn(V1[j]), -1); Y = VY[i:i + 64]
                ce += -lp.gather(-1, Y[..., None]).sum().item(); hit += (lp.argmax(-1) == Y).sum().item()
        return ce / VY.numel(), hit / VY.numel()
    ce_t, acc_t = score(lambda x1: rest(x1, x1))
    g = lambda ce: round((UNI - ce) / (UNI - ce_t), 4)
    print(json.dumps(dict(teacher_ce=ce_t, teacher_acc=acc_t, fit_tokens=len(Zfit), setup_sec=round(time.time() - t0))), flush=True)
    for M in [int(s) for s in a1.ms.split(",")]:
        t1 = time.time(); ds = net.d // M
        books = [kmeans(Zfit[:, j * ds:(j + 1) * ds], a1.codes, seed=j) for j in range(M)]

        def pq(x1):
            r = torch.sqrt((x1 * x1).mean(-1, keepdim=True) + 1e-6); u = (x1 / r).reshape(-1, net.d); out = torch.empty_like(u)
            for j, cb in enumerate(books):
                sl = slice(j * ds, (j + 1) * ds); out[:, sl] = cb[torch.cdist(u[:, sl], cb).argmin(1)]
            return out.reshape(x1.shape) * r
        with torch.no_grad():
            u = Zfit[:20000]; rec = pq(u[None])[0]; rel = ((rec - u) ** 2).sum() / (u ** 2).sum()
        ce_s, acc_s = score(lambda x1: rest(pq(x1), pq(x1)))
        ce_a, acc_a = score(lambda x1: rest(pq(x1), x1))
        print(json.dumps(dict(M=M, codes=a1.codes, bits_per_token=M * int(np.log2(a1.codes)), rel_mse=round(rel.item(), 4),
                              strict=dict(ce=round(ce_s, 4), acc=round(acc_s, 4), gain=g(ce_s)),
                              attn_only=dict(ce=round(ce_a, 4), acc=round(acc_a, 4), gain=g(ce_a)), sec=round(time.time() - t1))), flush=True)


if __name__ == "__main__":
    main()
