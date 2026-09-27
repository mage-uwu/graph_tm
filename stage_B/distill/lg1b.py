"""Layer-1 routing from BLT hash n-gram embeddings (the BLT method, not a frequency table): no layer-1 q.k from the
residual stream at all. Layer 0 stays exact softmax here (to isolate layer 1); the rest is the frozen float BitNet.

  features   z_side(i) = byte_emb[b_i] + sum_{n=3..8} E_n[RollPolyHash(n-gram ending at i) mod NB]   (BLT, Pagnini et
             al. 2024: hash n-gram embeddings summed with the byte embedding; RollPolyHash as in stage2/blt.py)
             side in {q, k}; 128 dims = 4 heads x 32; all tables learned
  training   stage 1: regress the teacher's layer-1 q and k (relative MSE); stage 2: routing loss (per-head recovery of
             the teacher's attention output + KL on next bytes), tables still learned
  routing    score = z_q . z_k / sqrt(32) over the window (16); weights round(16 exp(score - max)); retrieval: quantized
             weighted mean (Q = 8) of 7-threshold value thermometers (lgfield.py)

  python3 lg1b.py --steps1 1500 --steps2 500 [--buckets 65536]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import common as K
from inspect_attn import load_attn
from lgfield import Field

S = os.environ.get("SCRATCH", "/tmp")
NGRAMS = (3, 4, 5, 6, 7, 8)


def rollpoly(X, n, nb):
    """RollPolyHash of the n bytes ending at each position (stage2/blt.py), -1 where fewer than n bytes exist"""
    B, T = X.shape; A = X.astype(np.uint64); h = np.zeros((B, T), np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):                                                # same order as stage2/blt.py
            sh = np.zeros_like(A); sh[:, j:] = A[:, :T - j]
            h = h * np.uint64(1000003) + sh + np.uint64(1)
    out = (h % np.uint64(nb)).astype(np.int64); out[:, :n - 1] = -1
    return out


class HashEmb(torch.nn.Module):
    def __init__(self, nb, dim):
        super().__init__()
        self.nb = nb
        self.byte = torch.nn.Embedding(256, dim)
        self.ng = torch.nn.ModuleList([torch.nn.Embedding(nb + 1, dim, sparse=True) for _ in NGRAMS])
        torch.nn.init.zeros_(self.byte.weight)
        for e in self.ng: torch.nn.init.zeros_(e.weight)

    def forward(self, X):
        z = self.byte(torch.tensor(X))
        for k, n in enumerate(NGRAMS):
            idx = rollpoly(X, n, self.nb); idx[idx < 0] = self.nb           # padding bucket for short prefixes
            z = z + self.ng[k](torch.tensor(idx))
        return z


class L1B(Field):
    def teacher_l1(self, X):
        """teacher layer-1 q, k, v and the residual before layer 1 (layer 0 exact)"""
        x = self.embed(X); B, T = X.shape
        q, k, v = self.qkv(x, 0)
        p = torch.softmax((q @ k.transpose(-1, -2) / self.hd ** .5).masked_fill(~self.causal, float("-inf")), -1)
        x = x + self.bitlin((p @ v).transpose(1, 2).reshape(B, T, self.d), self.Wo[0])
        q1, k1, v1 = self.qkv(x, 1)
        return x, q1, k1, v1

    def forward_student(self, X, embq, embk, local=None):
        Xt = torch.tensor(X); B, T = Xt.shape; hd = self.hd; W = self.a.window
        x, q1, k1, v1 = self.teacher_l1(Xt)
        zq = embq(X).reshape(B, T, self.H, hd).transpose(1, 2); zk = embk(X).reshape(B, T, self.H, hd).transpose(1, 2)
        e = zq @ zk.transpose(-1, -2) / hd ** .5                            # [B, H, T, T]
        idx = torch.arange(T); rel = idx[:, None] - idx[None]; win = (rel >= 0) & (rel < W)
        e = e.masked_fill(~win, float("-inf"))
        E = torch.stack([torch.cat([torch.full((B, self.H, dl), float("-inf")), torch.diagonal(e, -dl, 2, 3)], -1)
                         for dl in range(W)], -1)                             # [B, H, T, W]: score at offset dl
        wr = torch.exp(E - E.amax(-1, keepdim=True)) * 16
        lv = wr + (torch.floor(wr + .5) - wr).detach()
        o, _ = self.retrieve(lv, v1)
        if local is not None:
            p = torch.softmax((q1 @ k1.transpose(-1, -2) / hd ** .5).masked_fill(~self.causal, float("-inf")), -1)
            ot = (p @ v1).transpose(1, 2).reshape(B, T, self.d)
            local.append(((o - ot) ** 2).mean() / (ot ** 2).mean())
        x = x + self.bitlin(o, self.Wo[1])
        return (x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.gf) @ self.head

    def forward_teacher(self, X):
        Xt = torch.tensor(X); B, T = Xt.shape
        x, q1, k1, v1 = self.teacher_l1(Xt)
        p = torch.softmax((q1 @ k1.transpose(-1, -2) / self.hd ** .5).masked_fill(~self.causal, float("-inf")), -1)
        x = x + self.bitlin((p @ v1).transpose(1, 2).reshape(B, T, self.d), self.Wo[1])
        return (x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.gf) @ self.head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, default=65536); ap.add_argument("--steps1", type=int, default=1500)
    ap.add_argument("--steps2", type=int, default=500); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=.05); ap.add_argument("--eval-windows", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=500)
    a1 = ap.parse_args(); torch.manual_seed(0)
    a = argparse.Namespace(layer=1, window=16, trees=8, levels=3, kbits=7, frac=8, rom_scale=16, seed=0, oracle_scale=0, route="table")
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); ntr = len(raw) * 9 // 10
    vb = np.frombuffer(raw[ntr:], np.uint8).astype(np.int64); nw = (len(vb) - 1) // T
    VX = vb[:nw * T].reshape(nw, T)[:a1.eval_windows]; VY = torch.tensor(vb[1:nw * T + 1].reshape(nw, T)[:a1.eval_windows])
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[VY.numpy()].mean())
    b = np.frombuffer(raw[:ntr], np.uint8).astype(np.int64); rng = np.random.default_rng(0)
    fitX = b[rng.integers(0, len(b) - T, 256)[:, None] + np.arange(T)]
    net = L1B(m, a, fitX)
    with torch.no_grad():                                                     # layer-1 value thermometers from layer-1 v
        v = torch.cat([net.teacher_l1(torch.tensor(fitX[i:i + 64]))[3].permute(0, 2, 1, 3).reshape(-1, net.d) for i in range(0, 256, 64)])
        th = torch.quantile(v[::3], torch.arange(1, 8).float() / 8, dim=0).T.contiguous(); lvl = (v[..., None] > th[None]).sum(-1)
        net.vth = th; net.vmean = torch.stack([(v * (lvl == i)).sum(0) / (lvl == i).sum(0).clamp(min=1) for i in range(8)], -1)
    dim = net.H * net.hd; embq, embk = HashEmb(a1.buckets, dim), HashEmb(a1.buckets, dim)

    def evaluate():
        ce = hit = 0.0; ce_t = 0.0
        with torch.no_grad():
            for i in range(0, len(VX), 64):
                X = VX[i:i + 64]; Y = VY[i:i + 64]
                lp = torch.log_softmax(net.forward_student(X, embq, embk), -1); ce += -lp.gather(-1, Y[..., None]).sum().item()
                hit += (lp.argmax(-1) == Y).sum().item()
                lt = torch.log_softmax(net.forward_teacher(X), -1); ce_t += -lt.gather(-1, Y[..., None]).sum().item()
        n = VY.numel(); return dict(ce=ce / n, acc=hit / n, gain=(UNI - ce / n) / (UNI - ce_t / n))
    dense = [embq.byte.weight, embk.byte.weight]; sparse = [p for e in (embq, embk) for t in e.ng for p in t.parameters()]
    opt_d = torch.optim.Adam(dense, lr=a1.lr); opt_s = torch.optim.SparseAdam(sparse, lr=a1.lr)
    t0 = time.time()
    for step in range(1, a1.steps1 + a1.steps2 + 1):
        st = rng.integers(0, len(b) - T - 1, a1.batch); X = b[st[:, None] + np.arange(T)]
        if step <= a1.steps1:                                                 # stage 1: regress teacher q, k
            with torch.no_grad():
                _, q1, k1, _ = net.teacher_l1(torch.tensor(X))
            zq = embq(X).reshape(len(X), T, net.H, net.hd).transpose(1, 2); zk = embk(X).reshape(len(X), T, net.H, net.hd).transpose(1, 2)
            loss = ((zq - q1) ** 2).mean() / (q1 ** 2).mean() + ((zk - k1) ** 2).mean() / (k1 ** 2).mean()
        else:                                                                 # stage 2: routing loss
            with torch.no_grad(): tp = torch.softmax(net.forward_teacher(X), -1)
            loc = []; lp = torch.log_softmax(net.forward_student(X, embq, embk, loc), -1)
            loss = (tp * (torch.log(tp + 1e-12) - lp)).sum(-1).mean() + loc[0]
        opt_d.zero_grad(); opt_s.zero_grad(); loss.backward(); opt_d.step(); opt_s.step()
        if step % a1.eval_every == 0 or step == a1.steps1 + a1.steps2:
            r = evaluate(); r.update(step=step, stage=1 if step <= a1.steps1 else 2, loss=round(loss.item(), 4), sec=round(time.time() - t0))
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
