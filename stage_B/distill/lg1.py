"""Layer-1 routing from lookups (BLT-style): context -> routing code (hashed byte n-gram table) -> code-pair score table.
No q.k, no softmax in layer 1; the frozen model elsewhere (layer 0 exact softmax here, to isolate layer 1).

  codes     per head, k-means (C centroids) on the teacher's layer-1 q vectors and, separately, k vectors (fit windows)
  scores    E[h, cq, ck] = centroid_q . centroid_k / sqrt(hd)  (a C x C table per head; fine-tuned in 'train')
  assign    'vq'   : nearest centroid of the token's true q / k (needs q, k: the vector-quantization bound)
            'ngram': hashed n-gram of the last n bytes (n = 8 .. 1, backoff to the longest seen >= 3 times) ->
                     most frequent code of that context in the fit data (no q / k computed)
  weights   round(S exp(E - max over the window)); retrieval: quantized weighted mean (Q = 8) of 7-threshold
            value thermometers (lgfield.py)

  python3 lg1.py [--codes 256] [--assign vq|ngram] [--steps 300]
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
NS = (8, 6, 5, 4, 3, 2, 1); NBK = 1 << 22; P = np.uint64(1000003)


def keys(X, n):
    B, T = X.shape; h = np.zeros((B, T), np.uint64)
    Xp = np.concatenate([np.full((B, n - 1), 256), X], 1).astype(np.uint64)
    for i in range(n): h = h * P + Xp[:, i:i + T] + np.uint64(1)
    return (h % np.uint64(NBK)).astype(np.int64)


def kmeans(Z, C, iters=15, seed=0):
    g = torch.Generator().manual_seed(seed); cen = Z[torch.randperm(len(Z), generator=g)[:C]].clone()
    for _ in range(iters):
        a = torch.cdist(Z, cen).argmin(1)
        for c in range(C):
            m = a == c
            if m.any(): cen[c] = Z[m].mean(0)
    return cen


class L1(Field):
    def setup(self, fitX, a):
        """k-means codes on teacher layer-1 q, k; n-gram -> code tables"""
        self.a1 = a; H, hd = self.H, self.hd
        with torch.no_grad():
            qs, ks = [], []
            for i in range(0, len(fitX), 64):
                X = torch.tensor(fitX[i:i + 64]); q, k = self.layer1_qk(X); qs.append(q); ks.append(k)
            q = torch.cat(qs); k = torch.cat(ks)                              # [N, H, T, hd]
            # value thermometers from layer 1's own v (the inherited ones were built on layer-0 inputs)
            vs = []
            for i in range(0, 512, 64):
                X = torch.tensor(fitX[i:i + 64]); x = self.embed(X); B, T = X.shape
                q0, k0, v0 = self.qkv(x, 0)
                p0 = torch.softmax((q0 @ k0.transpose(-1, -2) / hd ** .5).masked_fill(~self.causal, float("-inf")), -1)
                x = x + self.bitlin((p0 @ v0).transpose(1, 2).reshape(B, T, self.d), self.Wo[0])
                vs.append(self.qkv(x, 1)[2].permute(0, 2, 1, 3).reshape(-1, self.d))
            v = torch.cat(vs); K_ = self.a.kbits
            th = torch.quantile(v[::3], torch.arange(1, K_ + 1).float() / (K_ + 1), dim=0).T.contiguous()
            lvl = (v[..., None] > th[None]).sum(-1)
            self.vth = th; self.vmean = torch.stack([(v * (lvl == i)).sum(0) / (lvl == i).sum(0).clamp(min=1) for i in range(K_ + 1)], -1)
        self.cq = torch.stack([kmeans(q[:, h].reshape(-1, hd)[::2], a.codes, seed=h) for h in range(H)])
        self.ck = torch.stack([kmeans(k[:, h].reshape(-1, hd)[::2], a.codes, seed=10 + h) for h in range(H)])
        self.E = torch.nn.Parameter(torch.einsum("hcj,hdj->hcd", self.cq, self.ck) / hd ** .5)
        # n-gram tables: most frequent code per hashed context, per head, for q and k
        codes_q = torch.stack([torch.cdist(q[:, h].reshape(-1, hd), self.cq[h]).argmin(1) for h in range(H)], 1).numpy()
        codes_k = torch.stack([torch.cdist(k[:, h].reshape(-1, hd), self.ck[h]).argmin(1) for h in range(H)], 1).numpy()
        self.tab = {}
        for n in NS:
            kk = keys(fitX, n).ravel()
            for side, codes in (("q", codes_q), ("k", codes_k)):
                for h in range(H):
                    comb = kk * a.codes + codes[:, h]
                    u, cnt = np.unique(comb, return_counts=True)
                    key, code = u // a.codes, u % a.codes
                    order = np.lexsort((-cnt, key)); key, code, cnt = key[order], code[order], cnt[order]
                    first = np.r_[True, key[1:] != key[:-1]]
                    tot = np.bincount(np.searchsorted(np.unique(key), key), weights=cnt)
                    self.tab[(side, h, n)] = (key[first], code[first], tot)

    def layer1_qk(self, X):
        x = self.embed(X); B, T = X.shape
        q, k, v = self.qkv(x, 0)
        p = torch.softmax((q @ k.transpose(-1, -2) / self.hd ** .5).masked_fill(~self.causal, float("-inf")), -1)
        x = x + self.bitlin((p @ v).transpose(1, 2).reshape(B, T, self.d), self.Wo[0])
        q, k, _ = self.qkv(x, 1)
        return q, k

    def ngram_codes(self, X, side):
        """[B, H, T] codes from the backoff n-gram tables"""
        Xn = X.numpy(); B, T = Xn.shape; out = np.full((B, self.H, T), -1)
        for n in NS:
            kk = keys(Xn, n)
            for h in range(self.H):
                key, code, tot = self.tab[(side, h, n)]
                idx = np.searchsorted(key, kk); idx = np.clip(idx, 0, len(key) - 1)
                hit = (key[idx] == kk) & (tot[idx] >= (3 if n > 1 else 1)) & (out[:, h] < 0)
                out[:, h][hit] = code[idx][hit]
        out[out < 0] = 0
        return torch.tensor(out)

    def route(self, X, q, k, v):
        a = self.a1; B, H, T, hd = q.shape; W = self.a.window
        if a.assign == "vq":
            cq = torch.stack([torch.cdist(q[:, h], self.cq[h][None].expand(B, -1, -1)).argmin(-1) for h in range(H)], 1)
            ck = torch.stack([torch.cdist(k[:, h], self.ck[h][None].expand(B, -1, -1)).argmin(-1) for h in range(H)], 1)
        else:
            cq, ck = self.ngram_codes(X, "q"), self.ngram_codes(X, "k")
        E = torch.full((B, H, T, W), float("-inf"))
        hi = torch.arange(H)[None, :, None]
        for dl in range(W):
            kc = torch.cat([ck[..., :1].expand(B, H, dl), ck[..., :T - dl]], -1)
            e = self.E[hi, cq, kc]
            E[..., dl] = torch.where(torch.arange(T) >= dl, e, torch.full_like(e, float("-inf")))
        wr = torch.exp(E - E.amax(-1, keepdim=True)) * self.a.rom_scale
        lv = wr + (torch.floor(wr + .5) - wr).detach()
        return self.retrieve(lv, v)

    def forward(self, X, mode="student", local=None):
        B, T = X.shape; x = self.embed(X); hd = self.hd
        for l in range(len(self.g)):
            q, k, v = self.qkv(x, l)
            p = torch.softmax((q @ k.transpose(-1, -2) / hd ** .5).masked_fill(~self.causal, float("-inf")), -1)
            ot = (p @ v).transpose(1, 2).reshape(B, T, self.d)
            if mode == "student" and l == 1:
                o, _ = self.route(X, q, k, v)
                if local is not None: local.append(((o - ot.detach()) ** 2).mean() / (ot.detach() ** 2).mean())
            else:
                o = ot
            x = x + self.bitlin(o, self.Wo[l])
        return (x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.gf) @ self.head, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=int, default=256); ap.add_argument("--assign", default="ngram")
    ap.add_argument("--steps", type=int, default=0); ap.add_argument("--lr", type=float, default=.01)
    ap.add_argument("--fit-windows", type=int, default=4000); ap.add_argument("--eval-windows", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--batch", type=int, default=16)
    a1 = ap.parse_args()
    a = argparse.Namespace(layer=1, window=16, trees=8, levels=3, kbits=7, frac=8, rom_scale=16, seed=0,
                           oracle_scale=0, route="table")
    torch.manual_seed(0)
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); ntr = len(raw) * 9 // 10
    vb = np.frombuffer(raw[ntr:], np.uint8).astype(np.int64); nw = (len(vb) - 1) // T
    VX = torch.tensor(vb[:nw * T].reshape(nw, T)); VY = torch.tensor(vb[1:nw * T + 1].reshape(nw, T))
    if a1.eval_windows: VX, VY = VX[:a1.eval_windows], VY[:a1.eval_windows]
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[VY.numpy()].mean())
    b = np.frombuffer(raw[:ntr], np.uint8).astype(np.int64); rng = np.random.default_rng(0)
    fitX = b[rng.integers(0, len(b) - T, a1.fit_windows)[:, None] + np.arange(T)]
    t0 = time.time(); net = L1(m, a, fitX[:256]); net.setup(fitX, a1)
    print(f"setup (k-means {a1.codes} codes x 2 x {net.H} heads, n-gram tables) {time.time() - t0:.0f}s", flush=True)

    def evaluate(mode):
        ce = hit = 0.0
        with torch.no_grad():
            for i in range(0, len(VX), 64):
                lg, _ = net(VX[i:i + 64], mode); lp = torch.log_softmax(lg, -1)
                ce += -lp.gather(-1, VY[i:i + 64, :, None]).sum().item(); hit += (lg.argmax(-1) == VY[i:i + 64]).sum().item()
        return ce / VY.numel(), hit / VY.numel()
    ce_t, _ = evaluate("teacher"); ce, acc = evaluate("student")
    print(json.dumps(dict(args=vars(a1), teacher_ce=ce_t, step=0, ce=ce, acc=acc, gain=(UNI - ce) / (UNI - ce_t))), flush=True)
    if a1.steps:
        opt = torch.optim.Adam([net.E], lr=a1.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a1.steps, eta_min=0)
        for step in range(1, a1.steps + 1):
            st = rng.integers(0, len(b) - T - 1, a1.batch); X = torch.tensor(b[st[:, None] + np.arange(T)])
            with torch.no_grad(): tp = torch.softmax(net(X, "teacher")[0], -1)
            loc = []; lp = torch.log_softmax(net(X, "student", loc)[0], -1)
            kl = (tp * (torch.log(tp + 1e-12) - lp)).sum(-1).mean(); loss = kl + loc[0]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % a1.eval_every == 0 or step == a1.steps:
                ce, acc = evaluate("student")
                print(json.dumps(dict(step=step, kl=round(kl.item(), 4), local=round(loc[0].item(), 4), ce=ce, acc=acc,
                                      gain=round((UNI - ce) / (UNI - ce_t), 4))), flush=True)


if __name__ == "__main__":
    main()
