"""Attention as logic: a trainable hard "reader" over the frozen attention-only BitNet ("library").

Frozen (never updated): token/position embeddings, RMSNorm gains, BitLinear W_qkv and W_o (ternary + int8 inputs),
final norm, LM head, and the spectral read-out directions of each head's bilinear form
  M_h = W_q,h W_k,h^T / sqrt(hd),  Sigma^1/2 M_h Sigma^1/2 = U S V^T,  alpha = n Sigma^-1/2 U[:, :r],
  beta = n Sigma^-1/2 V[:, :r]   (n = the normalized input of the layer; Sigma from the fit windows)
Reader (the only trained part, hard in every forward pass):
  codes    a_i(t) = sum_j step_ij [alpha_i(t) > theta_ij]   (K thresholds per mode: a (K+1)-level thermometer)
           b_i(s) likewise for beta
  score    e(t,s) = sum_i sigma_i a_i(t) b_i(s)            (small-integer products: a (K+1)^2 table per mode)
  select   the top-k scores of each query (causal)
  weights  w_s = 2^-floor(kappa_h (e_max - e_s)) inside the set, normalized  (shifts, no exp)
  values   the frozen v (--vbits 0), or a thermometer with --vbits thresholds per dim (learned thresholds/levels)
Backward: straight-through. Thresholds by a sigmoid surrogate; the selected set's weights by a softmax surrogate at
the same kappa. Loss: KL(teacher || student) on next-byte distributions of training windows.

  python3 reader.py --rank 8 --levels 7 --topk 4 --steps 600 [--vbits 0|3|7] [--out runs/reader.pt]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import common as K
from inspect_attn import load_attn

S = os.environ.get("SCRATCH", "/tmp")
torch.set_default_dtype(torch.float32)


def tern(W):
    wq, g = K.tern(W); return torch.tensor(wq * g, dtype=torch.float32)


class Model(torch.nn.Module):
    def __init__(self, m, fitX, a):
        super().__init__()
        self.T, self.d, self.H = m["T"], m["d"], m["H"]; self.hd = self.d // self.H; self.a = a
        f = lambda x: torch.tensor(x, dtype=torch.float32)
        self.tok, self.pos, self.gf, self.head = f(m["tok"]), f(m["pos"]), f(m["gf"]), f(m["head"])
        self.g = [f(s["g"]) for s in m["subs"]]; self.Wqkv = [tern(s["qkv"]) for s in m["subs"]]
        self.Wo = [tern(s["o"]) for s in m["subs"]]
        self.register_buffer("mask", torch.triu(torch.full((self.T, self.T), float("-inf")), 1))
        # spectral read-out from the frozen library (fit windows, teacher attention)
        with torch.no_grad():
            ns = self.teacher_inputs(torch.tensor(fitX))
        self.A, self.B, self.sv = [], [], []
        r = a.rank
        for l in range(len(m["subs"])):
            n = ns[l].reshape(-1, self.d).double(); Sg = n.T @ n / len(n)
            w, U = torch.linalg.eigh(Sg); w = w.clamp(min=w.max() * 1e-6)
            Sh, Si = (U * w.sqrt()) @ U.T, (U * w.rsqrt()) @ U.T
            As, Bs, ss = [], [], []
            for h in range(self.H):
                W = self.Wqkv[l].double(); Wq = W[:, h * self.hd:(h + 1) * self.hd]
                Wk = W[:, self.d + h * self.hd:self.d + (h + 1) * self.hd]
                Uu, sv, Vt = torch.linalg.svd(Sh @ (Wq @ Wk.T / self.hd ** .5) @ Sh)
                As.append((Si @ Uu[:, :r]).float()); Bs.append((Si @ Vt.T[:, :r]).float()); ss.append(sv[:r].float())
            self.A.append(torch.stack(As)); self.B.append(torch.stack(Bs)); self.sv.append(torch.stack(ss))  # [H, d, r]
        self.lsv = torch.nn.ParameterList([torch.nn.Parameter(torch.log(s_)) for s_ in self.sv])   # learned mode weights
        # reader parameters: each spectral coordinate is a uniform b-bit integer (learned step and center); bits per
        # coordinate by reverse water-filling of its score-error weight, --budget bits per head per side
        P = torch.nn.Parameter; self.mu_a, self.ls_a, self.mu_b, self.ls_b, self.kappa = [], [], [], [], []
        self.bits_a, self.bits_b = [], []
        for l in range(len(m["subs"])):
            n = ns[l].reshape(-1, self.d)
            ca = torch.einsum("nd,hdr->nhr", n, self.A[l]); cb = torch.einsum("nd,hdr->nhr", n, self.B[l])
            s2 = self.sv[l] ** 2
            for which, c, other in (("a", ca, cb), ("b", cb, ca)):
                var = c.var(0); w = s2 * (other ** 2).mean(0) * var           # score-error weight per coordinate [H, r]
                bits = torch.zeros_like(w)
                for h in range(self.H):                                       # reverse water-filling per head
                    lo, hi = -60.0, 60.0
                    for _ in range(100):
                        th = (lo + hi) / 2; b_ = (0.5 * (torch.log2(w[h]) - th)).clamp(0, a.maxbits)
                        lo, hi = (th, hi) if b_.sum() > a.budget else (lo, th)
                    bits[h] = torch.round(0.5 * (torch.log2(w[h]) - hi)).clamp(0, a.maxbits)
                mu = c.mean(0); span = 2 * 2.5 * c.std(0)                     # +-2.5 sd covered by 2^bits levels
                step = span / (2 ** bits).clamp(min=1)
                getattr(self, "mu_" + which).append(P(mu)); getattr(self, "ls_" + which).append(P(torch.log(step)))
                getattr(self, "bits_" + which).append(bits)
            self.kappa.append(P(torch.full((self.H,), 1.4427)))              # log2(e): shifts reproduce exp at init
        for name in ("mu_a", "ls_a", "mu_b", "ls_b", "kappa"):
            setattr(self, name, torch.nn.ParameterList(getattr(self, name)))
        self.th_a, self.th_b = [], []
        # value thermometers (optional)
        self.vth, self.vst = torch.nn.ParameterList(), torch.nn.ParameterList()
        if a.vbits:
            qs = torch.arange(1, a.vbits + 1, dtype=torch.float32) / (a.vbits + 1)
            with torch.no_grad():
                vs = self.teacher_values(torch.tensor(fitX))
            for l in range(len(m["subs"])):
                v = vs[l].reshape(-1, self.d)[::7]
                th = torch.quantile(v, qs, dim=0).T.contiguous()             # [d, vbits]
                lvl = (v[..., None] > th[None]).sum(-1)
                means = torch.stack([(v * (lvl == i)).sum(0) / (lvl == i).sum(0).clamp(min=1) for i in range(a.vbits + 1)], -1)
                self.vth.append(P(th)); self.vst.append(P(torch.cat([means[:, :1], torch.diff(means, dim=-1)], -1)))

    # ---- frozen pieces
    def bitlin(self, x, W):
        s = 127 / x.abs().amax(-1, keepdim=True).clamp(min=1e-5)
        xq = x + (torch.round(x * s) - x * s).detach() / s                 # int8 activations (STE for the grads)
        return xq @ W

    def norm_in(self, x, l):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.g[l]

    def qkv(self, n, l):
        B, T = n.shape[:2]; y = self.bitlin(n, self.Wqkv[l]).reshape(B, T, 3, self.H, self.hd)
        return [y[:, :, i].transpose(1, 2) for i in range(3)]              # [B, H, T, hd]

    def teacher_inputs(self, X):
        out = []; x = self.tok[X] + self.pos[None, :X.shape[1]]
        for l in range(len(self.g)):
            n = self.norm_in(x, l); out.append(n)
            q, k, v = self.qkv(n, l); p = torch.softmax(q @ k.transpose(-1, -2) / self.hd ** .5 + self.mask, -1)
            x = x + self.bitlin((p @ v).transpose(1, 2).reshape(*x.shape), self.Wo[l])
        return out

    def teacher_values(self, X):
        out = []; x = self.tok[X] + self.pos[None, :X.shape[1]]
        for l in range(len(self.g)):
            n = self.norm_in(x, l); q, k, v = self.qkv(n, l); out.append(v.transpose(1, 2).reshape(*x.shape))
            p = torch.softmax(q @ k.transpose(-1, -2) / self.hd ** .5 + self.mask, -1)
            x = x + self.bitlin((p @ v).transpose(1, 2).reshape(*x.shape), self.Wo[l])
        return out

    # ---- the reader
    @staticmethod
    def uniq(c, mu, ls, bits):
        """uniform integer code: round((c - mu) / step) clipped to bits (0 bits: the center), straight-through"""
        step = torch.exp(ls); half = (2 ** bits) / 2
        z = (c - mu) / step; zq = torch.clamp(torch.round(z), -half, half - 1)
        zq = torch.where(bits > 0, zq, torch.zeros_like(zq))
        return mu + step * (z + (zq - z).detach())

    @staticmethod
    def thermo(c, th, st, width):
        """hard thermometer value (forward), sigmoid surrogate (backward); c [..., r], th/st per [H, r, L]"""
        x = c[..., None] - th
        w = width * (th[..., -1:] - th[..., :1]).abs().clamp(min=1e-3) / th.shape[-1]
        soft = torch.sigmoid(x / w); hard = (x > 0).float()
        bits = (hard - soft).detach() + soft
        return st[..., 0] + (bits * st[..., 1:]).sum(-1)

    def forward(self, X, mode="student"):
        B, T = X.shape; x = self.tok[X] + self.pos[None, :T]; a = self.a
        for l in range(len(self.g)):
            n = self.norm_in(x, l); q, k, v = self.qkv(n, l)
            if mode == "teacher":
                p = torch.softmax(q @ k.transpose(-1, -2) / self.hd ** .5 + self.mask, -1)
            else:
                al = torch.einsum("btd,hdr->bhtr", n, self.A[l]); be = torch.einsum("btd,hdr->bhtr", n, self.B[l])
                ca = self.uniq(al, self.mu_a[l][None, :, None], self.ls_a[l][None, :, None], self.bits_a[l][None, :, None])
                cb = self.uniq(be, self.mu_b[l][None, :, None], self.ls_b[l][None, :, None], self.bits_b[l][None, :, None])
                if "float_codes" in a.diag: ca, cb = al, be                   # diagnostic: no thermometer
                e = torch.einsum("bhtr,bhsr->bhts", ca * torch.exp(self.lsv[l])[None, :, None], cb) + self.mask
                emax = e.amax(-1, keepdim=True)
                kth = torch.topk(e, a.topk, -1).values[..., -1:]
                sel = (e >= kth) & torch.isfinite(e)
                kap = self.kappa[l][None, :, None, None]
                gap = (emax - e).clamp(max=60)
                hard = torch.where(sel, torch.pow(2.0, -torch.floor(kap * gap.detach()) ), torch.zeros_like(e))
                hard = hard / hard.sum(-1, keepdim=True)
                soft = torch.where(sel, torch.exp(-kap * gap * 0.6931472), torch.zeros_like(e))
                soft = soft / soft.sum(-1, keepdim=True)
                p = (hard - soft).detach() + soft
                if "softmax" in a.diag: p = torch.softmax(e, -1)              # diagnostic: exact softmax, no selection
            if len(self.vth) and mode != "teacher":
                vv = v.transpose(1, 2).reshape(B, T, self.d)
                vv = self.thermo(vv, self.vth[l], self.vst[l], a.width)
                v = vv.reshape(B, T, self.H, self.hd).transpose(1, 2)
            o = (p @ v).transpose(1, 2).reshape(B, T, self.d)
            x = x + self.bitlin(o, self.Wo[l])
        return (x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.gf) @ self.head


def windows(raw, lo, hi, T, n, seed):
    rng = np.random.default_rng(seed); b = np.frombuffer(raw[lo:hi], np.uint8).astype(np.int64)
    st = rng.integers(0, len(b) - T - 1, n)
    idx = st[:, None] + np.arange(T + 1)[None]
    return b[idx[:, :-1]], b[idx[:, 1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8); ap.add_argument("--levels", type=int, default=7)
    ap.add_argument("--budget", type=float, default=32); ap.add_argument("--maxbits", type=float, default=8)
    ap.add_argument("--topk", type=int, default=4); ap.add_argument("--vbits", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--width", type=float, default=.5)
    ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=""); ap.add_argument("--diag", default="")
    ap.add_argument("--train", default="mu,kappa,sv,v")
    a = ap.parse_args(); torch.manual_seed(a.seed)
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); ntr = len(raw) * 9 // 10
    vb = np.frombuffer(raw[ntr:], np.uint8).astype(np.int64); nw = (len(vb) - 1) // T; half = nw // 2
    VX = vb[:nw * T].reshape(nw, T); VY = vb[1:nw * T + 1].reshape(nw, T)
    fitX = VX[:512]                                                          # fit windows: first half of validation
    XS, YS = torch.tensor(VX[half:]), torch.tensor(VY[half:])               # scoring half (as measure_attn.py)
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[VY[half:]].mean())
    net = Model(m, fitX, a)

    def evaluate(mode="student"):
        ce = hit = 0.0
        with torch.no_grad():
            for i in range(0, len(XS), 256):
                lg = net(XS[i:i + 256], mode); lp = torch.log_softmax(lg, -1)
                ce += -lp.gather(-1, YS[i:i + 256, :, None]).sum().item(); hit += (lg.argmax(-1) == YS[i:i + 256]).sum().item()
        return ce / YS.numel(), hit / YS.numel()
    ce_t, acc_t = evaluate("teacher")
    gain = lambda ce: (UNI - ce) / (UNI - ce_t)
    rec = dict(args=vars(a), teacher=dict(ce=ce_t, acc=acc_t, unigram=UNI),
               bits=dict(a=[b.tolist() for b in net.bits_a], b=[b.tolist() for b in net.bits_b]))
    ce0, acc0 = evaluate(); rec["step0"] = dict(ce=ce0, acc=acc0, gain=gain(ce0))
    print(json.dumps(rec), flush=True)
    groups = dict(mu=list(net.mu_a) + list(net.mu_b), ls=list(net.ls_a) + list(net.ls_b), kappa=list(net.kappa), sv=list(net.lsv),
                  v=list(net.vth) + list(net.vst))
    params = [p for g_ in a.train.split(",") for p in groups[g_]]
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * .1)
    t0 = time.time(); best = None; r = rec["step0"]
    for step in range(1, a.steps + 1):
        X, _ = windows(raw, 0, ntr, T, a.batch, a.seed * 100003 + step); X = torch.tensor(X)
        with torch.no_grad():
            tp = torch.softmax(net(X, "teacher"), -1)
        lp = torch.log_softmax(net(X), -1)
        loss = (tp * (torch.log(tp + 1e-12) - lp)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for th in list(net.vth):
                th.copy_(torch.sort(th, -1).values)                          # keep each thermometer ordered
        if step % a.eval_every == 0 or step == a.steps:
            ce, acc = evaluate(); r = dict(step=step, kl=round(loss.item(), 4), ce=round(ce, 4), acc=round(acc, 4),
                                           gain=round(gain(ce), 4), sec=round(time.time() - t0, 1))
            print(json.dumps(r), flush=True)
            if best is None or ce < best["ce"]: best = r
    rec["final"] = r; rec["best"] = best
    if a.out:
        torch.save(dict(state=net.state_dict(), args=vars(a)), a.out)
        json.dump(rec, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
