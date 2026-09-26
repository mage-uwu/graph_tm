"""Distil the MLP-only BitNet into a threshold-logic network (TLN) whose forward pass is pure integer logic.

  u   input bits: thermometer code (--in-levels K per dim) of the token embedding and of the position embedding
  g1  L gates per layer-1 hidden unit j: r1[j,l] = [u . A1[:, j] > t1[j,l]]      same wiring, L thresholds
  g2  r2[j,l] = [[u, r1] . A2[:, j] > t2[j,l]]                                     (residual folded into wiring)
  out votes[v] = [u, r1, r2] . C[:, v] + bias[v];  logits = s * votes               (argmax needs no s)
  A1, A2, C: integers in [-Q, Q] (Q = 1: ternary; each |A| = one wire with a polarity), t, bias: integers.
  Every value in the forward pass is an integer: popcounts of wired bits, integer compares, integer vote sums.

Init (--init weights): the teacher's weights folded as in convert.py, hidden units split into L levels at
quantiles of their active pre-activation, level increments alpha from the teacher's ReLU^2 values.
Training: hard forward (the deployed network), straight-through backward (sigmoid surrogate for each threshold,
identity through rounding), loss = KL(teacher || student) weighted by the training frequency of each input.

  python3 student.py --levels 4 --steps 600 [--init weights|random] [--qa 1] [--qc 1] --out tln.npz
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import common as K
from convert import eff, thermo

S = os.environ.get("SCRATCH", "/tmp")
UNI = 3.1696   # unigram CE on the validation text


def wquant(w, q):
    return np.quantile(w, q)


def wq_levels(p, f, L):
    """per column: L thresholds on p (0 and the weighted quantiles of the active part) """
    t = np.zeros((p.shape[1], L))
    for j in range(p.shape[1]):
        on = p[:, j] > 0
        if L > 1 and on.sum() > 1:
            v, w = p[on, j], f[on]; w = w if w.sum() > 0 else np.ones_like(w); o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum()
            t[j, 1:] = [v[o][min(np.searchsorted(cw, k / L), len(v) - 1)] for k in range(1, L)]
        elif L > 1:
            t[j, 1:] = np.arange(1, L) * 1e-3
    return t


def level_values(p, a, t, f):
    """mean teacher ReLU^2 of the inputs in each level band -> increments (value added when gate l turns on)"""
    H, L = t.shape; val = np.zeros((H, L))
    for l in range(L):
        lo = p > t[:, l]; hi = p > t[:, l + 1] if l + 1 < L else np.zeros_like(lo)
        band = (lo & ~hi) * f[:, None]
        val[:, l] = (a * band).sum(0) / np.maximum(band.sum(0), 1e-12)
    inc = np.diff(np.concatenate([np.zeros((H, 1)), val], 1), axis=1)
    return inc


def convert(m, f, Kin, L):
    ids, pos = K.domain()
    _, acts = K.teacher(m, ids, pos, keep=True)
    bt, Rt, lot = thermo(m["tok"], Kin); bp, Rp, lop = thermo(m["pos"], Kin)
    u = np.concatenate([bt[ids], bp[pos]], 1); R = np.concatenate([Rt, Rp]); xlo = lot + lop
    s1, s2 = m["subs"]
    W1 = s1["g"][:, None] * eff(s1["up"]); W2 = s2["g"][:, None] * eff(s2["up"])
    D1, D2 = eff(s1["down"]), eff(s2["down"]); Hh = m["gf"][:, None] * m["head"]
    A1, b1 = R @ W1, xlo @ W1
    p1 = u @ A1 + b1; t1 = wq_levels(p1, f, L); inc1 = level_values(p1, np.maximum(acts["h0"], 0) ** 2, t1, f)
    r1 = (p1[:, :, None] > t1[None]).reshape(len(u), -1).astype(np.float64)
    M1 = (inc1[:, :, None] * D1[:, None, :]).reshape(-1, D1.shape[1])        # gate (j,l) -> residual direction
    A2 = np.concatenate([R @ W2, M1 @ W2]); b2 = xlo @ W2
    p2 = np.concatenate([u, r1], 1) @ A2 + b2; t2 = wq_levels(p2, f, L)
    inc2 = level_values(p2, np.maximum(acts["h1"], 0) ** 2, t2, f)
    M2 = (inc2[:, :, None] * D2[:, None, :]).reshape(-1, D2.shape[1])
    C = np.concatenate([R @ Hh, M1 @ Hh, M2 @ Hh]); bias = xlo @ Hh
    x2 = acts["x2"]; s = (1 / np.sqrt((x2 * x2).mean(1) + 1e-6) * f).sum()
    # thresholds absorb the constant terms: [u.A + b > t] == [u.A > t - b]
    return dict(u=u, A1=A1, t1=t1 - b1[:, None], A2=A2, t2=t2 - b2[:, None], C=C, bias=bias, s=s)


class TLN(torch.nn.Module):
    def __init__(self, P, qa, qc, width):
        super().__init__()
        self.qa, self.qc, self.width = qa, qc, width
        # latent weights in units of their quantization step (so rounding thresholds are at +-0.5)
        self.g1 = float(np.abs(P["A1"]).mean() * (1 if qa == 1 else 2 / qa))
        self.g2 = float(np.abs(P["A2"]).mean() * (1 if qa == 1 else 2 / qa))
        self.gc = float(np.abs(P["C"]).mean() * (1 if qc == 1 else 2 / qc))
        T = lambda x: torch.nn.Parameter(torch.tensor(x, dtype=torch.float32))
        self.A1 = T(P["A1"] / self.g1); self.t1 = T(P["t1"] / self.g1)
        self.A2 = T(P["A2"] / self.g2); self.t2 = T(P["t2"] / self.g2)
        self.C = T(P["C"] / self.gc); self.bias = T(P["bias"] / self.gc)
        self.logs = T(np.log(P["s"] * self.gc))
        self.register_buffer("u", torch.tensor(P["u"], dtype=torch.float32))

    @staticmethod
    def qround(w, q):     # integer in [-q, q], straight-through
        return w + (torch.clamp(torch.round(w), -q, q) - w).detach()

    def gate(self, pre, t):
        """hard [pre > t] forward; sigmoid surrogate backward with a per-unit width"""
        x = pre[:, :, None] - t[None]
        w = (pre.detach().std(0, keepdim=True)[:, :, None] * self.width).clamp(min=.5)
        soft = torch.sigmoid(x / w)
        return ((x > 0).float() - soft).detach() + soft

    def forward(self, idx, hard_int=True):   # every forward is the integer network
        u = self.u[idx]; n = len(idx)
        A1 = self.qround(self.A1, self.qa); t1 = self.t1   # integer pre: [pre > t] == [pre > floor(t)]
        r1 = self.gate(u @ A1, t1).reshape(n, -1)
        z1 = torch.cat([u, r1], 1)
        A2 = self.qround(self.A2, self.qa); t2 = self.t2
        r2 = self.gate(z1 @ A2, t2).reshape(n, -1)
        z = torch.cat([z1, r2], 1)
        votes = z @ self.qround(self.C, self.qc) + self.qround(self.bias, 1e9)
        return votes * torch.exp(self.logs), votes

    def export(self):
        """integer model: wiring/polarity matrices, integer thresholds (fire iff pre > t), integer bias"""
        r = lambda x, q: torch.clamp(torch.round(x), -q, q).detach().numpy().astype(np.int32)
        return dict(A1=r(self.A1, self.qa), A2=r(self.A2, self.qa), C=r(self.C, self.qc),
                    t1=np.floor(self.t1.detach().numpy()).astype(np.int32),
                    t2=np.floor(self.t2.detach().numpy()).astype(np.int32),
                    bias=np.round(self.bias.detach().numpy()).astype(np.int32),
                    scale=float(torch.exp(self.logs)), u=self.u.numpy().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=4); ap.add_argument("--in-levels", type=int, default=3)
    ap.add_argument("--qa", type=int, default=1); ap.add_argument("--qc", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=.01)
    ap.add_argument("--width", type=float, default=.25); ap.add_argument("--init", default="weights")
    ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=""); ap.add_argument("--eval-every", type=int, default=100)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg, m = K.load(os.path.join(K.ROOT, "models/bitnet/bitnet_mlp.bin"))
    Cv, Ct = np.load(f"{S}/Cv.npy"), np.load(f"{S}/Ct.npy")
    ids, pos = K.domain(); f = Ct.sum(1); f = f / f.sum()
    tl = K.teacher(m, ids, pos); ce_t, acc_t = K.score(tl, Cv)
    P = convert(m, f, a.in_levels, a.levels)
    if a.init == "random":   # same shapes and scales, weights shuffled: no information from the teacher
        rng = np.random.default_rng(a.seed)
        for k in ("A1", "A2", "C"):
            P[k] = rng.permutation(P[k].ravel()).reshape(P[k].shape)
        P["bias"] = np.zeros_like(P["bias"])
    net = TLN(P, a.qa, a.qc, a.width)
    tp = torch.tensor(np.exp(tl - tl.max(1, keepdims=True)), dtype=torch.float32); tp /= tp.sum(1, keepdim=True)
    live = np.flatnonzero(f > 0); fw = torch.tensor(f, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * .1)

    def evaluate():
        with torch.no_grad():
            L, _ = net(torch.arange(len(f)), hard_int=True)
        ce, acc = K.score(L.double().numpy(), Cv)
        return dict(ce=round(ce, 4), acc=round(acc, 4), gain_kept=round((UNI - ce) / (UNI - ce_t), 4),
                    acc_kept=round(acc / acc_t, 4))
    rec = dict(args=vars(a), teacher=dict(ce=round(ce_t, 4), acc=round(acc_t, 4)),
               sizes=dict(in_bits=P["u"].shape[1], gates1=P["t1"].size, gates2=P["t2"].size, outputs=256))
    rec["step0"] = evaluate(); print(json.dumps(rec), flush=True)
    t0 = time.time(); rng = np.random.default_rng(a.seed)
    for step in range(1, a.steps + 1):
        idx = torch.tensor(rng.choice(live, size=min(a.batch, len(live)), replace=False))
        L, _ = net(idx)
        lp = torch.log_softmax(L, 1); w = fw[idx] / fw[idx].sum()
        loss = (w * (tp[idx] * (torch.log(tp[idx] + 1e-12) - lp)).sum(1)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(); r.update(step=step, kl=round(loss.item(), 4), sec=round(time.time() - t0, 1))
            print(json.dumps(r), flush=True)
    if a.out:
        E = net.export(); np.savez_compressed(a.out, **E)
        with open(a.out + ".json", "w") as fh:
            json.dump(dict(rec, final=evaluate()), fh)


if __name__ == "__main__":
    main()
