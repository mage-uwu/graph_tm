"""A DDLGN-style routing field for ONE attention layer of the frozen attention-only BitNet (default: layer 0).
The rest of the model is the frozen float BitNet (other layers keep softmax), so this isolates one question: can
learned hard logic gates replace a layer's routing (q.k scores + softmax)?

Everything is laid out to compile bit-sliced later (bit t = position t, an offset is a shift):
  bits        per head, per token: sign and top-magnitude bit of each q / k dimension (frozen int8 projections)
  selection   per head, per offset delta in 0..W-1: N depth-3 trees (8 leaves, 7 learned 4-corner gates); leaves are
              (query bit, same-dimension key bit delta back) pairs; score = popcount of the N tree outputs;
              level = number of learned thresholds <= score (0..L); the current token (delta = 0) gets level + 1
  retrieval   value dims as K-threshold thermometers (quantiles of the teacher's v); each output bit = the fraction
              of selection weight whose value bit (delta back) is set, quantized with Q integer comparisons
              (count Q vs weight i; --frac 1 = weighted majority, a median); decoding is linear, so this is a
              weighted mean of the decoded values; decoded to the level means and fed to the frozen W_o
  training    hard forward always, straight-through backward (sigmoid surrogates), init: level-1 gates XNOR (bit
              agreement), upper gates pass-through; loss = KL(teacher || student) + local x per-head recovery of the
              teacher's softmax output

  python3 lgfield.py --layer 0 --steps 600 [--trees 8 --levels 3 --window 16 --kbits 7] [--out runs/lgf0.pt]
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


def tern(W):
    wq, g = K.tern(W); return torch.tensor(wq * g, dtype=torch.float32)


def hard(x_soft_logit, w=1.0):
    s = torch.sigmoid(x_soft_logit / w); return ((x_soft_logit > 0).float() - s).detach() + s


def gate(a, b, c):
    """learned 2-input gate, 4 corner logits c[..., 4] (00, 01, 10, 11); a, b in [0, 1] (hard forward via STE)"""
    p = hard(c)
    return (p[..., 0] * (1 - a) * (1 - b) + p[..., 1] * (1 - a) * b + p[..., 2] * a * (1 - b) + p[..., 3] * a * b)


class Field(torch.nn.Module):
    def __init__(self, m, a, fitX):
        super().__init__()
        self.a = a; self.T, self.d, self.H = m["T"], m["d"], m["H"]; self.hd = self.d // self.H
        f = lambda x: torch.tensor(x, dtype=torch.float32)
        self.tok, self.pos, self.gf, self.head = f(m["tok"]), f(m["pos"]), f(m["gf"]), f(m["head"])
        self.g = [f(s["g"]) for s in m["subs"]]; self.Wqkv = [tern(s["qkv"]) for s in m["subs"]]
        self.Wo = [tern(s["o"]) for s in m["subs"]]
        H, W, N = self.H, a.window, a.trees
        rng = np.random.default_rng(a.seed)
        # wiring: tree n of (head, delta) reads 4 dims; leaf pair i = (q bit, k bit) of dim j, bit type (sign / top)
        wires = np.zeros((H, W, N, 4, 2), np.int64)                       # [..., pair, (dim, bittype)]
        for h in range(H):
            for dl in range(W):
                dims = rng.permutation(np.tile(np.arange(self.hd), (N * 4 + self.hd - 1) // self.hd))[:N * 4]
                wires[h, dl, :, :, 0] = dims.reshape(N, 4); wires[h, dl, :, :, 1] = rng.integers(0, 2, (N, 4))
        self.register_buffer("wires", torch.tensor(wires))
        # gates: level 1 (4 per tree): XNOR init; levels 2 (2) and 3 (1): pass-through of the first input
        xnor = torch.tensor([3., -3., -3., 3.]); passa = torch.tensor([-3., -3., 3., 3.])
        self.g1 = torch.nn.Parameter(xnor.repeat(H, W, N, 4, 1).clone())
        self.g2 = torch.nn.Parameter(passa.repeat(H, W, N, 2, 1).clone())
        self.g3 = torch.nn.Parameter(passa.repeat(H, W, N, 1, 1).clone())
        L = a.levels
        self.th = torch.nn.Parameter((torch.arange(1, L + 1).float() * N / (L + 1))[None, None].repeat(H, W, 1).clone())
        idx = torch.arange(self.T); rel = idx[:, None] - idx[None]
        self.register_buffer("causal", rel >= 0)
        # value thermometers from the teacher's v on the fit windows (per head dim)
        with torch.no_grad():
            v = self.qkv(self.embed(torch.tensor(fitX)), a.layer)[2]          # [B, H, T, hd]
            v = v.permute(0, 2, 1, 3).reshape(-1, self.d)
            qs = torch.arange(1, a.kbits + 1).float() / (a.kbits + 1)
            th = torch.quantile(v[::3], qs, dim=0).T.contiguous()            # [d, K]
            lvl = (v[..., None] > th[None]).sum(-1)
            means = torch.stack([(v * (lvl == i)).sum(0) / (lvl == i).sum(0).clamp(min=1) for i in range(a.kbits + 1)], -1)
        self.register_buffer("vth", th); self.register_buffer("vmean", means)
        # q / k bit thresholds: sign, and |x| above its median (the top-magnitude bit)
        with torch.no_grad():
            q, k, _ = self.qkv(self.embed(torch.tensor(fitX)), a.layer)
            self.register_buffer("qmed", q.abs().median(dim=2).values.median(dim=0).values)   # [H, hd]
            self.register_buffer("kmed", k.abs().median(dim=2).values.median(dim=0).values)

    # ---- frozen library
    @staticmethod
    def bitlin(x, W):
        s = 127 / x.abs().amax(-1, keepdim=True).clamp(min=1e-5)
        return (x + (torch.round(x * s) - x * s).detach() / s) @ W

    def embed(self, X):
        return self.tok[X] + self.pos[None, :X.shape[1]]

    def qkv(self, x, l):
        B, T = x.shape[:2]
        n = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.g[l]
        y = self.bitlin(n, self.Wqkv[l]).reshape(B, T, 3, self.H, self.hd)
        return [y[:, :, i].transpose(1, 2) for i in range(3)]              # [B, H, T, hd]

    # ---- the routing field
    def field(self, q, k, v, oracle=None):
        a = self.a; B, H, T, hd = q.shape; W, N = a.window, a.trees
        qbits = torch.stack([(q > 0).float(), (q.abs() > self.qmed[None, :, None]).float()], -1)   # [B, H, T, hd, 2]
        kbits = torch.stack([(k > 0).float(), (k.abs() > self.kmed[None, :, None]).float()], -1)
        vt = v.permute(0, 2, 1, 3).reshape(B, T, self.d)
        vb = (vt[..., None] > self.vth[None, None]).float()                   # [B, T, d, K] hard (frozen thresholds)
        num = torch.zeros(B, T, self.d, a.kbits)
        qsel, ksel = qbits.permute(0, 1, 3, 4, 2), kbits.permute(0, 1, 3, 4, 2)   # [B, H, hd, 2, T]
        wmaps = []
        for dl in range(W):
            wr = self.wires[:, dl]                                            # [H, N, 4, 2]
            dim, bt = wr[..., 0], wr[..., 1]
            hi = torch.arange(H)[:, None, None].expand_as(dim)
            ql = qsel[:, hi, dim, bt]                                         # [B, H, N, 4, T]
            kl = ksel[:, hi, dim, bt]
            kl = torch.cat([torch.zeros_like(kl[..., :dl]), kl[..., :T - dl]], -1)   # key bits delta back
            l1 = gate(ql, kl, self.g1[:, dl][None, ..., None, :].expand(B, H, N, 4, T, 4))   # [B, H, N, 4, T]
            l2 = gate(l1[:, :, :, 0::2], l1[:, :, :, 1::2], self.g2[:, dl][None, ..., None, :].expand(B, H, N, 2, T, 4))
            l3 = gate(l2[:, :, :, 0], l2[:, :, :, 1], self.g3[:, dl][None, :, :, 0, None, :].expand(B, H, N, T, 4))
            score = l3.sum(2)                                                 # [B, H, T] popcount of N trees
            th = torch.sort(self.th[:, dl], -1).values                        # [H, L]
            lvl = hard(score[..., None] - th[None, :, None, :] + .5, .5).sum(-1)     # [B, H, T]
            if dl == 0: lvl = lvl + 1
            if oracle is not None:                                            # diagnostic: teacher weights, quantized
                pd = torch.stack([oracle[:, :, t, t - dl] if t >= dl else oracle[:, :, t, 0] * 0 for t in range(T)], -1)
                lvl = torch.round(self.a.oracle_scale * pd)
            valid = (torch.arange(T) >= dl).float()
            lvl = lvl * valid
            wmaps.append(lvl)
            # weights per head apply to that head's value dims
            wd = lvl.repeat_interleave(hd, 1).permute(0, 2, 1)[..., None]    # [B, T, d, 1]
            vs = torch.cat([torch.zeros_like(vb[:, :dl]), vb[:, :T - dl]], 1)
            num = num + wd * vs
        wsum = torch.stack(wmaps, 0).sum(0)                                   # [B, H, T]
        wsum_d = wsum.repeat_interleave(hd, 1).permute(0, 2, 1)[..., None]   # [B, T, d, 1]
        if a.frac <= 1:
            ob = hard(2 * num - wsum_d, 1.0)                                  # weighted majority per thermometer bit
        else:                                                                 # fraction of weight, Q comparisons
            fr = num / wsum_d.clamp(min=1e-6)
            fq = torch.floor(fr * a.frac + .5) / a.frac
            ob = fr + (fq - fr).detach()
        steps = torch.diff(self.vmean, dim=-1)                                # [d, K]
        o = self.vmean[None, None, :, 0] + (ob * steps[None, None]).sum(-1)   # [B, T, d]
        return o, torch.stack(wmaps, -1)

    def forward(self, X, mode="student", local=None):
        B, T = X.shape; x = self.embed(X); hd = self.hd; stats = None
        for l in range(len(self.g)):
            q, k, v = self.qkv(x, l)
            p = torch.softmax((q @ k.transpose(-1, -2) / hd ** .5).masked_fill(~self.causal, float("-inf")), -1)
            ot = (p @ v).transpose(1, 2).reshape(B, T, self.d)
            if mode == "student" and l == self.a.layer:
                o, stats = self.field(q, k, v, p if self.a.oracle_scale else None)
                if local is not None: local.append(((o - ot.detach()) ** 2).mean() / (ot.detach() ** 2).mean())
            else:
                o = ot
            x = x + self.bitlin(o, self.Wo[l])
        return (x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.gf) @ self.head, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0); ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--trees", type=int, default=8); ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--kbits", type=int, default=7); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=.02)
    ap.add_argument("--local", type=float, default=1.0); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=1024); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=""); ap.add_argument("--oracle-scale", type=float, default=0)
    ap.add_argument("--frac", type=int, default=4)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); ntr = len(raw) * 9 // 10
    vb = np.frombuffer(raw[ntr:], np.uint8).astype(np.int64); nw = (len(vb) - 1) // T
    VX = torch.tensor(vb[:nw * T].reshape(nw, T)); VY = torch.tensor(vb[1:nw * T + 1].reshape(nw, T))
    if a.eval_windows: VX, VY = VX[:a.eval_windows], VY[:a.eval_windows]
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[VY.numpy()].mean())
    b = np.frombuffer(raw[:ntr], np.uint8).astype(np.int64); rng = np.random.default_rng(a.seed)
    fitX = b[rng.integers(0, len(b) - T, 256)[:, None] + np.arange(T)]
    net = Field(m, a, fitX)

    def evaluate(mode):
        ce = hit = 0.0; lv = []
        with torch.no_grad():
            for i in range(0, len(VX), 64):
                lg, st = net(VX[i:i + 64], mode); lp = torch.log_softmax(lg, -1)
                ce += -lp.gather(-1, VY[i:i + 64, :, None]).sum().item(); hit += (lg.argmax(-1) == VY[i:i + 64]).sum().item()
                if st is not None: lv.append(st.mean((0, 2)))
        r = dict(ce=ce / VY.numel(), acc=hit / VY.numel())
        if lv: r["mean_level_by_offset"] = [round(x, 2) for x in torch.stack(lv).mean(0).mean(0).tolist()]
        return r
    ce_t = evaluate("teacher")["ce"]; gain = lambda ce: (UNI - ce) / (UNI - ce_t)
    r0 = evaluate("student"); r0["gain"] = gain(r0["ce"])
    print(json.dumps(dict(args=vars(a), teacher_ce=ce_t, unigram=UNI, step0=r0)), flush=True)
    opt = torch.optim.Adam([dict(params=[net.g1, net.g2, net.g3], lr=a.lr), dict(params=[net.th], lr=a.lr * 5)])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=0); t0 = time.time()
    for step in range(1, a.steps + 1):
        st = rng.integers(0, len(b) - T - 1, a.batch); X = torch.tensor(b[st[:, None] + np.arange(T)])
        with torch.no_grad():
            tp = torch.softmax(net(X, "teacher")[0], -1)
        loc = []; lg, _ = net(X, "student", loc); lp = torch.log_softmax(lg, -1)
        kl = (tp * (torch.log(tp + 1e-12) - lp)).sum(-1).mean(); loss = kl + a.local * loc[0]
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate("student"); r.update(step=step, kl=round(kl.item(), 4), local=round(loc[0].item(), 4),
                                             gain=round(gain(r["ce"]), 4), sec=round(time.time() - t0, 1))
            print(json.dumps(r), flush=True)
    if a.out: torch.save(dict(state=net.state_dict(), args=vars(a)), a.out)


if __name__ == "__main__":
    main()
