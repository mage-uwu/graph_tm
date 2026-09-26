"""Step 2: replace the frozen BitNet's softmax retrieval with a learned field of hard logic gates, with BLT-style
variable routing effort. Everything else is the frozen model (attention-only BitNet by default).

Interface (from the frozen library, per token per head): q8, k8, v8 = BitNet's int8 absmax quantization of q, k, v,
with per-token scales sq, sk, sv.  Sign-magnitude bits: sign s, magnitude bits m_0..m_6.

Match gates (per head)
  score(t,s) = sum_j s_q,j s_k,j sum_{a,b} w_ab G_ab(m_a(q_t,j), m_b(k_s,j))      (a, b = magnitude bit-planes)
  G_ab: a learned 2-input gate (4 corner logits; hard: corner > 0), w_ab: learned integer.
  Pass-through init: G = AND, w_ab = 2^(a+b)  ->  score == q8 . k8 exactly.
  e(t,s) = sq_t sk_s score / sqrt(hd)   (the frozen per-token scales)
Retrieval gates (replace softmax), window W (s in t-W+1..t)
  gap = max_window e - e;  level = number of learned thresholds theta_1 < ... < theta_L below gap
  weight = LUT[level] (learned integers, LUT[L] = 0: not selected);  o_t = sum_s weight sv_s v8_s / sum_s weight
BLT effort (--tau): coarse match from the plane pairs with a + b >= --coarse; where the coarse top-1 minus top-2 gap
  (score units) exceeds tau the query is certain and keeps the coarse scores; otherwise the full match runs.
  effort = mean share of plane-pair work spent per query (coarse always, fine only where uncertain).
Training: hard forward, straight-through backward (sigmoid surrogates for corners and thresholds, identity for
rounding). Loss = KL(teacher || student) on next bytes + --local x per-head recovery (MSE to the softmax output on
the same q, k, v, relative to its power).

  python3 gatefield.py --steps 600 --window 8 --levels 8 [--tau 1.0 --coarse 8] [--out runs/gf.pt]
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
NB = 7                                                     # magnitude bit-planes of an int8 value


def tern(W):
    wq, g = K.tern(W); return torch.tensor(wq * g, dtype=torch.float32)


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def hard_sig(x, w):
    s = torch.sigmoid(x / w); return ((x > 0).float() - s).detach() + s


class Net(torch.nn.Module):
    def __init__(self, m, a):
        super().__init__()
        self.a = a; self.T, self.d, self.H = m["T"], m["d"], m["H"]; self.hd = self.d // self.H
        f = lambda x: torch.tensor(x, dtype=torch.float32)
        self.tok, self.pos, self.gf, self.head = f(m["tok"]), f(m["pos"]), f(m["gf"]), f(m["head"])
        self.g = [f(s["g"]) for s in m["subs"]]; self.Wqkv = [tern(s["qkv"]) for s in m["subs"]]
        self.Wo = [tern(s["o"]) for s in m["subs"]]; nl = len(self.g); H = self.H
        P = torch.nn.Parameter
        ab = torch.arange(NB)[:, None] + torch.arange(NB)[None]
        corner = torch.full((4, NB, NB), -3.0); corner[3] = 3.0               # corners 00, 01, 10, 11: AND
        self.corner = torch.nn.ParameterList([P(corner[None].repeat(H, 1, 1, 1).clone()) for _ in range(nl)])
        self.register_buffer("w0", (2.0 ** ab).float())
        self.w = torch.nn.ParameterList([P(torch.ones(H, NB, NB)) for _ in range(nl)])   # w_ab = round(2^(a+b) r_ab)
        L = a.levels; th = (torch.arange(1, L + 1).float() * a.gapmax / L)       # thresholds on the gap
        centers = torch.cat([torch.zeros(1), (th[:-1] + th[1:]) / 2])            # level i covers [th_i-1, th_i)
        lut = torch.exp(-centers)                                               # x 255 in the forward pass
        self.th = torch.nn.ParameterList([P(th[None].repeat(H, 1).clone()) for _ in range(nl)])
        self.lut = torch.nn.ParameterList([P(lut[None].repeat(H, 1).clone()) for _ in range(nl)])
        idx = torch.arange(self.T)
        rel = idx[:, None] - idx[None]
        self.register_buffer("win", ((rel >= 0) & (rel < a.window)))
        self.register_buffer("coarse_mask", (ab >= a.coarse).float())
        self.effort = [0.0, 0]

    # ---- frozen library
    @staticmethod
    def bitlin(x, W):
        s = 127 / x.abs().amax(-1, keepdim=True).clamp(min=1e-5)
        return (x + (torch.round(x * s) - x * s).detach() / s) @ W

    def qkv8(self, x, l):
        B, T = x.shape[:2]
        n = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.g[l]
        y = self.bitlin(n, self.Wqkv[l]).reshape(B, T, 3, self.H, self.hd).transpose(1, 3)   # [B, H, 3, T, hd]
        sc = y.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
        return torch.round(y / sc).detach(), sc.detach(), y

    # ---- gate field
    def match(self, q8, k8, l, pairs_mask=None):
        """score_int [B, H, T, T] from sign-magnitude bit-planes through the learned plane-pair gates"""
        sq, sk = torch.sign(q8) + (q8 == 0), torch.sign(k8) + (k8 == 0)
        mq, mk = q8.abs().long(), k8.abs().long()
        bq = ((mq[..., None] >> torch.arange(NB)) & 1).float()                  # [B, H, T, hd, NB]
        bk = ((mk[..., None] >> torch.arange(NB)) & 1).float()
        c = hard_sig(self.corner[l], 1.0)                                        # [H, 4, NB, NB] in {0, 1}
        w = ste_round(self.w0 * self.w[l])
        if pairs_mask is not None: w = w * pairs_mask
        M = c * w[:, None]                                                       # [H, 4, NB, NB]
        tot = 0
        for ci, (x, y) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            A = (bq if x else 1 - bq) * sq[..., None]; Bm = (bk if y else 1 - bk) * sk[..., None]
            AM = torch.einsum("bhtja,hac->bhtjc", A, M[:, ci])
            tot = tot + torch.einsum("bhtjc,bhsjc->bhts", AM, Bm)
        return tot

    def retrieve(self, e, v8, sv, l, win=None):
        """gap -> level (learned thresholds) -> integer weight (learned LUT); weighted sum of the window's values"""
        win = self.win if win is None else win
        e = e.masked_fill(~win, float("-inf"))
        gap = (e.amax(-1, keepdim=True) - e).masked_fill(~win, 1e4)
        th = torch.sort(self.th[l], -1).values[None, :, None, None]             # [1, H, 1, 1, L]
        lut = ste_round(255 * self.lut[l]).clamp(min=0)                           # [H, L] integers
        steps = torch.cat([lut[:, 1:] - lut[:, :-1], -lut[:, -1:]], -1)         # passing threshold i adds steps[i]
        on = hard_sig(gap[..., None] - th, self.a.width)                        # [B, H, T, T, L]
        wgt = (lut[None, :, 0, None, None] + (on * steps[None, :, None, None]).sum(-1)).masked_fill(~win, 0)
        wgt = wgt.clamp(min=0)
        den = wgt.sum(-1, keepdim=True).clamp(min=1e-6)
        return (wgt / den) @ (v8 * sv)

    def forward(self, X, mode="student", local=None):
        B, T = X.shape; x = self.tok[X] + self.pos[None, :T]; hd = self.hd
        for l in range(len(self.g)):
            q8, sc, y = self.qkv8(x, l)
            if mode == "teacher":
                q, k, v = y[:, :, 0], y[:, :, 1], y[:, :, 2]
                cm = torch.ones(T, T, dtype=torch.bool).tril()
                p = torch.softmax((q @ k.transpose(-1, -2) / hd ** .5).masked_fill(~cm, float("-inf")), -1)
                o = p @ v
            else:
                qq, kk, vv = q8[:, :, 0], q8[:, :, 1], q8[:, :, 2]
                s_q, s_k, s_v = sc[:, :, 0], sc[:, :, 1], sc[:, :, 2]
                scale = s_q * s_k.transpose(-1, -2) / hd ** .5                   # [B, H, T, T]
                if self.a.tau is None:
                    e = self.match(qq, kk, l) * scale
                else:                                                            # BLT: effort where routing is uncertain
                    ec = self.match(qq, kk, l, self.coarse_mask) * scale
                    ecm = ec.masked_fill(~self.win, float("-inf"))
                    top2 = torch.topk(ecm, 2, -1).values
                    certain = ((top2[..., 0] - top2[..., 1]) > self.a.tau)[..., None]
                    ef = self.match(qq, kk, l) * scale
                    e = torch.where(certain, ec, ef)
                    frac = self.coarse_mask.mean().item()
                    self.effort[0] += (frac + (1 - frac) * (~certain).float().mean().item()); self.effort[1] += 1
                if self.a.diag == "softmax":                                     # ceiling: exact q8.k8, softmax, all positions
                    cm = torch.ones(T, T, dtype=torch.bool).tril()
                    o = torch.softmax(e.masked_fill(~cm, float("-inf")), -1) @ (vv * s_v)
                elif self.a.diag == "softmax_window":
                    o = torch.softmax(e.masked_fill(~self.win, float("-inf")), -1) @ (vv * s_v)
                elif self.a.effort == "select":                                   # BLT: graded weights only if uncertain
                    em = e.masked_fill(~self.win, float("-inf")); top2 = torch.topk(em, 2, -1)
                    certain = ((top2.values[..., 0] - top2.values[..., 1]) > self.a.etau)[..., None]
                    o1 = torch.gather(vv * s_v, 2, top2.indices[..., :1].expand(-1, -1, -1, vv.shape[-1]))
                    o = torch.where(certain, o1, self.retrieve(e, vv, s_v, l))
                    self.effort[0] += (~certain).float().mean().item(); self.effort[1] += 1
                elif self.a.effort == "reach":                                    # BLT: long reach only if uncertain
                    idx = torch.arange(T); rel = idx[:, None] - idx[None]
                    small = (rel >= 0) & (rel < self.a.small)
                    em = e.masked_fill(~small, float("-inf")); top2 = torch.topk(em, 2, -1).values
                    certain = ((top2[..., 0] - top2[..., 1]) > self.a.etau)[..., None]
                    o = torch.where(certain, self.retrieve(e, vv, s_v, l, small), self.retrieve(e, vv, s_v, l))
                    c = certain.float().mean().item()
                    self.effort[0] += (c * self.a.small + (1 - c) * self.a.window) / self.a.window; self.effort[1] += 1
                else:
                    o = self.retrieve(e, vv, s_v, l)
                if local is not None:                                            # softmax on the same q, k, v
                    q, k, v = y[:, :, 0], y[:, :, 1], y[:, :, 2]
                    cm = torch.ones(T, T, dtype=torch.bool).tril()
                    pt = torch.softmax((q @ k.transpose(-1, -2) / hd ** .5).masked_fill(~cm, float("-inf")), -1)
                    ot = (pt @ v).detach()
                    local.append(((o - ot) ** 2).mean() / (ot ** 2).mean())
            x = x + self.bitlin(o.transpose(1, 2).reshape(B, T, self.d), self.Wo[l])
        return (x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.gf) @ self.head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8); ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--gapmax", type=float, default=4.0); ap.add_argument("--width", type=float, default=.25)
    ap.add_argument("--tau", type=float, default=None); ap.add_argument("--coarse", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=.01); ap.add_argument("--local", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--eval-windows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--diag", default="")
    ap.add_argument("--load", default=""); ap.add_argument("--effort", default="")
    ap.add_argument("--etau", type=float, default=1.0); ap.add_argument("--small", type=int, default=8)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); ntr = len(raw) * 9 // 10
    vb = np.frombuffer(raw[ntr:], np.uint8).astype(np.int64); nw = (len(vb) - 1) // T
    VX = torch.tensor(vb[:nw * T].reshape(nw, T)); VY = torch.tensor(vb[1:nw * T + 1].reshape(nw, T))
    if a.eval_windows: VX, VY = VX[:a.eval_windows], VY[:a.eval_windows]
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[VY.numpy()].mean())
    net = Net(m, a)
    if a.load:
        st = torch.load(a.load)["state"]; st = {k: v for k, v in st.items() if k not in ("coarse_mask", "win")}
        net.load_state_dict(st, strict=False)
    if a.check:                                                                  # pass-through init == q8 . k8
        with torch.no_grad():
            x = net.tok[VX[:4]] + net.pos[None]; q8, sc, _ = net.qkv8(x, 0)
            s1 = net.match(q8[:, :, 0], q8[:, :, 1], 0); s2 = q8[:, :, 0] @ q8[:, :, 1].transpose(-1, -2)
            print(f"check: pass-through match == q8.k8 exactly: {bool((s1 == s2).all())} (max |diff| {float((s1 - s2).abs().max())})")

    def evaluate(mode="student"):
        ce = hit = 0.0; net.effort = [0.0, 0]
        with torch.no_grad():
            for i in range(0, len(VX), 128):
                lg = net(VX[i:i + 128], mode); lp = torch.log_softmax(lg, -1)
                ce += -lp.gather(-1, VY[i:i + 128, :, None]).sum().item(); hit += (lg.argmax(-1) == VY[i:i + 128]).sum().item()
        eff = net.effort[0] / net.effort[1] if net.effort[1] else 1.0
        return ce / VY.numel(), hit / VY.numel(), eff
    ce_t, acc_t, _ = evaluate("teacher"); gain = lambda ce: (UNI - ce) / (UNI - ce_t)
    ce0, acc0, eff0 = evaluate()
    rec = dict(args=vars(a), teacher=dict(ce=ce_t, acc=acc_t, unigram=UNI), step0=dict(ce=ce0, acc=acc0, gain=gain(ce0), effort=eff0))
    print(json.dumps(rec), flush=True)
    opt = torch.optim.Adam([dict(params=list(net.corner), lr=a.lr), dict(params=list(net.w), lr=a.lr * .1),
                            dict(params=list(net.th), lr=a.lr), dict(params=list(net.lut), lr=a.lr)])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(a.steps, 1), eta_min=0)
    rng = np.random.default_rng(a.seed); b = np.frombuffer(raw[:ntr], np.uint8).astype(np.int64); t0 = time.time()
    r = rec["step0"]
    for step in range(1, a.steps + 1):
        st = rng.integers(0, len(b) - T - 1, a.batch); X = torch.tensor(b[st[:, None] + np.arange(T)])
        with torch.no_grad():
            tp = torch.softmax(net(X, "teacher"), -1)
        loc = []; lp = torch.log_softmax(net(X, local=loc), -1)
        kl = (tp * (torch.log(tp + 1e-12) - lp)).sum(-1).mean(); loss = kl + a.local * sum(loc) / len(loc)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            ce, acc, eff = evaluate()
            r = dict(step=step, kl=round(kl.item(), 4), local=round(float(sum(loc) / len(loc)), 4), ce=round(ce, 4),
                     acc=round(acc, 4), gain=round(gain(ce), 4), effort=round(eff, 3), sec=round(time.time() - t0, 1))
            print(json.dumps(r), flush=True)
    rec["final"] = r
    if a.out:
        torch.save(dict(state=net.state_dict(), args=vars(a)), a.out)
        json.dump(rec, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
