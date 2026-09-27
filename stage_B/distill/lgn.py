"""LGN-attention BitNet: the frozen attention-only BitNet with every continuous operation replaced by integer / logic
operations and the softmax retrieval replaced by a learned hard gate field. This file is the exact specification:
one implementation in torch float64 whose every value is an integer or an integer times a power of two (all below
2^53, so float64 holds them exactly). exact=True is the true-integer reference that lgn.c must reproduce bit for bit;
training uses the same forward values with straight-through gradient surrogates.

Spec (d = 128, H = 4, hd = 32, 2 layers, T = 64; frozen ternary W with scales gamma from the BitNet file)
  residual     integers in units u = 2^-12; token / position tables: 8-bit quantized, re-expressed in units u
  minifloat    (m, e), m in [64, 127] (7-bit mantissa), value m 2^e; product: m1 m2 >> (6 or 7) rounded, renormalized
  rsqrt(S)     S = sum x^2 (integer): n = floor(log2 S), idx = top 8 bits after the leading one, parity of n ->
               RSQ[parity][idx] minifloat, exponent - floor(n / 2); rsq = sqrt(d) (minifloat) x rsqrt(S) = 1 / rms
  q6(y)        per row: k = min k >= 0 with round(max|y| / 2^k) <= 31; q = sign(y) floor((|y| + 2^(k-1)) / 2^k)
  layer l      y = x o g_int (g_int = round(g 2^8));  xq, k1 = q6(y);  z = xq . W_qkv  (exact integers)
               c = gamma_qkv (minifloat) x rsq  ->  real q/k/v = z m_c 2^(e_c + k1 - 8)
               q6, k6, v6 per token per head: q6(z_head) with shifts k2  ->  exponents e_c + k1 - 8 + k2
               score(t,s) = gate field on the 5 magnitude bit-planes of q6_t and k6_s (pass-through: q6 . k6)
               E(t,s) = score m_c(s) 2^(ek_s - emin_t)  (emin_t = min over the window of ek_s); gap = max E - E
               thr_i(t) = floor( mf(theta_i sqrt(hd)) x RM[m_c(t)] 2^(-eq_t - emin_t) );  level = #{i: gap > thr_i}
               weight = LUT[level] (integers), 0 at level L or outside the window (s in t-15 .. t)
               num = sum_s weight v6 m_c(s) 2^(ev_s - evmin_t);  o = floor(num / sum weight)   (per head)
               heads aligned to e_o = min_h evmin;  oq, k3 = q6(o);  zo = oq . W_o
               x += round( zo m_go 2^(e_go + e_o + k3) )  with (m_go, e_go) = minifloat(gamma_o / u)
  head         yf, kh = shift quantizer of x o gf_int to 16 bits (|yf| <= 32767)
               logit_int_v = (yf . head8_v) m_hv 2^(e_hv - min e_h)   (argmax: exact integers)
               probabilities (for CE only): logit_int 2^(min e_h - 8 + kh) x rsq(final)

  python3 lgn.py eval  [--gates runs/lgn.pt]           composed evaluation (full validation)
  python3 lgn.py train --steps 400 --out runs/lgn.pt   recovery training of the gate field on this network
  python3 lgn.py export runs/lgn.pt runs/lgn.bin       integer model file + reference logits for lgn.c
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

import common as K
from inspect_attn import load_attn

S = os.environ.get("SCRATCH", "/tmp")
DT = torch.float64
F, G, NB, QMAX, W_, LV = 12, 8, 5, 31, 16, 8
U = 2.0 ** -F


def ste(hard, soft):
    return soft + (hard - soft).detach()


def rshift_round(v, s):
    """round-half-away-from-zero division of integer-valued v by 2^s (s >= 0)"""
    p = torch.pow(2.0, s.to(DT) if torch.is_tensor(s) else torch.tensor(float(s), dtype=DT))
    return torch.sign(v) * torch.floor((v.abs() + p / 2) / p)


def shift_round(v, s):
    """v * 2^s for s >= 0, rounded v / 2^-s for s < 0"""
    s = s.to(DT)
    up = v * torch.pow(2.0, s.clamp(min=0))
    return torch.where(s >= 0, up, rshift_round(v, (-s).clamp(min=0)))


def bitlen(x):
    return torch.frexp(x.to(DT))[1].to(DT)                     # x >= 0 integer-valued; 0 -> 0


def q6(y, qmax=QMAX, nb=5):
    """shift quantizer to [-qmax, qmax] per row (last dim); returns q (STE), k"""
    mx = y.detach().abs().amax(-1, keepdim=True)
    k = (bitlen(mx) - nb).clamp(min=0)
    k = torch.where(rshift_round(mx, k) > qmax, k + 1, k)
    q = rshift_round(y.detach(), k).clamp(-qmax, qmax)
    return ste(q, y / torch.pow(2.0, k)), k


def mf(x):
    """float > 0 -> minifloat (m, e); the mantissa is integer-valued"""
    mant, ex = torch.frexp(x.to(DT))
    m = torch.floor(mant * 128 + .5); e = ex.to(DT) - 7
    over = m >= 128
    return torch.where(over, torch.full_like(m, 64), m), torch.where(over, e + 1, e)


def mf_mul(a, b):
    p = a[0] * b[0]
    s = torch.where(p >= 8192, torch.full_like(p, 7), torch.full_like(p, 6))
    m = rshift_round(p, s); e = a[1] + b[1] + s
    over = m >= 128
    return torch.where(over, torch.full_like(m, 64), m), torch.where(over, e + 1, e)


def tables():
    rsq = np.zeros((2, 256, 2)); rm = np.zeros((128, 2))
    for par in range(2):
        for i in range(256):
            m, e = mf(torch.tensor(((1 + (i + .5) / 256) * (1 + par)) ** -.5, dtype=DT)); rsq[par, i] = (m, e)
    for mm in range(64, 128):
        m, e = mf(torch.tensor(1.0 / mm, dtype=DT)); rm[mm] = (m, e)
    return torch.tensor(rsq, dtype=DT), torch.tensor(rm, dtype=DT)


RSQ, RM = tables()


def rsqrt(Ssum):
    """minifloat 1 / sqrt(S) for integer S >= 1 via RSQ"""
    Ssum = Ssum.clamp(min=1)
    n = bitlen(Ssum) - 1
    f = Ssum / torch.pow(2.0, n)                               # [1, 2), exact
    idx = torch.floor((f - 1) * 256).long().clamp(0, 255); par = (n % 2).long()
    ent = RSQ[par, idx]
    return ent[..., 0], ent[..., 1] - torch.floor(n / 2)


def hard_sig(x, w):
    s = torch.sigmoid(x / w); return ((x > 0).to(DT) - s).detach() + s


class LGN(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        f = lambda x: torch.tensor(x, dtype=DT)
        self.T, self.d, self.H = m["T"], m["d"], m["H"]; self.hd = self.d // self.H; nl = len(m["subs"])
        sc = max(np.abs(m["tok"]).max(), np.abs(m["pos"]).max()) / 127
        q8 = lambda t: np.round(np.round(t / sc) * sc / U)
        self.tok, self.pos = f(q8(m["tok"])), f(q8(m["pos"]))
        self.gi = [f(np.round(s["g"] * 2 ** G)) for s in m["subs"]]; self.gfi = f(np.round(m["gf"] * 2 ** G))
        self.Wq, self.Wo, gq, go = [], [], [], []
        for s in m["subs"]:
            w, g = K.tern(s["qkv"]); self.Wq.append(f(w)); gq.append(g)
            w, g = K.tern(s["o"]); self.Wo.append(f(w)); go.append(g)
        self.gq = [mf(f(g)) for g in gq]; self.go = [mf(f(g / U)) for g in go]
        self.sqd = mf(f(np.sqrt(self.d)))
        hs = np.abs(m["head"]).max(0) / 127; self.head8 = f(np.round(m["head"] / hs)); self.mh = mf(f(hs))
        idx = torch.arange(self.T); rel = idx[:, None] - idx[None]
        self.win = (rel >= 0) & (rel < W_)
        P = torch.nn.Parameter; H = self.H
        ab = torch.arange(NB)[:, None] + torch.arange(NB)[None]
        corner = torch.full((4, NB, NB), -3.0, dtype=DT); corner[3] = 3.0
        self.corner = torch.nn.ParameterList([P(corner[None].repeat(H, 1, 1, 1).clone()) for _ in range(nl)])
        self.w0 = (2.0 ** ab).to(DT)
        self.wr = torch.nn.ParameterList([P(torch.ones(H, NB, NB, dtype=DT)) for _ in range(nl)])
        th = torch.arange(1, LV + 1, dtype=DT) * 4.0 / LV
        cen = torch.cat([torch.zeros(1, dtype=DT), (th[:-1] + th[1:]) / 2])
        self.th = torch.nn.ParameterList([P(th[None].repeat(H, 1).clone()) for _ in range(nl)])
        self.lut = torch.nn.ParameterList([P(torch.exp(-cen)[None].repeat(H, 1).clone()) for _ in range(nl)])

    def load_gatefield(self, path):
        """thresholds / weight tables from a gatefield.py run (same window and levels); match stays pass-through"""
        st = torch.load(path)["state"]
        for l in range(len(self.th)):
            self.th[l].data = st[f"th.{l}"].to(DT).clone(); self.lut[l].data = st[f"lut.{l}"].to(DT).clone()

    # ---- integer parameters used by the forward pass (and exported)
    def ints(self, l):
        corner = (self.corner[l] > 0).to(DT)
        w = ste(torch.round(self.w0 * self.wr[l]), self.w0 * self.wr[l])
        th = torch.sort(self.th[l], -1).values
        thm = mf(th.detach() * self.hd ** .5)
        lut = ste(torch.round(255 * self.lut[l]).clamp(min=0), 255 * self.lut[l])
        return corner, w, th, thm, lut

    def match(self, q, k, l):
        sq, sk = torch.sign(q) + (q == 0), torch.sign(k) + (k == 0)
        bits = lambda a: (torch.floor(a.abs()[..., None] / 2.0 ** torch.arange(NB, dtype=DT)) % 2)
        bq, bk = bits(q.detach()), bits(k.detach())
        c = hard_sig(self.corner[l], 1.0)
        _, w, _, _, _ = self.ints(l); M = c * w[:, None]
        tot = 0
        for ci, (x, y) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            A = (bq if x else 1 - bq) * sq[..., None]; B = (bk if y else 1 - bk) * sk[..., None]
            tot = tot + torch.einsum("bhtjc,bhsjc->bhts", torch.einsum("bhtja,hac->bhtjc", A, M[:, ci]), B)
        if self.training:                                                     # gradient path to q, k (STE)
            tot = ste(tot, tot.detach() + (q @ k.transpose(-1, -2) - (q @ k.transpose(-1, -2)).detach()))
        return tot

    def layer(self, x, l):
        B, T = x.shape[:2]; H, hd, d = self.H, self.hd, self.d
        y = x * self.gi[l]; xq, k1 = q6(y)
        z = xq @ self.Wq[l]
        rs = mf_mul(self.sqd, rsqrt((x.detach() ** 2).sum(-1)))               # 1 / rms, [B, T]
        mc, ec = mf_mul(self.gq[l], rs)
        zq = z.reshape(B, T, 3, H, hd).permute(0, 3, 2, 1, 4)                  # [B, H, 3, T, hd]
        qkv, k2 = q6(zq)
        eb = (ec + k1[..., 0] - G)[:, None, None, :]                          # [B, 1, 1, T]
        ex = eb + k2[..., 0]                                                  # [B, H, 3, T]
        mcb = mc[:, None, :]                                                  # [B, 1, T]
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        eq, ek, ev = ex[:, :, 0], ex[:, :, 1], ex[:, :, 2]
        score = self.match(q, k, l)                                           # [B, H, T, T]
        win = self.win
        big = torch.tensor(1e9, dtype=DT)
        ekw = torch.where(win, ek[:, :, None, :], big); emin = ekw.amin(-1, keepdim=True)
        E = score * mcb[:, :, None, :] * torch.pow(2.0, (ek[:, :, None, :] - emin).clamp(max=60))
        E = torch.where(win, E, torch.full_like(E, -1e18))
        gap = E.amax(-1, keepdim=True) - E
        corner, w, th, thm, lut = self.ints(l)
        # per-query integer thresholds: floor( mf(theta sqrt(hd)) x RM[m_c(t)] 2^(-eq - emin) )
        rmt = RM[mc.long()]                                                   # [B, T, 2]
        A = mf_mul((thm[0][None, :, None, :], thm[1][None, :, None, :]),
                   (rmt[..., 0][:, None, :, None], rmt[..., 1][:, None, :, None]))   # [B, H, T, L]
        sh = A[1] - eq[..., None] - emin
        thr = torch.floor(A[0] * torch.pow(2.0, sh))
        on_hard = (gap[..., None] > thr[:, :, :, None, :]).to(DT)
        if self.training:
            greal = gap * torch.pow(2.0, emin) * (mcb[:, :, :, None] * torch.pow(2.0, eq[..., None])) / hd ** .5
            soft = torch.sigmoid((greal[..., None] - th[None, :, None, None, :]) / .25)
            on = (on_hard - soft).detach() + soft
        else:
            on = on_hard
        steps = torch.cat([lut[:, 1:] - lut[:, :-1], -lut[:, -1:]], -1)
        wgt = lut[None, :, 0, None, None] + (on * steps[None, :, None, None]).sum(-1)
        wgt = torch.where(win, wgt.clamp(min=0), torch.zeros_like(wgt))
        evw = torch.where(win, ev[:, :, None, :], big); evmin = evw.amin(-1, keepdim=True)
        fac = torch.where(win, mcb[:, :, None, :] * torch.pow(2.0, (ev[:, :, None, :] - evmin).clamp(max=60)), torch.zeros_like(wgt))
        num = (wgt * fac) @ v                                                 # [B, H, T, hd]
        den = wgt.sum(-1, keepdim=True).clamp(min=1)
        o = ste(torch.floor(num.detach() / den.detach()), num / den)
        eo = evmin.amin(1, keepdim=True)                                      # [B, 1, T, 1]
        o = o * torch.pow(2.0, evmin - eo)
        o = o.permute(0, 2, 1, 3).reshape(B, T, d)
        oq, k3 = q6(o)
        zo = oq @ self.Wo[l]
        mg, eg = self.go[l]
        shift = eg + eo[:, 0] + k3                                            # [B, T, 1]
        dx = ste(shift_round((zo * mg).detach(), shift.detach()), zo * mg * torch.pow(2.0, shift))
        return x + dx

    def forward(self, X):
        x = self.tok[X] + self.pos[None, :X.shape[1]]
        for l in range(len(self.gi)): x = self.layer(x, l)
        yf, kh = q6(x * self.gfi, 32767, 15)                                 # 16-bit head input (shift)
        li = yf @ self.head8
        mh, eh = self.mh; emin = eh.min()
        logit_int = li * mh * torch.pow(2.0, eh - emin)
        rs = mf_mul(self.sqd, rsqrt((x.detach() ** 2).sum(-1)))
        real = logit_int * torch.pow(2.0, emin - G + kh) * (rs[0] * torch.pow(2.0, rs[1]))[..., None]
        return logit_int, real


def teacher_logits(m, X):
    """frozen float BitNet (bitnet.c semantics) for the KL targets"""
    import int_skeleton as IS
    if not hasattr(teacher_logits, "sk"):
        teacher_logits.sk = IS.Skeleton(IS.load(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")), [])
    return torch.tensor(teacher_logits.sk.forward(X.numpy()), dtype=DT)


def data():
    T = 64; raw = open(f"{S}/wiki_11m.txt", "rb").read(); lo = len(raw) * 9 // 10
    b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); nw = (len(b) - 1) // T
    VX = torch.tensor(b[:nw * T].reshape(nw, T)); VY = torch.tensor(b[1:nw * T + 1].reshape(nw, T))
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[VY.numpy()].mean())
    tr = np.frombuffer(raw[:lo], np.uint8).astype(np.int64)
    return VX, VY, UNI, tr


TCE = {}


def teacher_ce(m, X, Y):
    key = len(X)
    if key not in TCE:
        ce = 0.0
        for i in range(0, len(X), 256):
            lp = torch.log_softmax(teacher_logits(m, X[i:i + 256]), -1); ce += -lp.gather(-1, Y[i:i + 256, :, None]).sum().item()
        TCE[key] = ce / Y.numel()
    return TCE[key]


def evaluate(net, VX, VY, UNI, n=None, bs=128, m=None):
    net.eval(); ce = hit = 0.0; X, Y = (VX, VY) if n is None else (VX[:n], VY[:n])
    with torch.no_grad():
        for i in range(0, len(X), bs):
            li, real = net(X[i:i + bs]); lp = torch.log_softmax(real, -1)
            ce += -lp.gather(-1, Y[i:i + bs, :, None]).sum().item(); hit += (li.argmax(-1) == Y[i:i + bs]).sum().item()
    ce /= Y.numel(); hit /= Y.numel(); ct = teacher_ce(m, X, Y)
    return dict(ce=ce, acc=hit, teacher_ce=ct, gain=(UNI - ce) / (UNI - ct))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("args", nargs="*")
    ap.add_argument("--gates", default=""); ap.add_argument("--init", default="runs/gf_w16.pt")
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=.01); ap.add_argument("--out", default="")
    ap.add_argument("--eval-windows", type=int, default=0); ap.add_argument("--eval-every", type=int, default=100)
    a = ap.parse_args(); torch.manual_seed(0)
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin"))
    net = LGN(m); VX, VY, UNI, tr = data(); nev = a.eval_windows or None
    if a.gates: net.load_state_dict(torch.load(a.gates)["state"])
    elif a.init and os.path.exists(a.init): net.load_gatefield(a.init)
    if a.cmd == "eval":
        t0 = time.time(); r = evaluate(net, VX, VY, UNI, nev, m=m); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r))
    elif a.cmd == "train":
        r = evaluate(net, VX, VY, UNI, nev, m=m); print(json.dumps(dict(step=0, **r)), flush=True)
        opt = torch.optim.Adam([dict(params=list(net.corner), lr=a.lr), dict(params=list(net.wr), lr=a.lr * .1),
                                dict(params=list(net.th), lr=a.lr), dict(params=list(net.lut), lr=a.lr)])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=0)
        rng = np.random.default_rng(1); t0 = time.time()
        for step in range(1, a.steps + 1):
            net.train(); st = rng.integers(0, len(tr) - 65, a.batch); X = torch.tensor(tr[st[:, None] + np.arange(64)])
            tp = torch.softmax(teacher_logits(m, X), -1)
            _, real = net(X); lp = torch.log_softmax(real, -1)
            loss = (tp * (torch.log(tp + 1e-12) - lp)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % a.eval_every == 0 or step == a.steps:
                r = evaluate(net, VX, VY, UNI, nev, m=m)
                print(json.dumps(dict(step=step, kl=round(loss.item(), 4), **r, sec=round(time.time() - t0, 1))), flush=True)
        if a.out: torch.save(dict(state=net.state_dict()), a.out)
    elif a.cmd == "export":
        export(net, a.args[0], a.args[1], VX)


def export(net, src, dst, VX):
    """integer model file for lgn.c and reference integer logits on the first 64 validation windows"""
    net.load_state_dict(torch.load(src)["state"]); net.eval()
    I = lambda t: np.asarray(t.detach().numpy() if torch.is_tensor(t) else t, dtype=np.int64)
    with open(dst, "wb") as f:
        f.write(b"LGNATT01"); f.write(struct.pack("<9i", net.T, net.d, net.H, len(net.gi), NB, W_, LV, F, G))
        wr = lambda a: f.write(np.ascontiguousarray(I(a), dtype=np.int64).tobytes())
        wr(net.tok); wr(net.pos); wr(net.gfi); wr(net.head8); wr(net.mh[0]); wr(net.mh[1])
        wr(torch.stack(net.sqd)); wr(RSQ); wr(RM)
        for l in range(len(net.gi)):
            corner, w, th, thm, lut = net.ints(l)
            wr(net.gi[l]); wr(net.Wq[l]); wr(net.Wo[l]); wr(torch.stack(net.gq[l])); wr(torch.stack(net.go[l]))
            wr(corner); wr(w); wr(thm[0]); wr(thm[1]); wr(lut)
    with torch.no_grad():
        li, _ = net(VX[:64])
    np.asarray(li.numpy(), dtype=np.int64).tofile(dst + ".ref")
    VX[:64].numpy().astype(np.uint8).tofile(dst + ".x")
    print(f"exported {dst}: reference integer logits for 64 windows ({li.abs().max().item():.3g} max |logit|)")


if __name__ == "__main__":
    main()
