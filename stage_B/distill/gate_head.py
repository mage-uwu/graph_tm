"""Can a sparse DDLGN-style gate circuit stand in for layer 1's output projection + the head of the LGN engine?

The site (lgn.py spec, lgn.c): logits = head( q16( gf o (x1 + dx1) ) ), dx1 = round(W_o1 oq1 x scale), where
  x1  = residual after layer 0 (lgn.c already quantizes it to 6 bits as layer 1's input: xq1 = q6(g1 o x1))
  oq1 = layer 1's 6-bit attention output (q6 of the retrieved heads)
Everything before the site is the exact LGN engine (98.30% of the float BitNet's gain on full validation).

Circuit input bits (as in lgn.c's popcount matmul): per value v in [-31, 31], 5 positive planes (bit b of |v|, v > 0)
and 5 negative planes (v < 0), so value = sum_b 2^b (P_b - N_b) and any linear readout is a weighted sum of bits:
  xq1: 128 x 10, oq1: 128 x 10, plus an 8-bit thermometer of the relative scale bitlen(max|dx1|) - bitlen(max|x1|)
  (2,568 bits; the runtime has k1 and the W_o shift, equivalent information).
Circuit (DDLGN-light): layers of 2-input gates, 4 corner logits each (hard: corner > 0), hard forward + straight-through
  (multilinear surrogate) from step 0, fixed wiring, pass-through init; last layer = 256 classes x k bits,
  votes = popcount per class, logits = tau x votes + bias.
  wiring 'random': every gate input uniform over the previous layer.
  wiring 'weights': layers keep lanes (gate u's first input is unit u of the previous layer, second input random); the
    first layer's lane (class v, slot j) reads an input bit drawn by |composite weight| for class v (head . gf . W),
    initialized as copy (weight > 0) or NOT (weight < 0): at init, votes are a sign-agreement readout.
Loss: KL(LGN site logits || circuit logits) on next bytes (site distillation; the LGN is the ceiling).
Controls: 'lgn' (exact site), 'linear' (exact float readout from the same bits: what the bit encoding costs).

  python3 gate_head.py --k 32 --widths 8192,8192 --wiring weights --steps 2000
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import common as K
import lgn as L
from inspect_attn import load_attn

S = os.environ.get("SCRATCH", "/tmp")
_rec = []
_q6 = L.q6


def _q6_rec(*a, **k):
    r = _q6(*a, **k); _rec.append(r); return r


L.q6 = _q6_rec


def site(net, X):
    """exact LGN: (xq1, k1, oq1, x1, dx1, real logits)"""
    with torch.no_grad():
        x = net.tok[X] + net.pos[None, :X.shape[1]]
        x1 = net.layer(x, 0); _rec.clear()
        x2 = net.layer(x1, 1)
        (xq1, k1), _, (oq1, _) = _rec[0], _rec[1], _rec[2]
        _rec.clear()
        yf, kh = L.q6(x2 * net.gfi, 32767, 15); _rec.clear()
        li = yf @ net.head8; mh, eh = net.mh; emin = eh.min()
        rs = L.mf_mul(net.sqd, L.rsqrt((x2 ** 2).sum(-1)))
        real = li * mh * torch.pow(2.0, eh - emin) * torch.pow(2.0, emin - L.G + kh) * (rs[0] * torch.pow(2.0, rs[1]))[..., None]
    return xq1, k1, oq1, x1, x2 - x1, real


def bitlen(t):
    return torch.floor(torch.log2(t.clamp(min=1))) + 1


def build(net, X, bs=64):
    xs, os_, rs, lg, lin = [], [], [], [], []
    for i in range(0, len(X), bs):
        xq1, k1, oq1, x1, dx1, real = site(net, X[i:i + bs])
        r = bitlen(dx1.abs().amax(-1)) - bitlen(x1.abs().amax(-1))
        # control: exact float readout from the 6-bit x1 and the exact dx1 (a function of oq1 and the scale)
        x2c = xq1 * torch.pow(2.0, k1) / net.gi[1] + dx1
        # real ~ (x2 gf) . head8 m_h 2^(e_h - G) / rms(x2)  (the 16-bit head quantizer and minifloat rsqrt dropped)
        lc = (x2c * net.gfi) @ (net.head8 * net.mh[0] * torch.pow(2.0, net.mh[1] - L.G)) / torch.sqrt((x2c ** 2).mean(-1, keepdim=True))
        xs.append(xq1.to(torch.int8)); os_.append(oq1.to(torch.int8)); rs.append(r.to(torch.int8))
        lg.append(real.float()); lin.append(lc.float())
    return torch.cat(xs), torch.cat(os_), torch.cat(rs), torch.cat(lg), torch.cat(lin)


def planes(v):
    """int values [..., n] -> [..., n * 10] float bits (5 positive planes, 5 negative planes)"""
    a = v.abs().long(); b = (a[..., None] >> torch.arange(5)) & 1
    return torch.cat([b * (v > 0)[..., None], b * (v < 0)[..., None]], -1).flatten(-2).float()


def bits(xq, oq, r, rlo, rhi):
    th = (r.float()[..., None] > torch.linspace(rlo, rhi, 8)).float()
    return torch.cat([planes(xq), planes(oq), th], -1)


class Gates(torch.nn.Module):
    def __init__(self, ia, ib, c0):
        super().__init__()
        self.register_buffer("ia", ia); self.register_buffer("ib", ib); self.c = torch.nn.Parameter(c0)

    def forward(self, h):
        a, b = h[..., self.ia], h[..., self.ib]
        s = torch.sigmoid(self.c); hc = (self.c > 0).float()
        f = lambda w: w[:, 0] * (1 - a) * (1 - b) + w[:, 1] * (1 - a) * b + w[:, 2] * a * (1 - b) + w[:, 3] * a * b
        soft = f(s)
        return soft + (f(hc) - soft).detach()


COPY = torch.tensor([-2., -2., 2., 2.]); NOT = -COPY


def circuit(nin, widths, k, wiring, wbit, g):
    """wbit: [256, nin] composite weight per input bit and class (for 'weights' wiring)"""
    layers, prev = [], nin
    widths = list(widths) + [256 * k]
    for li, w in enumerate(widths):
        if wiring == "random":
            ia = torch.randint(prev, (w,), generator=g); c0 = COPY[None].repeat(w, 1)
        elif li == 0:                                                     # lanes (class v, slot j) read weighted input bits
            lanes = 256 * k; ia = torch.empty(w, dtype=torch.long); c0 = COPY[None].repeat(w, 1)
            per = torch.multinomial(wbit.abs() + 1e-12, k, replacement=False, generator=g)     # [256, k]
            sg = torch.gather(wbit, 1, per) > 0
            n = min(w, lanes); fl = per.flatten()[:n]; ia[:n] = fl
            c0[:n] = torch.where(sg.flatten()[:n, None], COPY, NOT)
            if w > lanes: ia[lanes:] = torch.randint(prev, (w - lanes,), generator=g)
        else:
            ia = torch.arange(w) % prev; c0 = COPY[None].repeat(w, 1)
        ib = torch.randint(prev, (w,), generator=g)
        layers.append(Gates(ia, ib, c0)); prev = w
    return torch.nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32); ap.add_argument("--widths", default="8192")
    ap.add_argument("--wiring", default="weights"); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=.05)
    ap.add_argument("--train-windows", type=int, default=4096); ap.add_argument("--eval-windows", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(); torch.manual_seed(a.seed); g = torch.Generator().manual_seed(a.seed)
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin"))
    net = L.LGN(m); net.load_gatefield("runs/gf_w16.pt"); net.eval()
    VX, VY, UNI, tr = L.data(); VX, VY = VX[:a.eval_windows], VY[:a.eval_windows]
    cache = f"{S}/gate_head_{a.train_windows}_{a.eval_windows}.pt"
    if os.path.exists(cache):
        D = torch.load(cache)
    else:
        t0 = time.time(); rng = np.random.default_rng(1)
        st = rng.integers(0, len(tr) - 65, a.train_windows); TX = torch.tensor(tr[st[:, None] + np.arange(64)])
        D = dict(tr=build(net, TX), va=build(net, VX)); torch.save(D, cache)
        print(f"site data built in {time.time() - t0:.0f}s", flush=True)
    xq, oq, r, lg, _ = D["tr"]; vxq, voq, vr, vlg, vlin = D["va"]
    rlo, rhi = float(r.float().quantile(.02)), float(r.float().quantile(.98))
    ct = L.teacher_ce(m, VX, VY)

    def score(logits):
        lp = torch.log_softmax(logits.double(), -1); ce = -lp.gather(-1, VY[..., None]).mean().item()
        return dict(ce=round(ce, 5), acc=round((logits.argmax(-1) == VY).double().mean().item(), 5), gain=round((UNI - ce) / (UNI - ct), 5))
    base = dict(lgn=score(vlg), linear=score(vlin), teacher_ce=ct)
    # composite weights per input bit and class: x2 ~ xq1 2^k1 / g1 + W_o1^T oq1 s; logit_v ~ sum_j head_jv gf_j x2_j
    H = (net.head8 * net.gfi[:, None]).float()                               # [128, 256]
    A = (H / net.gi[1][:, None].float()).T                                   # [256, 128] per xq1 unit
    B = (net.Wo[1].float() @ H).T                                            # [256, 128] per oq1 unit
    # put both blocks on the scale they have in the data (typical |contribution| per unit)
    ca = (xq.float().abs().mean() * A.abs().mean()).item(); cb = (oq.float().abs().mean() * B.abs().mean()).item()
    A, B = A / ca, B / cb
    pw = 2.0 ** torch.arange(5)
    wx = torch.cat([A[..., None] * pw, -A[..., None] * pw], -1).flatten(1); wo = torch.cat([B[..., None] * pw, -B[..., None] * pw], -1).flatten(1)
    wbit = torch.cat([wx, wo, torch.zeros(256, 8)], 1)
    nin = wbit.shape[1]; widths = [int(w) for w in a.widths.split(",") if w]
    cir = circuit(nin, widths, a.k, a.wiring, wbit, g)
    ngates = sum(w for w in widths) + 256 * a.k
    tau = torch.nn.Parameter(torch.tensor(0.05))
    uni = torch.softmax(lg.reshape(-1, 256)[::97].double(), -1).mean(0)            # start from the LGN's mean prediction
    bias = torch.nn.Parameter(torch.log(uni).float())

    def run(xb, ob, rb):
        v = cir(bits(xb, ob, rb, rlo, rhi)).reshape(*xb.shape[:-1], 256, a.k).sum(-1)
        return tau * v + bias, v

    def evaluate():
        with torch.no_grad():
            out = torch.cat([run(vxq[i:i + 16], voq[i:i + 16], vr[i:i + 16])[0] for i in range(0, len(vxq), 16)])
        return score(out)
    info = dict(args=vars(a), gates=ngates, adder_ops_est=256 * a.k * 5, input_bits=nin, **base)
    print(json.dumps(info), flush=True)
    print(json.dumps(dict(step=0, **evaluate())), flush=True)
    opt = torch.optim.Adam([dict(params=cir.parameters(), lr=a.lr), dict(params=[tau, bias], lr=a.lr * .2)])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * .02)
    rng = np.random.default_rng(a.seed + 7); t0 = time.time()
    for step in range(1, a.steps + 1):
        i = torch.tensor(rng.integers(0, len(xq), a.batch))
        out, _ = run(xq[i], oq[i], r[i]); tp = torch.softmax(lg[i], -1)
        loss = (tp * (torch.log(tp + 1e-12) - torch.log_softmax(out, -1))).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            print(json.dumps(dict(step=step, kl=round(loss.item(), 4), **evaluate(), sec=round(time.time() - t0))), flush=True)
    if a.out: torch.save(dict(state=cir.state_dict(), tau=tau.data, bias=bias.data, args=vars(a)), a.out)


if __name__ == "__main__":
    main()
