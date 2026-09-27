"""How sparse can the layer-1 output projection + head readout be? (the bound behind gate_head.py)

The site's exact linear readout (gate_head.py 'linear' control) per class v:
  logit_v ~ [ sum_j A[v, j] xq1_j 2^k1  +  s sum_i B[v, i] oq1_i ] / rms(x2)
  A = (head8 o gf / g1)^T  (per xq1 unit), B = (W_o1 diag(gf) head8)^T  (per oq1 unit), s = W_o shift scale per token.
Prune each class to its n largest terms by expected |contribution| (|weight| x mean |value| of that unit) and measure
the kept share of the float BitNet's gain. rms(x2) stays exact (a separate cost). Dense = 256 terms per class.

  python3 sparse_readout.py [--eval-windows 1024]
"""
import argparse
import json
import os

import torch

import common as K
import gate_head as GH
import lgn as L
from inspect_attn import load_attn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eval-windows", type=int, default=1024); a = ap.parse_args()
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin"))
    net = L.LGN(m); net.load_gatefield("runs/gf_w16.pt"); net.eval()
    VX, VY, UNI, _ = L.data(); VX, VY = VX[:a.eval_windows], VY[:a.eval_windows]
    ct = L.teacher_ce(m, VX, VY)
    X1, O1, DX, RL = [], [], [], []
    for i in range(0, len(VX), 64):
        xq1, k1, oq1, x1, dx1, real = GH.site(net, VX[i:i + 64])
        X1.append(xq1 * torch.pow(2.0, k1)); O1.append(oq1); DX.append(dx1); RL.append(real)
    X1, O1, DX, RL = torch.cat(X1), torch.cat(O1), torch.cat(DX), torch.cat(RL)
    zo = O1 @ net.Wo[1]
    s = (zo * DX).sum(-1, keepdim=True) / (zo * zo).sum(-1, keepdim=True).clamp(min=1)     # dx1 ~ s zo (per token)
    x2 = X1 / net.gi[1] + DX; rms = torch.sqrt((x2 ** 2).mean(-1, keepdim=True))
    Hm = net.head8 * net.gfi[:, None] * net.mh[0] * torch.pow(2.0, net.mh[1] - L.G)          # [128, 256]
    A = (Hm / net.gi[1][:, None]).T; B = (net.Wo[1] @ Hm).T                                # [256 cls, 128]
    xin, oin = X1, O1 * s                                                                  # per-token scaled inputs

    def score(lg):
        lp = torch.log_softmax(lg, -1); ce = -lp.gather(-1, VY[..., None]).mean().item()
        return dict(ce=round(ce, 5), acc=round((lg.argmax(-1) == VY).double().mean().item(), 5), gain=round((UNI - ce) / (UNI - ct), 5))
    res = dict(lgn=score(RL), dense_composite=score((xin @ A.T + oin @ B.T) / rms), teacher_ce=ct)
    # the other readout: keep W_o1 exact (dx1), prune only the head over x2 (128 terms per class)
    imp = torch.cat([A.abs() * xin.abs().mean((0, 1)), B.abs() * oin.abs().mean((0, 1))], 1)    # [256, 256]
    W = torch.cat([A, B], 1); inp = torch.cat([xin, oin], -1)
    out = []
    for n in (4, 8, 16, 32, 64, 128, 256):
        keep = torch.zeros_like(W, dtype=torch.bool).scatter_(1, imp.topk(n, 1).indices, True)
        r = score((inp @ (W * keep).T) / rms); r.update(terms_per_class=n, macs=256 * n); out.append(r)
        print(json.dumps(r), flush=True)
    Hx = (Hm).T; imph = Hx.abs() * x2.abs().mean((0, 1))
    for n in (8, 16, 32, 64, 128):
        keep = torch.zeros_like(Hx, dtype=torch.bool).scatter_(1, imph.topk(n, 1).indices, True)
        r = score((x2 @ (Hx * keep).T) / rms); r.update(head_only_terms_per_class=n, macs=256 * n + 16384); out.append(r)
        print(json.dumps(r), flush=True)
    res["pruned"] = out
    print(json.dumps(res))


if __name__ == "__main__":
    main()
