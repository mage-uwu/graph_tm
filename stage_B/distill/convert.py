"""Weights -> threshold logic, with no training: how much of the MLP-only BitNet survives the conversion itself.

  input bits  u = thermometer code of the token embedding and of the position embedding (K levels per dim);
              x0 ~= u . R + x_lo reconstructs the embedding sum from the bits
  layer 1     r1 = [u . A1 + b1 > 0],  A1 = R . diag(g1) W_up1        (ReLU fires iff pre-activation > 0; norms and
                                                                         int8 scaling are positive per token)
  layer 2     r2 = [[u, r1] . A2 + b2 > 0],  residual folded: x1 ~= x0 + (alpha1 * r1) . W_down1
  output      votes = [u, r1, r2] . C + bias,  C = [R; alpha1 W_down1; alpha2 W_down2] diag(g_f) W_head
  alpha_j = E[relu(h_j)^2 | h_j > 0] over the training distribution (the magnitude a bit stands for).

Prints validation CE / accuracy for: the teacher; float folded matrices with binary hidden units; and every matrix
ternarized (per row for hidden units, whose threshold at 0 makes a per-row scale irrelevant).
"""
import os
import sys

import numpy as np

import common as K

S = os.environ.get("SCRATCH", "/tmp")


def eff(W):
    wq, gam = K.tern(W)
    return wq * gam


def thermo(E, K_):
    """thermometer code of each column of E at its K_ inner quantiles: bits [n, d*K_], decoder R [d*K_, d], x_lo"""
    q = np.quantile(E, (np.arange(K_) + 1) / (K_ + 1), axis=0)            # [K_, d]
    bits = (E[:, None, :] > q[None]).astype(np.float64)                    # [n, K_, d]
    # value reconstruction: level means between thresholds
    edges = np.concatenate([E.min(0, keepdims=True) - 1e-9, q, E.max(0, keepdims=True) + 1e-9])
    lev = np.zeros((K_ + 1, E.shape[1]))
    for i in range(K_ + 1):
        m = (E >= edges[i]) & (E <= edges[i + 1]) if i == 0 else (E > edges[i]) & (E <= edges[i + 1])
        lev[i] = np.where(m.sum(0) > 0, (E * m).sum(0) / np.maximum(m.sum(0), 1), (edges[i] + edges[i + 1]) / 2)
    R = np.diff(lev, axis=0)                                               # step when bit l turns on
    return bits.reshape(len(E), -1), R.reshape(-1, 1) * np.eye(E.shape[1])[None].repeat(K_, 0).reshape(-1, E.shape[1]), lev[0]


def build(m, Kl, freq):
    ids, pos = K.domain()
    _, acts = K.teacher(m, ids, pos, keep=True)
    bt, Rt, lot = thermo(m["tok"], Kl); bp, Rp, lop = thermo(m["pos"], Kl)
    u = np.concatenate([bt[ids], bp[pos]], 1); R = np.concatenate([Rt, Rp]); xlo = lot + lop
    s1, s2 = m["subs"]
    W1 = s1["g"][:, None] * eff(s1["up"]); W2 = s2["g"][:, None] * eff(s2["up"])
    D1, D2 = eff(s1["down"]), eff(s2["down"])
    f = freq / freq.sum()

    def alpha(h):
        a = np.maximum(h, 0) ** 2; on = (h > 0) * f[:, None]
        return (a * on).sum(0) / np.maximum(on.sum(0), 1e-12)
    a1, a2 = alpha(acts["h0"]), alpha(acts["h1"])
    A1, b1 = R @ W1, xlo @ W1
    A2, b2 = np.concatenate([R @ W2, (a1[:, None] * D1) @ W2]), xlo @ W2
    Hh = m["gf"][:, None] * m["head"]
    C = np.concatenate([R @ Hh, (a1[:, None] * D1) @ Hh, (a2[:, None] * D2) @ Hh]); bias = xlo @ Hh
    x2 = acts["x2"]; scale = 1 / (np.sqrt((x2 * x2).mean(1) + 1e-6) * f).sum()   # mean 1/rms folded into one scale
    return dict(u=u, A1=A1, b1=b1, A2=A2, b2=b2, C=C, bias=bias, scale=scale, r1t=acts["h0"] > 0, r2t=acts["h1"] > 0)


def run(P, tern_hidden=False, tern_out=False):
    A1, A2, C = P["A1"], P["A2"], P["C"]; b1, b2 = P["b1"], P["b2"]
    if tern_hidden:   # per-row (per hidden unit) absmean ternary; threshold rescaled with the row
        g1 = np.abs(A1).mean(0); g2 = np.abs(A2).mean(0)
        A1 = np.clip(np.round(A1 / g1), -1, 1); b1 = b1 / g1; A2 = np.clip(np.round(A2 / g2), -1, 1); b2 = b2 / g2
    r1 = (P["u"] @ A1 + b1 > 0).astype(np.float64)
    r2 = (np.concatenate([P["u"], r1], 1) @ A2 + b2 > 0).astype(np.float64)
    z = np.concatenate([P["u"], r1, r2], 1)
    Cm, sc = C, P["scale"]
    if tern_out:
        g = np.abs(C).mean(); Cm = np.clip(np.round(C / g), -1, 1); sc = sc * g
    logits = sc * (z @ Cm + P["bias"])
    return logits, (r1 == P["r1t"]).mean(), (r2 == P["r2t"]).mean()


if __name__ == "__main__":
    cfg, m = K.load(os.path.join(K.ROOT, "models/bitnet/bitnet_mlp.bin"))
    Cv, Ct = np.load(f"{S}/Cv.npy"), np.load(f"{S}/Ct.npy")
    ids, pos = K.domain()
    ce_t, acc_t = K.score(K.teacher(m, ids, pos), Cv)
    uni = 3.1696
    print(f"teacher: CE {ce_t:.4f} acc {acc_t:.4f}")
    for Kl in [int(a) for a in sys.argv[1:]] or [1, 3, 7]:
        P = build(m, Kl, Ct.sum(1))
        for th, to in [(False, False), (True, False), (True, True)]:
            L, m1, m2 = run(P, th, to)
            ce, acc = K.score(L, Cv)
            print(f"K={Kl} in_bits={P['u'].shape[1]} hidden {'tern' if th else 'float'} out {'tern' if to else 'float'}: "
                  f"CE {ce:.4f} acc {acc:.4f}  gain kept {(uni - ce) / (uni - ce_t):.3f}  "
                  f"r1 bits agree {m1:.3f} r2 {m2:.3f}")
