"""Did distillation preserve the MLP, or fit a new solution? Compare a trained TLN (student.py --out) with the teacher.

  units      teacher bit TB[j,l] = [h_j > q_jl] (the teacher's own hidden unit j above level l; q_j0 = 0, the other
             levels at the weighted quantiles of its active values). For each student gate (j,l): phi correlation
             with TB[j,l] (same index) and the best phi over all teacher bits of that layer (any index), weighted by
             the training frequency of each input.
  wiring     ternary wires: share of the conversion's nonzero wires that kept their sign, and nonzero overlap.
  transplant replace the student's layer-1 (and layer-2) gate bits with the teacher bits TB and score the student's
             output layer on validation: a student that still uses the teacher's units keeps working.

  python3 preserve.py runs/w4.npz [runs/r4.npz ...]
"""
import os
import sys

import numpy as np

import common as K
from student import convert, wq_levels
from export_tln import int_forward

S = os.environ.get("SCRATCH", "/tmp")
UNI = 3.1696


def phi(a, b, f):
    """weighted phi correlation between the columns of a [n, p] and b [n, q] (0/1) -> [p, q]"""
    ma, mb = f @ a, f @ b
    cov = (a * f[:, None]).T @ b - np.outer(ma, mb)
    sa, sb = np.sqrt(np.maximum(ma * (1 - ma), 1e-12)), np.sqrt(np.maximum(mb * (1 - mb), 1e-12))
    return cov / np.outer(sa, sb), ma, mb


def teacher_bits(h, f, L):
    q = wq_levels(h, f, L)
    return (h[:, :, None] > q[None]).reshape(len(h), -1).astype(np.float64)


def layers(E, u):
    A1, A2 = E["A1"].astype(np.int64), E["A2"].astype(np.int64); u = u.astype(np.int64); n = len(u)
    r1 = ((u @ A1)[:, :, None] > E["t1"][None]).reshape(n, -1)
    r2 = ((np.concatenate([u, r1], 1) @ A2)[:, :, None] > E["t2"][None]).reshape(n, -1)
    return r1.astype(np.float64), r2.astype(np.float64)


def out(E, u, r1, r2):
    z = np.concatenate([u, r1, r2], 1).astype(np.int64)
    return E["scale"] * (z @ E["C"].astype(np.int64) + E["bias"])


def tern(x):
    g = np.abs(x).mean(); return np.clip(np.round(x / g), -1, 1).astype(np.int64)


def main(paths):
    cfg, m = K.load(os.path.join(K.ROOT, "models/bitnet/bitnet_mlp.bin"))
    Cv, Ct = np.load(f"{S}/Cv.npy"), np.load(f"{S}/Ct.npy")
    ids, pos = K.domain(); fr = Ct.sum(1); f = fr / fr.sum()
    tl, acts = K.teacher(m, ids, pos, keep=True); ce_t, _ = K.score(tl, Cv)
    for p in paths:
        E = dict(np.load(p)); L = E["t1"].shape[1]; u = E["u"]
        TB1, TB2 = teacher_bits(acts["h0"], f, L), teacher_bits(acts["h1"], f, L)
        P = convert(m, f, 3, L)
        r1, r2 = layers(E, u)
        print(f"== {p}")
        for name, r, TB in (("layer 1", r1, TB1), ("layer 2", r2, TB2)):
            live = (f @ r > .001) & (f @ r < .999) & (f @ TB > .001) & (f @ TB < .999)
            ph, _, _ = phi(r[:, live], TB, f)
            idx = np.flatnonzero(live)
            same = ph[np.arange(len(idx)), idx]
            best = np.abs(ph).max(1)
            print(f"  {name}: {live.sum()} live gates | phi with the SAME teacher bit: median {np.median(same):.3f}, "
                  f"share > .5: {(same > .5).mean():.3f} | best phi over ANY teacher bit: median {np.median(best):.3f}, "
                  f"share > .5: {(best > .5).mean():.3f}")
        for k in ("A1", "A2", "C"):
            a0 = tern(P[k]); a1 = E[k].astype(np.int64); nz0 = a0 != 0
            print(f"  wires {k}: init nonzero {nz0.mean():.3f}, final nonzero {(a1 != 0).mean():.3f}, "
                  f"init wires kept with same sign {(a1[nz0] == a0[nz0]).mean():.3f}, "
                  f"exact entry agreement {(a1 == a0).mean():.3f} (chance {((a0 == -1).mean() * (a1 == -1).mean() + (a0 == 0).mean() * (a1 == 0).mean() + (a0 == 1).mean() * (a1 == 1).mean()):.3f})")
        for name, a, b in (("own gates", r1, r2), ("teacher layer-1 bits", TB1, layers_from(E, u, TB1)),
                           ("teacher layer-1 + layer-2 bits", TB1, TB2)):
            ce, acc = K.score(out(E, u, a, b), Cv)
            print(f"  output layer fed {name}: CE {ce:.4f} acc {acc:.4f} gain kept {(UNI - ce) / (UNI - ce_t):.3f}")


def layers_from(E, u, r1):
    """student layer 2 computed from given layer-1 bits"""
    n = len(u); z1 = np.concatenate([u, r1], 1).astype(np.int64)
    return ((z1 @ E["A2"].astype(np.int64))[:, :, None] > E["t2"][None]).reshape(n, -1).astype(np.float64)


if __name__ == "__main__":
    main(sys.argv[1:])
