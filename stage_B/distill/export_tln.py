"""Export a trained TLN (student.py --out X.npz) to the integer model file read by tln.c, plus reference votes.

  X.tln   "TLN00001", int32 header, then the integer network (see tln.c for the layout)
  X.votes int32 votes [256 * 64 inputs (byte * 64 + pos), 256] from a numpy integer forward pass, which also must
          reproduce the torch hard forward (checked here)
"""
import sys

import numpy as np


def int_forward(E, u):
    A1, A2, C = E["A1"].astype(np.int64), E["A2"].astype(np.int64), E["C"].astype(np.int64)
    t1, t2 = E["t1"].astype(np.int64), E["t2"].astype(np.int64)
    u = u.astype(np.int64); n = len(u)
    r1 = ((u @ A1)[:, :, None] > t1[None]).reshape(n, -1).astype(np.int64)
    z1 = np.concatenate([u, r1], 1)
    r2 = ((z1 @ A2)[:, :, None] > t2[None]).reshape(n, -1).astype(np.int64)
    z = np.concatenate([z1, r2], 1)
    return z @ C + E["bias"].astype(np.int64)


def main(npz):
    E = dict(np.load(npz)); u = E["u"]; T = 64
    nin = u.shape[1]; ntok = nin // 2                      # [token code | position code], equal halves
    tok = u[np.arange(256) * T, :ntok]; pos = u[np.arange(T), ntok:]
    assert (np.concatenate([tok[np.arange(256 * T) // T], pos[np.arange(256 * T) % T]], 1) == u).all()
    votes = int_forward(E, u)
    H1, L = E["t1"].shape; H2 = E["t2"].shape[0]
    qa = int(max(np.abs(E["A1"]).max(), np.abs(E["A2"]).max())); qc = int(np.abs(E["C"]).max())
    hdr = np.array([256, T, ntok, nin - ntok, H1, H2, L, qa, qc], np.int32)
    with open(npz.replace(".npz", ".tln"), "wb") as f:
        f.write(b"TLN00001"); f.write(hdr.tobytes()); f.write(np.float64(E["scale"]).tobytes())
        f.write(tok.astype(np.uint8).tobytes()); f.write(pos.astype(np.uint8).tobytes())
        for a in (E["A1"], E["t1"], E["A2"], E["t2"], E["C"], E["bias"]):
            f.write(np.ascontiguousarray(a, np.int32).tobytes())
    votes.astype(np.int32).tofile(npz.replace(".npz", ".votes"))
    nnz = lambda a: int((a != 0).sum())
    print(f"exported {npz.replace('.npz', '.tln')}: in {nin} bits, gates {H1 * L}+{H2 * L}, wires "
          f"A1 {nnz(E['A1'])} A2 {nnz(E['A2'])} C {nnz(E['C'])}, qa {qa} qc {qc}, scale {E['scale']:.5f}")
    return E, votes


if __name__ == "__main__":
    main(sys.argv[1])
