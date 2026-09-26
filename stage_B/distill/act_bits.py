"""Toward a pure-logic BitNet: how few activation bits does the frozen library need, and can every per-token scale
come from a BLT-style lookup table instead of arithmetic? Read-only, on top of the step-1 integer skeleton
(int_skeleton.py, 6-bit-mantissa scales), scored on the full validation text.

  abits b     every BitLinear input quantized to b bits per value (absmax; b = 2 is ternary {-1, 0, +1} x scale,
              b = 1 is sign x mean|y|). Popcount-compatible: b bit-planes against the ternary wiring.
  nonuni      the same number of levels, but level edges and values shared across dimensions and set from the
              pooled distribution of |y| / max|y| (a per-token-normalized log-like quantizer; still b shared planes)
  qkvbits b   q, k, v at b bits per value (per token per head absmax) for the attention interface
  lookup      per-token scales from tables instead of max / sqrt:
                layer 0 BitLinear input and q/k/v scales: exact functions of (byte, position) -> a 256 x 64 table
                all other scales (W_o inputs, layer-1 inputs, final 1/rms): BLT hashed n-gram tables keyed on the
                last n bytes and the position, n in {8, 6, 4, 3, 2, 1} with backoff to the longest n seen >= 2
                times in training; each entry is the mean log2 scale (then rounded to the 6-bit minifloat)

  python3 act_bits.py            (SCRATCH = folder with wiki_11m.txt and Ct.npy)
"""
import json
import os
import time

import numpy as np

import common as K
from int_skeleton import Skeleton, load, minifloat

S = os.environ.get("SCRATCH", "/tmp")
FULL = ["nonorm", "fixres", "fixgain", "minifl", "qkv8", "head8"]
NS = (8, 6, 4, 3, 2, 1); NBK = 1 << 20; P = np.uint64(1000003)


USEPOS = [True]; MINC = [2]


def ctx_keys(X, n, pos=None):
    """hashed (last n bytes[, position]) key per token [B, T]"""
    pos = USEPOS[0] if pos is None else pos
    B, T = X.shape; h = np.zeros((B, T), np.uint64)
    Xp = np.concatenate([np.full((B, n - 1), 256), X], 1).astype(np.uint64)
    for i in range(n):
        h = h * P + Xp[:, i:i + T] + np.uint64(1)
    if pos: h = h * np.uint64(64) + np.arange(T, dtype=np.uint64)[None]
    return (h % np.uint64(NBK)).astype(np.int64)


class Tables:
    def __init__(self):
        self.sum, self.cnt = {}, {}

    def add(self, name, X, logv):                        # logv [B, T, k]
        for n in NS:
            key = ctx_keys(X, n, pos=(n == 1) or USEPOS[0]).ravel(); v = logv.reshape(len(key), -1)
            s = self.sum.setdefault((name, n), np.zeros((NBK, v.shape[1]))); c = self.cnt.setdefault((name, n), np.zeros(NBK))
            np.add.at(s, key, v); np.add.at(c, key, 1)

    def get(self, name, X, maxn):
        B, T = X.shape; out = None; done = np.zeros(B * T, bool)
        for n in [n for n in NS if n <= maxn]:
            key = ctx_keys(X, n, pos=(n == 1) or USEPOS[0]).ravel(); c = self.cnt[(name, n)][key]
            ok = (c >= (MINC[0] if n > 1 else 1)) & ~done
            val = self.sum[(name, n)][key] / np.maximum(c, 1)[:, None]
            if out is None: out = np.zeros_like(val)
            out[ok] = val[ok]; done |= ok
        return out, done


class Bits(Skeleton):
    def __init__(self, m, knobs, abits=8, nonuni=None, qkvbits=8, tables=None, lookup=None, maxn=8, collect=None):
        super().__init__(m, knobs, mant=6)
        self.abits, self.nonuni, self.qkvbits = abits, nonuni, qkvbits
        self.tables, self.lookup, self.maxn, self.collect = tables, lookup, maxn, collect
        self.X = None; self.miss = [0, 0]

    def q_levels(self, y, mx):
        """quantize rows y (absmax mx) to self.abits; returns values in units where max -> 127 (keeps scales)"""
        b = self.abits
        if b >= 8: return np.round(127 * y / mx)
        if b == 1:
            a = np.abs(y).mean(-1, keepdims=True); return np.sign(y) * a * 127 / mx
        if self.nonuni is not None:                                   # shared edges / values on |y| / mx
            r = np.abs(y) / mx; lvl = np.searchsorted(self.nonuni[0], r); return np.sign(y) * self.nonuni[1][lvl] * 127
        L = 2 ** (b - 1) - 1; return np.round(L * y / mx) * 127 / L

    def sc(self, name, val, layer0_exact):
        """per-token scale: computed, or looked up (exact (byte, pos) table for layer-0 names)"""
        if self.collect is not None:
            self.collect.add(name, self.X, np.log2(np.maximum(val, 1e-30)).reshape(*self.X.shape, -1))
        if self.lookup and (self.lookup == "all" or layer0_exact):
            v, ok = self.tables.get(name, self.X, 1 if layer0_exact else self.maxn)
            self.miss[0] += (~ok).sum(); self.miss[1] += ok.size
            val = np.where(ok.reshape(*self.X.shape, *([1] * (val.ndim - 2))), (2.0 ** v).reshape(val.shape), val)
        return minifloat(val, self.mant)

    def forward(self, X):
        m = self.m; B, T = X.shape; d, H = m["d"], m["H"]; hd = d // H; self.X = X
        mask = np.triu(np.full((T, T), -np.inf), 1)
        x = self.tok[X] + self.pos[None, :T]
        for i, s in enumerate(m["subs"]):
            y = x * self.g[i]; rms = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6 / self.u ** 2)
            mx = np.maximum(np.abs(y).max(-1, keepdims=True), 1e-30)
            xq = self.q_levels(y, mx)
            c = self.sc(f"in{i}", (s["ga"] * mx / 127 / rms), i == 0)             # [B, T, 1]
            yi = (xq.reshape(-1, d) @ s["wa"]).reshape(B, T, 3, H, hd)
            my = np.maximum(np.abs(yi).max(-1, keepdims=True), 1e-30)          # per token per head
            L = 2 ** (self.qkvbits - 1) - 1
            q8 = np.round(L * yi / my)
            cs = self.sc(f"qkv{i}", c[..., None, None] * my / L, i == 0)     # [B, T, 3, H, 1]
            val = q8 * cs
            q, k, v = (val[:, :, j].transpose(0, 2, 1, 3) for j in range(3))
            e = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
            p = np.exp(e - e.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            o = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            mo = np.maximum(np.abs(o).max(-1, keepdims=True), 1e-30)
            oq = self.q_levels(o, mo)
            so = self.sc(f"o{i}", s["gb"] * mo / 127, False)
            x = x + np.round((oq.reshape(-1, d) @ s["wb"]).reshape(B, T, d) * so / self.u)
        rms = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6 / self.u ** 2)
        inv = self.sc("fin", 1 / rms, False)
        return ((x * self.gf) @ self.head) * inv


def main():
    m = load(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin"))
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); lo = len(raw) * 9 // 10; T = 64
    b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); nw = (len(b) - 1) // T
    X = b[:nw * T].reshape(nw, T); Y = b[1:nw * T + 1].reshape(nw, T)
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[Y].mean())
    tr = np.frombuffer(raw[:lo], np.uint8).astype(np.int64); rng = np.random.default_rng(3)
    TX = tr[rng.integers(0, len(tr) - T, 24000)[:, None] + np.arange(T)]      # 1.5M training bytes for the tables

    # non-uniform shared levels from the pooled |y| / max|y| of BitLinear inputs (fit on training windows)
    pool = []

    class Pool(Bits):
        def q_levels(self, y, mx):
            pool.append((np.abs(y) / mx).ravel()[::97]); return super().q_levels(y, mx)
    pk = Pool(m, FULL)
    for i in range(0, 2048, 256): pk.forward(TX[i:i + 256])
    pool = np.concatenate(pool)

    def nonuni_levels(bits):
        L = 2 ** (bits - 1); qs = np.quantile(pool, np.linspace(0, 1, L + 1)[1:-1])   # L magnitude bands
        edges = qs; vals = np.array([pool[(pool >= lo_) & (pool < hi_)].mean() if ((pool >= lo_) & (pool < hi_)).any() else lo_
                                     for lo_, hi_ in zip(np.r_[0, edges], np.r_[edges, 1.01])])
        return edges, vals

    t0 = time.time(); tables = Tables(); col = Bits(m, FULL, collect=tables)
    for i in range(0, len(TX), 256): col.forward(TX[i:i + 256])
    print(f"tables fitted on {TX.size:,} training bytes in {time.time() - t0:.0f}s", flush=True)
    ref = None; out = {}

    def run(label, **kw):
        nonlocal ref
        sk = Bits(m, FULL, tables=tables, **kw); ce = hit = agree = 0.0; t1 = time.time(); arg = []
        for i in range(0, nw, 256):
            lg = sk.forward(X[i:i + 256]); z = lg - lg.max(-1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(-1, keepdims=True)); yb = Y[i:i + 256]
            ce += -np.take_along_axis(lp, yb[..., None], -1).sum(); am = lp.argmax(-1); hit += (am == yb).sum()
            if ref is None: arg.append(am)
            else: agree += (am == ref[i // 256]).sum()
        n = Y.size; ce /= n; hit /= n
        if ref is None: ref = arg; agree = n
        rec = dict(ce=ce, acc=hit, gain=(UNI - ce) / (UNI - 1.954176), argmax_agree=agree / n,
                   lookup_miss=(sk.miss[0] / sk.miss[1]) if sk.miss[1] else 0.0)
        out[label] = rec
        print(f"  {label:44s} CE {ce:.4f} acc {hit:.4f} gain {rec['gain']:.4f} argmax==int-skeleton {rec['argmax_agree']:.4f}"
              f"  backoff-miss {rec['lookup_miss']:.3f}  ({time.time() - t1:.0f}s)", flush=True)

    print("gain vs the float teacher (1.954176); argmax agreement vs the 6-bit-scale integer skeleton (first row)")
    run("int skeleton, 8-bit activations, m=6 scales")
    for bb in (6, 4, 3, 2, 1): run(f"activations {bb} bits (absmax)", abits=bb)
    for bb in (4, 3, 2): run(f"activations {bb} bits (shared non-uniform levels)", abits=bb, nonuni=nonuni_levels(bb))
    for bb in (6, 4, 3): run(f"q/k/v {bb} bits", qkvbits=bb)
    run("scales: layer-0 exact (byte, pos) tables", lookup="layer0")
    for n in (2, 4, 8): run(f"scales: all from tables (n-gram <= {n})", lookup="all", maxn=n)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "act_bits.json"), "w"), indent=1)


def followup():
    """context tables without the position in the key (n >= 2), higher minimum counts, and one scale family at a time"""
    import sys
    USEPOS[0] = False
    m = load(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin"))
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); lo = len(raw) * 9 // 10; T = 64
    b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); nw = (len(b) - 1) // T
    X = b[:nw * T].reshape(nw, T); Y = b[1:nw * T + 1].reshape(nw, T)
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[Y].mean())
    tr = np.frombuffer(raw[:lo], np.uint8).astype(np.int64); rng = np.random.default_rng(3)
    TX = tr[rng.integers(0, len(tr) - T, 24000)[:, None] + np.arange(T)]
    tables = Tables(); col = Bits(m, FULL, collect=tables)
    for i in range(0, len(TX), 256): col.forward(TX[i:i + 256])
    out = {}

    class Only(Bits):
        only = None
        def sc(self, name, val, layer0_exact):
            if self.only and not name.startswith(self.only) and not layer0_exact:
                return minifloat(val, self.mant)
            return super().sc(name, val, layer0_exact)

    def run(label, only=None, **kw):
        sk = Only(m, FULL, tables=tables, **kw); sk.only = only; ce = hit = 0.0
        for i in range(0, nw, 256):
            lg = sk.forward(X[i:i + 256]); z = lg - lg.max(-1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(-1, keepdims=True)); yb = Y[i:i + 256]
            ce += -np.take_along_axis(lp, yb[..., None], -1).sum(); hit += (lp.argmax(-1) == yb).sum()
        n = Y.size; ce /= n; hit /= n; rec = dict(ce=ce, acc=hit, gain=(UNI - ce) / (UNI - 1.954176)); out[label] = rec
        print(f"  {label:52s} CE {ce:.4f} acc {hit:.4f} gain {rec['gain']:.4f}", flush=True)
    for mc in (2, 8, 32):
        MINC[0] = mc
        for n in (2, 4, 8): run(f"all scales, no position key, n<={n}, min count {mc}", lookup="all", maxn=n)
    MINC[0] = 8
    for fam in ("o", "in1", "qkv1", "fin"):
        run(f"only '{fam}' scales from tables (n<=4, min 8)", only=fam, lookup="all", maxn=4)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "act_bits_followup.json"), "w"), indent=1)


if __name__ == "__main__":
    import sys
    followup() if "followup" in sys.argv else main()
