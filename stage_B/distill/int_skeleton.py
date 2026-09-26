"""Step 1: the integer skeleton of a frozen BitNet (bitnet.c model file), softmax kept as the float reference.

Frozen source of truth: the ternary matrices (round(W / mean|W|) in {-1, 0, +1}) with their scales gamma, and
BitNet's int8 absmax activation quantizer. Everything between them is made integer / fixed point, one knob at a time,
and each rung of the ladder is scored against the float model on the full validation text.

  float     bitnet.c's forward pass re-implemented in float64 (must reproduce bitnet eval)
  nonorm    BitLinear input without the RMSNorm divide: xq = round(127 (x o g) / max|x o g|) — the RMS cancels in the
            absmax; it survives only in the output scale  gamma * max|x o g| / (127 rms(x))
  fixres    residual stream in fixed point: integers in units of 2^-F (--res-frac F); embedding tables quantized to
            E-bit integers (--emb-bits) and re-expressed in residual units
  fixgain   RMSNorm gains g as integers with G fractional bits (--gain-frac)
  minifl    every per-token scale (BitLinear outputs, 1/rms of the final norm) rounded to a minifloat with m mantissa bits
            (--mant); sqrt(mean x^2) is the only root left, and it only feeds these scales
  qkv8      q, k, v requantized per token per head to int8 by BitNet's own absmax quantizer (scores: int8 . int8)
  head8     LM head as int8 with one scale per output byte
  shift     (variant) absmax "divide by max" replaced by a power-of-two shift: xq = round(y / 2^k), max|y| / 2^k <= 127

Integers are held in float64 (exact below 2^53). Every per-token computation depends only on that token's row.

  python3 int_skeleton.py [MODEL.bin ...]      (SCRATCH = folder with wiki_11m.txt and Ct.npy)
"""
import json
import os
import sys
import time

import numpy as np

import common as K

S = os.environ.get("SCRATCH", "/tmp")


def load(path):
    b = open(path, "rb").read(); assert b[:8] == b"BITNET02"
    V, T, d, L, H, hid, arch, quant, task = map(int, np.frombuffer(b, np.int32, 9, 8))
    npar = int(np.frombuffer(b, np.uint64, 1, 44)[0]); w = np.frombuffer(b, np.float32, npar, 52).astype(np.float64)
    o = 0

    def take(*shape):
        nonlocal o
        n = int(np.prod(shape)); t = w[o:o + n].reshape(shape); o += n; return t
    m = dict(V=V, T=T, d=d, L=L, H=H, hid=hid, arch=arch, tok=take(V, d), pos=take(T, d), subs=[])
    nsub = L * (2 if arch == 2 else 1)
    for s in range(nsub):
        typ = "att" if arch == 0 or (arch == 2 and s % 2 == 0) else "mlp"
        if typ == "att":
            m["subs"].append(dict(type=typ, g=take(d), a=take(d, 3 * d), b=take(d, d)))
        else:
            m["subs"].append(dict(type=typ, g=take(d), a=take(d, hid), b=take(hid, d)))
    m["gf"] = take(d); m["head"] = take(d, V); assert o == npar
    for s in m["subs"]:
        s["wa"], s["ga"] = K.tern(s["a"]); s["wb"], s["gb"] = K.tern(s["b"])
    return m


def minifloat(x, mant):
    """round positive scales to mant mantissa bits (log-uniform relative error <= 2^-(mant+1))"""
    if mant is None: return x
    e = np.floor(np.log2(x)); f = x / 2 ** e
    return np.round(f * 2 ** mant) / 2 ** mant * 2 ** e


class Skeleton:
    def __init__(self, m, knobs, F=12, E=8, G=8, mant=4):
        self.m, self.k, self.F, self.G, self.mant = m, set(knobs), F, G, mant
        self.u = 2.0 ** -F                                                    # residual unit
        if "fixres" in self.k:
            sc = max(np.abs(m["tok"]).max(), np.abs(m["pos"]).max()) / (2 ** (E - 1) - 1)
            q = lambda t: np.round(np.round(t / sc) * sc / self.u)            # E-bit table -> residual units
            self.tok, self.pos = q(m["tok"]), q(m["pos"])
        else:
            self.tok, self.pos = m["tok"] / self.u, m["pos"] / self.u         # residual in units of u, exact
        gq = (lambda g: np.round(g * 2 ** G) / 2 ** G) if "fixgain" in self.k else (lambda g: g)
        self.g = [gq(s["g"]) for s in m["subs"]]; self.gf = gq(m["gf"])
        if "head8" in self.k:
            hs = np.abs(m["head"]).max(0) / 127; self.head = np.round(m["head"] / hs) * hs
        else:
            self.head = m["head"]

    def res(self, x):
        """residual update in units of u (rounded when the residual is fixed point)"""
        return np.round(x) if "fixres" in self.k else x

    def quant_in(self, x, g):
        """BitLinear input quantizer on residual rows x (units of u). Returns int8 values and the per-token factor c
        such that the real BitLinear output is gamma * c * (xq . wq)"""
        rms = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6 / self.u ** 2)
        if "nonorm" in self.k:
            y = x * g; mx = np.abs(y).max(-1, keepdims=True)
            if "shift" in self.k:
                kk = np.maximum(np.ceil(np.log2(np.maximum(mx, 1e-30) / 127)), -60); step = 2.0 ** kk
                xq = np.round(y / step); c = step / rms
            else:
                xq = np.round(127 * y / np.maximum(mx, 1e-30)); c = mx / 127 / rms
        else:                                                                 # bitnet.c exactly: normalize, absmax
            n = x / rms * g; mx = np.maximum(np.abs(n).max(-1, keepdims=True), 1e-5)
            xq = np.round(n * 127 / mx); c = mx / 127
        return xq, c

    def scale(self, c):
        return minifloat(c, self.mant) if "minifl" in self.k else c

    def forward(self, X):
        m = self.m; B, T = X.shape; d, H = m["d"], m["H"]; hd = d // H
        mask = np.triu(np.full((T, T), -np.inf), 1)
        x = self.tok[X] + self.pos[None, :T]
        for i, s in enumerate(m["subs"]):
            xq, c = self.quant_in(x, self.g[i])
            yi = xq.reshape(-1, d) @ s["wa"]                                  # exact integers
            cs = self.scale(s["ga"] * c.reshape(-1, 1))
            if s["type"] == "att":
                yi = yi.reshape(B, T, 3, H, hd); cs = cs.reshape(B, T, 1, 1)
                if "qkv8" in self.k:                                          # BitNet's absmax per token per head
                    mx = np.maximum(np.abs(yi).max(-1, keepdims=True), 1e-30)
                    q8 = np.round(127 * yi / mx); val = q8 * self.scale(cs[..., None] * mx / 127)
                else:
                    val = yi * cs[..., None]
                q, k, v = (val[:, :, j].transpose(0, 2, 1, 3) for j in range(3))
                e = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
                p = np.exp(e - e.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)   # float reference
                o = (p @ v).transpose(0, 2, 1, 3).reshape(B * T, d)
                mo = np.maximum(np.abs(o).max(-1, keepdims=True), 1e-5); oq = np.round(127 * o / mo)
                out = (oq @ s["wb"]) * self.scale(s["gb"] * mo / 127)
            else:
                ri = np.maximum(yi, 0) ** 2                                   # ReLU^2 of the integer, exact
                mr = np.maximum(ri.max(-1, keepdims=True), 1e-30); rq = np.round(127 * ri / mr)
                out = (rq @ s["wb"]) * self.scale(s["gb"] * cs ** 2 * mr / 127)
            x = x + self.res(out.reshape(B, T, d) / self.u)
        rms = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6 / self.u ** 2)
        inv = self.scale(1 / rms)
        return ((x * self.gf) @ self.head) * inv


def main(paths):
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); lo = len(raw) * 9 // 10
    b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); T = 64; nw = (len(b) - 1) // T
    X = b[:nw * T].reshape(nw, T); Y = b[1:nw * T + 1].reshape(nw, T)
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[Y].mean())
    ladder = [("float", []), ("nonorm", ["nonorm"]), ("+fixres (F=12, E=8)", ["nonorm", "fixres"]),
              ("+fixgain (G=8)", ["nonorm", "fixres", "fixgain"]),
              ("+minifloat scales (m=4)", ["nonorm", "fixres", "fixgain", "minifl"]),
              ("+qkv int8", ["nonorm", "fixres", "fixgain", "minifl", "qkv8"]),
              ("+head int8", ["nonorm", "fixres", "fixgain", "minifl", "qkv8", "head8"]),
              ("full integer, absmax by shift", ["nonorm", "fixres", "fixgain", "minifl", "qkv8", "head8", "shift"])]
    sweeps = [("full integer, F=8", dict(F=8)), ("full integer, F=16", dict(F=16)),
              ("full integer, m=2", dict(mant=2)), ("full integer, m=6", dict(mant=6)),
              ("full integer, E=6", dict(E=6))]
    full = ladder[6][1]
    out = {}
    for path in paths:
        m = load(path); name = os.path.basename(path); out[name] = {}
        if m["arch"] == 2 or m["arch"] == 1: pass
        print(f"\n== {name}: arch {['attn', 'mlp', 'full'][m['arch']]}, {len(m['subs'])} sublayers", flush=True)
        ref_lp = None
        runs = [(n_, k_, {}) for n_, k_ in ladder] + [(n_, full, kw) for n_, kw in sweeps]
        for label, knobs, kw in runs:
            if m["arch"] == 1 and "qkv8" in knobs and label.startswith("+qkv"): continue
            t0 = time.time(); sk = Skeleton(m, knobs, **kw); ce = hit = agree = kl = 0.0; lps = []
            for i in range(0, nw, 256):
                lg = sk.forward(X[i:i + 256]); z = lg - lg.max(-1, keepdims=True)
                lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
                yb = Y[i:i + 256]
                ce += -np.take_along_axis(lp, yb[..., None], -1).sum(); hit += (lp.argmax(-1) == yb).sum()
                if ref_lp is None: lps.append(lp.argmax(-1))
                else:
                    r = ref_lp[i // 256]; agree += (lp.argmax(-1) == r[0]).sum()
                    kl += (np.exp(r[1]) * (r[1] - lp)).sum()
                if ref_lp is None: lps[-1] = (lps[-1], lp.astype(np.float32))
            n = Y.size; ce /= n; hit /= n
            if ref_lp is None:
                ref_lp = lps; ce_f = ce; agree_r, kl_r = 1.0, 0.0
            else:
                agree_r, kl_r = agree / n, kl / n
            rec = dict(ce=ce, acc=hit, gain=(UNI - ce) / (UNI - ce_f), top1_agree=agree_r, kl=kl_r)
            out[name][label] = rec
            print(f"  {label:34s} CE {ce:.6f} acc {hit:.4f} gain {rec['gain']:.4f} argmax==float {agree_r:.4f} "
                  f"KL(float||.) {kl_r:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "int_skeleton.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(K.ROOT, "models/bitnet", f) for f in
                          ("bitnet_attn.bin", "bitnet_mlp.bin", "bitnet_full.bin")])
