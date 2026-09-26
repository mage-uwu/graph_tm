"""Shared pieces for distilling the MLP-only BitNet into logic: load a BITNET02 model, the exact teacher forward
(numpy float64 re-implementation of bitnet.c), and the (byte, position) -> next-byte count tables of the train and
validation text (the MLP-only model's output depends only on the current byte and its position)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(path):
    b = open(path, "rb").read()
    assert b[:8] == b"BITNET02"
    V, T, d, L, H, hid, arch, quant, task = np.frombuffer(b, np.int32, 9, 8)
    npar = int(np.frombuffer(b, np.uint64, 1, 44)[0])
    w = np.frombuffer(b, np.float32, npar, 52).astype(np.float64)
    cfg = dict(V=V, T=T, d=d, L=L, H=H, hid=hid, arch=arch, quant=quant, task=task)
    assert arch == 1, "MLP-only model expected"
    o = 0

    def take(*shape):
        nonlocal o
        n = int(np.prod(shape)); t = w[o:o + n].reshape(shape); o += n; return t
    m = {"tok": take(V, d), "pos": take(T, d), "subs": []}
    for _ in range(L):
        m["subs"].append({"g": take(d), "up": take(d, hid), "down": take(hid, d)})
    m["gf"] = take(d); m["head"] = take(d, V)
    assert o == npar
    return cfg, m


def rms(x, g):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * g


def tern(W):
    gam = np.abs(W).mean()
    return np.clip(np.round(W / (gam + 1e-8)), -1, 1), gam


def actq(x):
    s = 127 / np.maximum(np.abs(x).max(-1, keepdims=True), 1e-5)
    return np.round(x * s), 1 / s


def bitlin(x, W):
    q, is_ = actq(x); wq, gam = tern(W)
    return (q @ wq) * gam * is_


def teacher(m, ids, pos, keep=False):
    """logits [n, 256] for inputs (byte ids, positions); keep=True also returns intermediate activations"""
    x = m["tok"][ids] + m["pos"][pos]; acts = {"x0": x}
    for i, s in enumerate(m["subs"]):
        h = bitlin(rms(x, s["g"]), s["up"]); r = np.maximum(h, 0) ** 2
        x = x + bitlin(r, s["down"]); acts[f"h{i}"] = h; acts[f"x{i + 1}"] = x
    logits = rms(x, m["gf"]) @ m["head"]
    return (logits, acts) if keep else logits


def domain(T=64):
    """all (byte, position) inputs, index = byte * T + position"""
    ids, pos = np.divmod(np.arange(256 * T), T)
    return ids, pos


def counts(path, lo, hi, T=64):
    """C[byte * T + pos, next] over every non-overlapping T-window of bytes [lo, hi) (bitnet eval's windows)"""
    b = np.frombuffer(open(path, "rb").read()[lo:hi + 1], np.uint8).astype(np.int64)
    nw = (len(b) - 1) // T
    x = b[:nw * T].reshape(nw, T); y = b[1:nw * T + 1].reshape(nw, T)
    key = (x * T + np.arange(T)) * 256 + y
    return np.bincount(key.ravel(), minlength=256 * T * 256).reshape(256 * T, 256).astype(np.float64)


def score(logits, C):
    """cross-entropy (nats/byte) and next-byte accuracy of per-input logits under count table C"""
    z = logits - logits.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    n = C.sum(); ce = -(C * lp).sum() / n
    acc = C[np.arange(len(C)), logits.argmax(1)].sum() / n
    return ce, acc
