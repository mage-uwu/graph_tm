"""How much of the attention-only BitNet is content routing? Replace attention weights by fixed patterns:
  fixed(l)   head h of layer l uses its mean attention pattern P[h, t, s] (over the validation text) for every input:
             a fixed, content-independent convolution (DDLGN-style fixed wiring with learned tap weights)
  onehot(l)  each query attends only to its single most-used offset (a pure wire)
Scores on the validation text; the pattern means are taken on the first half, scored on the second half."""
import os
import numpy as np
import common as K
from inspect_attn import load_attn

S = os.environ.get("SCRATCH", "/tmp")


def forward(m, X, override):
    Bn, T = X.shape; d, H = m["d"], m["H"]; hd = d // H
    x = m["tok"][X] + m["pos"][None, :T]; mask = np.triu(np.full((T, T), -np.inf), 1); atts = []
    for l, s in enumerate(m["subs"]):
        qkv = K.bitlin(K.rms(x, s["g"]).reshape(-1, d), s["qkv"]).reshape(Bn, T, 3, H, hd)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); atts.append(p.mean(0))
        if l in override: p = np.broadcast_to(override[l], p.shape)
        out = (p @ v).transpose(0, 2, 1, 3).reshape(Bn * T, d)
        x = x + K.bitlin(out, s["o"]).reshape(Bn, T, d)
    return K.rms(x, m["gf"]) @ m["head"], atts


def score(m, X, Y, override):
    ce = hit = 0.0
    for i in range(0, len(X), 512):
        lg, _ = forward(m, X[i:i + 512], override)
        z = lg - lg.max(-1, keepdims=True); lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        ce += -np.take_along_axis(lp, Y[i:i + 512, :, None], -1).sum(); hit += (lg.argmax(-1) == Y[i:i + 512]).sum()
    return ce / Y.size, hit / Y.size


m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
raw = open(f"{S}/wiki_11m.txt", "rb").read(); lo = len(raw) * 9 // 10
b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); nw = (len(b) - 1) // T
X = b[:nw * T].reshape(nw, T); Y = b[1:nw * T + 1].reshape(nw, T); half = nw // 2
# mean patterns on the first half, with layer 0 fixed when measuring layer 1's mean under fixed layer 0 is not needed:
# use the true model's mean patterns
P = [np.zeros((m["H"], T, T)) for _ in m["subs"]]; n = 0
for i in range(0, half, 512):
    _, at = forward(m, X[i:min(i + 512, half)], {})
    w = min(i + 512, half) - i
    for l in range(len(P)): P[l] += at[l] * w
    n += w
P = [p / n for p in P]
onehot = []
for p in P:
    o = np.zeros_like(p); idx = p.argmax(-1)
    np.put_along_axis(o, idx[..., None], 1.0, -1); onehot.append(o)
Xs, Ys = X[half:], Y[half:]
uni = 3.1696
ce0, a0 = score(m, Xs, Ys, {})
print(f"true attention           CE {ce0:.4f} acc {a0:.4f}")
for name, ov in [("fixed pattern, layer 0", {0: P[0]}), ("fixed pattern, layer 1", {1: P[1]}),
                 ("fixed pattern, both", {0: P[0], 1: P[1]}), ("single wire, both", {0: onehot[0], 1: onehot[1]}),
                 ("no attention (self only)", {0: np.broadcast_to(np.eye(T), P[0].shape), 1: np.broadcast_to(np.eye(T), P[1].shape)})]:
    ce, a = score(m, Xs, Ys, ov)
    print(f"{name:25s} CE {ce:.4f} acc {a:.4f}  gain over unigram kept {(uni - ce) / (uni - ce0):.3f}")
