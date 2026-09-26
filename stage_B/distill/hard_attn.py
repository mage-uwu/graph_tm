"""Is attention's content routing logic-friendly? Modify the trained attention-only BitNet's attention, no retraining:
  window W   softmax over the last W positions only (locality)
  top-k      keep each query's k highest scores (softmax over them), k = 1 is hard argmax attention: a max comparator
             and a multiplexer on the value bits"""
import os
import numpy as np
import common as K
from inspect_attn import load_attn

S = os.environ.get("SCRATCH", "/tmp")


def forward(m, X, mode, layers):
    Bn, T = X.shape; d, H = m["d"], m["H"]; hd = d // H
    x = m["tok"][X] + m["pos"][None, :T]; mask = np.triu(np.full((T, T), -np.inf), 1)
    for l, s in enumerate(m["subs"]):
        qkv = K.bitlin(K.rms(x, s["g"]).reshape(-1, d), s["qkv"]).reshape(Bn, T, 3, H, hd)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
        if l in layers:
            kind, n = mode
            if kind == "window":
                sc = sc + np.tril(np.full((T, T), -np.inf), -n)
            else:
                kth = -np.sort(-sc, -1)[..., n - 1:n]
                sc = np.where(sc >= kth, sc, -np.inf)
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        out = (p @ v).transpose(0, 2, 1, 3).reshape(Bn * T, d)
        x = x + K.bitlin(out, s["o"]).reshape(Bn, T, d)
    return K.rms(x, m["gf"]) @ m["head"]


m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
raw = open(f"{S}/wiki_11m.txt", "rb").read(); lo = len(raw) * 9 // 10
b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); nw = (len(b) - 1) // T
X = b[:nw * T].reshape(nw, T); Y = b[1:nw * T + 1].reshape(nw, T)
X, Y = X[nw // 2:], Y[nw // 2:]          # same half as fixed_attn.py
uni, ce0 = 3.1696, None
for mode, layers in [(("window", 64), ()), (("window", 4), (0, 1)), (("window", 8), (0, 1)), (("window", 16), (0, 1)),
                     (("top", 1), (0,)), (("top", 1), (1,)), (("top", 1), (0, 1)), (("top", 2), (0, 1)), (("top", 4), (0, 1))]:
    ce = hit = 0.0
    for i in range(0, len(X), 512):
        lg = forward(m, X[i:i + 512], mode, layers)
        z = lg - lg.max(-1, keepdims=True); lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        ce += -np.take_along_axis(lp, Y[i:i + 512, :, None], -1).sum(); hit += (lg.argmax(-1) == Y[i:i + 512]).sum()
    ce /= Y.size; hit /= Y.size
    if ce0 is None: ce0 = ce
    print(f"{mode[0]} {mode[1]:2d} layers {str(layers):7s} CE {ce:.4f} acc {hit:.4f} gain kept {(uni - ce) / (uni - ce0):.3f}")
