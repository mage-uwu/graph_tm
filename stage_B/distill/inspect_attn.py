"""Open up the attention-only BitNet (models/bitnet/bitnet_attn.bin): exact numpy forward (checked against bitnet eval),
per-head attention statistics on the validation text, and the ternary weights.

  python3 inspect_attn.py [TEXT]      TEXT = the corpus bitnet was trained on (last 10% = validation)
"""
import os
import sys

import numpy as np

import common as K

S = os.environ.get("SCRATCH", "/tmp")


def load_attn(path):
    b = open(path, "rb").read(); assert b[:8] == b"BITNET02"
    V, T, d, L, H, hid, arch, quant, task = map(int, np.frombuffer(b, np.int32, 9, 8))
    npar = int(np.frombuffer(b, np.uint64, 1, 44)[0]); w = np.frombuffer(b, np.float32, npar, 52).astype(np.float64)
    assert arch == 0; o = 0

    def take(*shape):
        nonlocal o
        n = int(np.prod(shape)); t = w[o:o + n].reshape(shape); o += n; return t
    m = dict(V=V, T=T, d=d, L=L, H=H, tok=take(V, d), pos=take(T, d), subs=[])
    for _ in range(L):
        m["subs"].append(dict(g=take(d), qkv=take(d, 3 * d), o=take(d, d)))
    m["gf"] = take(d); m["head"] = take(d, V); assert o == npar
    return m


def forward(m, X):
    """X [B, T] bytes -> logits [B, T, V], attention [L, B, H, T, T]"""
    Bn, T = X.shape; d, H = m["d"], m["H"]; hd = d // H
    x = m["tok"][X] + m["pos"][None, :T]; atts = []
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for s in m["subs"]:
        qkv = K.bitlin(K.rms(x, s["g"]).reshape(-1, d), s["qkv"]).reshape(Bn, T, 3, H, hd)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))          # [B, H, T, hd]
        sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); atts.append(p)
        out = (p @ v).transpose(0, 2, 1, 3).reshape(Bn * T, d)
        x = x + K.bitlin(out, s["o"]).reshape(Bn, T, d)
    return K.rms(x, m["gf"]) @ m["head"], np.stack(atts)


def main(text):
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]
    raw = open(text, "rb").read(); lo = len(raw) * 9 // 10
    b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); nw = (len(b) - 1) // T
    X = b[:nw * T].reshape(nw, T); Y = b[1:nw * T + 1].reshape(nw, T)
    L, H = m["L"], m["H"]
    ce = hit = 0.0; off = np.zeros((L, H, T)); ent = np.zeros((L, H)); cnt = 0
    same_byte = np.zeros((L, H)); after_space = np.zeros((L, H)); word_start_mass = np.zeros((L, H))
    for i in range(0, nw, 512):
        xb, yb = X[i:i + 512], Y[i:i + 512]
        lg, at = forward(m, xb)
        z = lg - lg.max(-1, keepdims=True); lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        ce += -np.take_along_axis(lp, yb[..., None], -1).sum(); hit += (lg.argmax(-1) == yb).sum()
        Bn = len(xb)
        tq = np.arange(T)[:, None]; ts = np.arange(T)[None]
        rel = tq - ts                                                            # >= 0 on the causal part
        for r in range(T):
            off[:, :, r] += (at * (rel == r)).sum((1, 3, 4))
        ent += -(at * np.log(np.where(at > 0, at, 1))).sum(-1).sum((1, 3))
        same = (xb[:, :, None] == xb[:, None, :])                               # key byte == query byte
        same_byte += (at * same[None, :, None]).sum((1, 3, 4))
        prev_sp = np.concatenate([np.zeros((Bn, 1), bool), xb[:, :-1] == 32], 1)   # key position follows a space
        after_space += (at * prev_sp[None, :, None, None, :]).sum((1, 3, 4))
        cnt += Bn * T
    n = nw * T
    print(f"attention-only BitNet: {L} layers x {H} heads, d {m['d']}, seq {T}; params "
          f"{sum(a.size for a in [m['tok'], m['pos'], m['gf'], m['head']]) + sum(s['g'].size + s['qkv'].size + s['o'].size for s in m['subs']):,}")
    print(f"validation (every {T}-byte window): CE {ce / n:.6f} nats/byte (bitnet eval: 1.954175), acc {hit / n:.4f}")
    frac_sp = (X == 32).mean(); frac_same = np.mean([(xw[:, None] == xw[None]).sum() for xw in X[:200]]) / (T * (T + 1) / 2)
    print("\nwhere each head looks (share of attention mass by distance back; 'self' = the query position)")
    print("head    self    -1     -2     -3    -4..-8  -9..-63 | entropy(nats) | on same byte | on word starts")
    for l in range(L):
        for h in range(H):
            o = off[l, h] / cnt
            print(f"L{l}H{h}  {o[0]:.3f}  {o[1]:.3f}  {o[2]:.3f}  {o[3]:.3f}  {o[4:9].sum():.3f}   {o[9:].sum():.3f}  |"
                  f"     {ent[l, h] / cnt:.2f}      |    {same_byte[l, h] / cnt:.3f}     |    {after_space[l, h] / cnt:.3f}")
    print(f"(reference: bytes that are spaces {frac_sp:.3f})")
    print("\nternary weights (BitLinear: sign(round(W / mean|W|)))")
    for l, s in enumerate(m["subs"]):
        for name in ("qkv", "o"):
            wq, g = K.tern(s[name])
            print(f"L{l} {name:3s} {s[name].shape[0]}x{s[name].shape[1]}: -1 {np.mean(wq == -1):.3f}  0 {np.mean(wq == 0):.3f}  "
                  f"+1 {np.mean(wq == 1):.3f}  gamma {g:.4f}")
    return m


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{S}/wiki_11m.txt")
