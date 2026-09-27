"""Does BLT entropy patching supply attention's content routing? Read-only on the frozen attention-only BitNet.

  entropy     count-based order-4 byte model with backoff (orders 4..1, used when the context was seen >= 3 times),
              fitted on training text; H_i = entropy of p(x_i | x_{i-4..i-1}); patch boundary at i if H_i > theta
              (theta from quantiles: average patch length 3 / 4.5 / 6 bytes); position 0 of a window always starts a patch
  coords      per position: patch index p(t), position inside the patch r(t)
  stats       per head: attention mass on patch-start keys, on the current patch, the previous patch, older patches
  replace     each head's attention -> its mean pattern A[r(t), p(t) - p(s), r(s)] (r capped at 7, patch offset at 15),
              renormalized over the causal keys; compared with the byte-offset mean pattern A[t, s] (fixed_attn.py) and
              with the same patch pattern on random boundaries (same rate). Fit: first half of validation; score: second.

  python3 entropy_attn.py
"""
import json
import os
import time

import numpy as np

import common as K
from inspect_attn import load_attn

S = os.environ.get("SCRATCH", "/tmp")
P = np.uint64(1000003)
RC, PC = 8, 16                                                               # caps: position in patch, patch offset


def ctx_hash(b, i_idx, n):
    h = np.zeros(len(i_idx), np.uint64)
    for j in range(n, 0, -1):
        h = h * P + b[i_idx - j].astype(np.uint64) + np.uint64(1)
    return h * np.uint64(8) + np.uint64(n)


class Entropy:
    def __init__(self, tr, orders=(4, 3, 2, 1)):
        self.orders = orders; self.tab = {}
        idx = np.arange(max(orders), len(tr)); nxt = tr[idx].astype(np.uint64)
        for n in orders:
            h = ctx_hash(tr, idx, n); key = h * np.uint64(256) + nxt
            u, c = np.unique(key, return_counts=True); ctx = u // np.uint64(256)
            cu, start = np.unique(ctx, return_index=True)
            tot = np.add.reduceat(c, start).astype(np.float64)
            cf = c.astype(np.float64)                                     # entropy of each context's next-byte counts
            ent = np.log2(tot) - np.add.reduceat(cf * np.log2(cf), start) / tot
            self.tab[n] = (cu, tot, ent)

    def __call__(self, X):
        """entropy (bits) of predicting each byte of windows X [B, T] from up to 4 preceding bytes in the window"""
        B, T = X.shape; out = np.full((B, T), 8.0); done = np.zeros((B, T), bool)
        flat = X.ravel()
        for n in self.orders:
            cu, tot, ent = self.tab[n]
            pos = np.arange(B * T); t = pos % T; ok = t >= n
            h = ctx_hash(flat, np.where(ok, pos, n), n)
            k = np.clip(np.searchsorted(cu, h), 0, len(cu) - 1)
            hit = ok & (cu[k] == h) & (tot[k] >= 3) & ~done.ravel()
            out.ravel()[hit] = ent[k][hit]; done.ravel()[hit] = True
        return out


def coords(bound):
    """bound [B, T] bool (bound[:, 0] forced) -> patch index p, position in patch r"""
    b = bound.copy(); b[:, 0] = True
    p = np.cumsum(b, 1) - 1
    idx = np.arange(b.shape[1])[None].repeat(len(b), 0)
    start = np.maximum.accumulate(np.where(b, idx, 0), 1)
    return p, idx - start


def forward(m, X, override=None):
    """frozen float forward; override(l, B) -> attention [B, H, T, T] or None; returns logits, attention per layer"""
    Bn, T = X.shape; d, H = m["d"], m["H"]; hd = d // H
    x = m["tok"][X] + m["pos"][None, :T]; atts = []
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for l, s in enumerate(m["subs"]):
        qkv = K.bitlin(K.rms(x, s["g"]).reshape(-1, d), s["qkv"]).reshape(Bn, T, 3, H, hd)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); atts.append(p)
        if override is not None:
            ov = override(l)
            if ov is not None: p = ov
        out = (p @ v).transpose(0, 2, 1, 3).reshape(Bn * T, d)
        x = x + K.bitlin(out, s["o"]).reshape(Bn, T, d)
    return K.rms(x, m["gf"]) @ m["head"], atts


def main():
    t0 = time.time()
    m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin")); T = m["T"]; H = m["H"]; L = len(m["subs"])
    raw = open(f"{S}/wiki_11m.txt", "rb").read(); lo = len(raw) * 9 // 10
    tr = np.frombuffer(raw[:lo], np.uint8)[:6_000_000].astype(np.int64)
    b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); nw = (len(b) - 1) // T
    X = b[:nw * T].reshape(nw, T); Y = b[1:nw * T + 1].reshape(nw, T); half = nw // 2
    XF, XS, YS = X[:2048], X[half:], Y[half:]
    Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; UNI = float(-np.log(uni / uni.sum())[YS].mean())
    ent = Entropy(tr); print(f"entropy model on {len(tr):,} training bytes ({time.time() - t0:.0f}s)", flush=True)
    HF, HS = ent(XF), ent(XS)
    rng = np.random.default_rng(0); R = {}

    def score(override):
        ce = hit = 0.0
        for i in range(0, len(XS), 128):
            lg, _ = forward(m, XS[i:i + 128], override(i) if override else None)
            z = lg - lg.max(-1, keepdims=True); lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
            ce += -np.take_along_axis(lp, YS[i:i + 128, :, None], -1).sum(); hit += (lp.argmax(-1) == YS[i:i + 128]).sum()
        return ce / YS.size, hit / YS.size
    ce_t, acc_t = score(None); gain = lambda ce: (UNI - ce) / (UNI - ce_t)
    print(f"teacher (scoring half): CE {ce_t:.4f} acc {acc_t:.4f}; unigram {UNI:.4f}", flush=True)

    # teacher attention on the fit half
    AT = [[] for _ in range(L)]
    for i in range(0, len(XF), 128):
        _, at = forward(m, XF[i:i + 128])
        for l in range(L): AT[l].append(at[l])
    AT = [np.concatenate(a) for a in AT]                                      # [N, H, T, T]
    tq, ts = np.tril_indices(T)

    def pattern_table(p, r, A):
        """mean attention over (r(t), p(t) - p(s), r(s)) per head -> table [H, RC, PC, RC]"""
        dp = np.clip(p[:, :, None] - p[:, None, :], 0, PC - 1)          # [N, T, T]
        rt = np.minimum(r, RC - 1)[:, :, None].repeat(T, 2); rs = np.minimum(r, RC - 1)[:, None, :].repeat(T, 1)
        key = (rt * PC + dp) * RC + rs; caus = np.tril(np.ones((T, T), bool))[None]
        tab = np.zeros((H, RC * PC * RC)); cnt = np.zeros(RC * PC * RC)
        k = key[:, caus[0]]
        cnt += np.bincount(k.ravel(), minlength=len(cnt))
        for h in range(H):
            tab[h] = np.bincount(k.ravel(), weights=A[:, h][:, caus[0]].ravel(), minlength=len(cnt))
        return (tab / np.maximum(cnt, 1)).reshape(H, RC, PC, RC)

    def apply_table(tab, p, r):
        dp = np.clip(p[:, :, None] - p[:, None, :], 0, PC - 1)
        rt = np.minimum(r, RC - 1)[:, :, None].repeat(T, 2); rs = np.minimum(r, RC - 1)[:, None, :].repeat(T, 1)
        W = tab[:, rt, dp, rs].transpose(1, 0, 2, 3) * np.tril(np.ones((T, T)))[None, None]
        W = W + 1e-9 * np.tril(np.ones((T, T)))[None, None]
        return W / W.sum(-1, keepdims=True)

    byte_tab = [a.mean(0) for a in AT]                                        # [H, T, T] mean pattern (fixed_attn.py)
    for avg in (3.0, 4.5, 6.0):
        th = np.quantile(HF, 1 - 1 / avg)
        bf, bs = HF > th, HS > th
        pf, rf = coords(bf); ps, rs_ = coords(bs)
        # random-boundary control with the same rate
        rbf = rng.random(bf.shape) < bf.mean(); rbs = rng.random(bs.shape) < bs.mean()
        pfr, rfr = coords(rbf); psr, rsr = coords(rbs)
        real_len = T / (pf[:, -1] + 1).mean()
        print(f"\n== average patch {avg} bytes: theta {th:.2f} bits, measured {real_len:.2f} bytes/patch; "
              f"boundaries at a space {((XF[bf] == 32).mean() if bf.any() else 0):.2f}, after a space "
              f"{(np.concatenate([np.zeros((len(XF), 1), bool), XF[:, :-1] == 32], 1)[bf]).mean():.2f}", flush=True)
        rec = {"theta": float(th), "bytes_per_patch": float(real_len)}
        # stats: where attention goes in patch coordinates
        for l in range(L):
            for h in range(H):
                A = AT[l][:, h]; caus = np.tril(np.ones((T, T), bool))
                same = (pf[:, :, None] == pf[:, None, :]) & caus; prev = (pf[:, :, None] - pf[:, None, :] == 1) & caus
                start = np.broadcast_to(bf[:, None, :] | (np.arange(T) == 0)[None, None, :], A.shape) & caus
                sr = np.broadcast_to(rbf[:, None, :] | (np.arange(T) == 0)[None, None, :], A.shape) & caus
                n = A.shape[0] * A.shape[1]
                rec[f"L{l}H{h}"] = dict(current=float(A[same].sum() / n), previous=float(A[prev].sum() / n),
                                        starts=float(A[start].sum() / n), starts_random=float(A[sr].sum() / n))
                print(f"  L{l}H{h}: mass on current patch {rec[f'L{l}H{h}']['current']:.3f}, previous "
                      f"{rec[f'L{l}H{h}']['previous']:.3f}; on patch starts {rec[f'L{l}H{h}']['starts']:.3f} "
                      f"(random boundaries {rec[f'L{l}H{h}']['starts_random']:.3f})", flush=True)
        tabs = [pattern_table(pf, rf, AT[l]) for l in range(L)]
        tabs_r = [pattern_table(pfr, rfr, AT[l]) for l in range(L)]
        for name, layers in (("layer 0", (0,)), ("layer 1", (1,)), ("both", (0, 1))):
            for kind, tb, pp, rr in (("entropy patches", tabs, ps, rs_), ("random boundaries", tabs_r, psr, rsr)):
                def ov(i, tb=tb, pp=pp, rr=rr, layers=layers):
                    return lambda l: apply_table(tb[l], pp[i:i + 128], rr[i:i + 128]) if l in layers else None
                ce, acc = score(ov); rec[f"{name} {kind}"] = dict(ce=ce, acc=acc, gain=gain(ce))
                print(f"  fixed pattern in patch coords, {name:7s}, {kind:17s}: gain kept {gain(ce):.3f} (acc {acc:.4f})", flush=True)
        R[str(avg)] = rec
    for name, layers in (("layer 0", (0,)), ("layer 1", (1,)), ("both", (0, 1))):
        def ov(i, layers=layers):
            return lambda l: np.broadcast_to(byte_tab[l], (min(128, len(XS) - i), H, T, T)) if l in layers else None
        ce, acc = score(ov); R[f"byte offsets {name}"] = dict(ce=ce, acc=acc, gain=gain(ce))
        print(f"  fixed pattern in BYTE coords (reference), {name}: gain kept {gain(ce):.3f}", flush=True)
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "entropy_attn.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
