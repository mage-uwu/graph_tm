"""Read-only measurements of the frozen attention-only BitNet (models/bitnet/bitnet_attn.bin), before any training.

  M1  per head: attention entropy, p_max, top1-top2 score gap (softmax units), share of queries in the retrieval
      regime (p_max >= .9) and the mixing regime (H >= 1 nat), and how query sharpness tracks the query norm |q_t|
      (the per-token BitLinear scale makes |q_t| a per-query inverse temperature)
  M2  score = sum of source terms q_a . k_b, sources a, b in {tok, pos} (layer 0) or {tok, pos, ctx} (layer 1, ctx =
      the layer-0 attention output in the residual): within-query variance of each term, and gain kept when term
      groups are removed from one layer's scores
  M3  per head, the whitened bilinear form Sigma^1/2 (Wq_h Wk_h^T / sqrt(hd)) Sigma^1/2: singular energy by rank, and
      Hamming attention on r spectral sign bits (score = sum_i sigma_i c_i sgn(alpha_i) sgn(beta_i)) vs rank-r float
  M4  rate-distortion: shells (positions with e_s >= e_max - Delta) and fixed top-k, weights softmax-in-set or
      uniform-in-set, both layers: gain kept vs mean set size and routing bits log2 C(t+1, |S|)
  M5  value bits: v (per head dim) or the mixed attention output (the W_o input) thermometer-coded with b thresholds

Fitting (covariances, quantiles, bit scales) uses the first half of the validation windows; every score is on the
second half. gain kept = (CE_unigram - CE) / (CE_unigram - CE_teacher).

  python3 measure_attn.py            (SCRATCH = folder with wiki_11m.txt and Ct.npy)
"""
import json
import os
import time
from math import lgamma, log

import numpy as np

import common as K
from inspect_attn import load_attn

S = os.environ.get("SCRATCH", "/tmp")
m = load_attn(os.path.join(K.ROOT, "models/bitnet/bitnet_attn.bin"))
T, d, H = m["T"], m["d"], m["H"]; hd = d // H; SC = 1 / np.sqrt(hd)
MASK = np.triu(np.full((T, T), -np.inf), 1)
raw = open(f"{S}/wiki_11m.txt", "rb").read(); lo = len(raw) * 9 // 10
b = np.frombuffer(raw[lo:], np.uint8).astype(np.int64); nw = (len(b) - 1) // T
XA = b[:nw * T].reshape(nw, T); YA = b[1:nw * T + 1].reshape(nw, T); half = nw // 2
XF, XS, YS = XA[:1024], XA[half:], YA[half:]                      # fit subset (first half), scoring set
Ct = np.load(f"{S}/Ct.npy"); uni = Ct.sum(0) + 1; LPU = np.log(uni / uni.sum())
UNI = float(-LPU[YS].mean())
LAY = []
for s in m["subs"]:
    w, g = K.tern(s["qkv"]); LAY.append(dict(w=w, g=g))
R = {}                                                              # results


def softmax(e):
    z = e - e.max(-1, keepdims=True); p = np.exp(z); return p / p.sum(-1, keepdims=True)


def forward(X, hook=None, collect=None):
    """hook(ctx) -> None or dict with any of: e (new scores, masked + softmax), p (weights), v, o (fn on W_o input)"""
    B = len(X); tok = m["tok"][X]; pos = np.broadcast_to(m["pos"][None, :T], tok.shape)
    x = tok + pos; parts = {"tok": tok, "pos": pos}
    for l, s in enumerate(m["subs"]):
        L = LAY[l]; r = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6); n = x / r * s["g"]
        xq, is_ = K.actq(n.reshape(-1, d))
        qkv = ((xq @ L["w"]) * L["g"] * is_).reshape(B, T, 3, H, hd)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        e = q @ k.transpose(0, 1, 3, 2) * SC + MASK
        ctx = dict(l=l, x=x, n=n, r=r, parts=parts, q=q, k=k, v=v, e=e, B=B, s=s)
        p = softmax(e); ofn = None
        h = hook(ctx) if hook else None
        if h:
            if "e" in h: p = softmax(h["e"] + MASK)
            if "p" in h: p = h["p"]
            if "v" in h: v = h["v"]
            ofn = h.get("o")
        if collect: collect(ctx, p)
        o = (p @ v).transpose(0, 2, 1, 3).reshape(B * T, d)
        if ofn: o = ofn(o)
        add = K.bitlin(o, s["o"]).reshape(B, T, d)
        parts = dict(parts); parts["ctx"] = add
        x = x + add
    return K.rms(x, m["gf"]) @ m["head"]


def score(hook=None, X=None, Y=None):
    X = XS if X is None else X; Y = YS if Y is None else Y; ce = hit = 0.0
    for i in range(0, len(X), 256):
        lg = forward(X[i:i + 256], hook)
        z = lg - lg.max(-1, keepdims=True); lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        ce += -np.take_along_axis(lp, Y[i:i + 256, :, None], -1).sum(); hit += (lg.argmax(-1) == Y[i:i + 256]).sum()
    return ce / Y.size, hit / Y.size


t0 = time.time()
CE_T, ACC_T = score()
gain = lambda ce: (UNI - ce) / (UNI - CE_T)
print(f"teacher on scoring half: CE {CE_T:.4f} acc {ACC_T:.4f}; unigram {UNI:.4f}; {len(XS)} windows", flush=True)
R["teacher"] = dict(ce=CE_T, acc=ACC_T, unigram=UNI, windows=int(len(XS)))

# ---------------------------------------------------------------- M1
st = {(l, h): dict(H=[], pm=[], gap=[], qn=[]) for l in range(2) for h in range(H)}
nstat = [dict(sum=np.zeros(d), sq=np.zeros((d, d)), n=0) for _ in range(2)]
vs = [[] for _ in range(2)]; os_ = [[] for _ in range(2)]


def col_m1(ctx, p):
    l = ctx["l"]; pp = p[:, :, 1:]; e = ctx["e"][:, :, 1:]                      # queries t >= 1
    Hn = -(pp * np.log(np.where(pp > 0, pp, 1))).sum(-1); es = -np.sort(-e, -1)
    qn = np.linalg.norm(ctx["q"][:, :, 1:], axis=-1)
    for h in range(H):
        a = st[(l, h)]; a["H"].append(Hn[:, h].ravel()); a["pm"].append(pp[:, h].max(-1).ravel())
        a["gap"].append((es[:, h, :, 0] - es[:, h, :, 1]).ravel()); a["qn"].append(qn[:, h].ravel())
    nn = ctx["n"].reshape(-1, d); ns = nstat[l]; ns["sum"] += nn.sum(0); ns["sq"] += nn.T @ nn; ns["n"] += len(nn)
    vs[l].append(ctx["v"].transpose(0, 2, 1, 3).reshape(-1, d)[::7])
    os_[l].append((p @ ctx["v"]).transpose(0, 2, 1, 3).reshape(-1, d)[::7])


for i in range(0, len(XF), 256):
    forward(XF[i:i + 256], collect=col_m1)
R["M1"] = {}
print("\nM1  per head (queries t>=1): entropy nats [median / p90], p_max median, retrieval share (p_max>=.9), "
      "mixing share (H>=1), gap median, corr(|q|, p_max)")
for (l, h), a in st.items():
    Hn, pm, gap, qn = (np.concatenate(a[k]) for k in ("H", "pm", "gap", "qn"))
    rq = np.corrcoef(np.argsort(np.argsort(qn)), np.argsort(np.argsort(pm)))[0, 1]
    rec = dict(H_med=float(np.median(Hn)), H_p90=float(np.quantile(Hn, .9)), pmax_med=float(np.median(pm)),
               retrieval=float((pm >= .9).mean()), mixing=float((Hn >= 1).mean()), gap_med=float(np.median(gap)),
               gap_p10=float(np.quantile(gap, .1)), rank_corr_qnorm_pmax=float(rq))
    R["M1"][f"L{l}H{h}"] = rec
    print(f"  L{l}H{h}: H {rec['H_med']:.2f}/{rec['H_p90']:.2f}  pmax {rec['pmax_med']:.2f}  retrieval {rec['retrieval']:.2f}"
          f"  mixing {rec['mixing']:.2f}  gap {rec['gap_med']:.2f} (p10 {rec['gap_p10']:.2f})  corr(|q|,pmax) {rq:+.2f}",
          flush=True)
SIG = []
for ns in nstat:
    mu = ns["sum"] / ns["n"]; SIG.append(ns["sq"] / ns["n"])                    # second moment (scores are bilinear)
VS = [np.concatenate(v) for v in vs]; OS = [np.concatenate(o) for o in os_]

# ---------------------------------------------------------------- M2
SRC = [("tok", "pos"), ("tok", "pos", "ctx")]


def term_scores(ctx):
    l, s = ctx["l"], ctx["s"]; L = LAY[l]; out = {}
    qk = {}
    for a in SRC[l]:
        na = ctx["parts"][a] / ctx["r"] * s["g"]
        y = (na.reshape(-1, d) @ L["w"][:, :2 * d]) * L["g"]
        y = y.reshape(ctx["B"], T, 2, H, hd); qk[a] = (y[:, :, 0].transpose(0, 2, 1, 3), y[:, :, 1].transpose(0, 2, 1, 3))
    for a in SRC[l]:
        for c in SRC[l]:
            out[(a, c)] = qk[a][0] @ qk[c][1].transpose(0, 1, 3, 2) * SC
    return out


var = {l: {} for l in range(2)}


def col_m2(ctx, p):
    l = ctx["l"]; tm = term_scores(ctx); caus = np.isfinite(MASK)[None, None]
    tot = np.where(caus, ctx["e"], 0); resid = tot - sum(tm.values())
    items = dict(tm); items[("total", "")] = tot; items[("act-quant residual", "")] = resid
    for k_, v_ in items.items():
        vv = np.where(caus, v_, np.nan)[:, :, 1:]
        var[l][k_] = var[l].get(k_, 0) + np.nanvar(vv, -1).sum()


for i in range(0, 512, 256):
    forward(XF[i:i + 256], collect=col_m2)
R["M2"] = {"variance_share": {}, "gain": {}}
print("\nM2  within-query variance of each score term, as a share of the total score's variance")
for l in range(2):
    tv = var[l][("total", "")]
    line = "  ".join(f"{a}.{c}:{v_ / tv:.3f}" for (a, c), v_ in var[l].items() if a != "total")
    print(f"  layer {l}: {line}", flush=True)
    R["M2"]["variance_share"][f"L{l}"] = {f"{a}.{c}": float(v_ / tv) for (a, c), v_ in var[l].items()}


def drop_hook(layer, dropped):
    def hook(ctx):
        if ctx["l"] != layer: return None
        tm = term_scores(ctx)
        e = np.where(np.isfinite(MASK)[None, None], ctx["e"], 0) - sum(tm[k_] for k_ in dropped)
        return dict(e=e)
    return hook


print("  gain kept when a layer's scores lose term groups (other layer exact)")
for l in range(2):
    src = SRC[l]; content = [a for a in src if a != "pos"]
    allk = [(a, c) for a in src for c in src]
    groups = {"drop content.content": [(a, c) for a in content for c in content],
              "drop pos.pos": [("pos", "pos")],
              "drop cross (content.pos, pos.content)": [k_ for k_ in allk if (k_[0] == "pos") != (k_[1] == "pos")],
              "keep pos.pos only": [k_ for k_ in allk if k_ != ("pos", "pos")],
              "keep content.content only": [k_ for k_ in allk if "pos" in k_]}
    for name, dropped in groups.items():
        ce, acc = score(drop_hook(l, dropped))
        R["M2"]["gain"][f"L{l} {name}"] = dict(ce=ce, acc=acc, gain=gain(ce))
        print(f"    layer {l} {name:40s} CE {ce:.4f} acc {acc:.4f} gain kept {gain(ce):.3f}", flush=True)

# ---------------------------------------------------------------- M3
def psd_sqrt(Sg, inv=False):
    w, U = np.linalg.eigh(Sg); w = np.maximum(w, w.max() * 1e-6)
    return (U * (w ** (-.5 if inv else .5))) @ U.T


SPEC = []
R["M3"] = {"energy": {}, "hamming": {}}
print("\nM3  whitened bilinear form per head: cumulative singular energy at rank 1/2/4/8/16/32")
for l in range(2):
    Sh, Si = psd_sqrt(SIG[l]), psd_sqrt(SIG[l], True); w, g = LAY[l]["w"], LAY[l]["g"]; heads = []
    for h in range(H):
        Wq = g * w[:, h * hd:(h + 1) * hd]; Wk = g * w[:, d + h * hd:d + (h + 1) * hd]
        Mt = Sh @ (Wq @ Wk.T * SC) @ Sh; U, sv, Vt = np.linalg.svd(Mt)
        en = np.cumsum(sv ** 2) / (sv ** 2).sum()
        A, Bm = Si @ U, Si @ Vt.T                                   # alpha = n A, beta = n Bm
        heads.append(dict(sv=sv, A=A, B=Bm))
        R["M3"]["energy"][f"L{l}H{h}"] = [float(en[r - 1]) for r in (1, 2, 4, 8, 16, 32)]
        print(f"  L{l}H{h}: " + " ".join(f"{en[r - 1]:.2f}" for r in (1, 2, 4, 8, 16, 32)), flush=True)
    # bit scales c_i = E|alpha_i| E|beta_i| on the fit subset (n second moments are enough: use a sample of n)
    SPEC.append(heads)
nsamp = [None, None]


def col_n(ctx, p):
    if nsamp[ctx["l"]] is None: nsamp[ctx["l"]] = ctx["n"].reshape(-1, d)[::5].copy()


forward(XF[:256], collect=col_n)
for l in range(2):
    for h in range(H):
        hs = SPEC[l][h]; hs["ca"] = np.abs(nsamp[l] @ hs["A"]).mean(0); hs["cb"] = np.abs(nsamp[l] @ hs["B"]).mean(0)

agree = {}


def spec_hook(layer, r, binary):
    def hook(ctx):
        if ctx["l"] != layer: return None
        n = ctx["n"]; e = np.zeros_like(ctx["e"]); te = ctx["e"]
        for h in range(H):
            hs = SPEC[layer][h]; al = n @ hs["A"][:, :r]; be = n @ hs["B"][:, :r]
            if binary:
                al = np.sign(al) * hs["ca"][:r]; be = np.sign(be) * hs["cb"][:r]
            e[:, h] = (al * hs["sv"][:r]) @ be.transpose(0, 2, 1)
        em = e + MASK; tp = softmax(te)
        top1 = em.argmax(-1); t1 = te.argmax(-1)
        top2 = np.argsort(-em, -1)[..., :2]
        a = agree.setdefault((layer, r, binary), [0, 0, 0, 0])
        a[0] += (top1 == t1)[:, :, 1:].sum(); a[1] += np.take_along_axis(tp, top1[..., None], -1)[:, :, 1:].sum()
        a[2] += np.take_along_axis(tp, top2, -1)[:, :, 1:].sum(); a[3] += top1[:, :, 1:].size
        return dict(e=e)
    return hook


print("  Hamming attention on r spectral sign bits (and rank-r float), one layer replaced:")
print("  layer r  kind    top1 agree  teacher mass on top1 / top2   gain kept")
for l in range(2):
    for r, binary in [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (4, 0), (8, 0), (32, 0)]:
        ce, acc = score(spec_hook(l, r, binary)); a = agree[(l, r, binary)]
        rec = dict(ce=ce, acc=acc, gain=gain(ce), top1_agree=a[0] / a[3], mass_top1=a[1] / a[3], mass_top2=a[2] / a[3])
        R["M3"]["hamming"][f"L{l} r{r} {'bits' if binary else 'float'}"] = rec
        print(f"  {l}  {r:2d} {'bits ' if binary else 'float'}   {rec['top1_agree']:.3f}        {rec['mass_top1']:.3f} / "
              f"{rec['mass_top2']:.3f}          {rec['gain']:.3f}", flush=True)

# ---------------------------------------------------------------- M4
sizes = {}
LOG2C = np.array([[(lgamma(c + 1) - lgamma(k_ + 1) - lgamma(c - k_ + 1)) / log(2) if k_ <= c else 0 for k_ in range(T + 1)]
                  for c in range(T + 1)])                                   # log2 C(c, k)


def select_hook(kind, par, weights):
    def hook(ctx):
        e = ctx["e"]; emax = e.max(-1, keepdims=True)
        if kind == "shell":
            sel = e >= emax - par
        else:
            kth = -np.sort(-e, -1)[..., min(par, T) - 1:min(par, T)]
            sel = (e >= kth) & np.isfinite(e)
        if weights == "softmax":
            p = np.where(sel, np.exp(e - emax), 0)
        else:
            p = sel.astype(np.float64)
        p = p / p.sum(-1, keepdims=True)
        n_ = sel[:, :, 1:].sum(-1); cand = np.arange(1, T)[None, None]
        bits = LOG2C[np.broadcast_to(cand + 1, n_.shape), n_]
        z = sizes.setdefault((kind, par, weights), [0.0, 0.0, 0])
        z[0] += n_.sum(); z[1] += bits.sum(); z[2] += n_.size
        return dict(p=p)
    return hook


R["M4"] = {}
print("\nM4  rate-distortion, both layers: set rule, weights, mean |S|, routing bits/query/head, gain kept")
for kind, pars in [("shell", [0, .5, 1, 2, 3, 4, 6]), ("topk", [1, 2, 3, 4, 6, 8])]:
    for par in pars:
        for wts in ("softmax", "uniform"):
            ce, acc = score(select_hook(kind, par, wts)); z = sizes[(kind, par, wts)]
            rec = dict(ce=ce, acc=acc, gain=gain(ce), mean_set=z[0] / z[2], bits=z[1] / z[2])
            R["M4"][f"{kind} {par} {wts}"] = rec
            print(f"  {kind:5s} {par:>4} {wts:8s} |S| {rec['mean_set']:.2f}  bits {rec['bits']:.2f}  gain {rec['gain']:.3f}"
                  f"  acc {acc:.4f}", flush=True)

# ---------------------------------------------------------------- M5
def thermo_fit(Z, nb):
    """per column: nb thresholds (0 for nb == 1, else quantiles) and the mean of each level on the fit sample"""
    th = np.zeros((1, Z.shape[1])) if nb == 1 else np.quantile(Z, (np.arange(nb) + 1) / (nb + 1), axis=0)
    lvl = (Z[:, None, :] > th[None]).sum(1)
    means = np.stack([np.where((lvl == i).sum(0) > 0, (Z * (lvl == i)).sum(0) / np.maximum((lvl == i).sum(0), 1), 0)
                      for i in range(nb + 1)])
    return th, means


def thermo_apply(Z, th, means):
    lvl = (Z[..., None, :] > th).sum(-2)                                         # Z is [N, d]
    return means[lvl, np.arange(Z.shape[-1])]


R["M5"] = {}
print("\nM5  value bits (thermometer with b thresholds per dim; b = 1 is the sign)")
for what in ("v", "o"):
    for layers in ((0,), (1,), (0, 1)):
        for nb in (1, 2, 3, 7):
            if what == "o" and layers != (0, 1) and nb in (2,): continue
            fits = {l: thermo_fit((VS if what == "v" else OS)[l], nb) for l in layers}

            def hook(ctx, fits=fits, what=what):
                l = ctx["l"]
                if l not in fits: return None
                th, mn = fits[l]
                if what == "v":
                    v = ctx["v"].transpose(0, 2, 1, 3).reshape(-1, d)
                    v = thermo_apply(v, th, mn).reshape(ctx["B"], T, H, hd).transpose(0, 2, 1, 3)
                    return dict(v=v)
                return dict(o=lambda o: thermo_apply(o, th, mn))
            ce, acc = score(hook)
            R["M5"][f"{what} layers {layers} b{nb}"] = dict(ce=ce, acc=acc, gain=gain(ce))
            print(f"  {what} layers {str(layers):6s} b={nb}: CE {ce:.4f} acc {acc:.4f} gain kept {gain(ce):.3f}", flush=True)

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "attn_measure.json"), "w"),
          indent=1, default=float)
print(f"\ndone in {time.time() - t0:.0f} s")
