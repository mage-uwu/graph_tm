"""
Masked-token evaluation for a pretrained GraphTM (or any model that yields a score per
candidate token).

Metrics on an evalset (pretrain_data.py evalset):
  per-bit      accuracy of sign(vote sum) against the teacher code bits
  acc@1/@10    rank of the true token among all ~29.5k real vocabulary tokens, scored by
  mrr          sum_b votes_b * (2*code_b - 1) (vote-weighted agreement with each code)
  frequent/rare  acc@10 split at the 100 most frequent training tokens
Baselines printed alongside: the frequency-weighted majority code (per-bit) and the unigram
ranking (acc@k).

usage: python3 eval_mlm.py --model M.gtmm --evalset P [--threads N] [--tag name]
"""
import argparse
import json
import os

import numpy as np

import common as C

TOP_FREQ = 100


def train_freq(vocab_size):
    p = os.path.join(C.DATA, "train_freq.npy")
    if not os.path.exists(p):
        f = np.bincount(np.load(os.path.join(C.DATA, "wt103_train.npz"))["flat"], minlength=vocab_size)
        np.save(p, f)
    return np.load(p)


def ranks_from_scores(score_fn, true, cand, n, chunk=2048):
    """rank (0 = best) of each true token among cand; ties count half"""
    pos = np.full(int(cand.max()) + 1, -1)
    pos[cand] = np.arange(len(cand))
    r = np.zeros(n)
    for i in range(0, n, chunk):
        s = score_fn(i, min(n, i + chunk))                # (b, len(cand))
        t = s[np.arange(len(s)), pos[true[i:i + chunk]]][:, None]
        r[i:i + chunk] = (s > t).sum(1) + 0.5 * ((s == t).sum(1) - 1)
    return r


def summarize(r, true, freq_rank, tag, extra=None):
    ok = true >= 0
    top = freq_rank[true] < TOP_FREQ
    res = {"tag": tag, "n": int(len(r)), "acc@1": float((r < 1).mean()), "acc@10": float((r < 10).mean()),
           "mrr": float((1 / (1 + r)).mean()), "acc@10_frequent": float((r[top] < 10).mean()),
           "acc@10_rare": float((r[~top] < 10).mean()), "frac_frequent": float(top.mean())}
    if extra:
        res.update(extra)
    assert ok.all()
    return res


def show(res):
    s = " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items() if k != "tag")
    print(f"[{res['tag']}] {s}", flush=True)
    print("RESULT " + json.dumps(res), flush=True)


def context():
    t = np.load(os.path.join(C.DATA, "teacher.npz"))
    codes, cand = t["codes"].astype(np.int32), t["candidates"]
    freq = train_freq(len(codes))
    freq_rank = np.empty(len(freq), dtype=np.int64)
    freq_rank[np.argsort(-freq, kind="stable")] = np.arange(len(freq))
    return codes, cand, freq, freq_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="gtm")
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    codes, cand, freq, freq_rank = context()
    ev = np.load(a.evalset + ".npz")
    true = ev["true"]
    n = len(true)
    if a.baselines:
        prior = (freq[:, None] * (2 * codes - 1)).sum(0) > 0
        r = ranks_from_scores(lambda i, j: np.tile(freq[cand].astype(np.float64), (j - i, 1)), true, cand, n)
        show(summarize(r, true, freq_rank, "unigram", {"per-bit": float((prior[None] == codes[true]).mean())}))
    if a.model:
        sums, txt = C.score_sums(a.model, a.evalset, n, codes.shape[1], a.threads, a.tag)
        fill = [l.strip() for l in txt.splitlines() if "fill" in l]
        S = (2 * codes[cand] - 1).T.astype(np.float32)
        sf = sums.astype(np.float32)
        r = ranks_from_scores(lambda i, j: sf[i:j] @ S, true, cand, n)
        per_bit = float(((sums >= 0) == codes[true]).mean())
        show(summarize(r, true, freq_rank, a.tag, {"per-bit": per_bit, "fill": fill[0] if fill else ""}))


if __name__ == "__main__":
    main()
