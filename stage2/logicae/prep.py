"""
Data for the logic-bert (DDLGN) comparison run, in logic-bert's `ids` format ("label id id ..."),
on exactly our task: bert-base-uncased WordPiece ids (vocab 30,522 -- logic-bert's reserved ids
0 PAD / 1 MASK / 2 UNK coincide with [PAD] / [unused0] / [unused1], which never occur in text),
WikiText-103, +-8-token windows (17 positions, PAD outside the paragraph).

  pre_train.ids   N random training windows (logic-bert masks 15% of positions itself)
  pre_val.ids     5,000 validation windows (for val_mlm_ce)
  test_win.ids    our 20k test windows (the GraphTM's), centre replaced by MASK; test_true.txt
                  holds the true centre ids, candidates.txt the 29,522 ranked candidates
  {sst2,qnli}_{train,dev,test}.ids   our adapter splits (dev = 2,000 held out of train, test =
                  GLUE validation); qnli = question [SEP] sentence

usage: python3 prep.py OUT_DIR [--n 3000000]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import tasks as TK  # noqa: E402

PAD, MASK, SEP, WIN = 0, 1, 102, 8


def windows(flat, lens, n, seed):
    rng = np.random.default_rng(seed)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    g = rng.integers(0, len(flat), size=n)
    para = np.searchsorted(np.cumsum(lens), g, side="right")
    lo, hi = starts[para], starts[para] + lens[para]
    p = g[:, None] + np.arange(-WIN, WIN + 1)[None, :]
    ok = (p >= lo[:, None]) & (p < hi[:, None])
    return np.where(ok, flat[np.clip(p, 0, len(flat) - 1)], PAD).astype(np.int64)


def write_ids(path, labels, rows):
    with open(path + ".tmp", "w") as f:
        for y, r in zip(labels, rows):
            f.write(f"{int(y)} " + " ".join(map(str, r)) + "\n")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(os.path.join(C.DATA, "wt103_train.npz"))
    W = windows(z["flat"], z["lens"], a.n, 1)
    assert not ((W == 1) | (W == 2)).any(), "reserved ids in text"
    np.savetxt(os.path.join(a.out, "pre_train.ids"), np.hstack([-np.ones((len(W), 1), np.int64), W]), fmt="%d")
    zv = np.load(os.path.join(C.DATA, "wt103_validation.npz"))
    Wv = windows(zv["flat"], zv["lens"], 5000, 2)
    np.savetxt(os.path.join(a.out, "pre_val.ids"), np.hstack([-np.ones((len(Wv), 1), np.int64), Wv]), fmt="%d")
    ev = np.load(os.path.join(C.DATA, "test20k.gtmd.npz"))  # the GraphTM's test windows
    Wt = np.where(ev["windows"] >= 0, ev["windows"], PAD).astype(np.int64)
    Wt[:, WIN] = MASK
    np.savetxt(os.path.join(a.out, "test_win.ids"), np.hstack([-np.ones((len(Wt), 1), np.int64), Wt]), fmt="%d")
    np.savetxt(os.path.join(a.out, "test_true.txt"), ev["true"], fmt="%d")
    np.savetxt(os.path.join(a.out, "candidates.txt"), np.load(os.path.join(C.DATA, "teacher.npz"))["candidates"], fmt="%d")
    for task in ("sst2", "qnli"):
        D = TK.load(task)
        for split, d in D.items():
            rows = [list(x) for x in d["a"]] if task == "sst2" else \
                [list(q) + [SEP] + list(s) for q, s in zip(d["a"], d["b"])]
            write_ids(os.path.join(a.out, f"{task}_{split}.ids"), d["y"], rows)
    print(f"prep: {len(W)} train windows, {len(Wv)} val, {len(Wt)} test -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
