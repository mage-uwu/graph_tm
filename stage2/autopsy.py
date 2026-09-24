"""
Sender autopsy: trace what the context (sender) clauses do over training.

For a config, trains from init over a list of shards and at each checkpoint reports, from an
oracle pass over validation graphs and the automaton states:
  - context clauses: layer-0 positive/negative includes, fraction with NO positive layer-0
    literal (cannot send under --senders pos), fraction with an empty layer 0, includes at
    deeper layers
  - how often context clauses are true at layer 0 anywhere vs true at their final layer
    (i.e. they vote), senders per context node, centre message fill per layer
  - the mean layer-0 literal count of context clauses that DID vs did NOT reach their final
    layer on the validation graphs (erosion by Type Ib when deeper layers block the clause)

usage: python3 autopsy.py DATA_DIR "gtm config" [--val N]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

import common as C
from gtm_oracle import OracleGTM
from gtmcore import Dataset


def stats(model, ds, n_graphs):
    m = OracleGTM.load(model)
    c = m.cfg
    ctx_cl = np.arange(c.C) % 2 == 0
    inc0 = m.ta[0] >= m.half
    pos0 = inc0[:, :c.H].sum(1)
    tot0 = inc0.sum(1)
    send = pos0 > 0 if c.senders else np.ones(c.C, bool)
    r = {"ctx_L0_pos": pos0[ctx_cl].mean(), "ctx_L0_neg": (tot0 - pos0)[ctx_cl].mean(),
         "ctx_nonsender_frac": (~send[ctx_cl]).mean(), "ctx_L0_empty_frac": (tot0[ctx_cl] == 0).mean()}
    for d in range(1, c.D):
        inc = m.ta[d] >= m.half
        r[f"ctx_L{d}_pos"] = inc[ctx_cl, :c.MS].sum(1).mean()
        r[f"ctx_L{d}_neg"] = inc[ctx_cl, c.MS:].sum(1).mean()
    true0 = np.zeros(c.C)
    truef = np.zeros(c.C)
    acc = {}
    for g in range(n_graphs):
        n0, n1 = ds.node_offset[g], ds.node_offset[g + 1]
        types = ds.node_type[n0:n1]
        cen, ctx = np.nonzero(types == 1)[0][0], types == 0
        typeok = types[None, :] == (np.arange(c.C) % 2)[:, None]
        out = typeok & ~((inc0.astype(np.int32) @ (~ds.X[n0:n1]).astype(np.int32).T) > 0)
        true0 += out.any(1)
        acc.setdefault("senders_per_ctx_node", []).append((out[:, ctx] & send[:, None]).sum(0).mean())
        for d in range(1, c.D):
            Xm = m._messages(ds, n0, n1 - n0, out)
            acc.setdefault(f"fill{d}_centre", []).append(Xm[cen, :c.MS].mean())
            out = out & typeok & ~(((m.ta[d] >= m.half).astype(np.int32) @ (~Xm).astype(np.int32).T) > 0)
        truef += out.any(1)
    true0 /= n_graphs
    truef /= n_graphs
    r["ctx_true_L0_rate"] = true0[ctx_cl].mean()
    r["ctx_true_final_rate"] = truef[ctx_cl].mean()
    voting = ctx_cl & (truef > 0)
    silent = ctx_cl & (truef == 0)
    r["ctx_silent_frac"] = silent.sum() / ctx_cl.sum()
    r["L0_lits_voting"] = tot0[voting].mean() if voting.any() else float("nan")
    r["L0_lits_silent"] = tot0[silent].mean() if silent.any() else float("nan")
    for k, v in acc.items():
        r[k] = float(np.mean(v))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("cfg")
    ap.add_argument("--shards", default="c1,c2,c3,c4,c5,c6")
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--init", help="start from this model instead of gtm init")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    ds = Dataset.load(os.path.join(a.data, "val.gtmd"))
    model = os.path.join(a.data, f"autopsy_{a.tag}.gtmm")
    shards = [os.path.join(a.data, s + ".gtmd") for s in a.shards.split(",")]
    if a.init:
        subprocess.run(["cp", a.init, model], check=True)
    else:
        subprocess.run([C.GTM, "init", "--data", shards[0], "--out", model] + a.cfg.split(), check=True, capture_output=True)
    keys = None
    done = 0
    for i, sh in enumerate([None] + shards):
        if sh is not None:
            n = int(C.gtm("train", "--data", sh, "--model", model, "--epochs", 1, "--threads", a.threads,
                          "--save", model).split("train: ")[1].split(" graphs")[0])
            done += n
        r = stats(model, ds, a.val)
        if keys is None:
            keys = list(r)
            print("windows " + " ".join(keys), flush=True)
        print(f"{done:7d} " + " ".join(f"{r[k]:.3f}" for k in keys), flush=True)


if __name__ == "__main__":
    main()
