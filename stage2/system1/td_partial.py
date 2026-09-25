"""Score the stopped typed-decisions lae_gm arm (laya_td.py) on the questions it finished, next to majority / bag of
words / bert-tiny trained on the same questions. Accuracy = argmax vs the gold label (Laya's metric; temperature does
not change the argmax). One JSON line per question, then a pooled line. Reads the models laya_td.py left in <JEV_OUT>/td."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import laya_td as L  # noqa: E402

J = L.J


def lae_gm(e, key):
    """test logits from the saved lae_gm models, or None if the question was not finished"""
    ks = [None] if e["type"] == "noul" else list(range(len(e["options"])))
    tag = f"{key[0]}__{key[1]}__gm"
    lths = [os.path.join(L.TDIR, f"{tag}_{'bin' if k is None else k}.lth") for k in ks]
    test = os.path.join(L.TDIR, f"{key[0]}__{key[1]}_test.ids")
    if not all(os.path.exists(p) for p in lths) or not os.path.exists(test):
        return None
    z = [J.lae_votes(p, test, L.SEQ) for p in lths]
    return J.to_logits(z[0], 2, True) if ks == [None] else np.stack(z, 1)


def main():
    Q = L.load()
    arms = {"majority": L.arm_majority, "bow": L.arm_bow, "berttiny": L.arm_berttiny}
    tot = {a: [] for a in ["lae_gm", *arms]}
    for key, e in sorted(Q.items()):
        zl = lae_gm(e, key)
        if zl is None:
            continue
        y = e["test"]["y"]
        row = {"question": f"{key[0]}/{key[1]}", "type": e["type"], "K": len(e["options"]), "n": int(len(y))}
        for a, z in [("lae_gm", zl)] + [(a, f(e)[1]) for a, f in arms.items()]:
            c = (np.asarray(z).argmax(1) == y)
            tot[a].append(c)
            row[a] = round(float(c.mean()), 4)
        print(json.dumps(row), flush=True)
    print(json.dumps({"pooled": {a: round(float(np.concatenate(v).mean()), 4) for a, v in tot.items() if v},
                      "questions": len(tot["lae_gm"]), "decisions": int(sum(len(c) for c in tot["lae_gm"]))}), flush=True)


if __name__ == "__main__":
    main()
