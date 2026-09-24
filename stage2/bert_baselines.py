"""
bert-tiny (google/bert_uncased_L-2_H-128_A-2: 2 layers, hidden 128, ~4.4M params) baselines on
exactly the inputs the GraphTM sees.

  mlm   --evalset P    masked-token ranking on the same +-8-token windows and the same
                       candidate set as eval_mlm.py (bert-tiny's own MLM head)
  task  --task T --mode frozen|finetune
        frozen:   final-layer features ([CLS] + mean over tokens, or the word's first piece for
                  conll), logistic regression with L2 strength picked on dev (same head code as
                  the GraphTM linear probe)
        finetune: standard full fine-tuning, epoch picked on dev
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import common as C
import tasks as TK
import eval_mlm as EM

REPO = C.TOKENIZER_REPO
CLS, SEP, PAD, MASK = 101, 102, 0, 103
FT = {"sst2": dict(lr=3e-4, epochs=4, bs=32), "qnli": dict(lr=3e-4, epochs=3, bs=32),
      "conll": dict(lr=3e-4, epochs=5, bs=16)}


def batches(ids_list, bs):
    for i in range(0, len(ids_list), bs):
        chunk = ids_list[i:i + bs]
        L = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), L), PAD, dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.as_tensor(x)
        yield i, ids, (ids != PAD).long()


def mlm(evalset):
    from transformers import BertForMaskedLM
    m = BertForMaskedLM.from_pretrained(REPO, cache_dir=os.path.join(C.DATA, "hf")).eval()
    codes, cand, freq, freq_rank = EM.context()
    ev = np.load(evalset + ".npz")
    true, wt = ev["true"], ev["windows"]
    seqs, mpos = [], []
    for w in wt:
        v = w[w >= 0].tolist()
        c = int((w[:C.WIN] >= 0).sum())
        v[c] = MASK
        seqs.append([CLS] + v + [SEP])
        mpos.append(c + 1)
    scores = np.zeros((len(seqs), len(cand)), dtype=np.float32)
    with torch.no_grad():
        for i, ids, att in batches(seqs, 256):
            lg = m(input_ids=ids, attention_mask=att).logits
            rows = torch.arange(len(ids))
            scores[i:i + len(ids)] = lg[rows, torch.as_tensor(mpos[i:i + len(ids)])][:, torch.as_tensor(cand)].numpy()
    r = EM.ranks_from_scores(lambda i, j: scores[i:j], true, cand, len(true))
    EM.show(EM.summarize(r, true, freq_rank, "bert-tiny-mlm"))


def encode_task(task, d):
    """model inputs per example (+ word-start positions for conll)"""
    if task == "sst2":
        return [[CLS] + list(a) + [SEP] for a in d["a"]], None
    if task == "qnli":
        return [[CLS] + list(a) + [SEP] + list(b) + [SEP] for a, b in zip(d["a"], d["b"])], None
    return [[CLS] + list(a[:254]) + [SEP] for a in d["a"]], [st[st < 254] + 1 for st in d["starts"]]


def conll_labels(d, starts):
    return [y[:len(s)] for y, s in zip(d["tags"], starts)]


def load(task):
    D = TK.load(task)
    n = int(os.environ.get("BERT_LIMIT", 0))
    return {s: {k: v[:n] for k, v in d.items()} for s, d in D.items()} if n else D


def frozen(task):
    from transformers import BertModel
    import adapters as AD
    m = BertModel.from_pretrained(REPO, cache_dir=os.path.join(C.DATA, "hf")).eval()
    D = load(task)
    F, Y, Dm = {}, {}, {}
    for s in ("train", "dev", "test"):
        seqs, starts = encode_task(task, D[s])
        out = []
        with torch.no_grad():
            for i, ids, att in batches(seqs, 128):
                h = m(input_ids=ids, attention_mask=att).last_hidden_state
                if task == "conll":
                    for j in range(len(ids)):
                        out.append(h[j, torch.as_tensor(starts[i + j])].numpy())
                else:
                    mean = (h * att[..., None]).sum(1) / att.sum(1, keepdim=True)
                    out.append(torch.cat([h[:, 0], mean], 1).numpy())
        F[s] = np.concatenate(out)
        if task == "conll":
            ys = conll_labels(D[s], starts)
            Y[s] = np.concatenate(ys)
            Dm[s] = {"tags": ys}
        else:
            Y[s], Dm[s] = D[s]["y"], D[s]
    F = {s: (v - F["train"].mean(0)) / (F["train"].std(0) + 1e-6) for s, v in F.items()}
    from sklearn.linear_model import LogisticRegression
    best = None
    for c in AD.LOGREG_C:
        clf = LogisticRegression(C=c, max_iter=1000).fit(F["train"], Y["train"])
        mdev = AD.metric(task, Dm["dev"], clf.predict(F["dev"]))
        if best is None or AD.key_metric(mdev) > AD.key_metric(best[1]):
            best = (c, mdev, clf)
    return {"dev": best[1], "test": AD.metric(task, Dm["test"], best[2].predict(F["test"])), "C": best[0]}


def finetune(task):
    import adapters as AD
    from transformers import BertForSequenceClassification, BertForTokenClassification
    torch.manual_seed(0)
    hp = FT[task]
    nc = TK.N_CLASSES[task]
    cls = BertForTokenClassification if task == "conll" else BertForSequenceClassification
    m = cls.from_pretrained(REPO, num_labels=nc, cache_dir=os.path.join(C.DATA, "hf"))
    D = load(task)
    enc = {s: encode_task(task, D[s]) for s in ("train", "dev", "test")}

    def target(s, idx, L):
        if task != "conll":
            return torch.as_tensor(D[s]["y"][idx])
        t = torch.full((len(idx), L), -100, dtype=torch.long)
        for j, k in enumerate(idx):
            st = enc[s][1][k]
            t[j, torch.as_tensor(st)] = torch.as_tensor(D[s]["tags"][k][:len(st)])
        return t

    def predict(s):
        m.eval()
        seqs, starts = enc[s]
        preds = []
        with torch.no_grad():
            for i, ids, att in batches(seqs, 128):
                lg = m(input_ids=ids, attention_mask=att).logits
                if task == "conll":
                    for j in range(len(ids)):
                        preds.append(lg[j, torch.as_tensor(starts[i + j])].argmax(-1).numpy())
                else:
                    preds.append(lg.argmax(-1).numpy())
        m.train()
        if task == "conll":
            return AD.metric(task, {"tags": conll_labels(D[s], starts)}, np.concatenate(preds))
        return AD.metric(task, D[s], np.concatenate(preds))

    seqs = enc["train"][0]
    steps = hp["epochs"] * ((len(seqs) + hp["bs"] - 1) // hp["bs"])
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, k / (0.06 * steps)) * max(0.0, (steps - k) / steps))
    rng = np.random.default_rng(0)
    best, curve = None, []
    m.train()
    for ep in range(hp["epochs"]):
        t0 = time.time()
        perm = rng.permutation(len(seqs))
        for i in range(0, len(perm), hp["bs"]):
            idx = perm[i:i + hp["bs"]]
            _, ids, att = next(batches([seqs[k] for k in idx], len(idx)))
            loss = m(input_ids=ids, attention_mask=att, labels=target("train", idx, ids.shape[1])).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        mdev = predict("dev")
        curve.append((ep, round(AD.key_metric(mdev), 4), round(time.time() - t0)))
        if best is None or AD.key_metric(mdev) > AD.key_metric(best[1]):
            best = (ep, mdev, predict("test"))
    return {"dev": best[1], "test": best[2], "epochs": best[0] + 1, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mlm", "task"])
    ap.add_argument("--evalset")
    ap.add_argument("--task")
    ap.add_argument("--mode", choices=["frozen", "finetune"])
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    if a.cmd == "mlm":
        return mlm(a.evalset)
    t0 = time.time()
    r = frozen(a.task) if a.mode == "frozen" else finetune(a.task)
    r.update(task=a.task, features=f"bert-tiny-{a.mode}", head="logreg" if a.mode == "frozen" else "finetune",
             head_s=round(time.time() - t0, 1))
    print(f"[{a.task}/bert-tiny-{a.mode}] dev {r['dev']}  TEST {r['test']}  ({r['head_s']}s)", flush=True)
    print("RESULT " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
