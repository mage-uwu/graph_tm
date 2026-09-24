# How much does the 256-bit-code output + code-agreement ranking cap a model that knows bert-tiny's full
# distribution? Rank by vote_b = sum_t p(t) (2 code_tb - 1), the best a bit-vote model can express.
import sys
import os, numpy as np, torch
import bert_baselines as BB, common as C, eval_mlm as EM
from transformers import BertForMaskedLM
m = BertForMaskedLM.from_pretrained(BB.REPO, cache_dir=os.path.join(C.DATA, "hf")).eval()
codes, cand, freq, freq_rank = EM.context()
ev = np.load(os.path.join(C.DATA, "val10k.gtmd.npz")); true, wt = ev["true"], ev["windows"]
seqs, mpos = [], []
for w in wt:
    v = w[w >= 0].tolist(); c = int((w[:C.WIN] >= 0).sum()); v[c] = BB.MASK
    seqs.append([BB.CLS] + v + [BB.SEP]); mpos.append(c + 1)
S = np.zeros((len(seqs), len(cand)), np.float32)
with torch.no_grad():
    for i, ids, att in BB.batches(seqs, 256):
        lg = m(input_ids=ids, attention_mask=att).logits
        S[i:i+len(ids)] = lg[torch.arange(len(ids)), torch.as_tensor(mpos[i:i+len(ids)])][:, torch.as_tensor(cand)].numpy()
cb = codes[cand].astype(np.float32) * 2 - 1
P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
for name, votes in [("bert-direct", None), ("bert->expected-code-votes", P @ cb), ("bert->argmax-code", cb[S.argmax(1)]),
                    ("bert->sign(expected)", np.sign(P @ cb))]:
    f = (lambda i, j: S[i:j]) if votes is None else (lambda i, j, v=votes: v[i:j] @ cb.T)
    r = EM.ranks_from_scores(f, true, cand, len(true))
    res = EM.summarize(r, true, freq_rank, name)
    if votes is not None: res["per-bit"] = float(((votes > 0) == codes[true].astype(bool)).mean())
    print(name, {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
rng = np.random.default_rng(0)
for bits in (256, 1024, 4096):
    rc = np.where(rng.random((len(cand), bits)) < 0.5, 1.0, -1.0).astype(np.float32)
    v = P @ rc
    r = EM.ranks_from_scores(lambda i, j: v[i:j] @ rc.T, true, cand, len(true))
    res = EM.summarize(r, true, freq_rank, f"random{bits}")
    print(f"random-codes-{bits}", {k: round(x, 4) for k, x in res.items() if k in ("acc@1", "acc@10", "mrr")})
