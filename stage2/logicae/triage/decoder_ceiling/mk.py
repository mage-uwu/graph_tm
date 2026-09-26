# masked-centre windows (65 tokens, target at 32) streamed from WikiText-103: fit set (train) and test set (validation)
import os, sys, numpy as np
sys.path.insert(0, "/home/user/graph_tm/stage2"); sys.path.insert(0, "/home/user/graph_tm/stage2/fast_logicae")
os.environ["LR_OUT"] = os.path.dirname(os.path.abspath(__file__))
import longrun as L, common as C, pretrain_data as P
tok = C.tokenizer(); out = os.path.dirname(os.path.abspath(__file__))
def build(split, n, seed, per_para):
    rng = np.random.default_rng(seed); rows, true = [], []
    for p in L.stream_paragraphs(tok, P.SPLIT_FILES[split]):
        if len(p) < 65: continue
        for j in rng.choice(np.arange(32, len(p) - 32), min(per_para, len(p) - 64), replace=False):
            w = p[j - 32:j + 33].copy(); true.append(int(w[32])); w[32] = 1; rows.append(w)
        if len(rows) >= n: break
    L.write_rows(f"{out}/{split}.ids", rows[:n]); np.savetxt(f"{out}/{split}.true", true[:n], fmt="%d")
    print(split, len(rows[:n]), flush=True)
build("validation", 12000, 7, 4)
build("train", 200000, 8, 2)
