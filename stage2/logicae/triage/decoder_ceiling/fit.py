# decoder ceiling on frozen saved_fastae.pt features: untied softmax heads fitted on 200k masked targets,
# CE on 5.4k validation targets vs the model's own tied 128-bit Hamming decoder
import sys, time, numpy as np, torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(4)
W, C, V = 1024, 128, 30522
def load(s):
    b = np.unpackbits(np.fromfile(f"{s}.bits", np.uint8).reshape(-1, 656), axis=1, bitorder="little")
    return torch.from_numpy(b), torch.from_numpy(np.loadtxt(f"{s}.true", dtype=np.int64)), np.fromfile(f"{s}.ce", np.float32)
Xt, yt, cet = load("train"); Xv, yv, cev = load("validation")
cnt = np.bincount(yt.numpy(), minlength=V) + 0.1; lp = torch.tensor(np.log(cnt / cnt.sum()), dtype=torch.float32)
print(f"tied decoder CE: val {cev.mean():.4f}  train {cet.mean():.4f}   unigram(train counts) on val {(-lp[yv]).mean():.4f}", flush=True)
sel = {"proj128": slice(5 * W, 5 * W + C), "trunk1024": slice(2 * W, 3 * W), "trunk5x1024": slice(0, 5 * W)}
def run(name, rank):
    s = sel[name]; D = s.stop - s.start
    f = lambda X: X[:, s].float() * 2 - 1
    head = nn.Linear(D, V) if rank == 0 else nn.Sequential(nn.Linear(D, rank, bias=False), nn.Linear(rank, V))
    last = head if rank == 0 else head[1]
    with torch.no_grad(): last.weight.mul_(0.1); last.bias.copy_(lp)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0)
    best, bad, t0 = 9e9, 0, time.time()
    for ep in range(30):
        perm = torch.randperm(len(yt))
        head.train()
        for i in range(0, len(yt), 1024):
            idx = perm[i:i + 1024]; loss = nn.functional.cross_entropy(head(f(Xt[idx])), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad(): v = nn.functional.cross_entropy(head(f(Xv)), yv).item()
        print(f"  {name} rank={rank or 'full'} epoch {ep+1}: val CE {v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v < best - 1e-3: best, bad = v, 0
        else:
            bad += 1
            if bad >= 2: break
    print(f"RESULT {name} rank={rank or 'full'} best val CE {best:.4f}", flush=True)
for a in sys.argv[1:]:
    n, r = a.split(":"); run(n, int(r))
