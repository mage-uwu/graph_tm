"""
Ground-truth training targets in GraphTM-native form, written as .gtmd files.

  noisy_xor  - port of examples/NoisyXORDemo.py (2 nodes, depth 2). Expect ~100% test.
  sequence   - port of examples/SequenceClassificationDemo.py (count a run of 'A's,
               3 classes, chain graph with Left/Right edges, depth 3).
  masked --codes N - the same, with an N-bit code per token as target (teacher-LSH shape).
  masked     - NEW: first MLM-shaped target. A chain of tokens that walks +1 or -1
               (mod V), long enough that every token occurs (no bag-of-symbols leak;
               a depth-1 control scores chance). One interior node is replaced by MASK;
               the label is the hidden token. The masked node carries no information itself, and a single
               neighbor is ambiguous (left=3 -> 4 or 2), so the model must AND the
               messages from both neighbors. --bits switches the target from V-way
               multiclass to log2(V) independent output bits (multi-output, the shape
               the teacher-LSH-bit objective will have).

Usage: python3 datasets.py <task> <out_prefix> [--n N] [--seed S] [...]
Writes <out_prefix>.train.gtmd and <out_prefix>.test.gtmd (noise on train only, as in the demos).
"""
import argparse
import numpy as np
from gtmcore import Dataset


class GraphBuilder:
    def __init__(self, hv_size, symbol_hv, n_node_types=1):
        self.H = hv_size
        self.symbol_hv = symbol_hv
        self.n_node_types = n_node_types
        self.npg, self.ntype, self.epn, self.edges, self.X, self.Y = [], [], [], [], [], []

    def add_graph(self, node_symbols, adjacency, y, node_types=None):
        """node_symbols: list of lists of symbol ids; adjacency: per node list of (dst, edge_type)."""
        n = len(node_symbols)
        self.npg.append(n)
        for i in range(n):
            x = np.zeros(2 * self.H, dtype=bool)
            x[self.H:] = True
            for sym in node_symbols[i]:
                for k in self.symbol_hv[sym]:
                    x[k] = True
                    x[k + self.H] = False
            self.X.append(x)
            self.ntype.append(0 if node_types is None else node_types[i])
            self.epn.append(len(adjacency[i]))
            self.edges.extend(adjacency[i])
        self.Y.append(y)

    def build(self, n_edge_types, n_outputs, task_kind):
        return Dataset(self.H, self.n_node_types, n_edge_types, n_outputs, task_kind,
                       self.npg, self.ntype, self.epn, np.array(self.edges, dtype=np.uint32).reshape(-1, 2),
                       np.array(self.X), np.array(self.Y))


def make_symbol_hv(rng, n_symbols, hv_size, hv_bits):
    return np.stack([rng.choice(hv_size, size=hv_bits, replace=False) for _ in range(n_symbols)])


def one_hot(k, n):
    y = np.zeros(n, dtype=np.int32)
    y[k] = 1
    return y


def chain(n):
    """Left = type 0, Right = type 1, same edge order as the repo demo."""
    adj = []
    for i in range(n):
        a = []
        if i > 0:
            a.append((i - 1, 0))
        if i < n - 1:
            a.append((i + 1, 1))
        adj.append(a)
    return adj


def noisy_xor(rng, n, noise, hv_size=32, hv_bits=2):
    shv = make_symbol_hv(rng, 2, hv_size, hv_bits)  # A=0, B=1
    out = []
    for split_noise in (noise, 0.0):
        gb = GraphBuilder(hv_size, shv)
        for _ in range(n):
            x1, x2 = rng.integers(2), rng.integers(2)
            y = int(x1 != x2)
            if rng.random() <= split_noise:
                y = 1 - y
            gb.add_graph([[x1], [x2]], [[(1, 0)], [(0, 0)]], one_hot(y, 2))
        out.append(gb.build(1, 2, 0))
    return out


def sequence(rng, n, noise, classes=3, max_len=10, hv_size=256, hv_bits=2):
    shv = make_symbol_hv(rng, 1, hv_size, hv_bits)  # 'A'
    out = []
    for split_noise in (noise, 0.0):
        gb = GraphBuilder(hv_size, shv)
        for _ in range(n):
            L = rng.integers(classes, max_len + 1)
            y = rng.integers(classes)
            node = rng.integers(y, L)
            syms = [[] for _ in range(L)]
            for p in range(y + 1):
                syms[node - p] = [0]
            if rng.random() <= split_noise:
                y = rng.choice(np.setdiff1d(np.arange(classes), [y]))
            gb.add_graph(syms, chain(L), one_hot(y, classes))
        out.append(gb.build(2, classes, 0))
    return out


def masked(rng, n, noise, vocab=8, min_len=None, max_len=None, hv_size=128, hv_bits=2, bits=False, codes=0,
           code_seed=0):
    # Length >= 2V+1 so every token appears somewhere even after masking. With shorter
    # sequences the masked token is often the one *absent* from the graph, and a depth-1
    # GraphTM solves ~73% of the task from the bag of node symbols alone (graph output is
    # an OR over nodes). The depth-1 control must sit at chance for the task to be honest.
    min_len = min_len or 2 * vocab + 1
    max_len = max_len or 2 * vocab + 8
    shv = make_symbol_hv(rng, vocab + 1, hv_size, hv_bits)  # tokens 0..V-1, MASK = V
    nbits = int(np.ceil(np.log2(vocab)))
    # --codes N: each token gets a random N-bit code (a stand-in for teacher-embedding LSH bits;
    # random is the worst case, real LSH codes are correlated for similar tokens)
    code_table = np.random.default_rng(code_seed).integers(0, 2, size=(vocab, codes)).astype(np.int32) if codes else None
    out = []
    for split_noise in (noise, 0.0):
        gb = GraphBuilder(hv_size, shv)
        for _ in range(n):
            L = rng.integers(min_len, max_len + 1)
            step = 1 if rng.random() < 0.5 else -1
            start = rng.integers(vocab)
            toks = [(start + step * i) % vocab for i in range(L)]
            m = rng.integers(1, L - 1)  # interior only: both neighbors exist
            tgt = toks[m]
            if rng.random() <= split_noise:
                tgt = rng.choice(np.setdiff1d(np.arange(vocab), [tgt]))
            syms = [[t] for t in toks]
            syms[m] = [vocab]
            if code_table is not None:
                y = code_table[tgt]
            elif bits:
                y = np.array([(tgt >> b) & 1 for b in range(nbits)], dtype=np.int32)
            else:
                y = one_hot(tgt, vocab)
            gb.add_graph(syms, chain(L), y)
        n_out = codes if codes else (nbits if bits else vocab)
        out.append(gb.build(2, n_out, 1 if (bits or codes) else 0))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["noisy_xor", "sequence", "masked"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=8)
    ap.add_argument("--bits", action="store_true")
    ap.add_argument("--codes", type=int, default=0, help="masked: N-bit random code per token as target")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.task == "noisy_xor":
        tr, te = noisy_xor(rng, a.n or 10000, a.noise)
    elif a.task == "sequence":
        tr, te = sequence(rng, a.n or 40000, a.noise)
    else:
        tr, te = masked(rng, a.n or 20000, a.noise, vocab=a.vocab, bits=a.bits, codes=a.codes)
    tr.save(a.out + ".train.gtmd")
    te.save(a.out + ".test.gtmd")
    print(f"{a.task}: {tr.n_graphs} train / {te.n_graphs} test graphs, "
          f"H={tr.hv_size}, outputs={tr.n_outputs}, edge types={tr.n_edge_types}")
