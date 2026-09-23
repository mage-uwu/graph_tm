"""
Parity suite: stage0 oracle  <->  stage1 C engine.

  1. init parity       fresh models are byte-identical
  2. inference parity  random sparse models on random graphs: class sums + clause bits
  3. training parity   N steps of training -> identical TA states and weights
  4. invariance        C results identical for 1..8 threads and AVX-512 vs scalar build

Run: python3 tests/test_parity.py [--quick]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "stage0"))
from gtmcore import Dataset, ModelConfig, load_model, save_model  # noqa: E402
from gtm_oracle import OracleGTM  # noqa: E402

GTM = os.path.join(ROOT, "stage1", "gtm")
GTM_PORTABLE = os.path.join(ROOT, "stage1", "gtm_portable")
TMP = tempfile.mkdtemp(prefix="gtm_parity_")
FAIL = []


def run(*args):
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{args}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def random_dataset(rng, n_graphs, H, NT, NET, O, kind, max_nodes=12, n_symbols=20, hv_bits=3):
    shv = np.stack([rng.choice(H, size=hv_bits, replace=False) for _ in range(n_symbols)])
    npg, ntype, epn, edges, X, Y = [], [], [], [], [], []
    for _ in range(n_graphs):
        n = int(rng.integers(1, max_nodes + 1))
        npg.append(n)
        for i in range(n):
            x = np.zeros(2 * H, dtype=bool)
            x[H:] = True
            for s in rng.choice(n_symbols, size=int(rng.integers(0, 4)), replace=False):
                x[shv[s]] = True
                x[shv[s] + H] = False
            X.append(x)
            ntype.append(int(rng.integers(NT)))
            ne = int(rng.integers(0, 4))  # includes self loops + multi-edges
            epn.append(ne)
            for _ in range(ne):
                edges.append((int(rng.integers(n)), int(rng.integers(NET))))
        if kind == 0:
            y = np.zeros(O, dtype=np.int32); y[rng.integers(O)] = 1
        else:
            y = rng.integers(0, 2, size=O).astype(np.int32)
        Y.append(y)
    return Dataset(H, NT, NET, O, kind, npg, ntype, epn, np.array(edges, dtype=np.uint32).reshape(-1, 2),
                   np.array(X), np.array(Y))


def c_scores(model_path, data_path, n_graphs, O, C, threads=1, binary=GTM):
    sums_p, bits_p = os.path.join(TMP, "sums.i32"), os.path.join(TMP, "bits.u64")
    run(binary, "score", "--model", model_path, "--data", data_path, "--threads", threads,
        "--sums", sums_p, "--bits", bits_p)
    sums = np.fromfile(sums_p, dtype="<i4").reshape(n_graphs, O)
    Cw = (C + 63) // 64
    words = np.fromfile(bits_p, dtype="<u8").reshape(n_graphs, Cw)
    bits = np.unpackbits(words.view(np.uint8).reshape(n_graphs, Cw * 8), axis=1, bitorder="little")[:, :C]
    return sums, bits.astype(bool)


def noncomplementary(ds, rng):
    """flip some negated-half bits so X is no longer [x | not x]: exercises the dense layer-0 path"""
    X = ds.X.copy()
    H = ds.hv_size
    flip = rng.random((X.shape[0], H)) < 0.05
    X[:, H:] ^= flip
    return Dataset(ds.hv_size, ds.n_node_types, ds.n_edge_types, ds.n_outputs, ds.task_kind, ds.nodes_per_graph,
                   ds.node_type, ds.edges_per_node, ds.edges, X, ds.Y)


def cfg_args(cfg):
    return ["--clauses", cfg.C, "--T", cfg.T, "--s", ",".join(repr(x) for x in cfg.s), "--q", repr(cfg.q),
            "--depth", cfg.D, "--msg-size", cfg.MS, "--msg-bits", cfg.MB, "--max-inc", cfg.max_inc,
            "--state-bits", cfg.B, "--boost", cfg.boost, "--neg", cfg.neg, "--seed", cfg.seed, "--rho", repr(cfg.rho), "--senders", "pos" if cfg.senders else "all",
            "--forget", "layered" if cfg.layered else "all"]


def models_equal(pa, pb):
    ca, hva, wa, taa, sa = load_model(pa)
    cb, hvb, wb, tab, sb = load_model(pb)
    same = (sa == sb and np.array_equal(hva, hvb) and np.array_equal(wa, wb)
            and all(np.array_equal(x, y) for x, y in zip(taa, tab)))
    if same:
        return True, ""
    diffs = [f"layer{i}:{int((x != y).sum())}" for i, (x, y) in enumerate(zip(taa, tab)) if not np.array_equal(x, y)]
    return False, f"step {sa} vs {sb}, hv diffs {int((hva != hvb).sum())}, weight diffs {int((wa != wb).sum())}, ta diffs {diffs}"


# ---------------------------------------------------------------------------
def test_init(rng):
    print("1. init parity")
    for D, neg in ((1, 1), (3, 1), (2, 0)):
        ds = random_dataset(rng, 5, 64, 2, 2, 3, 1)
        dp = os.path.join(TMP, "init.gtmd"); ds.save(dp)
        cfg = ModelConfig(70, 3, 64, 2, depth=D, msg_size=128, msg_bits=3, negative_clauses=neg, seed=7 + D)
        o = OracleGTM(cfg); op = os.path.join(TMP, "o_init.gtmm"); o.save(op)
        cp = os.path.join(TMP, "c_init.gtmm")
        run(GTM, "init", "--data", dp, "--out", cp, *cfg_args(cfg))
        ok, why = models_equal(op, cp)
        check(f"init D={D} neg={neg}", ok and open(op, "rb").read() == open(cp, "rb").read(), why)


def random_sparse_model(rng, cfg, p0, pm):
    o = OracleGTM(cfg)
    half = 1 << (cfg.B - 1)
    for layer, p in zip(range(cfg.D), [p0] + [pm] * (cfg.D - 1)):
        shape = o.ta[layer].shape
        inc = rng.random(shape) < p
        o.ta[layer] = np.where(inc, rng.integers(half, 2 * half, shape), rng.integers(0, half, shape)).astype(np.int32)
    o.w = rng.integers(-50, 51, size=o.w.shape).astype(np.int64)
    return o


def test_inference(rng):
    print("2. inference parity (random sparse models)")
    for (H, NT, NET, O, kind, C, D) in ((64, 1, 1, 2, 0, 50, 1), (100, 3, 3, 4, 1, 130, 2),
                                        (256, 2, 4, 5, 0, 200, 3), (32, 1, 2, 3, 1, 64, 3)):
        ds = random_dataset(rng, 150, H, NT, NET, O, kind)
        dp = os.path.join(TMP, "inf.gtmd"); ds.save(dp)
        cfg = ModelConfig(C, O, H, NT, depth=D, msg_size=128, msg_bits=2, seed=int(rng.integers(1 << 30)))
        o = random_sparse_model(rng, cfg, p0=3.0 / (2 * H), pm=2.0 / 256)
        mp = os.path.join(TMP, "inf.gtmm"); o.save(mp)
        want_sums, want_bits = o.score(ds), o.transform(ds)
        got_sums, got_bits = c_scores(mp, dp, ds.n_graphs, O, C, threads=3)
        frac = want_bits.mean()
        check(f"score  H={H} NT={NT} NET={NET} C={C} D={D}", np.array_equal(want_sums, got_sums),
              f"(clause fire rate {frac:.2f})")
        check(f"clause bits", np.array_equal(want_bits, got_bits))


def train_case(name, ds, cfg, steps, threads_list=(1, 2, 3, 5, 8, 16)):
    dp = os.path.join(TMP, name + ".gtmd"); ds.save(dp)
    init = os.path.join(TMP, name + ".init.gtmm")
    run(GTM, "init", "--data", dp, "--out", init, *cfg_args(cfg))

    o = OracleGTM.load(init)
    o.fit(ds, epochs=10**9, max_steps=steps)
    op = os.path.join(TMP, name + ".oracle.gtmm"); o.save(op)

    hashes = []
    for t in threads_list:
        cp = os.path.join(TMP, f"{name}.c{t}.gtmm")
        run(GTM, "train", "--data", dp, "--model", init, "--steps", steps, "--threads", t, "--save", cp)
        if t == threads_list[0]:
            ok, why = models_equal(op, cp)
            check(f"train {name} ({steps} steps) oracle == C", ok, why)
        hashes.append(run(GTM, "hash", "--model", cp).strip())
    pp = os.path.join(TMP, f"{name}.portable.gtmm")
    run(GTM_PORTABLE, "train", "--data", dp, "--model", init, "--steps", steps, "--threads", 2, "--save", pp)
    hashes.append(run(GTM, "hash", "--model", pp).strip())
    check(f"train {name} thread/ISA invariance {list(threads_list)}+scalar", len(set(hashes)) == 1, str(set(hashes)))
    # the trained model must also score identically, via the indexed and the reference evaluator
    want = o.score(ds)
    got, _ = c_scores(op, dp, ds.n_graphs, cfg.O, cfg.C, threads=2)
    os.environ["GTM_DENSE"] = "1"
    got_dense, _ = c_scores(op, dp, ds.n_graphs, cfg.O, cfg.C, threads=2)
    del os.environ["GTM_DENSE"]
    check(f"score after training {name} (indexed + dense)", np.array_equal(want, got) and np.array_equal(want, got_dense))


def test_training(rng, quick):
    print("3+4. training parity and invariance")
    scale = 1 if quick else 3
    data = os.path.join(ROOT, "data")

    def sub(path, n):
        full = Dataset.load(path)
        e = full.node_offset[n]
        ee = full.edge_offset[e]
        return Dataset(full.hv_size, full.n_node_types, full.n_edge_types, full.n_outputs, full.task_kind,
                       full.nodes_per_graph[:n], full.node_type[:e], full.edges_per_node[:e],
                       full.edges[:ee], full.X[:e], full.Y[:n])

    # demo configs (s = 1.0: deterministic feedback path)
    train_case("xor", sub(os.path.join(data, "xor.train.gtmd"), 400),
               ModelConfig(10, 2, 32, 1, depth=2, msg_size=256, max_included_literals=4, T=100, s=1.0, seed=3),
               steps=400 * scale)
    train_case("seq", sub(os.path.join(data, "seq.train.gtmd"), 150),
               ModelConfig(60, 3, 256, 1, depth=3, msg_size=256, max_included_literals=4, T=600, s=1.0, seed=4),
               steps=150 * scale)
    # s > 1: hashed feedback path; per-layer s; multi-output
    train_case("maskbits", sub(os.path.join(data, "maskbits.train.gtmd"), 120),
               ModelConfig(80, 3, 128, 1, depth=2, msg_size=128, T=40, s=(3.9, 2.5), seed=5),
               steps=120 * scale)
    # everything unusual at once: 3 node types, 3 edge types, boost off, no negative clauses,
    # q != 1, 5 state bits, include cap, clause count not a multiple of 64
    ds = random_dataset(rng, 100, 64, 3, 3, 4, 0)
    for Y, n in zip(ds.Y, ds.nodes_per_graph):  # learnable-ish label: node count mod 4
        Y[:] = 0
        Y[n % 4] = 1
    train_case("weird", ds,
               ModelConfig(97, 4, 64, 3, depth=3, msg_size=64, msg_bits=3, state_bits=5, boost=0,
                           negative_clauses=0, max_included_literals=6, T=15, q=2.5, s=(5.0, 1.7, 9.0), seed=6),
               steps=100 * scale)
    # many outputs: exercises per-output selection masks, per-clause output lists, partial last word
    ds = random_dataset(rng, 60, 64, 1, 2, 100, 1)
    train_case("wide_out", ds, ModelConfig(150, 100, 64, 1, depth=2, msg_size=128, T=20, q=3.0, s=3.0, seed=9),
               steps=60 * scale)
    # decoupled feedback: weights see every selected pair, automata a rho-fraction
    ds = random_dataset(rng, 60, 64, 2, 2, 80, 1)
    train_case("decoupled", ds, ModelConfig(140, 80, 64, 2, depth=3, msg_size=64, T=30, q=79.0, s=(4.0, 2.0, 3.0),
                                            seed=10, rho=0.05), steps=60 * scale)
    # only clauses with a positive layer-0 literal send messages
    ds = random_dataset(rng, 80, 64, 2, 3, 5, 0)
    train_case("senders", ds, ModelConfig(110, 5, 64, 2, depth=3, msg_size=128, T=25, s=(3.0, 2.0, 4.0), seed=12,
                                          senders=1), steps=80 * scale)
    # layer-0 features not of the form [x | not x]: indexed evaluator must fall back
    ds = noncomplementary(random_dataset(rng, 80, 64, 2, 2, 4, 0), rng)
    train_case("noncomp", ds, ModelConfig(90, 4, 64, 2, depth=2, msg_size=64, T=20, s=3.0, seed=11), steps=80 * scale)
    ds = random_dataset(rng, 100, 96, 2, 2, 6, 1)
    train_case("multi_out", ds,
               ModelConfig(130, 6, 96, 2, depth=2, msg_size=192, T=25, q=0.5, s=2.0, seed=8), steps=100 * scale)
    # layered forget: a non-firing clause keeps the layers it still matched somewhere (depth 3,
    # positive senders, node types, decoupled feedback, so every code path of the flag is hit)
    ds = random_dataset(rng, 80, 64, 2, 3, 12, 1)
    train_case("layered", ds, ModelConfig(120, 12, 64, 2, depth=3, msg_size=128, T=25, q=11.0, s=(3.0, 2.0, 4.0),
                                          seed=13, rho=0.5, senders=1, layered=1), steps=80 * scale)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(12345)
    test_init(rng)
    test_inference(rng)
    test_training(rng, quick)
    print()
    print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)
