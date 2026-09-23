"""
cross_check_cuda.py - verify the C engine against the ORIGINAL CUDA GraphTsetlinMachine.

*** UNTESTED: written without GPU access. Run on a CUDA box. ***

    pip install git+https://github.com/cair/GraphTsetlinMachine   (needs pycuda + a GPU)
    python3 cross_check_cuda.py --data ../data/seq.test.gtmd --model ../data/seq.model.gtmm

Inference check (exact):
    Rebuilds a Graphs object with identical X / node types / edges, converts the .gtmm
    TA states into the CUDA bit-plane layout, loads it with tm.load(state_dict=...),
    and requires tm.score() == C engine class sums, element for element.

Training check (statistical, --train-epochs N):
    curand makes CUDA trajectories unreproducible, so this trains the CUDA model from
    scratch on --train-data with the model's hyperparameters and prints per-epoch test
    accuracy next to whatever you got from `gtm train`. Expect matching curves, not bits.
"""
import argparse
import subprocess
import sys

import numpy as np

from gtmcore import Dataset, load_model


def build_graphs(ds, Graphs, init_with=None):
    if init_with is None:
        g = Graphs(ds.n_graphs, symbols=["_"], hypervector_size=ds.hv_size, hypervector_bits=1)
        g.node_type_id = {f"t{k}": k for k in range(ds.n_node_types)}
        g.edge_type_id = {f"e{k}": k for k in range(ds.n_edge_types)}
    else:
        g = Graphs(ds.n_graphs, init_with=init_with)
    for gi in range(ds.n_graphs):
        g.set_number_of_graph_nodes(gi, int(ds.nodes_per_graph[gi]))
    g.prepare_node_configuration()
    for gi in range(ds.n_graphs):
        n0 = ds.node_offset[gi]
        for v in range(ds.nodes_per_graph[gi]):
            g.add_graph_node(gi, int(v), int(ds.edges_per_node[n0 + v]), f"t{ds.node_type[n0 + v]}")
    g.prepare_edge_configuration()
    for gi in range(ds.n_graphs):
        n0 = ds.node_offset[gi]
        for v in range(ds.nodes_per_graph[gi]):
            for dst, et in ds.edges[ds.edge_offset[n0 + v]:ds.edge_offset[n0 + v + 1]]:
                g.add_graph_node_edge(gi, int(v), int(dst), f"e{et}")
    # overwrite features with the exact literal bits: bit k -> uint32 chunk k//32, pos k%32
    chunks = g.number_of_hypervector_chunks
    bits = np.zeros((len(ds.node_type), chunks * 32), dtype=bool)
    bits[:, : 2 * ds.hv_size] = ds.X
    g.X = np.packbits(bits.reshape(-1, chunks, 32), axis=2, bitorder="little").reshape(-1, chunks * 4).view("<u4").copy()
    g.encode()
    return g


def to_bitplanes(states, n_literals, B):
    """[C][n_literals] ints -> flat uint32 [C][chunks][B] as in kernels.py."""
    C = states.shape[0]
    chunks = (n_literals - 1) // 32 + 1
    out = np.zeros((C, chunks, B), dtype=np.uint32)
    padded = np.full((C, chunks * 32), (1 << (B - 1)) - 1, dtype=np.int64)
    padded[:, :n_literals] = states
    for b in range(B):
        plane = ((padded >> b) & 1).astype(bool).reshape(C, chunks, 32)
        out[:, :, b] = np.packbits(plane, axis=2, bitorder="little").reshape(C, chunks, 4).view("<u4")[:, :, 0]
    return out.ravel()


def make_tm(cfg, kind, tm_mod):
    cls = tm_mod.MultiClassGraphTsetlinMachine if kind == 0 else tm_mod.MultiOutputGraphTsetlinMachine
    return cls(cfg.C, cfg.T, cfg.s if cfg.D > 1 else cfg.s[0], q=cfg.q, max_included_literals=cfg.max_inc,
               boost_true_positive_feedback=cfg.boost, number_of_state_bits=cfg.B, depth=cfg.D,
               message_size=cfg.MS, message_bits=cfg.MB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset to score (.gtmd)")
    ap.add_argument("--model", required=True, help="C-engine model (.gtmm)")
    ap.add_argument("--gtm", default="../stage1/gtm", help="C engine binary")
    ap.add_argument("--train-data", help="optional: train CUDA from scratch on this for a curve comparison")
    ap.add_argument("--train-epochs", type=int, default=0)
    a = ap.parse_args()

    from GraphTsetlinMachine.graphs import Graphs
    import GraphTsetlinMachine.tm as tm_mod

    ds = Dataset.load(a.data)
    cfg, hv, w, ta, step = load_model(a.model)
    if cfg.neg != 1:
        sys.exit("the CUDA classes hard-code negative_clauses=1")
    graphs = build_graphs(ds, Graphs)

    tm = make_tm(cfg, ds.task_kind, tm_mod)
    state = {
        "ta_state": to_bitplanes(ta[0], cfg.L, cfg.B),
        "message_ta_state": [to_bitplanes(ta[d], cfg.M, cfg.B) for d in range(1, cfg.D)],
        "clause_weights": w.astype(np.int32).ravel(),          # [O][C], same as CUDA
        "hypervectors": hv.astype(np.uint32),
        "number_of_outputs": cfg.O,
        "number_of_literals": cfg.L,
        "number_of_message_literals": cfg.M,
        "min_y": None, "max_y": None,
        "negative_clauses": 1,
        "max_number_of_graph_nodes": int(graphs.max_number_of_graph_nodes),
        "message_size": cfg.MS,
    }
    tm.load(state_dict=state)
    cuda_sums = tm.score(graphs).astype(np.int32)

    subprocess.run([a.gtm, "score", "--model", a.model, "--data", a.data, "--sums", "/tmp/xc_sums.i32"], check=True)
    c_sums = np.fromfile("/tmp/xc_sums.i32", dtype="<i4").reshape(ds.n_graphs, cfg.O)
    bad = np.nonzero((cuda_sums != c_sums).any(1))[0]
    print(f"inference: {ds.n_graphs - len(bad)}/{ds.n_graphs} graphs identical")
    if len(bad):
        g = bad[0]
        print(f"  first mismatch graph {g}: CUDA {cuda_sums[g]} vs C {c_sums[g]}")
        sys.exit(1)
    print("INFERENCE PARITY WITH CUDA: PASS")

    if a.train_data and a.train_epochs:
        tr = Dataset.load(a.train_data)
        gtr = build_graphs(tr, Graphs)
        gte = build_graphs(ds, Graphs, init_with=gtr)
        tm2 = make_tm(cfg, tr.task_kind, tm_mod)
        Ytr = tr.Y.argmax(1).astype(np.uint32) if tr.task_kind == 0 else tr.Y.astype(np.uint32)
        for e in range(a.train_epochs):
            tm2.fit(gtr, Ytr, epochs=1, incremental=True)
            s = tm2.score(gte)
            acc = (s.argmax(1) == ds.Y.argmax(1)).mean() if ds.task_kind == 0 else \
                  ((s >= 0).astype(int) == ds.Y).all(1).mean()
            print(f"CUDA epoch {e}: test acc {acc:.4f}")


if __name__ == "__main__":
    main()
