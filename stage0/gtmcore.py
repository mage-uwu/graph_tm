"""
gtmcore: the shared spec between the Python oracle (stage 0) and the C engine (stage 1).

Two things live here and MUST stay byte-for-byte identical to stage1/gtm.c:

1. Counter-based RNG. Every random decision in training is a pure function of
   (seed, tag, a, b, c). No sequential RNG state, so:
     - the oracle and the C engine produce identical training trajectories,
     - the C engine is deterministic regardless of thread count.

   Tags:
     1 INIT_W     (output k, clause c, 0)            -> initial weight sign
     2 NODE_SEL   (step t, clause c, 0)              -> which true node gets feedback (key % count)
     3 UPD_SEL    (step t, 0, 0) = hs; per output k: hk = splitmix64(hs ^ k);
                  per clause word cw: base = low32(splitmix64(hk ^ cw)); draw j of that stream
                  (16-bit, see below) decides clause cw*64+j:
                     weight feedback from output k     iff u16 < thr16(p_k)
                     automaton feedback from output k  iff u16 < thr16(p_k * rho)   (a subset)
                  p_k = err_k / 2T * (min(1, q/max(1,O-1)) if target_k = -1 else 1).
                  The CUDA code draws u0 (q) and u1 (error) separately; both probabilities depend
                  only on the output, so one Bernoulli(e*q') per pair has the same law and
                  vectorizes 64 clauses per output. rho = 1 makes the two sets identical.
     4 FEEDBACK   (step t, clause c, layer) = hl; per output k: base = low32(splitmix64(hl ^ k));
                  draw i of that stream decides literal i's 1/s feedback (thr16(1/s)).
     5 CLAUSE_HV  (clause c, j, 0) % msg_size, rejecting duplicates -> clause message bits

   Decoupled feedback (rho): every selected (clause, output) pair updates its weight, but only
   a rho-fraction also drives Type I/II automaton feedback. With O outputs, rho ~ 2/O keeps the
   per-step automaton learning rate independent of O. Without it a clause receives ~O*err/2T
   Type I updates per example and memorizes single nodes; weights, meanwhile, need the full
   stream (subsampling both plateaus at ~65% token recovery on 256-bit random codes).

   16-bit draws, two per 32-bit hash:
     draw i:  u16 = (lowbias32(base + ((i >> 5) << 4) + (i & 15)) >> (16 * ((i >> 4) & 1))) & 0xFFFF
     hit iff  u16 < thr16,  thr16 = min(floor(p * 65536), 65536)
   Probabilities are quantized to 1/65536 (s = 5: 0.2 -> 0.199997). One 16-lane AVX-512 hash
   yields 32 draws.

   Weight rule: for a selected pair whose clause fired anywhere in the graph, w += target_k
   (algebraically identical to CUDA's "target*sign > 0 ? w += sign : w -= sign"), except that
   growing |w| is blocked at INT_MAX, and without negative clauses w is clamped at >= 1.

   Layered forget (model flag, default off = original semantics): a clause that did not fire
   normally gets Type I "forget" on every layer. With layered forget it is forgotten only from
   the first layer at which it became false at every node onwards; the layers before that (in
   particular layer 0, which decides whether a clause sends messages) matched somewhere and are
   left alone. Per-layer credit assignment without gradients.

   Message senders (model flag): 0 = every clause true at a node sends (GraphTM); 1 = only
   clauses that include at least one positive layer-0 literal send.

2. File formats: .gtmd (datasets) and .gtmm (models). Little-endian.
   .gtmm v2 header: magic, ver, C, O, H, NT, D, MS, MB, B, boost, neg, max_inc, T, senders
   (u32 each, T i32; v1 files have 0 in the senders slot; bit 0 = senders, bit 1 = layered
   forget), then q, s[D], rho (v2 only) as f64,
   seed, step as u64, hv u32[C][MB], weights i32[O][C], TA states u16 [C][lits] per layer.
"""
import math
import struct
import numpy as np

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1
GOLD = np.uint64(0x9E3779B97F4A7C15)
SM1 = np.uint64(0xBF58476D1CE4E5B9)
SM2 = np.uint64(0x94D049BB133111EB)
TAGMUL = 0xD6E8FEB86659FD93

TAG_INIT_W, TAG_NODE_SEL, TAG_UPD_SEL, TAG_FEEDBACK, TAG_CLAUSE_HV = 1, 2, 3, 4, 5


def _u64(x):
    return np.asarray(x, dtype=np.uint64)


def splitmix64(x):
    with np.errstate(over="ignore"):
        z = _u64(x) + GOLD
        z = (z ^ (z >> np.uint64(30))) * SM1
        z = (z ^ (z >> np.uint64(27))) * SM2
        return z ^ (z >> np.uint64(31))


def key(seed, tag, a, b, c):
    h = splitmix64(_u64(seed) ^ _u64((tag * TAGMUL) & M64))
    h = splitmix64(h ^ _u64(a))
    h = splitmix64(h ^ _u64(b))
    return splitmix64(h ^ _u64(c))


def lowbias32(x):
    with np.errstate(over="ignore"):
        x = _u64(x) & np.uint64(M32)
        x ^= x >> np.uint64(16)
        x = (x * np.uint64(0x7FEB352D)) & np.uint64(M32)
        x ^= x >> np.uint64(15)
        x = (x * np.uint64(0x846CA68B)) & np.uint64(M32)
        x ^= x >> np.uint64(16)
        return x


def prob_threshold(p):
    """Integer threshold thr such that draw x in [0,2^32) 'hits' iff x + 1 <= thr,
    i.e. u = (x+1)/2^32 <= p. Computed in IEEE double exactly as in C."""
    t = math.floor(p * 4294967296.0)
    return min(t, 1 << 32)


def prob_threshold16(p):
    return min(math.floor(p * 65536.0), 1 << 16)


def u16_draws(base, n):
    """the n 16-bit uniforms of a draw stream starting at 32-bit counter `base`"""
    i = np.arange(n, dtype=np.uint64)
    u = lowbias32(np.uint64(base) + ((i >> np.uint64(5)) << np.uint64(4)) + (i & np.uint64(15)))
    return (u >> (np.uint64(16) * ((i >> np.uint64(4)) & np.uint64(1)))) & np.uint64(0xFFFF)


def feedback_mask(seed, step, clause, output, n_outputs, layer, n_literals, thr16):
    """Boolean mask over literals: True where the literal receives 1/s feedback."""
    if thr16 >= (1 << 16):
        return np.ones(n_literals, dtype=bool)
    base = int(splitmix64(key(seed, TAG_FEEDBACK, step, clause, layer) ^ np.uint64(output))) & M32
    return u16_draws(base, n_literals) < np.uint64(thr16)


def update_prob(err, T, target, q, n_outputs):
    p = err / (2 * T)
    if target == -1:
        p = p * min(1.0, q / max(1, n_outputs - 1))
    return p


def selection_masks(seed, step, output, n_clauses, thr16, thr16_ta):
    """(weight, automaton) bool[n_clauses] masks: which clauses receive feedback from `output`.
    Clause cw*64 + j uses draw j of the stream based at low32(splitmix64(hk ^ cw))."""
    if thr16 == 0:
        z = np.zeros(n_clauses, dtype=bool)
        return z, z
    Cw = (n_clauses + 63) // 64
    hk = splitmix64(key(seed, TAG_UPD_SEL, step, 0, 0) ^ np.uint64(output))
    bases = splitmix64(hk ^ np.arange(Cw, dtype=np.uint64)) & np.uint64(M32)
    u = np.concatenate([u16_draws(int(b), 64) for b in bases])[:n_clauses]
    return u < np.uint64(thr16), u < np.uint64(thr16_ta)


def clause_hypervectors(seed, n_clauses, msg_size, msg_bits):
    hv = np.zeros((n_clauses, msg_bits), dtype=np.uint32)
    for c in range(n_clauses):
        got, j = [], 0
        while len(got) < msg_bits:
            r = int(key(seed, TAG_CLAUSE_HV, c, j, 0)) % msg_size
            j += 1
            if r not in got:
                got.append(r)
        hv[c] = got
    return hv


# ----------------------------------------------------------------------------
# Dataset format (.gtmd)
# ----------------------------------------------------------------------------
GTMD_HDR = "<4sIIIIIIIQQII"


class Dataset:
    """CSR graph batch. X holds literal bits [x | not x], packed into uint64 words."""

    def __init__(self, hv_size, n_node_types, n_edge_types, n_outputs, task_kind,
                 nodes_per_graph, node_type, edges_per_node, edges, X_bool, Y):
        self.hv_size = int(hv_size)
        self.L = 2 * self.hv_size
        self.W = (self.L + 63) // 64
        self.n_node_types = int(n_node_types)
        self.n_edge_types = int(n_edge_types)
        self.n_outputs = int(n_outputs)
        self.task_kind = int(task_kind)  # 0 multiclass (one-hot Y), 1 multi-output
        self.nodes_per_graph = np.asarray(nodes_per_graph, dtype=np.uint32)
        self.node_type = np.asarray(node_type, dtype=np.uint32)
        self.edges_per_node = np.asarray(edges_per_node, dtype=np.uint32)
        self.edges = np.asarray(edges, dtype=np.uint32).reshape(-1, 2)
        self.X = np.asarray(X_bool, dtype=bool).reshape(-1, self.L)
        self.Y = np.asarray(Y, dtype=np.int32).reshape(-1, self.n_outputs)
        self.n_graphs = len(self.nodes_per_graph)
        self.node_offset = np.zeros(self.n_graphs + 1, dtype=np.int64)
        self.node_offset[1:] = np.cumsum(self.nodes_per_graph)
        self.edge_offset = np.zeros(len(self.node_type) + 1, dtype=np.int64)
        self.edge_offset[1:] = np.cumsum(self.edges_per_node)

    def save(self, path):
        total_nodes = len(self.node_type)
        Xw = np.zeros((total_nodes, self.W * 64), dtype=bool)
        Xw[:, : self.L] = self.X
        packed = np.packbits(Xw.reshape(total_nodes, self.W, 64), axis=2, bitorder="little")
        packed = packed.reshape(total_nodes, self.W * 8).view("<u8")
        with open(path, "wb") as f:
            f.write(struct.pack(GTMD_HDR, b"GTMD", 1, self.n_graphs, self.hv_size,
                                self.n_node_types, self.n_edge_types, self.n_outputs,
                                self.task_kind, total_nodes, len(self.edges), self.W, 0))
            for a, dt in ((self.nodes_per_graph, "<u4"), (self.node_type, "<u4"),
                          (self.edges_per_node, "<u4"), (self.edges, "<u4"),
                          (packed, "<u8"), (self.Y, "<i4")):
                f.write(np.ascontiguousarray(a, dtype=dt).tobytes())

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            buf = f.read()
        hs = struct.calcsize(GTMD_HDR)
        (mag, ver, ng, H, nnt, net, no, kind, tn, te, W, _) = struct.unpack(GTMD_HDR, buf[:hs])
        assert mag == b"GTMD" and ver == 1
        off = hs

        def take(n, dt):
            nonlocal off
            a = np.frombuffer(buf, dtype=dt, count=n, offset=off)
            off += a.nbytes
            return a

        npg = take(ng, "<u4"); nt = take(tn, "<u4"); epn = take(tn, "<u4")
        ed = take(2 * te, "<u4"); Xp = take(tn * W, "<u8"); Y = take(ng * no, "<i4")
        bits = np.unpackbits(Xp.view(np.uint8).reshape(tn, W * 8), axis=1, bitorder="little")
        X = bits[:, : 2 * H].astype(bool)
        return Dataset(H, nnt, net, no, kind, npg, nt, epn, ed, X, Y)


# ----------------------------------------------------------------------------
# Model format (.gtmm). TA states stored as plain integers (layout-agnostic).
# ----------------------------------------------------------------------------
GTMM_HDR = "<4s" + "I" * 12 + "iI"


class ModelConfig:
    def __init__(self, clauses, outputs, hv_size, n_node_types, depth=1, msg_size=256,
                 msg_bits=2, state_bits=8, boost=1, negative_clauses=1,
                 max_included_literals=None, T=100, q=1.0, s=1.0, seed=42, rho=1.0, senders=0, layered=0):
        self.C, self.O, self.H = int(clauses), int(outputs), int(hv_size)
        self.L = 2 * self.H
        self.NT = int(n_node_types)
        self.D = int(depth)
        self.MS = int(msg_size)
        self.M = 2 * self.MS
        self.MB = int(msg_bits)
        self.B = int(state_bits)
        self.boost = int(boost)
        self.neg = int(negative_clauses)
        self.max_inc = int(self.L if max_included_literals is None else max_included_literals)
        self.T = int(T)
        self.q = float(q)
        self.s = tuple(float(x) for x in (s if isinstance(s, (tuple, list)) else (s,) * self.D))
        assert len(self.s) == self.D
        self.seed = int(seed)
        self.rho = float(rho)
        self.senders = int(senders)  # 0: all clauses send messages (GraphTM); 1: only clauses with a positive layer-0 literal
        self.layered = int(layered)  # 1: forget only from the first layer at which a non-firing clause died


def save_model(path, cfg, hv, w, ta, step):
    with open(path, "wb") as f:
        f.write(struct.pack(GTMM_HDR, b"GTMM", 2, cfg.C, cfg.O, cfg.H, cfg.NT, cfg.D, cfg.MS,
                            cfg.MB, cfg.B, cfg.boost, cfg.neg, cfg.max_inc, cfg.T, cfg.senders | (cfg.layered << 1)))
        f.write(struct.pack("<d", cfg.q))
        f.write(struct.pack("<%dd" % cfg.D, *cfg.s))
        f.write(struct.pack("<d", cfg.rho))  # v2
        f.write(struct.pack("<QQ", cfg.seed & M64, step))
        f.write(np.ascontiguousarray(hv, dtype="<u4").tobytes())
        f.write(np.ascontiguousarray(w, dtype="<i4").tobytes())       # [O][C]
        for layer in ta:
            f.write(np.ascontiguousarray(layer, dtype="<u2").tobytes())  # [C][lits]


def load_model(path):
    with open(path, "rb") as f:
        buf = f.read()
    hs = struct.calcsize(GTMM_HDR)
    (mag, ver, C, O, H, NT, D, MS, MB, B, boost, neg, max_inc, T, senders) = struct.unpack(GTMM_HDR, buf[:hs])
    assert mag == b"GTMM" and ver in (1, 2)
    off = hs
    (q,) = struct.unpack_from("<d", buf, off); off += 8
    s = struct.unpack_from("<%dd" % D, buf, off); off += 8 * D
    rho = 1.0
    if ver >= 2:
        (rho,) = struct.unpack_from("<d", buf, off); off += 8
    seed, step = struct.unpack_from("<QQ", buf, off); off += 16
    cfg = ModelConfig(C, O, H, NT, D, MS, MB, B, boost, neg, max_inc, T, q, s, seed, rho, senders & 1, (senders >> 1) & 1)

    def take(n, dt):
        nonlocal off
        a = np.frombuffer(buf, dtype=dt, count=n, offset=off).copy()
        off += a.nbytes
        return a

    hv = take(C * MB, "<u4").reshape(C, MB)
    w = take(O * C, "<i4").reshape(O, C).astype(np.int64)
    ta = [take(C * cfg.L, "<u2").reshape(C, cfg.L).astype(np.int32)]
    for _ in range(D - 1):
        ta.append(take(C * cfg.M, "<u2").reshape(C, cfg.M).astype(np.int32))
    return cfg, hv, w, ta, step
