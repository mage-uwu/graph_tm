"""
Byte Latent Transformer (Pagnini et al., Meta 2024, arXiv:2412.09871) front end, made
Tsetlin-native (no gradients):

  BLT                                          here
  -------------------------------------------  ---------------------------------------------------
  byte embedding x_i (learned, 256 x h)        fixed random +-1 vector per byte value
  hash n-gram embeddings, n = 3..8,            the same n-grams ending at byte i, RollPolyHash into
    RollPolyHash into ~500K buckets (learned)    NB buckets per n, fixed random +-1 vector per bucket
  e_i = x_i + sum_n E_n(hash(g_i,n)), normed   e_i = x_i + sum_n E_n(...) (int sum)
  entropy patching: small byte LM, boundary    count-based order-4 char LM with backoff; boundary
    where H(x_i) > theta_g                       where H(x_i) > theta_g, theta_g set for PATCH_BYTES
  local encoder: cross-attention pools bytes   patch code = sign(sum of e_i over the patch): a
    into patch vectors                           majority bundle, dense and similarity-preserving
  latent global transformer over patches       the GraphTM over patch nodes (edges +-1/+-2)

MLM alignment: a window is the text of +-WIN WordPiece tokens around a target token (the same
(paragraph, position) samples as pretrain_data.py, so validation windows are identical). The
target token's characters become one [MASK] node; the text on each side is patched and the
nearest WIN patches per side become context nodes. Target = the teacher code of the token, as
before.

Layouts (GTM_LAYOUT): blt       entropy patches, graph as window_graphs (centre type 1)
                      blt_flat  one node: concatenated codes of the patches at FLAT_OFFSETS
                      tokhash   patches = the WordPiece tokens themselves (isolates patching)
"""
import os
import time

import numpy as np

import common as C

NGRAMS = (3, 4, 5, 6, 7, 8)
NB = 1 << 16                 # buckets per n-gram size (BLT ablates 100K-500K total)
PATCH_BYTES = float(os.environ.get("GTM_BLT_PATCH", 4.5))
ORDER = 4                    # entropy model context length
MASK_BYTE = 1                # sentinel inside the stream for the masked span
SEED = 12345


def _tables(h):
    rng = np.random.default_rng(SEED)
    byte_emb = rng.integers(0, 2, (256, h), dtype=np.int8) * 2 - 1
    ng_emb = rng.integers(0, 2, (len(NGRAMS), NB, h), dtype=np.int8) * 2 - 1
    return byte_emb, ng_emb


def _shift(A, j):
    """A delayed by j positions, zero-filled, same length"""
    n = len(A)
    return np.concatenate([np.zeros(min(j, n), A.dtype), A[:max(n - j, 0)]])


def to_bytes(s):
    return np.frombuffer(s.encode("utf-8", "replace"), dtype=np.uint8)


# ------------------------------------------------------------------ entropy model
def build_entropy_model(texts, max_bytes=30_000_000):
    """order-ORDER char model: entropy per seen context (sorted keys), order-2 dense fallback"""
    buf, n = [], 0
    for t in texts:
        b = to_bytes(t + "\n")
        buf.append(b)
        n += len(b)
        if n > max_bytes:
            break
    A = np.concatenate(buf).astype(np.int64)
    ctx = np.zeros(len(A), np.int64)
    for j in range(1, ORDER + 1):
        ctx += _shift(A, j) << (8 * (j - 1))
    pair, cnt = np.unique(ctx * 256 + A, return_counts=True)
    pc = pair // 256
    ukeys, start = np.unique(pc, return_index=True)
    tot = np.add.reduceat(cnt, start)
    p = cnt / np.repeat(tot, np.diff(np.append(start, len(cnt))))
    ent = -np.add.reduceat(p * np.log2(p), start)
    # order-2 dense fallback
    c2 = _shift(A, 1) + (_shift(A, 2) << 8)
    M = np.zeros((65536, 256))
    np.add.at(M, (c2, A), 1)
    P = M / np.maximum(M.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e2 = -np.nansum(np.where(P > 0, P * np.log2(P), 0), 1)
    e2[M.sum(1) == 0] = 8.0
    return {"keys": ukeys, "ent": ent, "tot": tot, "e2": e2}


def entropies(model, b):
    """next-byte entropy H(x_i | preceding ORDER bytes) for each position of stream b"""
    A = b.astype(np.int64)
    ctx = np.zeros(len(A), np.int64)
    for j in range(1, ORDER + 1):
        ctx += _shift(A, j) << (8 * (j - 1))
    k = model["keys"]
    i = np.clip(np.searchsorted(k, ctx), 0, len(k) - 1)
    hit = (k[i] == ctx) & (model["tot"][i] >= 4)
    c2 = ctx & 0xFFFF
    return np.where(hit, model["ent"][i], model["e2"][c2])


# ------------------------------------------------------------------ byte codes
def byte_vectors(b, byte_emb, ng_emb):
    """e_i = x_i + sum_n E_n(RollPolyHash(g_i,n)) for every byte of stream b: (len, h) int16"""
    A = b.astype(np.uint64)
    e = byte_emb[b].astype(np.int16)
    for k, n in enumerate(NGRAMS):
        h = np.zeros(len(A), np.uint64)
        with np.errstate(over="ignore"):
            for j in range(n):  # polynomial rolling hash of the n bytes ending at i
                sh = _shift(A, j)
                h = h * np.uint64(1000003) + sh + np.uint64(1)
        valid = np.arange(len(A)) >= n - 1
        e[valid] += ng_emb[k, (h[valid] % np.uint64(NB)).astype(np.int64)]
    return e


def patch_code(vecs, tie):
    s = vecs.sum(0) + tie
    return s > 0


# ------------------------------------------------------------------ windows
def _load_texts(split):
    import pyarrow.parquet as pq
    import pretrain_data as PD
    lines = []
    for f in PD.SPLIT_FILES[split]:
        lines += pq.read_table(C.fetch(PD.WIKI, f)).column("text").to_pylist()
    return [s.strip() for s in lines if s.strip() and not s.strip().startswith("=")]


def prep(split):
    """kept paragraph texts aligned with wt103_<split>.npz (same filter as pretrain_data.prep)"""
    import fcntl
    with open(os.path.join(C.DATA, "blt_prep.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        _prep(split)


def _prep(split):
    p = os.path.join(C.DATA, f"blt_{split}_kept.npy")
    if not os.path.exists(p):
        tok = C.tokenizer()
        paras = _load_texts(split)
        lens = np.array([len(e.ids) for i in range(0, len(paras), 20000)
                         for e in tok.encode_batch(paras[i:i + 20000], add_special_tokens=False)])
        keep = np.nonzero(lens >= 2)[0]
        ref = np.load(os.path.join(C.DATA, f"wt103_{split}.npz"))["lens"]
        assert (lens[keep] == ref).all(), "paragraph alignment with wt103 npz failed"
        np.save(p, keep)
    mp = os.path.join(C.DATA, "blt_entropy.npz")
    if not os.path.exists(mp):
        paras = _load_texts("train")
        m = build_entropy_model(paras[::7])
        # global threshold for the target mean patch length, on held-out text
        ent = np.concatenate([entropies(m, to_bytes(t)) for t in paras[1::997][:3000]])
        theta = float(np.quantile(ent, 1 - 1 / PATCH_BYTES))
        np.savez(mp, theta=theta, **m)
        print(f"entropy model: {len(m['keys'])} contexts, theta={theta:.3f} bits for ~{PATCH_BYTES} bytes/patch", flush=True)


def load_entropy():
    z = np.load(os.path.join(C.DATA, "blt_entropy.npz"))
    return {k: z[k] for k in z.files}


def split_patches(b, ent, theta, layout):
    """boundary before position i when H(x_i) > theta (BLT global constraint); returns start offsets"""
    if len(b) == 0:
        return np.zeros(0, np.int64)
    cut = ent > theta
    cut[0] = True
    return np.nonzero(cut)[0]


def build(split, n, seed, out, codes, layout):
    import pretrain_data as PD
    prep(split)
    em = load_entropy()
    theta = float(em["theta"])
    keep = np.load(os.path.join(C.DATA, f"blt_{split}_kept.npy"))
    texts = _load_texts(split)
    lens = np.load(os.path.join(C.DATA, f"wt103_{split}.npz"))["lens"]
    z = np.load(os.path.join(C.DATA, f"wt103_{split}.npz"))
    flat, offs = z["flat"], np.concatenate([[0], np.cumsum(lens)])
    cent = PD.sample_centres(lens, n, seed)
    byte_emb, ng_emb = _tables(C.H)
    tie = (np.random.default_rng(SEED + 1).integers(0, 2, C.H) * 2 - 1).astype(np.int16)
    mask_code = np.random.default_rng(SEED + 2).integers(0, 2, C.H).astype(bool)
    tok = C.tokenizer()
    W = C.WIN
    feats, seq_nodes, centre_idx, true = [], [], [], []
    uniq, inv = np.unique(cent[:, 0], return_inverse=True)
    enc = {}
    for i in range(0, len(uniq), 20000):
        batch = [texts[keep[p]] for p in uniq[i:i + 20000]]
        for p, e in zip(uniq[i:i + 20000], tok.encode_batch(batch, add_special_tokens=False)):
            enc[int(p)] = e.offsets
    for (p, pos) in cent:
        p, pos = int(p), int(pos)
        text = texts[keep[p]]
        o = enc[p]
        assert len(o) == lens[p]
        true.append(int(flat[offs[p] + pos]))
        s0, s1 = o[pos]
        a = o[max(0, pos - W)][0]
        z1 = o[min(len(o) - 1, pos + W)][1]
        left, right = text[a:s0], text[s1:z1]
        stream = np.concatenate([to_bytes(left), np.array([MASK_BYTE], np.uint8), to_bytes(right)])
        vec = byte_vectors(stream, byte_emb, ng_emb)
        nl = len(to_bytes(left))
        if layout == "tokhash":
            # patches = the WordPiece tokens (byte spans from the tokenizer offsets)
            def spans(lo, hi, base):
                sp = []
                for q in range(lo, hi):
                    c0, c1 = o[q][0] - base, o[q][1] - base
                    sp.append((len(to_bytes(text[base:base + c0])), len(to_bytes(text[base:base + c1]))))
                return sp
            lsp = spans(max(0, pos - W), pos, a)
            rsp = [(x + nl + 1, y + nl + 1) for x, y in spans(pos + 1, min(len(o), pos + W + 1), s1)]
        else:
            ent = entropies(em, stream)
            def cuts(lo, hi):
                st = split_patches(stream[lo:hi], ent[lo:hi], theta, layout) + lo
                return list(zip(st, list(st[1:]) + [hi]))
            lsp = cuts(0, nl)[-W:]
            rsp = cuts(nl + 1, len(stream))[:W]
        nodes = [patch_code(vec[x:y], tie) for x, y in lsp] + [mask_code] + \
                [patch_code(vec[x:y], tie) for x, y in rsp]
        centre_idx.append(len(lsp))
        seq_nodes.append(len(nodes))
        feats.extend(nodes)
    true = np.array(true)
    feats = np.array(feats)
    if layout == "blt_flat":
        # one node: codes of the patches at FLAT_OFFSETS from the mask (pad = all-zero code)
        base = np.concatenate([[0], np.cumsum(seq_nodes)])
        rows = []
        for g in range(n):
            parts = []
            for off in C.FLAT_OFFSETS:
                j = centre_idx[g] + off
                parts.append(feats[base[g] + j] if 0 <= j < seq_nodes[g] else np.zeros(C.H, bool))
            rows.append(np.concatenate(parts))
        X = C.literal_rows(np.array(rows))
        h = C.H * len(C.FLAT_OFFSETS)
        C.write_gtmd(out, h, 1, 1, codes.shape[1], 1, np.ones(n), np.zeros(n), np.zeros((0, 2)), X, codes[true])
    else:
        seqs = []
        start = 0
        for k in seq_nodes:
            seqs.append(np.arange(start, start + k, dtype=np.int32))
            start += k
        rows = C.literal_rows(feats)
        npg, epn, edges, X, nt, _ = C.window_graphs(seqs, np.stack([np.arange(n), centre_idx], 1), None, rows)
        C.write_gtmd(out, C.H, C.N_NODE_TYPES, C.N_EDGE_TYPES, codes.shape[1], 1, npg, epn, edges, X, codes[true], nt)
    wt = np.full((n, 2 * W + 1), -1, np.int32)
    wt[:, W] = true
    return true, wt, {"nodes_per_graph": float(np.mean(seq_nodes)),
                      "bytes_per_patch": None}
