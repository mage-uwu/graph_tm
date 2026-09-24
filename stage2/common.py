"""
Stage 2 shared pieces: real text -> GraphTM graphs.

  - Tokenizer: bert-base-uncased WordPiece (the vocabulary bert-tiny uses), so the GraphTM and
    the bert-tiny baselines see identical tokens.
  - Node features: every token id gets a fixed random sparse hypervector (HV_BITS of H bits),
    written as literals [x | not x]. [MASK] is just another symbol. No teacher information
    enters the input.
  - Window graphs: one node per token of a +-WIN window; edges to the neighbors at distance
    1, 2 and 4 on each side, one edge type per (distance, side). Two message hops at depth 3
    therefore see +-8 tokens.
  - .gtmd writer that builds packed literal words and CSR edges with numpy (the stage-0
    GraphBuilder is a per-node Python loop, far too slow for millions of graphs).

The format is the one defined in stage0/gtmcore.py; nothing here changes engine semantics.
"""
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "stage0"))
from gtmcore import GTMD_HDR  # noqa: E402

DATA = os.environ.get("GTM_STAGE2_DATA", os.path.join(ROOT, "data", "stage2"))
GTM = os.path.join(ROOT, "stage1", "gtm")

TOKENIZER_REPO = "google/bert_uncased_L-2_H-128_A-2"  # bert-tiny; same vocab as bert-base-uncased
H = 128          # node hypervector bits
# bits set per token symbol. Sparse codes (3 of 128) make nearly every negated literal true for
# nearly every token, and the TM then learns "NOT bit k" clauses that fire everywhere; dense
# codes (H/2) make positive and negative literals equally informative. Env GTM_HV_BITS.
HV_BITS = int(os.environ.get("GTM_HV_BITS", 64))
HV_SEED = 7
WIN = 8          # tokens on each side of the window centre
# edge distances (env GTM_DISTS, e.g. "1,2,4"); in-degree is 2 * len(DISTS) away from the edges
DISTS = tuple(int(x) for x in os.environ.get("GTM_DISTS", "1,2,4").split(","))
N_EDGE_TYPES = 2 * len(DISTS)  # type 2k: to the left at DISTS[k], 2k+1: to the right
# GTM_LAYOUT=flat: no graph, one node per example whose features are the concatenated codes of
# the tokens at GTM_FLAT_OFFSETS (e.g. "-2:-1:1:2"). A depth-1 model on it sees exactly the
# local context that message passing would have to deliver: the upper bound for the TM head.
CODES = os.environ.get("GTM_CODES", "random")  # random | dist (learned co-occurrence codes)
LAYOUT = os.environ.get("GTM_LAYOUT", "window")
FLAT_OFFSETS = tuple(int(x) for x in os.environ.get("GTM_FLAT_OFFSETS", "-2:-1:1:2").split(":"))
# Node types: the window centre is type 1, context tokens type 0. A clause only fires on nodes
# of type clause % NT, so odd clauses can only fire at the centre ([MASK]) and must predict from
# the messages they receive there; even clauses are context-token detectors that send those
# messages. Without this, graph output = OR over 17 nodes makes nearly every clause fire on
# every window (a bias term), and feedback lands on random context nodes.
N_NODE_TYPES = 2


def fetch(repo, filename, repo_type="dataset", revision=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type=repo_type, revision=revision,
                           cache_dir=os.path.join(DATA, "hf"))


def tokenizer():
    from tokenizers import BertWordPieceTokenizer
    tok = BertWordPieceTokenizer(fetch(TOKENIZER_REPO, "vocab.txt", repo_type="model"), lowercase=True)
    return tok


def encode(tok, texts, batch=10000):
    """list of str -> list of int32 arrays, no [CLS]/[SEP]"""
    out = []
    for i in range(0, len(texts), batch):
        for e in tok.encode_batch(texts[i:i + batch], add_special_tokens=False):
            out.append(np.asarray(e.ids, dtype=np.int32))
    return out


def symbol_table(vocab_size, h=H, hv_bits=HV_BITS, seed=HV_SEED):
    """(V, hv_bits) bit positions per token id; deterministic in (vocab_size, h, hv_bits, seed)"""
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((vocab_size, h)), axis=1)[:, :hv_bits].astype(np.int64)


def literal_rows(feat_bool):
    """(n, F) bool features -> (n, W) uint64 packed literals [x | not x], W = ceil(2F / 64)"""
    n, F = feat_bool.shape
    W = (2 * F + 63) // 64
    lit = np.zeros((n, W * 64), dtype=bool)
    lit[:, :F] = feat_bool
    lit[:, F:2 * F] = ~feat_bool
    return np.packbits(lit.reshape(n, W, 64), axis=2, bitorder="little").reshape(n, W * 8).view("<u8")


# GTM_FEAT=ngram: BLT-style hashed n-gram token codes, made Tsetlin-friendly. Each component
# (whole-token hash x NG_WHOLE, char 3..6-grams of the token with boundary marks, a "##"
# continuation flag) gets a fixed random +-1 vector; the token code is the sign of their sum
# (a hyperdimensional majority bundle): dense (~50% ones, so negated literals stay informative)
# and similarity-preserving (tokens sharing n-grams share bits). No gradients, no teacher.
FEAT = os.environ.get("GTM_FEAT", "random")
NG_RANGE = (3, 6)
NG_WHOLE = float(os.environ.get("GTM_NG_WHOLE", 4))


def _hvec(key, h):
    import hashlib
    seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(seed).integers(0, 2, h, dtype=np.int8) * 2 - 1


def ngram_bits(vocab, h=H):
    out = np.zeros((len(vocab), h), dtype=bool)
    cache = {}
    for i, tok in enumerate(vocab):
        cont = tok.startswith("##")
        w = tok[2:] if cont else tok
        acc = NG_WHOLE * _hvec("W:" + tok, h).astype(np.float64)
        s = ("" if cont else "<") + w + ">"
        for n in range(NG_RANGE[0], NG_RANGE[1] + 1):
            for j in range(len(s) - n + 1):
                g = s[j:j + n]
                if g not in cache:
                    cache[g] = _hvec("G:" + g, h)
                acc += cache[g]
        if cont:
            acc += _hvec("F:cont", h)
        acc += 1e-3 * _hvec("T:" + tok, h)  # tie-break
        out[i] = acc > 0
    return out


def symbol_bits(vocab_size, h=H):
    """(V, h) bool code per token id"""
    if CODES == "dist":  # learned, gradient free, teacher free: see code_ceiling.py
        import code_ceiling as CC
        assert h == 128, "dist codes are 128 bits"
        flat = np.load(os.path.join(DATA, "wt103_train.npz"))["flat"]
        freq = np.bincount(flat, minlength=vocab_size)
        return CC.balanced_sign(CC.dist_embedding(flat, vocab_size, 40_000_000), freq)
    if FEAT == "ngram":
        p = os.path.join(DATA, f"ngram_bits_{h}_{NG_WHOLE:g}.npy")
        if not os.path.exists(p):
            vocab = [l.rstrip("\n") for l in open(fetch(TOKENIZER_REPO, "vocab.txt", repo_type="model"), encoding="utf-8")]
            np.save(p, ngram_bits(vocab[:vocab_size], h))
        return np.load(p)
    f = np.zeros((vocab_size, h), dtype=bool)
    np.put_along_axis(f, symbol_table(vocab_size, h), True, axis=1)
    return f


# GTM_MASK_EMPTY=1: [MASK] gets the all-zero code (x = 0, not x = 1). Centre clauses then hold
# no positive layer-0 literal, so with --senders pos they do not broadcast from the centre.
MASK_EMPTY = os.environ.get("GTM_MASK_EMPTY") == "1"


def symbol_rows(vocab_size, h=H, mask_id=None):
    """(V, W) packed literal rows, one per token id"""
    f = symbol_bits(vocab_size, h)
    if MASK_EMPTY and mask_id is not None:
        f[mask_id] = False
    return literal_rows(f)


def flat_graphs(seqs, centres, bits, pad_id, offsets=FLAT_OFFSETS, win=WIN):
    """one node per example: concatenated codes of the tokens at `offsets` from the centre
    (pad_id outside the sequence). Returns (npg, epn, edges, X, node_type, window_tokens, h)."""
    _, _, _, _, _, wt = window_graphs(seqs, centres, None, np.zeros((bits.shape[0], 1), np.uint64), win)
    ids = np.stack([np.where(wt[:, win + o] >= 0, wt[:, win + o], pad_id) for o in offsets], 1)
    feat = bits[ids].reshape(len(wt), -1)
    n = len(wt)
    return (np.ones(n, np.int64), np.zeros(n, np.int64), np.zeros((0, 2), np.int64), literal_rows(feat),
            np.zeros(n, np.uint32), wt, feat.shape[1])


def write_gtmd(path, h, n_node_types, n_edge_types, n_outputs, kind, npg, epn, edges, X, Y, node_type=None):
    npg = np.asarray(npg, dtype="<u4")
    epn = np.asarray(epn, dtype="<u4")
    tn = int(npg.sum())
    assert len(epn) == tn and X.shape[0] == tn
    W = (2 * h + 63) // 64
    assert X.shape[1] == W
    if node_type is None:
        node_type = np.zeros(tn, dtype="<u4")
    edges = np.asarray(edges, dtype="<u4").reshape(-1, 2)
    Y = np.asarray(Y, dtype="<i4").reshape(len(npg), n_outputs)
    with open(path, "wb") as f:
        f.write(struct.pack(GTMD_HDR, b"GTMD", 1, len(npg), h, n_node_types, n_edge_types, n_outputs,
                            kind, tn, len(edges), W, 0))
        for a, dt in ((npg, "<u4"), (node_type, "<u4"), (epn, "<u4"), (edges, "<u4"),
                      (X, "<u8"), (Y, "<i4")):
            f.write(np.ascontiguousarray(a, dtype=dt).tobytes())


def centre_policy(true, mask_id, seed, cand):
    """centre token shown to the model for each training window (GTM_MASK_POLICY=bert): BERT's
    80/10/10 -- [MASK] 80%, a random real token 10%, the true token 10% -- so centre clauses learn
    to use the word when it is visible (downstream features show it). Deterministic in seed."""
    rng = np.random.default_rng([seed, 8010])
    u = rng.random(len(true))
    return np.where(u < 0.8, mask_id, np.where(u < 0.9, rng.choice(cand, len(true)), true)).astype(np.int32)


MASK_POLICY = os.environ.get("GTM_MASK_POLICY", "mask")  # mask: centre always [MASK]; bert: 80/10/10


def window_graphs(seqs, centres, mask_id, rows, win=WIN, centre_ids=None):
    """Build window graphs.

    seqs:    list of int32 token arrays (paragraphs / sentences)
    centres: (N, 2) int array of (sequence index, position); the centre token is replaced by
             [MASK] when mask_id is not None, or by centre_ids (N,) when given
    rows:    (V, W) packed literal rows from symbol_rows
    Returns (npg, epn, edges, X, node_type, window_tokens) where window_tokens is (N, 2*win+1)
    int32 of the ORIGINAL tokens (-1 = outside the sequence) and the centre sits at column win."""
    N = len(centres)
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    offs = np.zeros(len(seqs) + 1, dtype=np.int64)
    offs[1:] = np.cumsum(lens)
    flat = np.concatenate(seqs) if len(seqs) else np.zeros(0, np.int32)
    si, pos = centres[:, 0].astype(np.int64), centres[:, 1].astype(np.int64)
    rel = np.arange(-win, win + 1)
    p = pos[:, None] + rel[None, :]
    ok = (p >= 0) & (p < lens[si][:, None])
    wt = np.where(ok, flat[np.clip(offs[si][:, None] + p, 0, max(len(flat) - 1, 0))], -1).astype(np.int32)
    toks = wt.copy()
    if centre_ids is not None:
        toks[:, win] = centre_ids
    elif mask_id is not None:
        toks[:, win] = mask_id
    # nodes = valid window cells, in order; local index = column - first valid column
    first = ok.argmax(1)
    npg = ok.sum(1)
    gi, col = np.nonzero(ok)
    local = col - first[gi]
    n_g = npg[gi]
    node_base = np.zeros(N + 1, dtype=np.int64)
    node_base[1:] = np.cumsum(npg)
    src_all, dst_all, typ_all = [], [], []
    for k, d in enumerate(DISTS):
        for side, sgn in ((0, -1), (1, 1)):
            dst = local + sgn * d
            m = (dst >= 0) & (dst < n_g)
            src_all.append(node_base[gi[m]] + local[m])
            dst_all.append(dst[m])
            typ_all.append(np.full(m.sum(), 2 * k + side))
    src = np.concatenate(src_all)
    order = np.argsort(src * N_EDGE_TYPES + np.concatenate(typ_all), kind="stable")
    edges = np.stack([np.concatenate(dst_all)[order], np.concatenate(typ_all)[order]], 1)
    epn = np.bincount(src, minlength=len(gi))
    X = rows[toks[gi, col]]
    node_type = (col == win).astype(np.uint32)
    return npg, epn, edges, X, node_type, wt


def gtm(*args, capture=True):
    cmd = [GTM] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def score_bits(model, gtmd, n_graphs, n_clauses, threads, tag):
    """clause-output bits per graph from `gtm score --bits`: (n_graphs, C) bool"""
    out = os.path.join(DATA, f"_bits_{tag}.u64")
    gtm("score", "--model", model, "--data", gtmd, "--threads", threads, "--bits", out)
    Cw = (n_clauses + 63) // 64
    w = np.fromfile(out, dtype="<u8").reshape(n_graphs, Cw)
    os.remove(out)
    return np.unpackbits(w.view(np.uint8), axis=1, bitorder="little")[:, :n_clauses].astype(bool)


def score_sums(model, gtmd, n_graphs, n_outputs, threads, tag):
    out = os.path.join(DATA, f"_sums_{tag}.i32")
    txt = gtm("score", "--model", model, "--data", gtmd, "--threads", threads, "--sums", out)
    s = np.fromfile(out, dtype="<i4").reshape(n_graphs, n_outputs)
    os.remove(out)
    return s, txt
