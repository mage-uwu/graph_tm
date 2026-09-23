/*
 * gtm.c - CPU-native Graph Tsetlin Machine. See gtm.h for layout notes and
 * stage0/gtmcore.py for the RNG + file-format spec this must match exactly.
 */
#define _GNU_SOURCE
#include "gtm.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#if defined(__x86_64__)
#include <immintrin.h>
#define CPU_RELAX() _mm_pause()
#else
#define CPU_RELAX() ((void)0)
#endif

#define M32 0xFFFFFFFFull
#define ALIGN 64

static void *xalloc(size_t bytes) {
    size_t n = (bytes + ALIGN - 1) / ALIGN * ALIGN;
    if (n == 0) n = ALIGN;
    void *p = aligned_alloc(ALIGN, n);
    if (!p) { fprintf(stderr, "gtm: out of memory (%zu bytes)\n", bytes); exit(1); }
    memset(p, 0, n);
    return p;
}

/* ------------------------------------------------------------------------- */
/* RNG spec (must match gtmcore.py)                                          */
/* ------------------------------------------------------------------------- */
static inline uint64_t splitmix64(uint64_t x) {
    uint64_t z = x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline uint64_t gkey(uint64_t seed, uint64_t tag, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t h = splitmix64(seed ^ (tag * 0xD6E8FEB86659FD93ull));
    h = splitmix64(h ^ a);
    h = splitmix64(h ^ b);
    return splitmix64(h ^ c);
}

enum { TAG_INIT_W = 1, TAG_NODE_SEL = 2, TAG_UPD_SEL = 3, TAG_FEEDBACK = 4, TAG_CLAUSE_HV = 5 };

static inline uint32_t lowbias32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

static __attribute__((unused)) uint64_t prob_threshold(double p) {
    double t = floor(p * 4294967296.0);
    if (t >= 4294967296.0) return 1ull << 32;
    if (t <= 0.0) return 0;
    return (uint64_t)t;
}

/* 64 feedback bits for literals [w*64, w*64+64): bit j set iff lowbias32(base+i)+1 <= thr. */
#if defined(__AVX512F__)
static inline __m512i lowbias32_v(__m512i x) {
    x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
    x = _mm512_mullo_epi32(x, _mm512_set1_epi32(0x7FEB352D));
    x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 15));
    x = _mm512_mullo_epi32(x, _mm512_set1_epi32((int)0x846CA68Bu));
    x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
    return x;
}
#endif

static uint64_t prob_threshold16(double p) {
    double t = floor(p * 65536.0);
    if (t >= 65536.0) return 1ull << 16;
    if (t <= 0.0) return 0;
    return (uint64_t)t;
}

/*
 * 16-bit coin flips, two per 32-bit hash (spec in gtmcore.py): draw i of a stream based at
 * `base` is  (lowbias32(base + ((i >> 5) << 4) + (i & 15)) >> 16*((i >> 4) & 1)) & 0xFFFF,
 * a hit iff < thr16. Word w of a stream covers draws [64w, 64w + 64): two 16-lane hashes.
 */
#if defined(__AVX512F__)
static inline void draws64(uint32_t base, uint32_t w, uint32_t thr_a, uint32_t thr_b, uint64_t *ma, uint64_t *mb) {
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i lo16 = _mm512_set1_epi32(0xFFFF);
    const __m512i ta = _mm512_set1_epi32((int)thr_a), tb = _mm512_set1_epi32((int)thr_b);
    uint64_t a = 0, b = 0;
    for (uint32_t q = 0; q < 2; q++) {
        const __m512i u = lowbias32_v(_mm512_add_epi32(_mm512_set1_epi32((int)(base + (2 * w + q) * 16)), iota));
        const __m512i lo = _mm512_and_si512(u, lo16), hi = _mm512_srli_epi32(u, 16);
        a |= ((uint64_t)_mm512_cmplt_epu32_mask(lo, ta) | (uint64_t)_mm512_cmplt_epu32_mask(hi, ta) << 16) << (32 * q);
        if (mb) b |= ((uint64_t)_mm512_cmplt_epu32_mask(lo, tb) | (uint64_t)_mm512_cmplt_epu32_mask(hi, tb) << 16) << (32 * q);
    }
    *ma = a;
    if (mb) *mb = b;
}
#else
static inline void draws64(uint32_t base, uint32_t w, uint32_t thr_a, uint32_t thr_b, uint64_t *ma, uint64_t *mb) {
    uint64_t a = 0, b = 0;
    for (uint32_t j = 0; j < 64; j++) {
        const uint32_t i = w * 64 + j;
        const uint32_t u = (lowbias32(base + ((i >> 5) << 4) + (i & 15)) >> (16 * ((i >> 4) & 1))) & 0xFFFF;
        if (u < thr_a) a |= 1ull << j;
        if (u < thr_b) b |= 1ull << j;
    }
    *ma = a;
    if (mb) *mb = b;
}
#endif

/* 1/s feedback bits for literals [64w, 64w + 64) */
static inline uint64_t fb_word(uint32_t base, uint32_t w, uint64_t thr16) {
    if (thr16 >= (1ull << 16)) return ~0ull;
    uint64_t m;
    draws64(base, w, (uint32_t)thr16, 0, &m, NULL);
    return m;
}

/* selection: one draw per clause, two thresholds (weights, automata; thr2 <= thr1) */
static inline void sel_word(uint32_t base, uint64_t thr1, uint64_t thr2, uint64_t *m1, uint64_t *m2) {
    if (thr2 >= (1ull << 16)) { *m1 = *m2 = ~0ull; return; }
    draws64(base, 0, thr1 >= (1ull << 16) ? 0x10000u : (uint32_t)thr1, (uint32_t)thr2, m1, m2);
}

/* ------------------------------------------------------------------------- */
/* Bit-sliced Tsetlin automata: saturating +-1 on 64 automata at once        */
/* ------------------------------------------------------------------------- */
/* Plane-major layout for one clause and layer: plane b of word w at p[b * W + w]. Each carry
 * step runs across all W words of the layer, so it vectorizes (8 words = one AVX-512 op). */
static inline __attribute__((always_inline)) void layer_inc_impl(uint64_t *p, uint32_t B, uint32_t W, uint64_t *carry) {
    for (uint32_t b = 0; b < B; b++) {
        uint64_t *pb = p + (size_t)b * W, any = 0;
        for (uint32_t w = 0; w < W; w++) {
            const uint64_t nx = pb[w] & carry[w];
            pb[w] ^= carry[w];
            carry[w] = nx;
            any |= nx;
        }
        if (!any) return;
    }
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t w = 0; w < W; w++) p[(size_t)b * W + w] |= carry[w]; /* saturate at max */
}

static inline __attribute__((always_inline)) void layer_dec_impl(uint64_t *p, uint32_t B, uint32_t W, uint64_t *carry) {
    for (uint32_t b = 0; b < B; b++) {
        uint64_t *pb = p + (size_t)b * W, any = 0;
        for (uint32_t w = 0; w < W; w++) {
            const uint64_t nx = ~pb[w] & carry[w];
            pb[w] ^= carry[w];
            carry[w] = nx;
            any |= nx;
        }
        if (!any) return;
    }
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t w = 0; w < W; w++) p[(size_t)b * W + w] &= ~carry[w]; /* saturate at 0 */
}

#define LAYER_OP_DISPATCH(NAME, IMPL)                                                   \
    static void NAME(uint64_t *p, uint32_t B, uint32_t W, uint64_t *carry) {            \
        switch (W) {                                                                    \
        case 1: IMPL(p, B, 1, carry); break;                                            \
        case 2: IMPL(p, B, 2, carry); break;                                            \
        case 4: IMPL(p, B, 4, carry); break;                                            \
        case 8: IMPL(p, B, 8, carry); break;                                            \
        case 16: IMPL(p, B, 16, carry); break;                                          \
        default: IMPL(p, B, W, carry); break;                                           \
        }                                                                               \
    }
LAYER_OP_DISPATCH(layer_inc, layer_inc_impl)
LAYER_OP_DISPATCH(layer_dec, layer_dec_impl)

/* position of the r-th set bit of v (r < popcount(v)) */
static inline uint32_t select_bit(uint64_t v, uint64_t r) {
#if defined(__BMI2__)
    return (uint32_t)__builtin_ctzll(_pdep_u64(1ull << r, v));
#else
    while (r--) v &= v - 1;
    return (uint32_t)__builtin_ctzll(v);
#endif
}

/* ------------------------------------------------------------------------- */
/* Model                                                                     */
/* ------------------------------------------------------------------------- */
/*
 * Literal -> clause inverted index, one per layer. Literal vectors are always [x | not x]
 * (message layers by construction, layer 0 checked at data load), so with S = set bits of x
 * and Z = its zero bits, clause c is true iff  pos(c) is a subset of S  and  neg(c) misses S.
 *   sparse node (|S| small):  c is false if it includes "not x_i" for some i in S (row N[i]);
 *                             c with no positive literals (NP0) is then true; others need
 *                             their witness positive literal in S (row WP[i]) + a full check.
 *   dense node (|Z| small):   mirror image over Z with rows P, WN and NN0.
 * Cost per node ~ min(|S|,|Z|) row ORs + a few full checks, instead of one check per clause.
 * Maintained incrementally when include decisions flip; a thread only touches its own words.
 */
typedef struct {
    uint32_t Hl;                  /* literals per half */
    uint64_t *P, *N, *WP, *WN;    /* [Hl][Cw] */
    uint64_t *NP0, *NN0;          /* [Cw] */
    int32_t *wp, *wn;             /* [C] current witnesses, -1 if none */
} lit_index;

typedef struct {
    gtm_model m;
    uint64_t *tmask;              /* [NT][Cw] clause type masks */
    lit_index idx[GTM_MAX_DEPTH];
    uint64_t *lut;                /* [Cw][16 nibbles][16 values][MS/64] bundled clause hypervectors */
    uint64_t *lut8;               /* [Cw][8 bytes][256 values][MS/64], only if it fits in ~1 MB */
    int dense;                    /* GTM_DENSE=1: reference evaluator (A/B testing) */
} gtm_model_ext;

static uint64_t *tmask_of(const gtm_model *m) { return ((const gtm_model_ext *)m)->tmask; }
static const lit_index *index_of(const gtm_model *m, uint32_t l) { return &((const gtm_model_ext *)m)->idx[l]; }

static int32_t first_set(const uint64_t *v, uint32_t a, uint32_t b) {
    for (uint32_t i = a; i < b;) {
        uint64_t w = v[i >> 6] >> (i & 63);
        if (w) {
            uint32_t j = i + (uint32_t)__builtin_ctzll(w);
            return j < b ? (int32_t)j : -1;
        }
        i = (i | 63) + 1;
    }
    return -1;
}

static void set_witness(uint64_t *rows, uint64_t *none, int32_t *cur, int32_t nw, uint32_t Cw, uint32_t c) {
    const uint32_t cw = c >> 6;
    const uint64_t bit = 1ull << (c & 63);
    if (*cur == nw) return;
    if (*cur >= 0) rows[(size_t)*cur * Cw + cw] &= ~bit;
    if (nw >= 0) { rows[(size_t)nw * Cw + cw] |= bit; none[cw] &= ~bit; }
    else none[cw] |= bit;
    *cur = nw;
}

static void refresh_clause(gtm_model *m, uint32_t c) {
    gtm_model_ext *e = (gtm_model_ext *)m;
    const uint32_t Cw = m->Cw, cw = c >> 6;
    const uint64_t bit = 1ull << (c & 63);
    uint32_t total = 0;
    for (uint32_t l = 0; l < m->D; l++) {
        uint32_t W = m->Wl[l];
        const uint64_t *pl = m->ta[l] + (size_t)c * W * m->B;
        uint64_t *inc = m->inc[l] + (size_t)c * W;
        const uint64_t *v = m->valid[l];
        lit_index *ix = &e->idx[l];
        int changed = 0;
        for (uint32_t w = 0; w < W; w++) {
            const uint64_t nw = pl[(size_t)(m->B - 1) * W + w] & v[w];
            uint64_t diff = inc[w] ^ nw;
            if (diff) {
                changed = 1;
                inc[w] = nw;
                while (diff) {
                    const uint32_t i = w * 64 + (uint32_t)__builtin_ctzll(diff);
                    diff &= diff - 1;
                    if (i < ix->Hl) ix->P[(size_t)i * Cw + cw] ^= bit;
                    else ix->N[(size_t)(i - ix->Hl) * Cw + cw] ^= bit;
                }
            }
            total += (uint32_t)__builtin_popcountll(nw);
        }
        if (changed) {
            set_witness(ix->WP, ix->NP0, &ix->wp[c], first_set(inc, 0, ix->Hl), Cw, c);
            int32_t n = first_set(inc, ix->Hl, 2 * ix->Hl);
            set_witness(ix->WN, ix->NN0, &ix->wn[c], n < 0 ? -1 : n - (int32_t)ix->Hl, Cw, c);
        }
    }
    m->ninc[c] = total;
}

/* refresh only the words of layer l whose include bits (MSB plane) flipped. Witnesses are the
 * lowest included literal of each half; they only move if the old one was excluded or a lower
 * one was included, both visible in the flip diff, so the rescans are rare. */
#define GTM_MAX_WORDS 256 /* per layer: 16384 literals, i.e. H, MS <= 8192 */
static void refresh_words(gtm_model *m, uint32_t c, uint32_t l, const uint64_t *wordmask) {
    gtm_model_ext *e = (gtm_model_ext *)m;
    const uint32_t Cw = m->Cw, cw = c >> 6, W = m->Wl[l];
    const uint64_t bit = 1ull << (c & 63);
    const uint64_t *pl = m->ta[l] + (size_t)c * W * m->B;
    uint64_t *inc = m->inc[l] + (size_t)c * W;
    lit_index *ix = &e->idx[l];
    const uint32_t Hl = ix->Hl;
    int pos_dirty = 0, neg_dirty = 0;
    for (uint32_t wm = 0; wm < (W + 63) / 64; wm++)
    for (uint64_t bits = wordmask[wm]; bits; bits &= bits - 1) {
        const uint32_t w = wm * 64 + (uint32_t)__builtin_ctzll(bits);
        const uint64_t nw = pl[(size_t)(m->B - 1) * W + w] & m->valid[l][w];
        uint64_t diff = inc[w] ^ nw;
        if (!diff) continue;
        m->ninc[c] += (uint32_t)__builtin_popcountll(nw) - (uint32_t)__builtin_popcountll(inc[w]);
        inc[w] = nw;
        while (diff) {
            const uint32_t i = w * 64 + (uint32_t)__builtin_ctzll(diff);
            const int now_in = (nw >> (i & 63)) & 1;
            diff &= diff - 1;
            if (i < Hl) {
                ix->P[(size_t)i * Cw + cw] ^= bit;
                const int32_t wp = ix->wp[c];
                if (now_in ? (wp < 0 || (int32_t)i < wp) : (int32_t)i == wp) pos_dirty = 1;
            } else {
                ix->N[(size_t)(i - Hl) * Cw + cw] ^= bit;
                const int32_t wn = ix->wn[c];
                if (now_in ? (wn < 0 || (int32_t)(i - Hl) < wn) : (int32_t)(i - Hl) == wn) neg_dirty = 1;
            }
        }
    }
    if (pos_dirty) set_witness(ix->WP, ix->NP0, &ix->wp[c], first_set(inc, 0, Hl), Cw, c);
    if (neg_dirty) {
        const int32_t n = first_set(inc, Hl, 2 * Hl);
        set_witness(ix->WN, ix->NN0, &ix->wn[c], n < 0 ? -1 : n - (int32_t)Hl, Cw, c);
    }
}

static void build_lut(gtm_model *m) {
    gtm_model_ext *e = (gtm_model_ext *)m;
    if (m->D < 2) return;
    const uint32_t half = m->Wl[1] / 2;
    free(e->lut8);
    e->lut8 = NULL;
    const size_t bytes8 = (size_t)m->Cw * 8 * 256 * half * 8;
    const char *env = getenv("GTM_LUT8");
    if (env ? atoi(env) > 0 : bytes8 <= (1u << 20)) {
        e->lut8 = xalloc(bytes8);
        for (uint32_t cw = 0; cw < m->Cw; cw++)
            for (uint32_t by = 0; by < 8; by++)
                for (uint32_t v = 0; v < 256; v++) {
                    uint64_t *ent = e->lut8 + (((size_t)cw * 8 + by) * 256 + v) * half;
                    for (uint32_t b = 0; b < 8; b++) {
                        const uint32_t c = cw * 64 + by * 8 + b;
                        if (!((v >> b) & 1) || c >= m->C) continue;
                        for (uint32_t k = 0; k < m->MB; k++) {
                            const uint32_t p = m->hv[(size_t)c * m->MB + k];
                            ent[p >> 6] |= 1ull << (p & 63);
                        }
                    }
                }
    }
    free(e->lut);
    e->lut = xalloc((size_t)m->Cw * 256 * half * 8);
    for (uint32_t cw = 0; cw < m->Cw; cw++)
        for (uint32_t nib = 0; nib < 16; nib++)
            for (uint32_t v = 0; v < 16; v++) {
                uint64_t *ent = e->lut + (((size_t)cw * 16 + nib) * 16 + v) * half;
                for (uint32_t b = 0; b < 4; b++) {
                    const uint32_t c = cw * 64 + nib * 4 + b;
                    if (!((v >> b) & 1) || c >= m->C) continue;
                    for (uint32_t k = 0; k < m->MB; k++) {
                        const uint32_t p = m->hv[(size_t)c * m->MB + k];
                        ent[p >> 6] |= 1ull << (p & 63);
                    }
                }
            }
}

gtm_model *gtm_new(const gtm_config *cfg) {
    if (cfg->O < 1 || cfg->O > 32768) { fprintf(stderr, "gtm: outputs must be 1..32768\n"); return NULL; }
    if (cfg->D < 1 || cfg->D > GTM_MAX_DEPTH) { fprintf(stderr, "gtm: depth must be 1..%d\n", GTM_MAX_DEPTH); return NULL; }
    if (cfg->B < 2 || cfg->B > 16) { fprintf(stderr, "gtm: state bits must be 2..16\n"); return NULL; }
    if (cfg->D > 1 && (cfg->MS == 0 || cfg->MS % 64)) { fprintf(stderr, "gtm: message size must be a positive multiple of 64\n"); return NULL; }
    if (cfg->MB > cfg->MS && cfg->D > 1) { fprintf(stderr, "gtm: message bits > message size\n"); return NULL; }
    if (cfg->NT < 1) { fprintf(stderr, "gtm: need >= 1 node type\n"); return NULL; }
    if (2 * cfg->H > 64 * GTM_MAX_WORDS || 2 * cfg->MS > 64 * GTM_MAX_WORDS) {
        fprintf(stderr, "gtm: at most %d literals per layer (H, MS <= %d)\n", 64 * GTM_MAX_WORDS, 32 * GTM_MAX_WORDS);
        return NULL;
    }

    gtm_model_ext *e = xalloc(sizeof(*e));
    gtm_model *m = &e->m;
    m->C = cfg->C; m->O = cfg->O; m->H = cfg->H; m->L = 2 * cfg->H; m->NT = cfg->NT;
    m->D = cfg->D; m->MS = cfg->MS; m->M = 2 * cfg->MS; m->MB = cfg->MB; m->B = cfg->B;
    m->boost = cfg->boost; m->neg = cfg->neg; m->max_inc = cfg->max_inc; m->T = cfg->T;
    m->q = cfg->q; m->seed = cfg->seed; m->step = 0;
    for (uint32_t l = 0; l < m->D; l++) m->s[l] = cfg->s[l];
    m->rho = cfg->rho > 0 ? cfg->rho : 1.0;
    m->senders = cfg->senders;
    m->layered = cfg->layered;

    m->Cw = (m->C + 63) / 64;
    m->Wl[0] = (m->L + 63) / 64;
    for (uint32_t l = 1; l < m->D; l++) m->Wl[l] = m->M / 64;
    for (uint32_t l = 0; l < m->D; l++) m->thr_s[l] = prob_threshold16(1.0 / m->s[l]);

    /* valid-literal masks */
    for (uint32_t l = 0; l < m->D; l++) {
        uint32_t W = m->Wl[l], bits = l == 0 ? m->L : m->M;
        m->valid[l] = xalloc(W * 8);
        for (uint32_t w = 0; w < W; w++) {
            uint32_t lo = w * 64;
            m->valid[l][w] = bits >= lo + 64 ? ~0ull : ((1ull << (bits - lo)) - 1);
        }
    }

    /* TA states: all planes but the MSB set -> 2^(B-1) - 1 */
    for (uint32_t l = 0; l < m->D; l++) {
        size_t words = (size_t)m->C * m->Wl[l];
        m->ta[l] = xalloc(words * m->B * 8);
        m->inc[l] = xalloc(words * 8);
        for (uint32_t c = 0; c < m->C; c++)
            for (uint32_t b = 0; b + 1 < m->B; b++)
                for (uint32_t w = 0; w < m->Wl[l]; w++) m->ta[l][((size_t)c * m->B + b) * m->Wl[l] + w] = ~0ull;
    }
    m->ninc = xalloc((size_t)m->C * 4);

    m->w = xalloc((size_t)m->O * m->Cw * 64 * 4);
    for (uint32_t k = 0; k < m->O; k++)
        for (uint32_t c = 0; c < m->C; c++)
            m->w[(size_t)k * m->Cw * 64 + c] = m->neg ? 1 - 2 * (int32_t)(gkey(m->seed, TAG_INIT_W, k, c, 0) & 1) : 1;

    m->hv = xalloc((size_t)m->C * (m->MB ? m->MB : 1) * 4);
    if (m->MS >= m->MB && m->MS > 0) { /* always generated, like the CUDA original */
        for (uint32_t c = 0; c < m->C; c++) {
            uint32_t got = 0;
            for (uint64_t j = 0; got < m->MB; j++) {
                uint32_t r = (uint32_t)(gkey(m->seed, TAG_CLAUSE_HV, c, j, 0) % m->MS);
                int dup = 0;
                for (uint32_t i = 0; i < got; i++) dup |= m->hv[c * m->MB + i] == r;
                if (!dup) m->hv[c * m->MB + got++] = r;
            }
        }
    }

    e->tmask = xalloc((size_t)m->NT * m->Cw * 8);
    for (uint32_t c = 0; c < m->C; c++) e->tmask[(size_t)(c % m->NT) * m->Cw + c / 64] |= 1ull << (c % 64);

    /* empty clauses: no includes anywhere */
    for (uint32_t l = 0; l < m->D; l++) {
        lit_index *ix = &e->idx[l];
        ix->Hl = l == 0 ? m->H : m->MS;
        const size_t rows = (size_t)ix->Hl * m->Cw * 8;
        ix->P = xalloc(rows); ix->N = xalloc(rows); ix->WP = xalloc(rows); ix->WN = xalloc(rows);
        ix->NP0 = xalloc((size_t)m->Cw * 8); ix->NN0 = xalloc((size_t)m->Cw * 8);
        ix->wp = xalloc((size_t)m->C * 4); ix->wn = xalloc((size_t)m->C * 4);
        for (uint32_t c = 0; c < m->C; c++) {
            ix->wp[c] = ix->wn[c] = -1;
            ix->NP0[c / 64] |= 1ull << (c % 64);
            ix->NN0[c / 64] |= 1ull << (c % 64);
        }
    }
    build_lut(m);
    { const char *d = getenv("GTM_DENSE"); e->dense = d && atoi(d) > 0; }

    return m;
}

void gtm_free(gtm_model *m) {
    if (!m) return;
    for (uint32_t l = 0; l < m->D; l++) { free(m->ta[l]); free(m->inc[l]); free(m->valid[l]); }
    gtm_model_ext *e = (gtm_model_ext *)m;
    for (uint32_t l = 0; l < m->D; l++) {
        lit_index *ix = &e->idx[l];
        free(ix->P); free(ix->N); free(ix->WP); free(ix->WN); free(ix->NP0); free(ix->NN0); free(ix->wp); free(ix->wn);
    }
    free(e->lut); free(e->lut8);
    free(m->ninc); free(m->w); free(m->hv); free(tmask_of(m));
    free(m);
}

uint32_t gtm_ta_state(const gtm_model *m, int layer, uint32_t c, uint32_t lit) {
    uint32_t W = m->Wl[layer];
    const uint64_t *p = m->ta[layer] + (size_t)c * W * m->B + lit / 64;
    uint32_t s = 0;
    for (uint32_t b = 0; b < m->B; b++) s |= (uint32_t)((p[(size_t)b * W] >> (lit % 64)) & 1) << b;
    return s;
}

static void ta_set_state(gtm_model *m, int layer, uint32_t c, uint32_t lit, uint32_t s) {
    uint32_t W = m->Wl[layer];
    uint64_t *p = m->ta[layer] + (size_t)c * W * m->B + lit / 64;
    uint64_t bit = 1ull << (lit % 64);
    for (uint32_t b = 0; b < m->B; b++) p[(size_t)b * W] = (s >> b) & 1 ? (p[(size_t)b * W] | bit) : (p[(size_t)b * W] & ~bit);
}

uint64_t gtm_state_hash(const gtm_model *m) {
    uint64_t h = splitmix64(m->step);
    for (uint32_t l = 0; l < m->D; l++) {
        uint32_t n = l == 0 ? m->L : m->M;
        for (uint32_t c = 0; c < m->C; c++)
            for (uint32_t i = 0; i < n; i++) h = splitmix64(h ^ gtm_ta_state(m, (int)l, c, i));
    }
    for (uint32_t c = 0; c < m->C; c++)
        for (uint32_t k = 0; k < m->O; k++) h = splitmix64(h ^ (uint32_t)m->w[(size_t)k * m->Cw * 64 + c]);
    return h;
}

/* ------------------------------------------------------------------------- */
/* Model IO (.gtmm)                                                          */
/* ------------------------------------------------------------------------- */
#pragma pack(push, 1)
typedef struct {
    char magic[4];
    uint32_t ver, C, O, H, NT, D, MS, MB, B, boost, neg, max_inc;
    int32_t T;
    uint32_t pad;
} gtmm_hdr;
typedef struct {
    char magic[4];
    uint32_t ver, n_graphs, H, NT, NET, O, kind;
    uint64_t total_nodes, total_edges;
    uint32_t W, reserved;
} gtmd_hdr;
#pragma pack(pop)

int gtm_save(const gtm_model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    gtmm_hdr h = {{'G', 'T', 'M', 'M'}, 2, m->C, m->O, m->H, m->NT, m->D, m->MS, m->MB, m->B,
                  m->boost, m->neg, m->max_inc, m->T, m->senders | (m->layered << 1)};
    fwrite(&h, sizeof h, 1, f);
    fwrite(&m->q, 8, 1, f);
    fwrite(m->s, 8, m->D, f);
    fwrite(&m->rho, 8, 1, f);
    fwrite(&m->seed, 8, 1, f);
    fwrite(&m->step, 8, 1, f);
    fwrite(m->hv, 4, (size_t)m->C * m->MB, f);
    for (uint32_t k = 0; k < m->O; k++) fwrite(&m->w[(size_t)k * m->Cw * 64], 4, m->C, f);
    for (uint32_t l = 0; l < m->D; l++) {
        uint32_t n = l == 0 ? m->L : m->M;
        uint16_t *row = malloc((size_t)n * 2);
        for (uint32_t c = 0; c < m->C; c++) {
            for (uint32_t i = 0; i < n; i++) row[i] = (uint16_t)gtm_ta_state(m, (int)l, c, i);
            fwrite(row, 2, n, f);
        }
        free(row);
    }
    fclose(f);
    return 0;
}

static void *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = xalloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

gtm_model *gtm_load(const char *path) {
    size_t len;
    uint8_t *buf = slurp(path, &len);
    if (!buf) return NULL;
    gtmm_hdr h;
    memcpy(&h, buf, sizeof h);
    if (memcmp(h.magic, "GTMM", 4) || (h.ver != 1 && h.ver != 2)) { fprintf(stderr, "%s: not a gtmm v1/v2 file\n", path); free(buf); return NULL; }
    size_t off = sizeof h;
    gtm_config cfg = {h.C, h.O, h.H, h.NT, h.D, h.MS, h.MB, h.B, h.boost, h.neg, h.max_inc, h.T, 0, {0}, 0, 1.0, h.pad & 1, (h.pad >> 1) & 1};
    memcpy(&cfg.q, buf + off, 8); off += 8;
    memcpy(cfg.s, buf + off, 8 * (size_t)h.D); off += 8 * (size_t)h.D;
    if (h.ver >= 2) { memcpy(&cfg.rho, buf + off, 8); off += 8; }
    memcpy(&cfg.seed, buf + off, 8); off += 8;
    uint64_t step;
    memcpy(&step, buf + off, 8); off += 8;
    gtm_model *m = gtm_new(&cfg);
    if (!m) { free(buf); return NULL; }
    m->step = step;
    memcpy(m->hv, buf + off, (size_t)m->C * m->MB * 4); off += (size_t)m->C * m->MB * 4;
    build_lut(m);
    for (uint32_t k = 0; k < m->O; k++) { memcpy(&m->w[(size_t)k * m->Cw * 64], buf + off, (size_t)m->C * 4); off += (size_t)m->C * 4; }
    for (uint32_t l = 0; l < m->D; l++) {
        uint32_t n = l == 0 ? m->L : m->M;
        for (uint32_t c = 0; c < m->C; c++)
            for (uint32_t i = 0; i < n; i++) {
                uint16_t s;
                memcpy(&s, buf + off, 2); off += 2;
                ta_set_state(m, (int)l, c, i, s);
            }
    }
    for (uint32_t c = 0; c < m->C; c++) refresh_clause(m, c);
    if (off != len) fprintf(stderr, "%s: warning, %zu trailing bytes\n", path, len - off);
    free(buf);
    return m;
}

/* ------------------------------------------------------------------------- */
/* Data IO (.gtmd)                                                           */
/* ------------------------------------------------------------------------- */
gtm_data *gtm_data_load(const char *path) {
    size_t len;
    uint8_t *buf = slurp(path, &len);
    if (!buf) return NULL;
    gtmd_hdr h;
    memcpy(&h, buf, sizeof h);
    if (memcmp(h.magic, "GTMD", 4) || h.ver != 1) { fprintf(stderr, "%s: not a gtmd v1 file\n", path); free(buf); return NULL; }
    gtm_data *d = xalloc(sizeof *d);
    d->blob = buf;
    d->n_graphs = h.n_graphs; d->H = h.H; d->W = h.W; d->NT = h.NT; d->NET = h.NET; d->O = h.O;
    d->kind = h.kind; d->total_nodes = h.total_nodes; d->total_edges = h.total_edges;
    size_t off = sizeof h;
#define TAKE(ptr, type, count) do { ptr = (type *)(buf + off); off += sizeof(type) * (size_t)(count); } while (0)
    TAKE(d->npg, uint32_t, h.n_graphs);
    TAKE(d->ntype, uint32_t, h.total_nodes);
    TAKE(d->epn, uint32_t, h.total_nodes);
    TAKE(d->edges, uint32_t, 2 * h.total_edges);
    /* X must be 8-byte aligned for uint64 access; copy to be safe */
    uint64_t *X = xalloc(h.total_nodes * h.W * 8);
    memcpy(X, buf + off, h.total_nodes * h.W * 8);
    off += h.total_nodes * h.W * 8;
    d->X = X;
    TAKE(d->Y, int32_t, (size_t)h.n_graphs * h.O);
#undef TAKE
    if (off != len) { fprintf(stderr, "%s: size mismatch (%zu vs %zu)\n", path, off, len); gtm_data_free(d); return NULL; }

    for (uint64_t i = 0; i < d->total_nodes; i++)
        if (d->ntype[i] >= d->NT) { fprintf(stderr, "%s: node %lu has type %u >= %u\n", path, (unsigned long)i, d->ntype[i], d->NT); exit(1); }
    d->comp = 1;
    for (uint64_t i = 0; i < d->total_nodes && d->comp; i++) {
        const uint64_t *x = d->X + i * d->W;
        for (uint32_t k = 0; k < d->H; k++) {
            const int a = (x[k >> 6] >> (k & 63)) & 1, b = (x[(k + d->H) >> 6] >> ((k + d->H) & 63)) & 1;
            if (a == b) { d->comp = 0; break; }
        }
    }
    d->node_off = xalloc(((size_t)d->n_graphs + 1) * 8);
    d->max_nodes = 0;
    for (uint32_t g = 0; g < d->n_graphs; g++) {
        d->node_off[g + 1] = d->node_off[g] + d->npg[g];
        if (d->npg[g] > d->max_nodes) d->max_nodes = d->npg[g];
    }
    d->edge_off = xalloc((d->total_nodes + 1) * 8);
    for (uint64_t i = 0; i < d->total_nodes; i++) d->edge_off[i + 1] = d->edge_off[i] + d->epn[i];

    /* inbound adjacency, per graph, ordered by source */
    d->in_off = xalloc((d->total_nodes + 1) * 8);
    d->in_edges = xalloc(d->total_edges * 2 * 4 + 8);
    uint32_t *indeg = xalloc(d->total_nodes * 4 + 4);
    for (uint32_t g = 0; g < d->n_graphs; g++) {
        uint64_t n0 = d->node_off[g];
        for (uint32_t s = 0; s < d->npg[g]; s++)
            for (uint64_t e = d->edge_off[n0 + s]; e < d->edge_off[n0 + s + 1]; e++) {
                uint32_t dst = d->edges[2 * e];
                if (dst >= d->npg[g]) { fprintf(stderr, "%s: graph %u edge to node %u out of range\n", path, g, dst); exit(1); }
                indeg[n0 + dst]++;
            }
    }
    for (uint64_t i = 0; i < d->total_nodes; i++) d->in_off[i + 1] = d->in_off[i] + indeg[i];
    memset(indeg, 0, d->total_nodes * 4);
    for (uint32_t g = 0; g < d->n_graphs; g++) {
        uint64_t n0 = d->node_off[g];
        for (uint32_t s = 0; s < d->npg[g]; s++)
            for (uint64_t e = d->edge_off[n0 + s]; e < d->edge_off[n0 + s + 1]; e++) {
                uint32_t dst = d->edges[2 * e], t = d->edges[2 * e + 1];
                uint64_t slot = d->in_off[n0 + dst] + indeg[n0 + dst]++;
                d->in_edges[2 * slot] = s;
                d->in_edges[2 * slot + 1] = t;
            }
    }
    free(indeg);
    return d;
}

void gtm_data_free(gtm_data *d) {
    if (!d) return;
    free(d->X); free(d->node_off); free(d->edge_off); free(d->in_off); free(d->in_edges); free(d->blob);
    free(d);
}

static int check_compat(const gtm_model *m, const gtm_data *d) {
    if (m->H != d->H || m->Wl[0] != d->W) { fprintf(stderr, "gtm: hypervector size mismatch (model %u, data %u)\n", m->H, d->H); return 0; }
    if (m->NT != d->NT) { fprintf(stderr, "gtm: node type count mismatch (model %u, data %u)\n", m->NT, d->NT); return 0; }
    if (m->O != d->O) { fprintf(stderr, "gtm: output count mismatch (model %u, data %u)\n", m->O, d->O); return 0; }
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Evaluation kernels                                                        */
/* ------------------------------------------------------------------------- */

/* all(inc & ~x == 0) with the word count known at compile time where it matters */
static inline int match_n(const uint64_t *inc, const uint64_t *x, uint32_t W) {
    uint64_t acc = 0;
    for (uint32_t w = 0; w < W; w++) acc |= inc[w] & ~x[w];
    return acc == 0;
}
#define MATCH_FIXED(N) static inline int match_##N(const uint64_t *inc, const uint64_t *x) { \
    uint64_t acc = 0; for (uint32_t w = 0; w < N; w++) acc |= inc[w] & ~x[w]; return acc == 0; }
MATCH_FIXED(1) MATCH_FIXED(2) MATCH_FIXED(4) MATCH_FIXED(8) MATCH_FIXED(16)

#define EVAL_BODY(MATCH)                                                                 \
    for (uint32_t cw = cw0; cw < cw1; cw++) {                                             \
        uint32_t cbase = cw * 64;                                                         \
        for (uint32_t node = 0; node < n; node++) {                                       \
            const uint64_t *x = Xin + (size_t)node * W;                                   \
            uint64_t cand = tmask[(size_t)ntype[node] * Cw + cw];                         \
            if (layer) cand &= out[(size_t)node * Cw + cw];                               \
            uint64_t word = 0;                                                            \
            while (cand) {                                                                \
                int j = __builtin_ctzll(cand);                                            \
                cand &= cand - 1;                                                         \
                if (MATCH) word |= 1ull << j;                                             \
            }                                                                             \
            out[(size_t)node * Cw + cw] = word;                                           \
        }                                                                                 \
    }

/* reference evaluator: one full check per (candidate clause, node).
 * layer 0 writes out; layer d>0 ANDs into out (the conditional chain) */
static void eval_layer_dense(const gtm_model *m, const uint32_t *ntype, uint32_t n, uint32_t layer,
                             const uint64_t *Xin, uint32_t cw0, uint32_t cw1, uint64_t *out) {
    const uint32_t W = m->Wl[layer], Cw = m->Cw;
    const uint64_t *incs = m->inc[layer];
    const uint64_t *tmask = tmask_of(m);
#define INC_OF(j) (incs + (size_t)(cbase + (j)) * W)
    switch (W) {
    case 1: EVAL_BODY(match_1(INC_OF(j), x)) break;
    case 2: EVAL_BODY(match_2(INC_OF(j), x)) break;
    case 4: EVAL_BODY(match_4(INC_OF(j), x)) break;
    case 8: EVAL_BODY(match_8(INC_OF(j), x)) break;
    case 16: EVAL_BODY(match_16(INC_OF(j), x)) break;
    default: EVAL_BODY(match_n(INC_OF(j), x, W)) break;
    }
#undef INC_OF
}

static inline int match_any(const uint64_t *inc, const uint64_t *x, uint32_t W) {
    switch (W) {
    case 1: return match_1(inc, x);
    case 2: return match_2(inc, x);
    case 4: return match_4(inc, x);
    case 8: return match_8(inc, x);
    case 16: return match_16(inc, x);
    default: return match_n(inc, x, W);
    }
}

/* indexed evaluator (see lit_index). Chooses per node between the index and dense checks
 * by estimated cost, so it is never much worse than the reference. Results are identical. */
static void eval_layer_indexed(const gtm_model *m, const uint32_t *ntype, uint32_t n, uint32_t layer,
                               const uint64_t *Xin, uint32_t cw0, uint32_t cw1, uint64_t *out, uint64_t *scratch) {
    const uint32_t W = m->Wl[layer], Cw = m->Cw;
    const lit_index *ix = index_of(m, layer);
    const uint32_t Hl = ix->Hl, Hw = (Hl + 63) / 64, span = cw1 - cw0;
    const uint64_t hlast = (Hl & 63) ? ((1ull << (Hl & 63)) - 1) : ~0ull;
    const uint64_t *incs = m->inc[layer], *tmask = tmask_of(m);
    uint64_t *V = scratch, *Cd = scratch + Cw;
    if (!span) return;
    for (uint32_t node = 0; node < n; node++) {
        const uint64_t *x = Xin + (size_t)node * W;
        const uint64_t *tm = tmask + (size_t)ntype[node] * Cw;
        uint64_t *o = out + (size_t)node * Cw;

        uint32_t nbase = 0;
        for (uint32_t cw = cw0; cw < cw1; cw++) nbase += (uint32_t)__builtin_popcountll(layer ? tm[cw] & o[cw] : tm[cw]);
        if (!nbase) { for (uint32_t cw = cw0; cw < cw1; cw++) o[cw] = 0; continue; }

        uint32_t npos = 0;
        for (uint32_t w = 0; w < Hw; w++) npos += (uint32_t)__builtin_popcountll(x[w] & (w == Hw - 1 ? hlast : ~0ull));
        const int sparse = 2 * npos <= Hl;
        const uint32_t rows = sparse ? npos : Hl - npos;

        if ((uint64_t)nbase * W <= (uint64_t)rows * 2 * span) { /* few candidates: check them directly */
            for (uint32_t cw = cw0; cw < cw1; cw++) {
                uint64_t cand = layer ? tm[cw] & o[cw] : tm[cw], res = 0;
                while (cand) {
                    const int j = __builtin_ctzll(cand);
                    cand &= cand - 1;
                    if (match_any(incs + (size_t)(cw * 64 + j) * W, x, W)) res |= 1ull << j;
                }
                o[cw] = res;
            }
            continue;
        }

        memset(V + cw0, 0, (size_t)span * 8);
        memset(Cd + cw0, 0, (size_t)span * 8);
        const uint64_t *R = sparse ? ix->N : ix->P, *Wt = sparse ? ix->WP : ix->WN;
        for (uint32_t w = 0; w < Hw; w++) {
            uint64_t bits = (sparse ? x[w] : ~x[w]) & (w == Hw - 1 ? hlast : ~0ull);
            while (bits) {
                const size_t i = (size_t)(w * 64 + (uint32_t)__builtin_ctzll(bits));
                bits &= bits - 1;
                const uint64_t *r = R + i * Cw, *t = Wt + i * Cw;
                for (uint32_t cw = cw0; cw < cw1; cw++) { V[cw] |= r[cw]; Cd[cw] |= t[cw]; }
            }
        }
        const uint64_t *Z0 = sparse ? ix->NP0 : ix->NN0;
        for (uint32_t cw = cw0; cw < cw1; cw++) {
            const uint64_t ok = (layer ? tm[cw] & o[cw] : tm[cw]) & ~V[cw];
            uint64_t res = ok & Z0[cw], chk = ok & Cd[cw];
            while (chk) {
                const int j = __builtin_ctzll(chk);
                chk &= chk - 1;
                if (match_any(incs + (size_t)(cw * 64 + j) * W, x, W)) res |= 1ull << j;
            }
            o[cw] = res;
        }
    }
}

static void eval_layer(const gtm_model *m, const gtm_data *ds, const uint32_t *ntype, uint32_t n, uint32_t layer,
                       const uint64_t *Xin, uint32_t cw0, uint32_t cw1, uint64_t *out, uint64_t *scratch) {
    if (((const gtm_model_ext *)m)->dense || (layer == 0 && !ds->comp))
        eval_layer_dense(m, ntype, n, layer, Xin, cw0, cw1, out);
    else
        eval_layer_indexed(m, ntype, n, layer, Xin, cw0, cw1, out, scratch);
}

/*
 * Message passing, two phases.
 *
 * The CUDA exchange_messages sets bit (hv_c[b] + edge_type) % MS at the destination for
 * every clause c true at the source, per edge. Rotation by edge type is a permutation, so
 * it distributes over OR:  OR_c rot(hv_c, t) = rot(OR_c hv_c, t).  So:
 *   phase A (bundle_sources): P[src] = OR of hv_c over clauses true at src, unrotated.
 *   phase B (build_messages): msg[dst] = OR over inbound edges of rot(P[src], type).
 * Per-edge cost drops from (#fired clauses x MB) to MS/64 word ops, with no division.
 */
static void bundle_sources_scatter(const gtm_model *m, uint32_t s0, uint32_t s1, const uint64_t *out, uint64_t *P) {
    const uint32_t half = m->Wl[1] / 2, Cw = m->Cw, MB = m->MB;
    for (uint32_t src = s0; src < s1; src++) {
        uint64_t *p = P + (size_t)src * half;
        memset(p, 0, (size_t)half * 8);
        const uint64_t *o = out + (size_t)src * Cw;
        const uint64_t *np0 = index_of(m, 0)->NP0;
        for (uint32_t cw = 0; cw < Cw; cw++) {
            uint64_t bits = o[cw] & (m->senders ? ~np0[cw] : ~0ull);
            while (bits) {
                const uint32_t *hv = m->hv + (size_t)(cw * 64 + (uint32_t)__builtin_ctzll(bits)) * MB;
                bits &= bits - 1;
                for (uint32_t b = 0; b < MB; b++) p[hv[b] >> 6] |= 1ull << (hv[b] & 63);
            }
        }
    }
}

/* LUT bundling: one half-vector OR per nonzero group of 8 (or 4) clauses, accumulated in
 * registers when the half width is a compile-time constant, stopping once the bundle is all
 * ones (a saturated Bloom-style bundle cannot change any more). */
#define BUNDLE_BODY(HALF, GBITS, LUTP)                                                            \
    for (uint32_t src = s0; src < s1; src++) {                                                     \
        uint64_t acc[HALF > 0 ? HALF : 1];                                                          \
        const uint32_t hw = HALF > 0 ? HALF : half;                                                \
        uint64_t *accp = HALF > 0 ? acc : P + (size_t)src * half;                                  \
        for (uint32_t w = 0; w < hw; w++) accp[w] = 0;                                             \
        const uint64_t *o = out + (size_t)src * Cw;                                                \
        for (uint32_t cw = 0; cw < Cw; cw++) {                                                     \
            uint64_t f = o[cw] & (m->senders ? ~np0[cw] : ~0ull);                                  \
            if (!f) continue;                                                                      \
            const uint64_t *base = LUTP + (size_t)cw * (64 / GBITS) * (1u << GBITS) * hw;          \
            while (f) {                                                                            \
                const uint32_t g = (uint32_t)__builtin_ctzll(f) / GBITS;                           \
                const uint64_t *ent = base + ((size_t)g * (1u << GBITS) +                          \
                                              ((f >> (g * GBITS)) & ((1u << GBITS) - 1))) * hw;    \
                f &= ~((((uint64_t)1 << GBITS) - 1) << (g * GBITS));                               \
                for (uint32_t w = 0; w < hw; w++) accp[w] |= ent[w];                               \
            }                                                                                      \
            uint64_t all = ~0ull;                                                                  \
            for (uint32_t w = 0; w < hw; w++) all &= accp[w];                                      \
            if (all == ~0ull) break;                                                               \
        }                                                                                          \
        if (HALF > 0) memcpy(P + (size_t)src * half, acc, sizeof(uint64_t) * hw);                  \
    }

static void bundle_sources(const gtm_model *m, uint32_t s0, uint32_t s1, const uint64_t *out, uint64_t *P) {
    const gtm_model_ext *e = (const gtm_model_ext *)m;
    if (e->dense) { bundle_sources_scatter(m, s0, s1, out, P); return; }
    const uint32_t half = m->Wl[1] / 2, Cw = m->Cw;
    const uint64_t *np0 = index_of(m, 0)->NP0;
    if (e->lut8) {
        switch (half) {
        case 2: BUNDLE_BODY(2, 8, e->lut8) return;
        case 4: BUNDLE_BODY(4, 8, e->lut8) return;
        case 8: BUNDLE_BODY(8, 8, e->lut8) return;
        default: BUNDLE_BODY(0, 8, e->lut8) return;
        }
    }
    switch (half) {
    case 2: BUNDLE_BODY(2, 4, e->lut) return;
    case 4: BUNDLE_BODY(4, 4, e->lut) return;
    case 8: BUNDLE_BODY(8, 4, e->lut) return;
    default: BUNDLE_BODY(0, 4, e->lut) return;
    }
}

/* dst |= src rotated left by r bits within an n-word ring (bit p -> (p + r) mod 64n) */
static inline void rotate_or(uint64_t *dst, const uint64_t *src, uint32_t n, uint32_t r) {
    const uint32_t rw = r >> 6, rb = r & 63;
    for (uint32_t w = 0; w < n; w++) {
        uint32_t i = w >= rw ? w - rw : w + n - rw;
        if (rb == 0) {
            dst[w] |= src[i];
        } else {
            uint32_t j = i ? i - 1 : n - 1;
            dst[w] |= (src[i] << rb) | (src[j] >> (64 - rb));
        }
    }
}

static void build_messages(const gtm_model *m, const gtm_data *ds, uint64_t n0, uint32_t d0, uint32_t d1,
                           const uint64_t *P, uint64_t *msg) {
    const uint32_t WM = m->Wl[1], half = WM / 2, MS = m->MS;
    for (uint32_t dst = d0; dst < d1; dst++) {
        uint64_t *mv = msg + (size_t)dst * WM;
        memset(mv, 0, (size_t)half * 8);
        for (uint64_t e = ds->in_off[n0 + dst]; e < ds->in_off[n0 + dst + 1]; e++) {
            const uint32_t src = ds->in_edges[2 * e], et = ds->in_edges[2 * e + 1];
            rotate_or(mv, P + (size_t)src * half, half, et % MS);
        }
        for (uint32_t w = 0; w < half; w++) mv[half + w] = ~mv[w];
    }
}

/* graph-level clause outputs for words [cw0,cw1) and their weighted votes */
static void clause_votes(const gtm_model *m, uint32_t n, const uint64_t *out, uint32_t cw0, uint32_t cw1,
                         uint64_t *fired, int32_t *sums) {
    const uint32_t Cw = m->Cw, O = m->O;
    for (uint32_t cw = cw0; cw < cw1; cw++) {
        uint64_t f = 0;
        for (uint32_t node = 0; node < n; node++) f |= out[(size_t)node * Cw + cw];
        fired[cw] = f;
    }
    for (uint32_t k = 0; k < O; k++) {
        const int32_t *wk = m->w + (size_t)k * Cw * 64;
#if defined(__AVX512F__)
        __m512i acc = _mm512_setzero_si512();
        for (uint32_t cw = cw0; cw < cw1; cw++) {
            const uint64_t f = fired[cw];
            if (!f) continue;
            for (int q = 0; q < 4; q++) {
                const __mmask16 mk = (__mmask16)(f >> (16 * q));
                if (mk) acc = _mm512_mask_add_epi32(acc, mk, acc, _mm512_loadu_si512(wk + cw * 64 + 16 * q));
            }
        }
        sums[k] += _mm512_reduce_add_epi32(acc);
#else
        int32_t acc = 0;
        for (uint32_t cw = cw0; cw < cw1; cw++)
            for (uint64_t f = fired[cw]; f; f &= f - 1) acc += wk[cw * 64 + __builtin_ctzll(f)];
        sums[k] += acc;
#endif
    }
}

/* inference votes from a clause-major weight mirror [C][O]: cost ~ fired clauses x O/16,
 * independent of how many clause words there are */
static void clause_votes_rows(const gtm_model *m, const int32_t *wcm, uint32_t n, const uint64_t *out,
                              uint64_t *fired, int32_t *sums) {
    const uint32_t Cw = m->Cw, O = m->O;
    for (uint32_t cw = 0; cw < Cw; cw++) {
        uint64_t f = 0;
        for (uint32_t node = 0; node < n; node++) f |= out[(size_t)node * Cw + cw];
        fired[cw] = f;
        while (f) {
            const int32_t *row = wcm + (size_t)(cw * 64 + (uint32_t)__builtin_ctzll(f)) * O;
            f &= f - 1;
            for (uint32_t k = 0; k < O; k++) sums[k] += row[k];
        }
    }
}

/* select_clause_updates weight rule, for the clauses in `upd` (selected and fired). The CUDA
 * rule "target*sign > 0 ? w += sign : w -= sign" is always w += target (sign^2 = 1); the magnitude
 * guard applies when growing, and without negative clauses weights are clamped at 1. */
static inline void update_weights(int32_t *wv, uint64_t upd, int target, int neg) {
#if defined(__AVX512F__)
    const __m512i t = _mm512_set1_epi32(target), one = _mm512_set1_epi32(1), zero = _mm512_setzero_si512();
    const __m512i imax = _mm512_set1_epi32(INT_MAX), nimax = _mm512_set1_epi32(-INT_MAX);
    for (int q = 0; q < 4; q++) {
        __mmask16 mk = (__mmask16)(upd >> (16 * q));
        if (!mk) continue;
        __m512i v = _mm512_loadu_si512(wv + 16 * q);
        const __mmask16 pos = _mm512_cmpge_epi32_mask(v, zero);
        const __mmask16 grow = target > 0 ? pos : (__mmask16)~pos;
        const __mmask16 huge = _mm512_cmpeq_epi32_mask(v, imax) | _mm512_cmple_epi32_mask(v, nimax);
        mk &= (__mmask16)~(grow & huge);
        v = _mm512_mask_add_epi32(v, mk, v, t);
        if (!neg) v = _mm512_mask_max_epi32(v, mk, v, one);
        _mm512_storeu_si512(wv + 16 * q, v);
    }
#else
    while (upd) {
        const int j = __builtin_ctzll(upd);
        upd &= upd - 1;
        int32_t w = wv[j];
        const int grow = target == (w >= 0 ? 1 : -1);
        if (grow && (w == INT_MAX || w <= -INT_MAX)) continue;
        w += target;
        if (!neg && w < 1) w = 1;
        wv[j] = w;
    }
#endif
}

/* ------------------------------------------------------------------------- */
/* Learning                                                                  */
/* ------------------------------------------------------------------------- */

/* select_clause_updates, output side. The CUDA code draws two uniforms per (clause, output):
 * u0 vs q/(O-1) (negative targets only) and u1 vs err/2T. Both probabilities depend only on
 * the output, so each clause gets feedback from output k with probability p_k = e_k * q'_k,
 * independently. One Bernoulli(p_k) bit per pair has the same law and lets us draw 64 clauses
 * per output with a single vectorized hash (fb_word), instead of O*C scalar draws. */
static void output_feedback_plan(const gtm_model *m, uint64_t step, const int32_t *cs, const int32_t *yenc,
                                 int8_t *target, uint64_t *thr, uint64_t *thr_ta, uint64_t *hk) {
    const double qprob = fmin(1.0, m->q / (double)(m->O > 1 ? m->O - 1 : 1));
    const uint64_t hs = gkey(m->seed, TAG_UPD_SEL, step, 0, 0);
    for (uint32_t k = 0; k < m->O; k++) {
        const int32_t y = yenc[k], s = cs[k];
        target[k] = s > y ? -1 : 1;
        double p = (double)(y > s ? y - s : s - y) / (2.0 * (double)m->T);
        if (target[k] == -1) p = p * qprob;
        thr[k] = prob_threshold16(p);
        thr_ta[k] = prob_threshold16(p * m->rho);
        hk[k] = splitmix64(hs ^ k);
    }
}

/* one clause: node selection and sequential Type I / II feedback for the outputs in `ks`
 * (ascending; bit 15 set = Type II sign) that selected it for automaton feedback */
static void update_clause(gtm_model *m, uint64_t step, uint32_t c, uint32_t n, const uint64_t *out,
                          const uint64_t *X0, uint64_t *const *msg, const uint16_t *ks, uint32_t nk,
                          uint64_t nodemask, uint32_t dead_at) {
    if (nk == 0) return; /* counter-based RNG: skipping draws never shifts anyone else's */
    const uint32_t Cw = m->Cw, B = m->B;
    const uint32_t cw = c >> 6;
    const uint64_t bit = 1ull << (c & 63);

    /* select_clause_node: uniform over nodes where the clause is true. For n <= 64 the caller
     * passes this clause's node bitmask (transposed from `out`), otherwise scan. */
    int64_t sel = -1;
    if (n <= 64) {
        const uint32_t cnt = (uint32_t)__builtin_popcountll(nodemask);
        if (cnt) sel = select_bit(nodemask, gkey(m->seed, TAG_NODE_SEL, step, c, 0) % cnt);
    } else {
        uint32_t cnt = 0;
        for (uint32_t node = 0; node < n; node++) cnt += (out[(size_t)node * Cw + cw] & bit) != 0;
        if (cnt) {
            uint64_t r = gkey(m->seed, TAG_NODE_SEL, step, c, 0) % cnt;
            for (uint32_t node = 0; node < n; node++)
                if (out[(size_t)node * Cw + cw] & bit) {
                    if (r == 0) { sel = node; break; }
                    r--;
                }
        }
    }

    /* Type I / Type II feedback on every layer, against the selected node's input */
    const int fired = sel != -1;
    const int grow = fired && m->ninc[c] <= m->max_inc;
    uint64_t a[GTM_MAX_WORDS], d[GTM_MAX_WORDS]; /* per-word active lanes (bounded at model creation) */
    for (uint32_t l = 0; l < m->D; l++) {
        /* layered forget: a non-firing clause keeps the layers it still matched somewhere
         * (skipping their draws is exact: draws are indexed, not streamed) */
        if (!fired && m->layered && l < dead_at) continue;
        const uint32_t W = m->Wl[l];
        uint64_t *pl = m->ta[l] + (size_t)c * W * B;
        const uint64_t *msb = pl + (size_t)(B - 1) * W;
        const uint64_t *valid = m->valid[l];
        const uint64_t *x = fired ? (l == 0 ? X0 + (size_t)sel * W : msg[l] + (size_t)sel * W) : NULL;
        const uint64_t thr = m->thr_s[l];
        uint64_t hl = 0;
        int have_hl = 0;
        for (uint32_t i = 0; i < nk; i++) {
            const uint32_t k = ks[i] & 0x7FFF;
            if (!(ks[i] >> 15)) { /* Type I */
                if (!have_hl) { hl = gkey(m->seed, TAG_FEEDBACK, step, c, l); have_hl = 1; }
                const uint32_t base = (uint32_t)(splitmix64(hl ^ k) & M32);
                if (grow) {
                    uint64_t anyd = 0;
                    for (uint32_t w = 0; w < W; w++) {
                        const uint64_t xw = x[w], vx = xw & valid[w], nx = ~xw & valid[w];
                        /* feedback bits only matter on x=0 literals (and x=1 without boost);
                         * skipping the rest is exact because draws are indexed, not streamed */
                        const uint64_t fb = (nx || (!m->boost && vx)) ? fb_word(base, w, thr) : 0;
                        a[w] = m->boost ? vx : (vx & ~fb);
                        d[w] = nx & fb;
                        anyd |= d[w];
                    }
                    layer_inc(pl, B, W, a);
                    if (anyd) layer_dec(pl, B, W, d);
                } else { /* clause did not fire: forget */
                    for (uint32_t w = 0; w < W; w++) d[w] = fb_word(base, w, thr) & valid[w];
                    layer_dec(pl, B, W, d);
                }
            } else if (fired) { /* Type II */
                for (uint32_t w = 0; w < W; w++) a[w] = ~x[w] & ~msb[w] & valid[w];
                layer_inc(pl, B, W, a);
            }
        }
        /* include decisions that flipped: compare the MSB plane with the mirror */
        const uint64_t *inc = m->inc[l] + (size_t)c * W;
        uint64_t flips[GTM_MAX_WORDS / 64] = {0}, any = 0;
        for (uint32_t w = 0; w < W; w++)
            if ((msb[w] & valid[w]) != inc[w]) { flips[w >> 6] |= 1ull << (w & 63); any = 1; }
        if (any) refresh_words(m, c, l, flips);
    }
}

/* ------------------------------------------------------------------------- */
/* Threading                                                                 */
/* ------------------------------------------------------------------------- */
/*
 * Dissemination barrier: in round r thread t signals thread (t + 2^r) mod P and waits for
 * thread (t - 2^r) mod P. Every flag has exactly one writer (plain release stores of a
 * monotonically increasing episode number) and lives on its own cache line, so there is no
 * contended read-modify-write: log2(P) rounds instead of P serialized RMWs on one line.
 */
#define BAR_MAX_ROUNDS 16
typedef struct {
    _Alignas(64) _Atomic uint64_t v;
    char pad[64 - sizeof(uint64_t)];
} bar_flag;

typedef struct {
    uint32_t n, rounds;
    bar_flag *flags; /* [n][rounds] */
} spin_barrier;

static void barrier_init(spin_barrier *b, uint32_t n) {
    b->n = n;
    b->rounds = 0;
    while ((1u << b->rounds) < n) b->rounds++;
    b->flags = aligned_alloc(64, sizeof(bar_flag) * (size_t)n * (b->rounds ? b->rounds : 1));
    for (size_t i = 0; i < (size_t)n * (b->rounds ? b->rounds : 1); i++) atomic_init(&b->flags[i].v, 0);
}

static void barrier_free(spin_barrier *b) { free(b->flags); }

/* *episode is thread-local, starts at 0 */
static void barrier_wait(spin_barrier *b, uint32_t tid, uint64_t *episode) {
    if (b->n <= 1) return;
    const uint64_t e = ++*episode;
    for (uint32_t r = 0; r < b->rounds; r++) {
        const uint32_t partner = (tid + (1u << r)) % b->n;
        atomic_store_explicit(&b->flags[(size_t)partner * b->rounds + r].v, e, memory_order_release);
        _Atomic uint64_t *mine = &b->flags[(size_t)tid * b->rounds + r].v;
        if (atomic_load_explicit(mine, memory_order_acquire) >= e) continue;
        /* spin ~20 us, then yield: with threads <= cores the wait is usually sub-microsecond, and
         * if the partner was descheduled, spinning longer only burns its CPU */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint32_t spins = 0;
        int yielding = 0;
        while (atomic_load_explicit(mine, memory_order_acquire) < e) {
            if (yielding) { sched_yield(); continue; }
            CPU_RELAX();
            if ((++spins & 63) == 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                yielding = (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec) > 20000;
            }
        }
    }
}

typedef struct {
    gtm_model *m;
    const gtm_data *ds;
    uint32_t P;
    int64_t total;
    spin_barrier bar;
    uint64_t *out;                 /* [max_nodes][Cw] shared, clause-word partitioned */
    uint64_t *msg[GTM_MAX_DEPTH];  /* [max_nodes][WM] shared, node partitioned */
    uint64_t *bundle;              /* [max_nodes][WM/2] bundled source hypervectors */
    int32_t *partial;              /* [2][P][Opad] */
    uint32_t Opad;
} train_ctx;

typedef struct { train_ctx *ctx; uint32_t tid; } train_arg;

/* GTM_PIN=1: pin thread tid to cpu tid (mod online cpus). Performance only. */
static void maybe_pin(uint32_t tid) {
    static _Atomic int pin = -1; /* idempotent lazy init; atomic so concurrent first calls are defined */
    int p = atomic_load_explicit(&pin, memory_order_relaxed);
    if (p < 0) {
        const char *e = getenv("GTM_PIN");
        p = e && atoi(e) > 0;
        atomic_store_explicit(&pin, p, memory_order_relaxed);
    }
    if (!p) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_SET(tid % (uint32_t)(ncpu > 0 ? ncpu : 1), &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

/* clause-word range for thread tid: 8-word (512-clause, one cache line per node row) blocks
 * when there are enough of them, single words otherwise. Results never depend on this. */
static void clause_range(uint32_t Cw, uint32_t P, uint32_t tid, uint32_t *cw0, uint32_t *cw1) {
    const uint32_t unit = Cw >= 8 * P ? 8 : 1;
    const uint32_t units = (Cw + unit - 1) / unit;
    uint32_t u0 = (uint32_t)((uint64_t)tid * units / P), u1 = (uint32_t)((uint64_t)(tid + 1) * units / P);
    *cw0 = u0 * unit < Cw ? u0 * unit : Cw;
    *cw1 = u1 * unit < Cw ? u1 * unit : Cw;
}

/* OR over nodes of this thread's clause words after a layer: which clauses are still true somewhere */
static void layer_alive(const uint64_t *out, uint32_t n, uint32_t Cw, uint32_t cw0, uint32_t cw1, uint64_t *alive) {
    for (uint32_t cw = cw0; cw < cw1; cw++) alive[cw] = 0;
    for (uint32_t v = 0; v < n; v++)
        for (uint32_t cw = cw0; cw < cw1; cw++) alive[cw] |= out[(size_t)v * Cw + cw];
}

static void *train_worker(void *argp) {
    train_arg *a = argp;
    train_ctx *ctx = a->ctx;
    gtm_model *m = ctx->m;
    const gtm_data *ds = ctx->ds;
    const uint32_t P = ctx->P, tid = a->tid, O = m->O, Cw = m->Cw;
    uint32_t cw0, cw1;
    clause_range(Cw, P, tid, &cw0, &cw1);
    maybe_pin(tid);
    const uint32_t c0 = cw0 * 64, c1 = cw1 * 64 < m->C ? cw1 * 64 : m->C;
    int32_t *sums = xalloc((size_t)O * 4), *yenc = xalloc((size_t)O * 4);
    int8_t *target = xalloc(O);
    uint64_t *thr = xalloc((size_t)O * 8), *thr_ta = xalloc((size_t)O * 8), *hk = xalloc((size_t)O * 8);
    uint64_t *fired = xalloc((size_t)Cw * 8);
    const uint32_t own = c1 > c0 ? c1 - c0 : 0;
    uint16_t *lists = xalloc((size_t)(own ? own : 1) * O * 2);
    uint32_t *nlist = xalloc((size_t)(own ? own : 1) * 4);
    uint64_t *scratch = xalloc((size_t)2 * Cw * 8);
    uint64_t *nodemask = xalloc((size_t)(own ? own : 1) * 8);
    uint64_t *alive = xalloc((size_t)m->D * Cw * 8); /* [layer][cw]: clause true at some node (own words) */
    uint64_t sense = 0; /* barrier episode */
    const uint64_t step0 = m->step;

    for (int64_t i = 0; i < ctx->total; i++) {
        const uint32_t g = (uint32_t)(i % ds->n_graphs);
        const uint64_t n0 = ds->node_off[g];
        const uint32_t n = ds->npg[g];
        const uint64_t *X0 = ds->X + n0 * ds->W;
        const uint32_t *ntype = ds->ntype + n0;

        eval_layer(m, ds, ntype, n, 0, X0, cw0, cw1, ctx->out, scratch);
        if (m->layered) layer_alive(ctx->out, n, Cw, cw0, cw1, alive);
        for (uint32_t d = 1; d < m->D; d++) {
            const uint32_t s0 = (uint32_t)((uint64_t)tid * n / P), s1 = (uint32_t)((uint64_t)(tid + 1) * n / P);
            barrier_wait(&ctx->bar, tid, &sense);
            bundle_sources(m, s0, s1, ctx->out, ctx->bundle);
            barrier_wait(&ctx->bar, tid, &sense);
            build_messages(m, ds, n0, s0, s1, ctx->bundle, ctx->msg[d]);
            barrier_wait(&ctx->bar, tid, &sense);
            eval_layer(m, ds, ntype, n, d, ctx->msg[d], cw0, cw1, ctx->out, scratch);
            if (m->layered) layer_alive(ctx->out, n, Cw, cw0, cw1, alive + (size_t)d * Cw);
        }

        int32_t *mine = ctx->partial + ((size_t)(i & 1) * P + tid) * ctx->Opad;
        memset(mine, 0, (size_t)O * 4);
        clause_votes(m, n, ctx->out, cw0, cw1, fired, mine);
        if (n <= 64) { /* transpose this thread's clause outputs into per-clause node masks */
            memset(nodemask, 0, (size_t)own * 8);
            for (uint32_t v = 0; v < n; v++)
                for (uint32_t cw = cw0; cw < cw1; cw++)
                    for (uint64_t b = ctx->out[(size_t)v * Cw + cw]; b; b &= b - 1)
                        nodemask[cw * 64 + (uint32_t)__builtin_ctzll(b) - c0] |= 1ull << v;
        }
        barrier_wait(&ctx->bar, tid, &sense);

        memset(sums, 0, (size_t)O * 4);
        for (uint32_t t = 0; t < P; t++) {
            const int32_t *pt = ctx->partial + ((size_t)(i & 1) * P + t) * ctx->Opad;
            for (uint32_t k = 0; k < O; k++) sums[k] += pt[k];
        }
        const int32_t *Y = ds->Y + (size_t)g * O;
        for (uint32_t k = 0; k < O; k++) {
            if (sums[k] > m->T) sums[k] = m->T;
            else if (sums[k] < -m->T) sums[k] = -m->T;
            yenc[k] = Y[k] == 1 ? m->T : -m->T;
        }
        const uint64_t step = step0 + (uint64_t)i;
        output_feedback_plan(m, step, sums, yenc, target, thr, thr_ta, hk);
        memset(nlist, 0, (size_t)own * 4);
        const uint64_t last = (m->C & 63) ? (1ull << (m->C & 63)) - 1 : ~0ull;
        for (uint32_t k = 0; k < O; k++) {
            if (!thr[k]) continue;
            int32_t *wk = m->w + (size_t)k * Cw * 64;
            for (uint32_t cw = cw0; cw < cw1; cw++) {
                uint64_t sel, keep;
                sel_word((uint32_t)(splitmix64(hk[k] ^ cw) & M32), thr[k], thr_ta[k], &sel, &keep);
                if (cw == Cw - 1) { sel &= last; keep &= last; }
                if (!sel) continue;
                int32_t *wv = wk + cw * 64;
                while (keep) { /* automaton feedback sign uses the weight before its update */
                    const uint32_t j = (uint32_t)__builtin_ctzll(keep);
                    keep &= keep - 1;
                    const uint32_t r = cw * 64 + j - c0;
                    const int ts = target[k] * (wv[j] >= 0 ? 1 : -1);
                    lists[(size_t)r * O + nlist[r]++] = (uint16_t)(k | (ts < 0 ? 0x8000u : 0));
                }
                update_weights(wv, sel & fired[cw], target[k], (int)m->neg);
            }
        }
        for (uint32_t c = c0; c < c1; c++) {
            uint32_t dead_at = m->D;
            if (m->layered)
                for (uint32_t d = 0; d < m->D; d++)
                    if (!(alive[(size_t)d * Cw + (c >> 6)] >> (c & 63) & 1)) { dead_at = d; break; }
            update_clause(m, step, c, n, ctx->out, X0, ctx->msg, lists + (size_t)(c - c0) * O, nlist[c - c0],
                          n <= 64 ? nodemask[c - c0] : 0, dead_at);
        }
    }
    free(sums); free(yenc); free(target); free(thr); free(thr_ta); free(hk); free(fired); free(lists); free(nlist); free(scratch); free(nodemask); free(alive);
    return NULL;
}

int64_t gtm_fit(gtm_model *m, const gtm_data *ds, int epochs, int64_t max_steps, int threads) {
    if (!check_compat(m, ds)) return -1;
    int64_t total = (int64_t)epochs * ds->n_graphs;
    if (max_steps >= 0 && max_steps < total) total = max_steps;
    if (total <= 0) return 0;
    uint32_t P = threads < 1 ? 1 : (uint32_t)threads;

    train_ctx ctx = {0};
    ctx.m = m; ctx.ds = ds; ctx.P = P; ctx.total = total;
    barrier_init(&ctx.bar, P);
    ctx.out = xalloc((size_t)ds->max_nodes * m->Cw * 8);
    for (uint32_t d = 1; d < m->D; d++) ctx.msg[d] = xalloc((size_t)ds->max_nodes * m->Wl[d] * 8);
    ctx.bundle = xalloc((size_t)ds->max_nodes * (m->D > 1 ? m->Wl[1] / 2 : 1) * 8);
    ctx.Opad = (m->O + 15) / 16 * 16;
    ctx.partial = xalloc((size_t)2 * P * ctx.Opad * 4);

    train_arg *args = xalloc(sizeof(train_arg) * P);
    pthread_t *th = xalloc(sizeof(pthread_t) * P);
    for (uint32_t t = 0; t < P; t++) {
        args[t].ctx = &ctx; args[t].tid = t;
        if (t > 0) pthread_create(&th[t], NULL, train_worker, &args[t]);
    }
    train_worker(&args[0]);
    for (uint32_t t = 1; t < P; t++) pthread_join(th[t], NULL);

    m->step += (uint64_t)total;
    free(ctx.out); free(ctx.bundle); free(ctx.partial); free(args); free(th);
    barrier_free(&ctx.bar);
    for (uint32_t d = 1; d < m->D; d++) free(ctx.msg[d]);
    return total;
}

/* ------------------------------------------------------------------------- */
/* Inference: example-parallel, no synchronization                            */
/* ------------------------------------------------------------------------- */
typedef struct {
    const gtm_model *m;
    const gtm_data *ds;
    int32_t *sums;
    uint64_t *bits;
    const int32_t *wcm;  /* clause-major weight mirror */
    _Atomic uint32_t next;
    _Atomic uint64_t fill_set, fill_total; /* message bits set / possible, layers >= 1 */
} score_ctx;

/*
 * Inference-only memo: a node's layer-0 clause outputs, and its bundled message, are a pure
 * function of (feature words, node type). Token graphs repeat nodes constantly, so a per-thread
 * direct-mapped cache (full key stored, so no false hits) turns them into a copy.
 * Entry layout: [tag: type+1][key: W words][out0: Cw words][bundle: half words].
 */
typedef struct {
    uint64_t *slots;
    uint32_t nslots, W, Cw, half, stride;
} memo_t;

static void memo_init(memo_t *mc, const gtm_model *m) {
    mc->W = m->Wl[0]; mc->Cw = m->Cw; mc->half = m->D > 1 ? m->Wl[1] / 2 : 0;
    mc->stride = 1 + mc->W + mc->Cw + mc->half;
    uint32_t n = 256;
    while (n < 8192 && (size_t)n * 2 * mc->stride * 8 <= (512u << 10)) n *= 2;
    mc->nslots = n;
    mc->slots = xalloc((size_t)n * mc->stride * 8); /* tag 0 = empty */
}

static inline uint64_t *memo_slot(const memo_t *mc, const uint64_t *x, uint32_t type, int *hit) {
    uint64_t h = splitmix64(type);
    for (uint32_t w = 0; w < mc->W; w++) h = splitmix64(h ^ x[w]);
    uint64_t *e = mc->slots + (size_t)(h & (mc->nslots - 1)) * mc->stride;
    *hit = e[0] == (uint64_t)type + 1 && !memcmp(e + 1, x, (size_t)mc->W * 8);
    return e;
}

#define SCORE_CHUNK 64

static void *score_worker(void *p) {
    score_ctx *ctx = p;
    static _Atomic uint32_t next_tid = 0;
    maybe_pin(atomic_fetch_add(&next_tid, 1));
    const gtm_model *m = ctx->m;
    const gtm_data *ds = ctx->ds;
    uint64_t *out = xalloc((size_t)ds->max_nodes * m->Cw * 8);
    uint64_t *msg[GTM_MAX_DEPTH] = {0};
    for (uint32_t d = 1; d < m->D; d++) msg[d] = xalloc((size_t)ds->max_nodes * m->Wl[d] * 8);
    uint64_t *fired = xalloc((size_t)m->Cw * 8);
    uint64_t *scratch = xalloc((size_t)2 * m->Cw * 8);
    memo_t mc;
    memo_init(&mc, m);
    const int use_memo = !((const gtm_model_ext *)m)->dense;
    uint64_t fill_set = 0, fill_total = 0;
    uint64_t *Pb = xalloc((size_t)ds->max_nodes * (m->D > 1 ? m->Wl[1] / 2 : 1) * 8);
    for (;;) {
        uint32_t g0 = atomic_fetch_add(&ctx->next, SCORE_CHUNK);
        if (g0 >= ds->n_graphs) break;
        uint32_t g1 = g0 + SCORE_CHUNK < ds->n_graphs ? g0 + SCORE_CHUNK : ds->n_graphs;
        for (uint32_t g = g0; g < g1; g++) {
            const uint64_t n0 = ds->node_off[g];
            const uint32_t n = ds->npg[g];
            const uint32_t *ntype = ds->ntype + n0;
            const uint64_t *X0 = ds->X + n0 * ds->W;
            if (use_memo) { /* layer 0 + first bundle through the memo */
                const uint32_t W = mc.W, Cw = m->Cw, half = mc.half;
                for (uint32_t v = 0; v < n; v++) {
                    int hit;
                    uint64_t *e = memo_slot(&mc, X0 + (size_t)v * W, ntype[v], &hit);
                    if (!hit) {
                        eval_layer(m, ds, ntype + v, 1, 0, X0 + (size_t)v * W, 0, Cw, out + (size_t)v * Cw, scratch);
                        if (half) bundle_sources(m, v, v + 1, out, Pb);
                        e[0] = (uint64_t)ntype[v] + 1;
                        memcpy(e + 1, X0 + (size_t)v * W, (size_t)W * 8);
                        memcpy(e + 1 + W, out + (size_t)v * Cw, (size_t)Cw * 8);
                        if (half) memcpy(e + 1 + W + Cw, Pb + (size_t)v * half, (size_t)half * 8);
                    } else {
                        memcpy(out + (size_t)v * Cw, e + 1 + W, (size_t)Cw * 8);
                        if (half) memcpy(Pb + (size_t)v * half, e + 1 + W + Cw, (size_t)half * 8);
                    }
                }
            } else {
                eval_layer(m, ds, ntype, n, 0, X0, 0, m->Cw, out, scratch);
            }
            for (uint32_t d = 1; d < m->D; d++) {
                if (d > 1 || !use_memo) bundle_sources(m, 0, n, out, Pb);
                build_messages(m, ds, n0, 0, n, Pb, msg[d]);
                for (uint32_t v = 0; v < n; v++) /* positive half of each node's message */
                    for (uint32_t w = 0; w < m->Wl[d] / 2; w++) fill_set += (uint64_t)__builtin_popcountll(msg[d][(size_t)v * m->Wl[d] + w]);
                fill_total += (uint64_t)n * m->MS;
                eval_layer(m, ds, ntype, n, d, msg[d], 0, m->Cw, out, scratch);
            }
            int32_t *s = ctx->sums + (size_t)g * m->O;
            memset(s, 0, (size_t)m->O * 4);
            clause_votes_rows(m, ctx->wcm, n, out, fired, s);
            if (ctx->bits) memcpy(ctx->bits + (size_t)g * m->Cw, fired, (size_t)m->Cw * 8);
        }
    }
    atomic_fetch_add(&ctx->fill_set, fill_set);
    atomic_fetch_add(&ctx->fill_total, fill_total);
    free(out); free(fired); free(Pb); free(scratch); free(mc.slots);
    for (uint32_t d = 1; d < m->D; d++) free(msg[d]);
    return NULL;
}

void gtm_score(const gtm_model *m, const gtm_data *ds, int32_t *sums, uint64_t *bits, int threads) {
    gtm_score_ex(m, ds, sums, bits, threads, NULL);
}

void gtm_score_ex(const gtm_model *m, const gtm_data *ds, int32_t *sums, uint64_t *bits, int threads, double *msg_fill) {
    if (!check_compat(m, ds)) return;
    uint32_t P = threads < 1 ? 1 : (uint32_t)threads;
    int32_t *wcm = xalloc((size_t)m->C * m->O * 4);
    for (uint32_t k = 0; k < m->O; k++)
        for (uint32_t c = 0; c < m->C; c++) wcm[(size_t)c * m->O + k] = m->w[(size_t)k * m->Cw * 64 + c];
    score_ctx ctx = {m, ds, sums, bits, wcm, 0, 0, 0};
    pthread_t *th = xalloc(sizeof(pthread_t) * P);
    for (uint32_t t = 1; t < P; t++) pthread_create(&th[t], NULL, score_worker, &ctx);
    score_worker(&ctx);
    for (uint32_t t = 1; t < P; t++) pthread_join(th[t], NULL);
    if (msg_fill) *msg_fill = ctx.fill_total ? (double)ctx.fill_set / (double)ctx.fill_total : 0.0;
    free(th); free(wcm);
}

double gtm_accuracy(const gtm_data *d, const int32_t *sums) {
    uint64_t ok = 0;
    for (uint32_t g = 0; g < d->n_graphs; g++) {
        const int32_t *s = sums + (size_t)g * d->O, *y = d->Y + (size_t)g * d->O;
        if (d->kind == 0) {
            uint32_t best = 0, truth = 0;
            for (uint32_t k = 1; k < d->O; k++) {
                if (s[k] > s[best]) best = k;
                if (y[k] > y[truth]) truth = k;
            }
            ok += best == truth;
        } else {
            int all = 1;
            for (uint32_t k = 0; k < d->O; k++) all &= (s[k] >= 0) == (y[k] == 1);
            ok += (uint64_t)all;
        }
    }
    return (double)ok / d->n_graphs;
}
