/* fastlae: fast inference for hardened LogicAE (logic-bert .lth) models.
 *
 * Same model, same scores: every command checks against, or reproduces exactly, logic_text.c's
 * hard_predict (integer class votes). What changes is the data layout:
 *
 *   reference   bit b of a 64-bit word = text b of a 64-text batch, one word per (position, channel).
 *               A call costs the same for 1 text as for 64.
 *   fastlae     bit t of a word = token position t of ONE text, one word per channel (NW words
 *               when T > 64). A convolution tap at offset d is a shift by d, so a whole block for
 *               one text is O x (R shifts + N gates) word operations, and a text's activations
 *               (width x NW x 8 bytes, 8 KB at width 1024) stay in L1.
 *   lanes       with --lanes L (1, 2, 4 or 8; GCC vector extensions) L texts run side by side in
 *               one vector register: 1 for single-request latency, 4 (AVX2) / 8 (AVX-512) for
 *               throughput. Batches are split across threads by lane groups.
 *
 * Gates: a 2-input gate with truth table f (bit 2a+b) is evaluated branch-free as a mux,
 *   t0 = m0 ^ (b & (m0^m1)),  t1 = m2 ^ (b & (m2^m3)),  out = t0 ^ (a & (t0^t1)),
 * with the four all-ones / all-zeros masks m_k read from a 16-entry table (512 bytes, L1).
 *
 *   fastlae verify MODEL.lth DATA.ids SEQ [--threads N]                 scores == hard_predict, every record
 *   fastlae predict MODEL.lth DATA.ids SEQ [--threads N] [--lanes L]    {"row","label","votes"} lines (lt predict's format)
 *   fastlae bench MODEL.lth DATA.ids SEQ --batch B [--threads N] [--lanes L] [--repeats R]
 *                                                                        median ms per call, both engines
 * Build: gcc -O3 -march=native -std=c11 -fopenmp -I<logic-bert>/src fastlae.c -lm -o fastlae
 */
#define LOGIC_NO_MAIN
#include "logic_text.c"

#define MAXNW 8   /* T <= 512 */

typedef struct {
    int T, NW, U, NWU;          /* positions, words per channel; pooled positions / words */
    Hard *h;
    uint64_t gm[16][4];         /* gate masks per truth table */
} Fast;

static Fast *fast_new(Hard *h, int T) {
    if (T < 1 || T > 64 * MAXNW) die("fastlae: SEQ must be 1..%d", 64 * MAXNW);
    Fast *F = alloc(1, sizeof *F);
    F->h = h; F->T = T; F->NW = (T + 63) / 64; F->U = (T + 1) / 2; F->NWU = (F->U + 63) / 64;
    for (int f = 0; f < 16; f++) for (int k = 0; k < 4; k++) F->gm[f][k] = (f >> k) & 1 ? ~UINT64_C(0) : 0;
    if (h->l[h->c.blocks].kernel != 1) die("fastlae: head kernel must be 1");
    return F;
}

/* ---------------------------------------------------------------- kernel, generic in lanes */
typedef uint64_t v1 __attribute__((vector_size(8)));
typedef uint64_t v2 __attribute__((vector_size(16)));
typedef uint64_t v4 __attribute__((vector_size(32)));
typedef uint64_t v8 __attribute__((vector_size(64)));
#define KERNEL(L, V)                                                                                         \
/* out = in read at offset d (position t takes position t+d; zero outside the words) */                      \
static inline void shift_##L(const V *in, V *out, int d, int NW) {                                           \
    if (NW == 1) {                                                                                           \
        out[0] = d >= 64 || d <= -64 ? (V){0} : d >= 0 ? in[0] >> (uint64_t)d : in[0] << (uint64_t)(-d);  \
        return;                                                                                              \
    }                                                                                                        \
    for (int w = 0; w < NW; w++) {                                                                           \
        int s = w * 64 + d; /* first source bit */                                                           \
        int q = s >= 0 ? s / 64 : -((-s + 63) / 64), r = s - q * 64;                                         \
        V lo = q >= 0 && q < NW ? in[q] : (V){0}, hi = q + 1 >= 0 && q + 1 < NW ? in[q + 1] : (V){0};      \
        out[w] = r ? (lo >> (uint64_t)r) | (hi << (uint64_t)(64 - r)) : lo;                                  \
    }                                                                                                        \
}                                                                                                            \
static inline V gate_##L(V a, V b, const uint64_t *m) {                                                     \
    V m0 = (V){0} + m[0], m1 = (V){0} + m[1], m2 = (V){0} + m[2], m3 = (V){0} + m[3];                      \
    V t0 = m0 ^ (b & (m0 ^ m1)), t1 = m2 ^ (b & (m2 ^ m3));                                                 \
    return t0 ^ (a & (t0 ^ t1));                                                                             \
}                                                                                                            \
static void layer_##L(const Fast *F, const HLayer *l, const V *x, V *y, const V *valid, int NW) {           \
    V leaf[MAX_LEAVES * MAXNW], v[2 * MAX_LEAVES - 1];                                                      \
    for (int o = 0; o < l->O; o++) {                                                                         \
        const int32_t *ch = l->ch + (size_t)o * l->R, *off = l->off + (size_t)o * l->R;                     \
        const uint8_t *f = l->f + (size_t)o * l->N;                                                          \
        for (int k = 0; k < l->R; k++) shift_##L(x + (size_t)ch[k] * NW, leaf + k * NW, off[k] * l->dilation, NW); \
        for (int w = 0; w < NW; w++) {                                                                       \
            for (int k = 0; k < l->R; k++) v[l->N + k] = leaf[k * NW + w];                                   \
            for (int j = l->N - 1; j >= 0; j--) v[j] = gate_##L(v[2 * j + 1], v[2 * j + 2], F->gm[f[j]]);    \
            y[(size_t)o * NW + w] = v[0] & valid[w];                                                         \
        }                                                                                                    \
    }                                                                                                        \
}                                                                                                            \
/* L texts (ids: L rows of T; rows beyond n are empty) -> scores[2L] */                                     \
static void run_##L(const Fast *F, const uint32_t *ids, int n, V *a, V *b, int64_t *scores) {               \
    const Hard *h = F->h;                                                                                    \
    int T = F->T, NW = F->NW, U = F->U, NWU = F->NWU, C = h->c.bits, Q = (C + 63) / 64;                     \
    V valid[MAXNW], pvalid[MAXNW];                                                                           \
    memset(a, 0, sizeof(V) * (size_t)C * NW);                                                                \
    memset(valid, 0, sizeof valid);                                                                          \
    for (int e = 0; e < n; e++) for (int t = 0; t < T; t++) {                                               \
        uint32_t id = ids[(size_t)e * T + t];                                                                \
        if (id >= (uint32_t)h->c.vocab) die("inference token outside vocabulary");                          \
        if (!id) continue;                                                                                   \
        uint64_t bit = UINT64_C(1) << (t % 64);                                                              \
        valid[t / 64][e] |= bit;                                                                             \
        for (int q = 0; q < Q; q++) {                                                                        \
            uint64_t code = h->codes[(size_t)id * Q + q];                                                    \
            while (code) { int c = q * 64 + lowbit(code); if (c < C) a[(size_t)c * NW + t / 64][e] |= bit; code &= code - 1; } \
        }                                                                                                    \
    }                                                                                                        \
    V *cur = a, *nxt = b;                                                                                    \
    if (GEN_OK(NW)) GEN_BLOCKS(L)(cur, nxt, valid[0]);                                                      \
    else for (int i = 0; i < h->c.blocks; i++) { layer_##L(F, h->l + i, cur, nxt, valid, NW); V *z = cur; cur = nxt; nxt = z; } \
    if (GEN_OK(NW) && h->c.blocks % 2) { V *z = cur; cur = nxt; nxt = z; }                                   \
    /* pairwise OR pool over positions: pooled bit u = bit 2u | bit 2u+1 */                                  \
    int W = h->c.width;                                                                                      \
    memset(pvalid, 0, sizeof pvalid);                                                                        \
    for (int e = 0; e < L; e++) {                                                                            \
        for (int w = 0; w < NW; w++) {                                                                       \
            uint64_t yv = valid[w][e] | (valid[w][e] >> 1);                                                   \
            pvalid[w / 2][e] |= (uint64_t)_pext_u64(yv, UINT64_C(0x5555555555555555)) << (32 * (w % 2));    \
        }                                                                                                    \
        for (int c = 0; c < W; c++) for (int w = 0; w < NW; w++) {                                           \
            if (w % 2 == 0) nxt[(size_t)c * NWU + w / 2][e] = 0;                                             \
            uint64_t xv = cur[(size_t)c * NW + w][e]; xv |= xv >> 1;                                         \
            nxt[(size_t)c * NWU + w / 2][e] |= (uint64_t)_pext_u64(xv, UINT64_C(0x5555555555555555)) << (32 * (w % 2)); \
        }                                                                                                    \
    }                                                                                                        \
    { V *z = cur; cur = nxt; nxt = z; }                                                                      \
    const HLayer *hl = h->l + h->c.blocks;                                                                   \
    layer_##L(F, hl, cur, nxt, pvalid, NWU);                                                                 \
    int votes = h->c.votes;                                                                                  \
    for (int e = 0; e < L; e++) {                                                                            \
        int64_t s[2] = {0, 0};                                                                               \
        for (int o = 0; o < 2 * votes; o++) for (int w = 0; w < NWU; w++)                                    \
            s[o / votes] += __builtin_popcountll(nxt[(size_t)o * NWU + w][e]);                              \
        scores[2 * e] = s[0]; scores[2 * e + 1] = s[1];                                                      \
    }                                                                                                        \
    (void)U;                                                                                                 \
}                                                                                                            \
static void batch_##L(const Fast *F, const uint32_t *ids, int B, int64_t *scores) {                         \
    int T = F->T, maxC = F->h->c.bits > F->h->c.width ? F->h->c.bits : F->h->c.width;                      \
    if (maxC < 2 * F->h->c.votes) maxC = 2 * F->h->c.votes;                                                   \
    int G = (B + L - 1) / L;                                                                                 \
    _Pragma("omp parallel")                                                                                  \
    {                                                                                                        \
        V *a = aligned_alloc(64, ((sizeof(V) * (size_t)maxC * F->NW + 63) / 64) * 64);                      \
        V *b = aligned_alloc(64, ((sizeof(V) * (size_t)maxC * F->NW + 63) / 64) * 64);                      \
        uint32_t *tmp = alloc((size_t)L * T, 4);                                                             \
        _Pragma("omp for schedule(static)")                                                                  \
        for (int g = 0; g < G; g++) {                                                                        \
            int n = B - g * L < L ? B - g * L : L;                                                           \
            memset(tmp, 0, (size_t)L * T * 4); memcpy(tmp, ids + (size_t)g * L * T, (size_t)n * T * 4);    \
            int64_t sc[2 * L];                                                                               \
            run_##L(F, tmp, n, a, b, sc);                                                                    \
            memcpy(scores + (size_t)g * L * 2, sc, (size_t)n * 2 * 8);                                       \
        }                                                                                                    \
        free(a); free(b); free(tmp);                                                                         \
    }                                                                                                        \
}

#include <immintrin.h>
/* fastgen.c defines FASTLAE_GEN and includes the network compiled to C (fastlae gen): all blocks
 * as straight-line word operations for T <= 64. Without it the interpreter runs every layer. */
#ifdef FASTLAE_GEN
#define GEN_OK(NW) ((NW) == 1)
#define GEN_BLOCKS(L) gen_blocks_##L
#define GEN_DECL(L, V) static void gen_blocks_##L(V *a, V *b, V valid);
GEN_DECL(1, v1) GEN_DECL(2, v2) GEN_DECL(4, v4) GEN_DECL(8, v8)
#else
#define GEN_OK(NW) 0
#define GEN_BLOCKS(L) gen_none
#define gen_none(a, b, v) ((void)0)
#endif
KERNEL(1, v1)
KERNEL(2, v2)
KERNEL(4, v4)
KERNEL(8, v8)

static void fast_batch(const Fast *F, const uint32_t *ids, int B, int64_t *scores, int lanes) {
    switch (lanes) {
    case 1: batch_1(F, ids, B, scores); break;
    case 2: batch_2(F, ids, B, scores); break;
    case 4: batch_4(F, ids, B, scores); break;
    case 8: batch_8(F, ids, B, scores); break;
    default: die("--lanes must be 1, 2, 4 or 8");
    }
}

/* ---------------------------------------------------------------- commands */
static int opt_int(int argc, char **argv, const char *key, int def) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], key)) return (int)parse_long(argv[i + 1], 1, 1 << 20);
    return def;
}

static void ref_scores(Hard *h, const uint32_t *ids, int B, int T, int64_t *out) {
    HWork *w = hwork_new(h, B, T);
    hard_predict(h, w, ids);
    memcpy(out, w->scores, (size_t)B * 2 * 8);
    hwork_free(w);
}

static double median_ms(double *t, int n) {
    qsort(t, n, sizeof(double), doublecmp);
    return 1000 * (n % 2 ? t[n / 2] : .5 * (t[n / 2 - 1] + t[n / 2]));
}

int main(int argc, char **argv) {
    if (argc < 5) die("usage: fastlae verify|predict|bench MODEL.lth DATA.ids SEQ [--threads N] [--lanes L] [--batch B] [--repeats R]");
    const char *cmd = argv[1];
    int T = (int)parse_long(argv[4], 1, 65536);
    int threads = opt_int(argc, argv, "--threads", 1), lanes = opt_int(argc, argv, "--lanes", 1);
    threads_set(threads);
    Hard *h = load_hard(argv[2]);
    Data d = data_load(argv[3], "ids", NULL, h->c.vocab, T);
    Fast *F = fast_new(h, T);
    if (!strcmp(cmd, "verify")) {
        int64_t *s = alloc((size_t)d.n * 2, 8), *r = alloc((size_t)d.n * 2, 8);
        long bad = 0;
        for (int L = 1; L <= 8; L *= 2) {
            fast_batch(F, d.x, d.n, s, L);
            for (int i0 = 0; i0 < d.n; i0 += 64) {
                int take = d.n - i0 < 64 ? d.n - i0 : 64;
                ref_scores(h, d.x + (size_t)i0 * T, take, T, r + (size_t)i0 * 2);
            }
            long m = 0;
            for (long i = 0; i < 2L * d.n; i++) m += s[i] != r[i];
            printf("{\"lanes\":%d,\"records\":%d,\"mismatched_scores\":%ld}\n", L, d.n, m);
            bad += m;
        }
        printf("{\"verify\":\"%s\"}\n", bad ? "FAILED" : "passed");
        return bad ? 1 : 0;
    }
    if (!strcmp(cmd, "predict")) {
        int64_t *s = alloc((size_t)d.n * 2, 8);
        fast_batch(F, d.x, d.n, s, lanes);
        int hit = 0, labeled = 0;
        for (int i = 0; i < d.n; i++) {
            int pred = s[2 * i + 1] > s[2 * i];
            if (d.y[i] >= 0) { labeled++; hit += pred == d.y[i]; }
            printf("{\"row\":%d,\"label\":%d,\"votes\":[%lld,%lld]}\n", i, pred, (long long)s[2 * i], (long long)s[2 * i + 1]);
        }
        fprintf(stderr, "labeled=%d correct=%d accuracy=%.6f\n", labeled, hit, labeled ? (double)hit / labeled : 0.);
        return 0;
    }
    if (!strcmp(cmd, "gen")) {  /* fastlae gen MODEL.lth - SEQ OUT.inc: blocks as straight-line C */
        const char *ops[16] = {"Z", "~(A|B)", "(~A&B)", "~A", "(A&~B)", "~B", "(A^B)", "~(A&B)",
                               "(A&B)", "~(A^B)", "B", "(~A|B)", "A", "(A|~B)", "(A|B)", "~Z"};
        FILE *o = open_file(argv[5], "w");
        fprintf(o, "/* generated by fastlae gen from %s: %d blocks x %d outputs, T <= 64 */\n", argv[2], h->c.blocks, h->c.width);
        for (int i = 0; i < h->c.blocks; i++) {
            HLayer *l = h->l + i;
            fprintf(o, "static void GEN_L(blk%d)(const V *restrict x, V *restrict y, V valid) {\n  const V Z = {0};\n", i);
            for (int oo = 0; oo < l->O; oo++) {
                fprintf(o, "  { ");
                for (int k = 0; k < l->R; k++) {
                    int c = l->ch[(size_t)oo * l->R + k], dd = l->off[(size_t)oo * l->R + k] * l->dilation;
                    if (dd == 0) fprintf(o, "V g%d = x[%d]; ", l->N + k, c);
                    else if (dd >= 64 || dd <= -64) fprintf(o, "V g%d = Z; ", l->N + k);
                    else fprintf(o, "V g%d = x[%d] %s %d; ", l->N + k, c, dd > 0 ? ">>" : "<<", dd > 0 ? dd : -dd);
                }
                for (int j = l->N - 1; j >= 0; j--) {
                    char e[64]; const char *p = ops[l->f[(size_t)oo * l->N + j]]; int n = 0;
                    for (; *p; p++) {
                        if (*p == 'A') n += sprintf(e + n, "g%d", 2 * j + 1);
                        else if (*p == 'B') n += sprintf(e + n, "g%d", 2 * j + 2);
                        else e[n++] = *p;
                    }
                    e[n] = 0;
                    fprintf(o, "V g%d = %s; ", j, e);
                }
                fprintf(o, "y[%d] = g0 & valid; }\n", oo);
            }
            fprintf(o, "}\n");
        }
        fprintf(o, "static void GEN_L(blocks)(V *a, V *b, V valid) {\n");
        for (int i = 0; i < h->c.blocks; i++) fprintf(o, "  GEN_L(blk%d)(%s, %s, valid);\n", i, i % 2 ? "b" : "a", i % 2 ? "a" : "b");
        fprintf(o, "}\n");
        fclose(o);
        printf("{\"generated\":\"%s\",\"blocks\":%d,\"outputs\":%d}\n", argv[5], h->c.blocks, h->c.blocks * h->c.width);
        return 0;
    }
    if (!strcmp(cmd, "bench")) {
        int B = opt_int(argc, argv, "--batch", 1), R = opt_int(argc, argv, "--repeats", 30);
        uint32_t *x = alloc((size_t)B * T, 4);
        for (int b = 0; b < B; b++) memcpy(x + (size_t)b * T, d.x + (size_t)(b % d.n) * T, (size_t)T * 4);
        int64_t *s = alloc((size_t)B * 2, 8), *r = alloc((size_t)B * 2, 8);
        double *tf = alloc(R, 8), *tr = alloc(R, 8);
        HWork *w = hwork_new(h, B, T);
        for (int k = -3; k < R; k++) {
            double t0 = now(); fast_batch(F, x, B, s, lanes); double t1 = now(); hard_predict(h, w, x); double t2 = now();
            if (k >= 0) { tf[k] = t1 - t0; tr[k] = t2 - t1; }
        }
        long m = 0;
        memcpy(r, w->scores, (size_t)B * 16);
        for (long i = 0; i < 2L * B; i++) m += s[i] != r[i];
        double mf = median_ms(tf, R), mr = median_ms(tr, R);
        printf("{\"batch\":%d,\"seq\":%d,\"threads\":%d,\"lanes\":%d,\"fast_ms\":%.4f,\"reference_ms\":%.4f,\"speedup\":%.2f,"
               "\"fast_texts_per_s\":%.0f,\"reference_texts_per_s\":%.0f,\"mismatched_scores\":%ld}\n",
               B, T, thread_count(), lanes, mf, mr, mr / mf, B / (mf / 1000), B / (mr / 1000), m);
        return m ? 1 : 0;
    }
    die("unknown command %s", cmd);
    return 1;
}
