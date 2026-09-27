/* lgn.c: LGN-attention BitNet runtime. The frozen attention-only BitNet with every continuous operation replaced by
 * integer / logic operations and softmax retrieval replaced by a learned hard gate field (spec: lgn.py). No floating
 * point in the forward pass: shifts, integer products of 6-bit values with ternary weights, popcounts over bit-plane
 * masks, comparators, small tables. (Floating point is used only to report cross-entropy.)
 *
 *   gcc -O3 -march=native -fopenmp lgn.c -lm -o lgn
 *   ./lgn verify MODEL.bin                 integer logits == lgn.py reference (MODEL.bin.ref, MODEL.bin.x)
 *   ./lgn eval   MODEL.bin TEXT            next-byte CE / accuracy on the last 10% of TEXT (every 64-byte window)
 *   ./lgn bench  MODEL.bin TEXT            ns per token: 1 thread and all threads (full 64-token windows)
 *   options: --mm popcount|int (ternary matmul as bit-plane popcounts or as int8 dot products; same integers)
 *            --table 0|1       (layer 0 from the exact (byte, position) table; default 1)
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

typedef int64_t i64;
static void die(const char *m) { fprintf(stderr, "lgn: %s\n", m); exit(1); }
static void *xal(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) die("out of memory"); return p; }

enum { MAXL = 4, D = 128, NH = 4, HD = 32, T = 64, QMAX = 31 };
typedef struct {
    int T, d, H, L, NB, W, LV, F, G;
    i64 *tok, *pos, *gfi, *head8, *mh_m, *mh_e, sqd[2], rsq[2][256][2], rm[128][2];
    i64 *gi[MAXL], *Wq[MAXL], *Wo[MAXL], gq[MAXL][2], go[MAXL][2], *corner[MAXL], *w[MAXL], *thm_m[MAXL], *thm_e[MAXL], *lut[MAXL];
    /* derived */
    int8_t *WqT[MAXL], *WoT[MAXL];                     /* [out][in] */
    uint64_t *Wqp[MAXL][2], *Woq[MAXL][2];             /* sign-split column masks [out][2 words] */
    i64 tab[MAXL][NH][2][2][5][5];                     /* corner x weight per (head, qbit, kbit, a, c) */
    int passthrough[MAXL][NH], need_masks[MAXL];
    int32_t *headT;                                    /* [256][D] */
    int8_t *Wqv[MAXL], *Wov[MAXL]; int32_t *Wqs[MAXL], *Wos[MAXL];   /* VNNI packs [cb][j4][16][4], column sums */
    int16_t *headv;                                    /* [cb][j2][16][2] */
    i64 emin_h;
} Model;

/* ---------------- integer helpers (exactly lgn.py) ---------------- */
static inline i64 rshift_round(i64 v, int s) {
    if (s <= 0) return v;
    i64 a = v < 0 ? -v : v; a = (a + ((i64)1 << (s - 1))) >> s; return v < 0 ? -a : a;
}
static inline i64 shift_round(i64 v, int s) { return s >= 0 ? v * ((i64)1 << s) : rshift_round(v, -s); }
static inline int bitlen(i64 x) { return x ? 64 - __builtin_clzll((uint64_t)x) : 0; }
static inline i64 floordiv(i64 a, i64 b) { i64 q = a / b; if ((a % b) && ((a < 0) != (b < 0))) q--; return q; }
static inline void mf_mul(i64 m1, i64 e1, i64 m2, i64 e2, i64 *m, i64 *e) {
    i64 p = m1 * m2; int s = p >= 8192 ? 7 : 6; i64 mm = rshift_round(p, s), ee = e1 + e2 + s;
    if (mm >= 128) { mm = 64; ee++; }
    *m = mm; *e = ee;
}
static int q16(const i64 *y, int n, int16_t *q) {        /* shift quantizer to [-32767, 32767] */
    i64 mx = 0; for (int i = 0; i < n; i++) { i64 a = y[i] < 0 ? -y[i] : y[i]; if (a > mx) mx = a; }
    int k = bitlen(mx) - 15; if (k < 0) k = 0;
    if (rshift_round(mx, k) > 32767) k++;
    for (int i = 0; i < n; i++) { i64 v = rshift_round(y[i], k); q[i] = (int16_t)(v > 32767 ? 32767 : v < -32767 ? -32767 : v); }
    return k;
}
static int q6(const i64 *y, int n, int8_t *q) {
    i64 mx = 0; for (int i = 0; i < n; i++) { i64 a = y[i] < 0 ? -y[i] : y[i]; if (a > mx) mx = a; }
    int k = bitlen(mx) - 5; if (k < 0) k = 0;
    if (rshift_round(mx, k) > QMAX) k++;
    for (int i = 0; i < n; i++) { i64 v = rshift_round(y[i], k); q[i] = (int8_t)(v > QMAX ? QMAX : v < -QMAX ? -QMAX : v); }
    return k;
}
static void rsqrt_mf(const Model *M, i64 S, i64 *m, i64 *e) {
    if (S < 1) S = 1;
    int n = bitlen(S) - 1; i64 r = S - ((i64)1 << n);
    i64 idx = n >= 8 ? r >> (n - 8) : r << (8 - n);
    int par = n & 1; *m = M->rsq[par][idx][0]; *e = M->rsq[par][idx][1] - (n >> 1);
}

/* ---------------- model file ---------------- */
static void rd(FILE *f, i64 *p, size_t n) { if (fread(p, 8, n, f) != n) die("truncated model"); }
static i64 *rda(FILE *f, size_t n) { i64 *p = xal(n * 8); rd(f, p, n); return p; }
static Model *load(const char *path) {
    FILE *f = fopen(path, "rb"); char mg[8]; int32_t h[9]; if (!f) die("cannot open model");
    if (fread(mg, 1, 8, f) != 8 || memcmp(mg, "LGNATT01", 8) || fread(h, 4, 9, f) != 9) die("not an LGN model");
    Model *M = xal(sizeof(Model));
    M->T = h[0]; M->d = h[1]; M->H = h[2]; M->L = h[3]; M->NB = h[4]; M->W = h[5]; M->LV = h[6]; M->F = h[7]; M->G = h[8];
    if (M->d != D || M->H != NH || M->T != T || M->NB != 5 || M->L > MAXL) die("unsupported shape");
    int d = D, H = NH, LV = M->LV;
    M->tok = rda(f, 256 * d); M->pos = rda(f, (size_t)T * d); M->gfi = rda(f, d); M->head8 = rda(f, (size_t)d * 256);
    M->mh_m = rda(f, 256); M->mh_e = rda(f, 256); rd(f, M->sqd, 2); rd(f, &M->rsq[0][0][0], 2 * 256 * 2); rd(f, &M->rm[0][0], 128 * 2);
    for (int l = 0; l < M->L; l++) {
        M->gi[l] = rda(f, d); M->Wq[l] = rda(f, (size_t)d * 3 * d); M->Wo[l] = rda(f, (size_t)d * d);
        rd(f, M->gq[l], 2); rd(f, M->go[l], 2);
        M->corner[l] = rda(f, (size_t)H * 4 * 25); M->w[l] = rda(f, (size_t)H * 25);
        M->thm_m[l] = rda(f, (size_t)H * LV); M->thm_e[l] = rda(f, (size_t)H * LV); M->lut[l] = rda(f, (size_t)H * LV);
    }
    if (fgetc(f) != EOF) die("trailing bytes in model");
    fclose(f);
    M->headT = xal((size_t)256 * d * 4);
    for (int j = 0; j < d; j++) for (int v = 0; v < 256; v++) M->headT[(size_t)v * d + j] = (int32_t)M->head8[(size_t)j * 256 + v];
    M->headv = xal((size_t)256 * d * 2);
    for (int v = 0; v < 256; v++) for (int j = 0; j < d; j++)
        M->headv[((((size_t)(v / 16) * (d / 2) + j / 2) * 16 + v % 16) * 2) + j % 2] = (int16_t)M->head8[(size_t)j * 256 + v];
    M->emin_h = M->mh_e[0]; for (int v = 1; v < 256; v++) if (M->mh_e[v] < M->emin_h) M->emin_h = M->mh_e[v];
    for (int l = 0; l < M->L; l++) {
        M->WqT[l] = xal(3 * d * d); M->WoT[l] = xal(d * d);
        for (int c = 0; c < 2; c++) { M->Wqp[l][c] = xal(3 * d * 2 * 8); M->Woq[l][c] = xal(d * 2 * 8); }
        for (int j = 0; j < d; j++) for (int c = 0; c < 3 * d; c++) {
            int wv = (int)M->Wq[l][(size_t)j * 3 * d + c]; M->WqT[l][(size_t)c * d + j] = (int8_t)wv;
            if (wv) M->Wqp[l][wv < 0][(size_t)c * 2 + (j >> 6)] |= 1ull << (j & 63);
        }
        for (int j = 0; j < d; j++) for (int c = 0; c < d; c++) {
            int wv = (int)M->Wo[l][(size_t)j * d + c]; M->WoT[l][(size_t)c * d + j] = (int8_t)wv;
            if (wv) M->Woq[l][wv < 0][(size_t)c * 2 + (j >> 6)] |= 1ull << (j & 63);
        }
        for (int pass = 0; pass < 2; pass++) {
            int nout = pass ? d : 3 * d; const int8_t *WT = pass ? M->WoT[l] : M->WqT[l];
            int8_t *pk = xal((size_t)nout * d); int32_t *cs = xal((size_t)nout * 4);
            for (int c = 0; c < nout; c++) for (int j = 0; j < d; j++) {
                int8_t wv = WT[(size_t)c * d + j]; cs[c] += wv;
                pk[((((size_t)(c / 16) * (d / 4) + j / 4) * 16 + c % 16) * 4) + j % 4] = wv;
            }
            if (pass) { M->Wov[l] = pk; M->Wos[l] = cs; } else { M->Wqv[l] = pk; M->Wqs[l] = cs; }
        }
        for (int h = 0; h < H; h++) {
            int pt = 1;
            for (int a = 0; a < 5; a++) for (int c = 0; c < 5; c++) for (int x = 0; x < 2; x++) for (int y = 0; y < 2; y++) {
                i64 cr = M->corner[l][(((size_t)h * 4 + (x * 2 + y)) * 5 + a) * 5 + c], wv = M->w[l][((size_t)h * 5 + a) * 5 + c];
                M->tab[l][h][x][y][a][c] = cr * wv;
                if ((x & y) ? (cr != 1 || wv != ((i64)1 << (a + c))) : cr != 0) pt = 0;
            }
            M->passthrough[l][h] = pt && !getenv("LGN_GENERAL");
            if (!M->passthrough[l][h]) M->need_masks[l] = 1;
        }
    }
    return M;
}

/* ---------------- ternary matmul: int dot products or bit-plane popcounts (identical integers) ---------------- */
static int MM_POPCOUNT = 0, MM_VNNI = 1;
static void mm_vnni(const int8_t *xq, int n_out, const int8_t *pk, const int32_t *cs, i64 *z) {
#ifdef __AVX512VNNI__
    uint8_t xu[D]; for (int j = 0; j < D; j++) xu[j] = (uint8_t)(xq[j] + 32);
    for (int cb = 0; cb < n_out / 16; cb++) {
        __m512i acc = _mm512_setzero_si512(); const int8_t *w = pk + (size_t)cb * D * 16;
        for (int j4 = 0; j4 < D / 4; j4++) {
            int32_t b; memcpy(&b, xu + 4 * j4, 4);
            acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(b), _mm512_loadu_si512(w + (size_t)j4 * 64));
        }
        int32_t r[16]; _mm512_storeu_si512(r, acc);
        for (int i = 0; i < 16; i++) z[cb * 16 + i] = (i64)r[i] - 32 * (i64)cs[cb * 16 + i];
    }
#else
    (void)xq; (void)n_out; (void)pk; (void)cs; (void)z; die("built without AVX512-VNNI");
#endif
}
static void mm(const int8_t *xq, int n_out, const int8_t *WT, uint64_t *const Wp[2], i64 *z) {
    if (!MM_POPCOUNT) {
        for (int c = 0; c < n_out; c++) {
            const int8_t *w = WT + (size_t)c * D; int32_t s = 0;
            for (int j = 0; j < D; j++) s += (int32_t)xq[j] * w[j];
            z[c] = s;
        }
        return;
    }
    uint64_t P[5][2], N[5][2]; memset(P, 0, sizeof P); memset(N, 0, sizeof N);
    for (int j = 0; j < D; j++) {
        int v = xq[j], a = v < 0 ? -v : v;
        for (int b = 0; b < 5; b++) if (a >> b & 1) (v < 0 ? N : P)[b][j >> 6] |= 1ull << (j & 63);
    }
    for (int c = 0; c < n_out; c++) {
        const uint64_t *wp = Wp[0] + (size_t)c * 2, *wn = Wp[1] + (size_t)c * 2; i64 s = 0;
        for (int b = 0; b < 5; b++) {
            int t = __builtin_popcountll(P[b][0] & wp[0]) + __builtin_popcountll(P[b][1] & wp[1])
                  + __builtin_popcountll(N[b][0] & wn[0]) + __builtin_popcountll(N[b][1] & wn[1])
                  - __builtin_popcountll(P[b][0] & wn[0]) - __builtin_popcountll(P[b][1] & wn[1])
                  - __builtin_popcountll(N[b][0] & wp[0]) - __builtin_popcountll(N[b][1] & wp[1]);
            s += (i64)t << b;
        }
        z[c] = s;
    }
}

/* ---------------- per-token layer inputs: q6/k6/v6 + scales (tabulated for layer 0) ---------------- */
typedef struct {
    int8_t qkv[3][NH][HD]; uint32_t mag[3][NH][5], neg[3][NH];
    i64 mc; i64 ex[3][NH];                             /* scale mantissa (shared), exponents per part / head */
} Tok;

static void tok_inputs(const Model *M, int l, const i64 *x, Tok *o) {
    i64 y[D], z[3 * D]; int8_t xq[D];
    for (int j = 0; j < D; j++) y[j] = x[j] * M->gi[l][j];
    int k1 = q6(y, D, xq);
    if (MM_VNNI && !MM_POPCOUNT) mm_vnni(xq, 3 * D, M->Wqv[l], M->Wqs[l], z); else mm(xq, 3 * D, M->WqT[l], M->Wqp[l], z);
    i64 S = 0; for (int j = 0; j < D; j++) S += x[j] * x[j];
    i64 rm_, re_, mr, er, mc, ec;
    rsqrt_mf(M, S, &rm_, &re_); mf_mul(M->sqd[0], M->sqd[1], rm_, re_, &mr, &er); mf_mul(M->gq[l][0], M->gq[l][1], mr, er, &mc, &ec);
    o->mc = mc;
    for (int p = 0; p < 3; p++) for (int h = 0; h < NH; h++) {
        int k2 = q6(z + p * D + h * HD, HD, o->qkv[p][h]);
        o->ex[p][h] = ec + k1 - M->G + k2;
        if (!M->need_masks[l]) continue;
        uint32_t neg = 0, mag[5] = {0};
        for (int j = 0; j < HD; j++) {
            int v = o->qkv[p][h][j], a = v < 0 ? -v : v; if (v < 0) neg |= 1u << j;
            for (int b = 0; b < 5; b++) if (a >> b & 1) mag[b] |= 1u << j;
        }
        o->neg[p][h] = neg; memcpy(o->mag[p][h], mag, sizeof mag);
    }
}

/* gate-field match: sum over plane pairs of w_ab x (popcount of corner mask on same-sign dims - on opposite-sign dims) */
static inline i64 match(const Model *M, int l, int h, const Tok *a, const Tok *b) {
    if (M->passthrough[l][h]) {                        /* pass-through gates == the integer dot product */
        int32_t s = 0; for (int j = 0; j < HD; j++) s += (int32_t)a->qkv[0][h][j] * b->qkv[1][h][j]; return s;
    }
    uint32_t same = ~(a->neg[0][h] ^ b->neg[1][h]), diff = a->neg[0][h] ^ b->neg[1][h];
    i64 s = 0;
    for (int x = 0; x < 2; x++) for (int y = 0; y < 2; y++) for (int i = 0; i < 5; i++) for (int c = 0; c < 5; c++) {
        i64 tv = M->tab[l][h][x][y][i][c]; if (!tv) continue;
        uint32_t qa = x ? a->mag[0][h][i] : ~a->mag[0][h][i], kb = y ? b->mag[1][h][c] : ~b->mag[1][h][c], g = qa & kb;
        s += tv * (__builtin_popcount(g & same) - __builtin_popcount(g & diff));
    }
    return s;
}

/* one window of T bytes -> integer logits [T][256] (and the final 1/rms minifloats for CE) */
static void forward(const Model *M, const Tok *tab0, const uint8_t *X, i64 *logits, i64 *rs_m, i64 *rs_e) {
    i64 x[T][D]; Tok tk[T]; int khv[T];
    for (int t = 0; t < T; t++) for (int j = 0; j < D; j++) x[t][j] = M->tok[(size_t)X[t] * D + j] + M->pos[(size_t)t * D + j];
    for (int l = 0; l < M->L; l++) {
        for (int t = 0; t < T; t++) {
            if (l == 0 && tab0) tk[t] = tab0[(size_t)X[t] * T + t];
            else tok_inputs(M, l, x[t], &tk[t]);
        }
        for (int t = 0; t < T; t++) {
            int s0 = t - M->W + 1 < 0 ? 0 : t - M->W + 1, n = t - s0 + 1;
            i64 o[NH][HD], evmin_h[NH];
            for (int h = 0; h < NH; h++) {
                i64 sc[16], E[16], emin = INT64_MAX, evmin = INT64_MAX, Emax = INT64_MIN, wgt[16], den = 0;
                for (int i = 0; i < n; i++) {
                    const Tok *ks = &tk[s0 + i];
                    sc[i] = match(M, l, h, &tk[t], ks);
                    if (ks->ex[1][h] < emin) emin = ks->ex[1][h];
                    if (ks->ex[2][h] < evmin) evmin = ks->ex[2][h];
                }
                for (int i = 0; i < n; i++) {
                    const Tok *ks = &tk[s0 + i]; E[i] = sc[i] * ks->mc * ((i64)1 << (ks->ex[1][h] - emin));
                    if (E[i] > Emax) Emax = E[i];
                }
                i64 thr[16]; i64 eq = tk[t].ex[0][h];
                for (int v = 0; v < M->LV; v++) {
                    i64 am, ae; mf_mul(M->thm_m[l][h * M->LV + v], M->thm_e[l][h * M->LV + v], M->rm[tk[t].mc][0], M->rm[tk[t].mc][1], &am, &ae);
                    i64 sh = ae - eq - emin; thr[v] = sh >= 0 ? am << sh : (sh <= -63 ? 0 : am >> (-sh));
                }
                const i64 *lut = M->lut[l] + h * M->LV;
                for (int i = 0; i < n; i++) {
                    i64 gap = Emax - E[i], wv = lut[0];
                    for (int v = 0; v < M->LV; v++) if (gap > thr[v]) wv += (v + 1 < M->LV ? lut[v + 1] - lut[v] : -lut[v]);
                    wgt[i] = wv < 0 ? 0 : wv; den += wgt[i];
                }
                if (den < 1) den = 1;
                i64 num[HD]; for (int j = 0; j < HD; j++) num[j] = 0;
                for (int i = 0; i < n; i++) {
                    const Tok *ks = &tk[s0 + i]; if (!wgt[i]) continue;
                    i64 f = wgt[i] * ks->mc * ((i64)1 << (ks->ex[2][h] - evmin)); const int8_t *vv = ks->qkv[2][h];
                    for (int j = 0; j < HD; j++) num[j] += f * vv[j];
                }
                for (int j = 0; j < HD; j++) o[h][j] = floordiv(num[j], den);
                evmin_h[h] = evmin;
            }
            i64 eo = evmin_h[0]; for (int h = 1; h < NH; h++) if (evmin_h[h] < eo) eo = evmin_h[h];
            i64 of[D]; int8_t oq[D]; i64 zo[D];
            for (int h = 0; h < NH; h++) for (int j = 0; j < HD; j++) of[h * HD + j] = o[h][j] * ((i64)1 << (evmin_h[h] - eo));
            int k3 = q6(of, D, oq);
            if (MM_VNNI && !MM_POPCOUNT) mm_vnni(oq, D, M->Wov[l], M->Wos[l], zo); else mm(oq, D, M->WoT[l], M->Woq[l], zo);
            int sh = (int)(M->go[l][1] + eo + k3);
            for (int c = 0; c < D; c++) x[t][c] += shift_round(zo[c] * M->go[l][0], sh);   /* updated after all heads */
        }
    }
    for (int t = 0; t < T; t++) {
        i64 yf[D], S = 0;
        for (int j = 0; j < D; j++) { yf[j] = x[t][j] * M->gfi[j]; S += x[t][j] * x[t][j]; }
        int16_t y16[D]; int kh = q16(yf, D, y16); (void)kh;
#ifdef __AVX512VNNI__
        if (MM_VNNI) {
            for (int cb = 0; cb < 16; cb++) {
                __m512i acc = _mm512_setzero_si512(); const int16_t *hw = M->headv + (size_t)cb * D * 16;
                for (int j2 = 0; j2 < D / 2; j2++) {
                    int32_t b; memcpy(&b, y16 + 2 * j2, 4);
                    acc = _mm512_dpwssd_epi32(acc, _mm512_set1_epi32(b), _mm512_loadu_si512(hw + (size_t)j2 * 32));
                }
                int32_t r[16]; _mm512_storeu_si512(r, acc);
                for (int i = 0; i < 16; i++) { int v = cb * 16 + i; logits[(size_t)t * 256 + v] = (i64)r[i] * M->mh_m[v] * ((i64)1 << (M->mh_e[v] - M->emin_h)); }
            }
        } else
#endif
        for (int v = 0; v < 256; v++) {
            i64 s = 0; const int32_t *hw = M->headT + (size_t)v * D;
            for (int j = 0; j < D; j++) s += (i64)y16[j] * hw[j];
            logits[(size_t)t * 256 + v] = s * M->mh_m[v] * ((i64)1 << (M->mh_e[v] - M->emin_h));
        }
        khv[t] = kh;
        if (rs_m) { i64 a, b; rsqrt_mf(M, S, &a, &b); mf_mul(M->sqd[0], M->sqd[1], a, b, &rs_m[t], &rs_e[t]); rs_e[t] += khv[t]; }
    }
}

static Tok *layer0_table(const Model *M) {
    Tok *tab = xal((size_t)256 * T * sizeof(Tok));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 256 * T; i++) {
        i64 x[D]; int b = i / T, t = i % T;
        for (int j = 0; j < D; j++) x[j] = M->tok[(size_t)b * D + j] + M->pos[(size_t)t * D + j];
        tok_inputs(M, 0, x, &tab[i]);
    }
    return tab;
}
static uint8_t *read_all(const char *p, long *n) {
    FILE *f = fopen(p, "rb"); if (!f) die("cannot open file");
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); uint8_t *b = xal(*n);
    if (fread(b, 1, *n, f) != (size_t)*n) die("read failed");
    fclose(f); return b;
}

int main(int argc, char **argv) {
    if (argc < 3) die("usage: lgn verify|eval|bench MODEL.bin [TEXT] [--mm popcount|int] [--table 0|1]");
    int use_table = 1;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--mm") && i + 1 < argc) MM_POPCOUNT = !strcmp(argv[++i], "popcount");
        else if (!strcmp(argv[i], "--table") && i + 1 < argc) use_table = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vnni") && i + 1 < argc) MM_VNNI = atoi(argv[++i]);
    }
    Model *M = load(argv[2]);
    Tok *tab0 = use_table ? layer0_table(M) : NULL;
    if (!strcmp(argv[1], "verify")) {
        char p[4096]; long nx, nr; snprintf(p, sizeof p, "%s.x", argv[2]); uint8_t *X = read_all(p, &nx);
        snprintf(p, sizeof p, "%s.ref", argv[2]); i64 *ref = (i64 *)read_all(p, &nr);
        long nw = nx / T, bad = 0, same_arg = 0; i64 *lg = xal((size_t)T * 256 * 8);
        for (long w = 0; w < nw; w++) {
            forward(M, tab0, X + w * T, lg, NULL, NULL);
            for (int i = 0; i < T * 256; i++) bad += lg[i] != ref[w * T * 256 + i];
        }
        printf("{\"windows\":%ld,\"logits\":%ld,\"mismatched_integer_logits\":%ld,\"verify\":\"%s\"}\n",
               nw, nw * T * 256, bad, bad ? "FAILED" : "passed");
        (void)same_arg; return bad != 0;
    }
    if (argc < 4) die("eval / bench need TEXT");
    long len; uint8_t *txt = read_all(argv[3], &len); long lo = len * 9 / 10, nw = (len - lo - 1) / T;
    if (!strcmp(argv[1], "eval")) {
        double ce = 0; long hit = 0;
        #pragma omp parallel for schedule(dynamic, 16) reduction(+:ce, hit)
        for (long w = 0; w < nw; w++) {
            i64 lg[T * 256], rm[T], re[T]; const uint8_t *X = txt + lo + w * T;
            forward(M, tab0, X, lg, rm, re);
            for (int t = 0; t < T; t++) {
                int y = X[t + 1], am = 0; double sc = ldexp((double)rm[t], (int)(re[t] + M->emin_h - M->G)), mx = -1e300, z = 0;
                for (int v = 0; v < 256; v++) { if (lg[t * 256 + v] > lg[t * 256 + am]) am = v; double r = lg[t * 256 + v] * sc; if (r > mx) mx = r; }
                for (int v = 0; v < 256; v++) z += exp(lg[t * 256 + v] * sc - mx);
                ce += mx + log(z) - lg[t * 256 + y] * sc; hit += am == y;
            }
        }
        long n = nw * T;
        printf("{\"bytes\":%ld,\"ce\":%.6f,\"acc\":%.6f}\n", n, ce / n, (double)hit / n);
        return 0;
    }
    if (!strcmp(argv[1], "bench")) {
        long nb = nw < 2048 ? nw : 2048; i64 *lg = xal((size_t)T * 256 * 8);
        double t0 = omp_get_wtime(); long done = 0;
        while (omp_get_wtime() - t0 < 2.0) { forward(M, tab0, txt + lo + (done % nb) * T, lg, NULL, NULL); done++; }
        double one = (omp_get_wtime() - t0) / done;
        t0 = omp_get_wtime();
        #pragma omp parallel
        {
            i64 *l2 = xal((size_t)T * 256 * 8);
            #pragma omp for schedule(dynamic, 8)
            for (long w = 0; w < nb; w++) forward(M, tab0, txt + lo + w * T, l2, NULL, NULL);
            free(l2);
        }
        double all = (omp_get_wtime() - t0) / nb;
        printf("{\"mm\":\"%s\",\"layer0_table\":%d,\"ns_per_token_1thread\":%.0f,\"ns_per_token_all_threads\":%.0f,"
               "\"threads\":%d,\"us_per_64byte_window_1thread\":%.1f}\n", MM_POPCOUNT ? "popcount" : "int", use_table,
               one / T * 1e9, all / T * 1e9, omp_get_max_threads(), one * 1e6);
        return 0;
    }
    die("unknown command");
}
