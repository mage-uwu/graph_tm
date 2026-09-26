/* bitnet.c: a small BitNet b1.58 transformer that trains on the CPU, in one file. Built to train toy tasks in
 * seconds, not for full-size runs.
 *
 *   gcc -O3 -march=native -fopenmp bitnet.c -lm -o bitnet
 *   ./bitnet train --task copy --arch full --steps 300
 *   ./bitnet train --task text --data input.txt --arch full --steps 500
 *   ./bitnet selftest                  gradient check (float mode, all 3 archs) + thread-count determinism
 *
 * MODEL
 *   tokens -> float embedding + learned positions -> S residual sublayers (pre-norm) -> RMSNorm -> float LM head
 *     attention  RMSNorm -> BitLinear qkv (d -> 3d) -> causal softmax attention (float) -> BitLinear o (d -> d)
 *     mlp        RMSNorm -> BitLinear up (d -> hidden) -> ReLU^2 -> BitLinear down (hidden -> d)
 *   --arch full   layers x (attention, mlp)
 *   --arch attn   layers x attention            (no MLP layers)
 *   --arch mlp    layers x mlp                  (no attention: no mixing across positions)
 *   BitLinear (b1.58): weights ternary {-1,0,+1} x gamma, gamma = mean |W| over the matrix; inputs int8 per token
 *   (absmax -> 127). The product is an integer matrix product, scaled by gamma * absmax / 127. It runs as float
 *   arithmetic on integer values, which is exact (|sum| < 2^24), so the forward pass is the deployed integer model.
 *   Embeddings, norms, attention scores and the LM head stay float, as in BitNet.
 *   Training: float latent weights, AdamW, straight-through estimator for both quantizers (gradients pass the
 *   rounding as identity). --quant 0 turns the quantizers off (plain float transformer, used by the gradient check).
 *
 * SPEED: every matrix product goes through one register-blocked GEMM (4 rows x 32 columns of accumulators, vector
 *   extensions, OpenMP over row blocks); weights are quantized once per step; attention is parallel over
 *   (sequence, head).
 * DETERMINISM: randomness is counter-based (a pure function of seed, tag, index); every parallel loop writes
 *   disjoint outputs and every sum has a fixed order, so training is identical at any thread count.
 *
 * TASKS (built in; train and validation batches are drawn from disjoint counter streams)
 *   copy   32 random symbols (alphabet 16), a separator, the same 32 symbols again; loss on the copied half only.
 *          Chance 2.773 nats. Needs attention: --arch mlp cannot beat chance.
 *   text   byte-level language model on --data FILE (first 90% train, last 10% validation).
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *fmt, ...) {
    va_list a; va_start(a, fmt); fprintf(stderr, "bitnet: "); vfprintf(stderr, fmt, a); fprintf(stderr, "\n");
    va_end(a); exit(1);
}
static void *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) die("out of memory (%zu bytes)", n);
    memset(p, 0, n ? n : 64); return p;
}

/* ------------------------------ counter-based randomness ------------------------------ */
static uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}
static uint64_t rnd(uint64_t seed, uint64_t tag, uint64_t i) { return mix(mix(mix(seed) ^ tag) ^ i); }
static double unif(uint64_t seed, uint64_t tag, uint64_t i) { return ((rnd(seed, tag, i) >> 11) + .5) * 0x1p-53; }
static float gauss(uint64_t seed, uint64_t tag, uint64_t i) {
    return (float)(sqrt(-2 * log(unif(seed, tag, 2 * i))) * cos(6.283185307179586 * unif(seed, tag, 2 * i + 1)));
}
enum { TAG_INIT = 1000, TAG_TRAIN = 1, TAG_VAL = 2 };

/* ------------------------------------ GEMM ------------------------------------ */
typedef float v16 __attribute__((vector_size(64)));
typedef float v16u __attribute__((vector_size(64), aligned(4)));   /* for unaligned loads / stores */
/* C[N x M] = A . B, B row-major [K x M], A element (i, k) at A[i * ai + k * ak] (ai = K, ak = 1: A row-major;
 * ai = 1, ak = K: A^T of a row-major [K x N] matrix... see gemm_tn). BLAS-style packing: A into 4-row panels
 * (k-major), B into 32-column panels (zero-padded), so the 4 x 32 register kernel streams contiguous memory and ragged
 * edges take the same path. Jobs are column-panel major (a thread reuses its B panel). Each C element sums k in
 * order, so results do not depend on the thread count. */
static float *PA, *PB; static size_t PAn, PBn;
static float *scratch(float **p, size_t *cap, size_t n) {
    if (n > *cap) { free(*p); *p = xalloc(n * 4); *cap = n; }
    return *p;
}
static void gemm_s(float *restrict C, const float *restrict A, size_t ai, size_t ak, const float *restrict B,
                   int N, int K, int M) {
    int ni = (N + 3) / 4, nj = (M + 31) / 32;
    float *pa = scratch(&PA, &PAn, (size_t)ni * K * 4), *pb = scratch(&PB, &PBn, (size_t)nj * K * 32);
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int ib = 0; ib < ni; ib++) {
            float *d = pa + (size_t)ib * K * 4; size_t i0 = (size_t)ib * 4; int nr = N - (int)i0 < 4 ? N - (int)i0 : 4;
            for (int k = 0; k < K; k++, d += 4)
                for (int r = 0; r < 4; r++) d[r] = r < nr ? A[(i0 + r) * ai + k * ak] : 0;
        }
        #pragma omp for schedule(static)
        for (int jb = 0; jb < nj; jb++) {
            float *d = pb + (size_t)jb * K * 32; int j0 = jb * 32, w = M - j0 < 32 ? M - j0 : 32;
            for (int k = 0; k < K; k++) {
                const float *src = B + (size_t)k * M + j0; float *o = d + (size_t)k * 32; int j = 0;
                for (; j < w; j++) o[j] = src[j];
                for (; j < 32; j++) o[j] = 0;
            }
        }
        #pragma omp for schedule(static)
        for (int job = 0; job < ni * nj; job++) {
            int ib = job % ni, jb = job / ni, i0 = ib * 4, j0 = jb * 32;
            int nr = N - i0 < 4 ? N - i0 : 4, w = M - j0 < 32 ? M - j0 : 32;
            const float *a = pa + (size_t)ib * K * 4, *b = pb + (size_t)jb * K * 32;
            v16 c0a = {0}, c0b = {0}, c1a = {0}, c1b = {0}, c2a = {0}, c2b = {0}, c3a = {0}, c3b = {0};
            for (int k = 0; k < K; k++, a += 4, b += 32) {
                v16 ba = *(const v16u *)b, bb = *(const v16u *)(b + 16);
                c0a += a[0] * ba; c0b += a[0] * bb; c1a += a[1] * ba; c1b += a[1] * bb;
                c2a += a[2] * ba; c2b += a[2] * bb; c3a += a[3] * ba; c3b += a[3] * bb;
            }
            v16 acc[8] = {c0a, c0b, c1a, c1b, c2a, c2b, c3a, c3b};
            for (int r = 0; r < nr; r++) {
                float *c = C + (size_t)(i0 + r) * M + j0;
                if (w == 32) { *(v16u *)c = acc[2 * r]; *(v16u *)(c + 16) = acc[2 * r + 1]; }
                else for (int j = 0; j < w; j++) c[j] = j < 16 ? acc[2 * r][j] : acc[2 * r + 1][j - 16];
            }
        }
    }
}
static void gemm(float *C, const float *A, const float *B, int N, int K, int M) { gemm_s(C, A, K, 1, B, N, K, M); }
/* C[K x M] = A^T . B, A row-major [N x K], B row-major [N x M] */
static void gemm_tn(float *C, const float *A, const float *B, int N, int K, int M) { gemm_s(C, A, 1, K, B, K, N, M); }
/* dst[C x R] = src[R x C]^T */
static void transpose(float *restrict dst, const float *restrict src, int R, int C) {
    #pragma omp parallel for schedule(static)
    for (int j0 = 0; j0 < C; j0 += 16)
        for (int i = 0; i < R; i++)
            for (int j = j0; j < j0 + 16 && j < C; j++) dst[(size_t)j * R + i] = src[(size_t)i * C + j];
}

/* ------------------------------------ parameters ------------------------------------ */
typedef struct { float *w, *g, *m, *v; size_t n; int decay; } Ten;
typedef struct { Ten *W; int K, M; float *wq, *wqT, gamma; } Lin;   /* W is [K in x M out] */
enum { ATT = 0, MLP = 1 };
typedef struct {
    int type; Ten *g; Lin a, b;                  /* ATT: a = qkv (d -> 3d), b = o (d -> d); MLP: a = up, b = down */
    float *x, *rstd, *xa, *sa, *h, *r, *xb, *sb, *P;   /* saved activations (xa/xb: int8 values, sa/sb: 1/scale) */
} Sub;
typedef struct {
    int V, T, d, L, H, hid, arch, quant, B, steps, warm, eval_every, eval_batches, task, threads;
    float lr, wd, clip; uint64_t seed; const char *data, *save;
} Cfg;
typedef struct {
    Cfg c; int S, N; Ten ts[256]; int nt; Ten *tok, *pos, *gf, *head; Sub *sub;
    float *xf, *rstdf, *nf, *logits, *dlog, *loss_i, *hit, *wt; int *ids, *tgt;
    float *d0, *dn, *dh, *dr, *t0, *t1;     /* gradient and scratch buffers */
} Net;

static Ten *ten(Net *n, size_t len, int decay, float std, uint64_t tag) {
    if (n->nt == 256) die("too many tensors");
    Ten *t = n->ts + n->nt++; t->n = len; t->decay = decay;
    t->w = xalloc(len * 4); t->g = xalloc(len * 4); t->m = xalloc(len * 4); t->v = xalloc(len * 4);
    for (size_t i = 0; i < len; i++) t->w[i] = std < 0 ? 1.f : std * gauss(n->c.seed, TAG_INIT + tag, i);
    return t;
}
static Lin lin(Net *n, int K, int M, float std, uint64_t tag) {
    Lin l = {ten(n, (size_t)K * M, 1, std, tag), K, M, xalloc((size_t)K * M * 4), xalloc((size_t)K * M * 4), 1};
    return l;
}
static size_t nparams(Net *n) { size_t s = 0; for (int i = 0; i < n->nt; i++) s += n->ts[i].n; return s; }

static Net *net_new(Cfg c) {
    Net *n = xalloc(sizeof(Net)); n->c = c;
    if (c.d % c.H) die("--dim must be divisible by --heads");
    int per = c.arch == 2 ? 2 : 1; n->S = c.L * per; n->N = c.B * c.T;
    int N = n->N, d = c.d, W3 = 3 * d > c.hid ? 3 * d : c.hid, WX = W3 > c.V ? W3 : c.V;
    float res = .02f / sqrtf(2.f * n->S);
    n->tok = ten(n, (size_t)c.V * d, 0, .02f, 1); n->pos = ten(n, (size_t)c.T * d, 0, .02f, 2);
    n->sub = xalloc(sizeof(Sub) * n->S);
    for (int s = 0; s < n->S; s++) {
        Sub *u = n->sub + s; u->type = c.arch == 0 ? ATT : c.arch == 1 ? MLP : s % 2 ? MLP : ATT;
        uint64_t tg = 100 + 10 * (uint64_t)s; u->g = ten(n, d, 0, -1, tg);
        int H1 = u->type == ATT ? 3 * d : c.hid, R1 = u->type == ATT ? d : c.hid;
        u->a = lin(n, d, H1, .02f, tg + 1); u->b = lin(n, R1, d, res, tg + 2);
        u->x = xalloc((size_t)N * d * 4); u->rstd = xalloc(N * 4); u->xa = xalloc((size_t)N * d * 4);
        u->sa = xalloc(N * 4); u->h = xalloc((size_t)N * H1 * 4); u->r = xalloc((size_t)N * R1 * 4);
        u->xb = xalloc((size_t)N * R1 * 4); u->sb = xalloc(N * 4);
        if (u->type == ATT) u->P = xalloc((size_t)c.B * c.H * c.T * c.T * 4);
    }
    n->gf = ten(n, d, 0, -1, 3); n->head = ten(n, (size_t)d * c.V, 1, .02f, 4);
    n->xf = xalloc((size_t)N * d * 4); n->rstdf = xalloc(N * 4); n->nf = xalloc((size_t)N * d * 4);
    n->logits = xalloc((size_t)N * c.V * 4); n->dlog = xalloc((size_t)N * c.V * 4);
    n->loss_i = xalloc(N * 4); n->hit = xalloc(N * 4); n->wt = xalloc(N * 4); n->ids = xalloc(N * 4); n->tgt = xalloc(N * 4);
    n->d0 = xalloc((size_t)N * d * 4); n->dn = xalloc((size_t)N * d * 4);
    n->dh = xalloc((size_t)N * W3 * 4); n->dr = xalloc((size_t)N * W3 * 4);
    size_t tsz = (size_t)N * WX > (size_t)WX * d ? (size_t)N * WX : (size_t)WX * d;
    n->t0 = xalloc(tsz * 4); n->t1 = xalloc(tsz * 4);
    return n;
}

static void net_free(Net *n) {
    for (int t = 0; t < n->nt; t++) { free(n->ts[t].w); free(n->ts[t].g); free(n->ts[t].m); free(n->ts[t].v); }
    for (int s = 0; s < n->S; s++) {
        Sub *u = n->sub + s; float *f[] = {u->a.wq, u->a.wqT, u->b.wq, u->b.wqT, u->x, u->rstd, u->xa, u->sa, u->h, u->r, u->xb, u->sb, u->P};
        for (size_t i = 0; i < sizeof f / sizeof *f; i++) free(f[i]);
    }
    float *f[] = {n->xf, n->rstdf, n->nf, n->logits, n->dlog, n->loss_i, n->hit, n->wt, n->d0, n->dn, n->dh, n->dr, n->t0, n->t1};
    for (size_t i = 0; i < sizeof f / sizeof *f; i++) free(f[i]);
    free(n->ids); free(n->tgt); free(n->sub); free(n);
}

/* ------------------------------------ BitLinear ------------------------------------ */
/* round to nearest, ties to even (|x| < 2^22): exact, and it vectorizes where nearbyintf does not */
static inline float rne(float x) { return (x + 12582912.f) - 12582912.f; }
static void lin_quant(Lin *l, int quant) {
    size_t n = (size_t)l->K * l->M; const float *w = l->W->w;
    if (!quant) { memcpy(l->wq, w, n * 4); l->gamma = 1; }
    else {
        enum { BLK = 4096 }; size_t nb = (n + BLK - 1) / BLK; double *part = xalloc(nb * 8), s = 0;
        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < nb; b++) {
            float t = 0; for (size_t i = b * BLK; i < n && i < (b + 1) * BLK; i++) t += fabsf(w[i]);
            part[b] = t;
        }
        for (size_t b = 0; b < nb; b++) s += part[b];     /* fixed blocks, fixed order */
        free(part); l->gamma = (float)(s / n); float inv = 1.f / (l->gamma + 1e-8f);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) { float t = w[i] * inv; l->wq[i] = fminf(1.f, fmaxf(-1.f, rne(fminf(4.f, fmaxf(-4.f, t))))); }
    }
    transpose(l->wqT, l->wq, l->K, l->M);
}
typedef int vi16 __attribute__((vector_size(64)));
typedef int vi16u __attribute__((vector_size(64), aligned(4)));
static void act_quant(const float *x, float *xq, float *is, int N, int K, int quant) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        const float *a = x + (size_t)i * K; float *q = xq + (size_t)i * K;
        if (!quant) { memcpy(q, a, K * 4); is[i] = 1; continue; }
        float mx = 0; int k = 0;
        if (K >= 16) {
            v16 m = {0};
            for (; k + 16 <= K; k += 16) {
                v16 v = (v16)(*(const vi16u *)(a + k) & 0x7fffffff); vi16 gt = v > m;
                m = (v16)(((vi16)v & gt) | ((vi16)m & ~gt));
            }
            for (int j = 0; j < 16; j++) mx = fmaxf(mx, m[j]);
        }
        for (; k < K; k++) mx = fmaxf(mx, fabsf(a[k]));
        float s = 127.f / fmaxf(mx, 1e-5f); is[i] = 1.f / s;   /* |a * s| <= 127: no clamp needed */
        for (k = 0; k + 16 <= K; k += 16) { v16 t = *(const v16u *)(a + k) * s; *(v16u *)(q + k) = (t + 12582912.f) - 12582912.f; }
        for (; k < K; k++) q[k] = rne(a[k] * s);
    }
}
static void lin_fwd(Lin *l, const float *x, float *xq, float *is, float *y, int N, int quant) {
    act_quant(x, xq, is, N, l->K, quant);
    gemm(y, xq, l->wq, N, l->K, l->M);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float s = l->gamma * is[i]; float *r = y + (size_t)i * l->M;
        for (int j = 0; j < l->M; j++) r[j] *= s;
    }
}
/* straight-through: dx = gamma * dy . wq^T ; dW = (xq * is)^T . dy */
static void lin_bwd(Net *n, Lin *l, const float *xq, const float *is, const float *dy, float *dx, int N) {
    int K = l->K, M = l->M;
    if (dx) {
        gemm(dx, dy, l->wqT, N, M, K);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)N * K; i++) dx[i] *= l->gamma;
    }
    float *dys = n->t0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) dys[(size_t)i * M + j] = dy[(size_t)i * M + j] * is[i];
    gemm_tn(l->W->g, xq, dys, N, K, M);
}

/* ------------------------------------ layers ------------------------------------ */
static void rms_fwd(const float *x, const float *g, float *y, float *rstd, int N, int d) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        const float *a = x + (size_t)i * d; float s = 0;
        for (int k = 0; k < d; k++) s += a[k] * a[k];
        float r = 1.f / sqrtf(s / d + 1e-6f); rstd[i] = r;
        for (int k = 0; k < d; k++) y[(size_t)i * d + k] = a[k] * r * g[k];
    }
}
/* dx = (acc ? dx : 0) + d rms/dx ; dg = sum over rows */
static void rms_bwd(const float *x, const float *rstd, Ten *g, const float *dy, float *dx, int N, int d, int acc) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        const float *a = x + (size_t)i * d, *e = dy + (size_t)i * d; float r = rstd[i], dot = 0;
        for (int k = 0; k < d; k++) dot += g->w[k] * e[k] * a[k];
        float c = r * r * dot / d; float *o = dx + (size_t)i * d;
        for (int k = 0; k < d; k++) { float v = r * (g->w[k] * e[k] - a[k] * c); o[k] = acc ? o[k] + v : v; }
    }
    #pragma omp parallel for schedule(static)
    for (int k0 = 0; k0 < d; k0 += 16) {
        int w = d - k0 < 16 ? d - k0 : 16; float s[16] = {0};
        for (int i = 0; i < N; i++) {
            const float *e = dy + (size_t)i * d + k0, *a = x + (size_t)i * d + k0; float r = rstd[i];
            for (int k = 0; k < w; k++) s[k] += e[k] * a[k] * r;
        }
        for (int k = 0; k < w; k++) g->g[k0 + k] = s[k];
    }
}
/* Causal attention per (sequence, head), parallel over (sequence, head). Q, K^T, V^T are gathered into
 * contiguous buffers so every inner loop is a vectorisable axpy; each output sums in a fixed order. */
static void gather(const Cfg *c, const float *qkv, int b, int h, float *Q, float *KT, float *V, float *VT) {
    int T = c->T, d = c->d, hd = d / c->H;
    for (int t = 0; t < T; t++) {
        const float *r = qkv + (size_t)(b * T + t) * 3 * d + h * hd;
        for (int j = 0; j < hd; j++) {
            Q[t * hd + j] = r[j]; KT[(size_t)j * T + t] = r[d + j];
            if (V) V[t * hd + j] = r[2 * d + j];
            if (VT) VT[(size_t)j * T + t] = r[2 * d + j];
        }
    }
}
static void attn_fwd(const Cfg *c, const float *qkv, float *P, float *out) {
    int T = c->T, d = c->d, H = c->H, hd = d / H; float sc = 1.f / sqrtf((float)hd);
    #pragma omp parallel
    {
        float *Q = xalloc((size_t)T * hd * 4), *KT = xalloc((size_t)T * hd * 4), *V = xalloc((size_t)T * hd * 4);
        #pragma omp for schedule(static)
        for (int bh = 0; bh < c->B * H; bh++) {
            int b = bh / H, h = bh % H; float *Pb = P + (size_t)bh * T * T;
            gather(c, qkv, b, h, Q, KT, V, NULL);
            for (int t = 0; t < T; t++) {
                float *p = Pb + (size_t)t * T, mx = -INFINITY; const float *q = Q + t * hd;
                for (int s = 0; s <= t; s++) p[s] = 0;
                for (int j = 0; j < hd; j++) { const float *k = KT + (size_t)j * T; float qj = q[j]; for (int s = 0; s <= t; s++) p[s] += qj * k[s]; }
                for (int s = 0; s <= t; s++) { p[s] *= sc; mx = fmaxf(mx, p[s]); }
                float z = 0; for (int s = 0; s <= t; s++) { p[s] = expf(p[s] - mx); z += p[s]; }
                float iz = 1.f / z; for (int s = 0; s <= t; s++) p[s] *= iz;
                for (int s = t + 1; s < T; s++) p[s] = 0;
                float *o = out + (size_t)(b * T + t) * d + h * hd;
                for (int j = 0; j < hd; j++) o[j] = 0;
                for (int s = 0; s <= t; s++) { const float *v = V + s * hd; float ps = p[s]; for (int j = 0; j < hd; j++) o[j] += ps * v[j]; }
            }
        }
        free(Q); free(KT); free(V);
    }
}
static void attn_bwd(const Cfg *c, const float *qkv, const float *P, const float *dout, float *dqkv) {
    int T = c->T, d = c->d, H = c->H, hd = d / H; float sc = 1.f / sqrtf((float)hd);
    #pragma omp parallel
    {
        float *Q = xalloc((size_t)T * hd * 4), *KT = xalloc((size_t)T * hd * 4), *VT = xalloc((size_t)T * hd * 4);
        float *K = xalloc((size_t)T * hd * 4), *dQ = xalloc((size_t)T * hd * 4), *dK = xalloc((size_t)T * hd * 4);
        float *dV = xalloc((size_t)T * hd * 4), *dp = xalloc((size_t)T * 4);
        #pragma omp for schedule(static)
        for (int bh = 0; bh < c->B * H; bh++) {
            int b = bh / H, h = bh % H; const float *Pb = P + (size_t)bh * T * T;
            gather(c, qkv, b, h, Q, KT, NULL, VT);
            for (int t = 0; t < T; t++) for (int j = 0; j < hd; j++) K[t * hd + j] = KT[(size_t)j * T + t];
            memset(dQ, 0, (size_t)T * hd * 4); memset(dK, 0, (size_t)T * hd * 4); memset(dV, 0, (size_t)T * hd * 4);
            for (int t = 0; t < T; t++) {
                const float *p = Pb + (size_t)t * T, *go = dout + (size_t)(b * T + t) * d + h * hd, *q = Q + t * hd;
                for (int s = 0; s <= t; s++) dp[s] = 0;
                for (int j = 0; j < hd; j++) { const float *v = VT + (size_t)j * T; float g = go[j]; for (int s = 0; s <= t; s++) dp[s] += g * v[s]; }
                float sum = 0; for (int s = 0; s <= t; s++) sum += p[s] * dp[s];
                float *dq = dQ + t * hd;
                for (int s = 0; s <= t; s++) {
                    float ps = p[s], g = ps * (dp[s] - sum) * sc; const float *k = K + s * hd;
                    float *dk = dK + s * hd, *dv = dV + s * hd;
                    for (int j = 0; j < hd; j++) { dv[j] += ps * go[j]; dq[j] += g * k[j]; dk[j] += g * q[j]; }
                }
            }
            for (int t = 0; t < T; t++) {
                float *z = dqkv + (size_t)(b * T + t) * 3 * d + h * hd;
                for (int j = 0; j < hd; j++) { z[j] = dQ[t * hd + j]; z[d + j] = dK[t * hd + j]; z[2 * d + j] = dV[t * hd + j]; }
            }
        }
        free(Q); free(KT); free(VT); free(K); free(dQ); free(dK); free(dV); free(dp);
    }
}

/* ------------------------------------ model ------------------------------------ */
static void quantize_all(Net *n) {
    for (int s = 0; s < n->S; s++) { lin_quant(&n->sub[s].a, n->c.quant); lin_quant(&n->sub[s].b, n->c.quant); }
}
/* forward pass; returns the weighted mean cross-entropy; *acc = weighted argmax accuracy */
static double forward(Net *n, double *acc) {
    const Cfg *c = &n->c; int N = n->N, d = c->d, T = c->T, V = c->V; float *x;
    float *x0 = n->S ? n->sub[0].x : n->xf;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) x0[(size_t)i * d + k] = n->tok->w[(size_t)n->ids[i] * d + k] + n->pos->w[(size_t)(i % T) * d + k];
    for (int s = 0; s < n->S; s++) {
        Sub *u = n->sub + s; x = u->x; float *y = s + 1 < n->S ? n->sub[s + 1].x : n->xf;
        rms_fwd(x, u->g->w, n->dn, u->rstd, N, d);
        lin_fwd(&u->a, n->dn, u->xa, u->sa, u->h, N, c->quant);
        if (u->type == ATT) attn_fwd(c, u->h, u->P, u->r);
        else {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < (size_t)N * c->hid; i++) { float v = fmaxf(u->h[i], 0.f); u->r[i] = v * v; }
        }
        lin_fwd(&u->b, u->r, u->xb, u->sb, y, N, c->quant);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)N * d; i++) y[i] += x[i];
    }
    rms_fwd(n->xf, n->gf->w, n->nf, n->rstdf, N, d);
    gemm(n->logits, n->nf, n->head->w, N, d, V);
    double W = 0; for (int i = 0; i < N; i++) W += n->wt[i];
    float iw = (float)(1 / W);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float *l = n->logits + (size_t)i * V, *g = n->dlog + (size_t)i * V, mx = -INFINITY; int am = 0;
        for (int j = 0; j < V; j++) if (l[j] > mx) { mx = l[j]; am = j; }
        float z = 0; for (int j = 0; j < V; j++) z += expf(l[j] - mx);
        float lse = mx + logf(z), w = n->wt[i] * iw;
        n->loss_i[i] = n->wt[i] ? lse - l[n->tgt[i]] : 0;
        for (int j = 0; j < V; j++) g[j] = w * (expf(l[j] - lse) - (j == n->tgt[i]));
        n->hit[i] = (float)(am == n->tgt[i]);
    }
    double L = 0, A = 0; for (int i = 0; i < N; i++) { L += n->wt[i] * n->loss_i[i]; A += n->wt[i] * n->hit[i]; }
    if (acc) *acc = A / W;
    return L / W;
}
static void backward(Net *n) {
    const Cfg *c = &n->c; int N = n->N, d = c->d, T = c->T, V = c->V;
    gemm_tn(n->head->g, n->nf, n->dlog, N, d, V);
    transpose(n->t1, n->head->w, d, V); gemm(n->dn, n->dlog, n->t1, N, V, d);
    float *dx = n->d0; rms_bwd(n->xf, n->rstdf, n->gf, n->dn, dx, N, d, 0);
    for (int s = n->S - 1; s >= 0; s--) {
        Sub *u = n->sub + s; int R1 = u->type == ATT ? d : c->hid;
        lin_bwd(n, &u->b, u->xb, u->sb, dx, n->dr, N);
        if (u->type == ATT) attn_bwd(c, u->h, u->P, n->dr, n->dh);
        else {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < (size_t)N * R1; i++) n->dh[i] = n->dr[i] * 2 * fmaxf(u->h[i], 0.f);
        }
        lin_bwd(n, &u->a, u->xa, u->sa, n->dh, n->dn, N);
        rms_bwd(u->x, u->rstd, u->g, n->dn, dx, N, d, 1);   /* residual: dx += d(sublayer)/dx */
    }
    memset(n->tok->g, 0, n->tok->n * 4); memset(n->pos->g, 0, n->pos->n * 4);
    #pragma omp parallel for schedule(static)
    for (int k0 = 0; k0 < d; k0 += 16)
        for (int i = 0; i < N; i++) {
            float *a = n->tok->g + (size_t)n->ids[i] * d, *p = n->pos->g + (size_t)(i % T) * d; const float *e = dx + (size_t)i * d;
            for (int k = k0; k < k0 + 16 && k < d; k++) { a[k] += e[k]; p[k] += e[k]; }
        }
}
static float lr_at(const Cfg *c, int step) {
    if (step < c->warm) return c->lr * (step + 1) / c->warm;
    double p = (double)(step - c->warm) / fmax(1, c->steps - c->warm);
    return (float)(c->lr * (.1 + .45 * (1 + cos(3.141592653589793 * p))));
}
static double adamw(Net *n, int step) {
    const Cfg *c = &n->c; double norm = 0;
    for (int t = 0; t < n->nt; t++) for (size_t i = 0; i < n->ts[t].n; i++) norm += (double)n->ts[t].g[i] * n->ts[t].g[i];
    norm = sqrt(norm); if (!isfinite(norm)) die("non-finite gradient at step %d", step);
    float clip = c->clip > 0 && norm > c->clip ? (float)(c->clip / norm) : 1.f, lr = lr_at(c, step);
    float b1c = 1 - powf(.9f, step + 1), b2c = 1 - powf(.95f, step + 1);
    for (int t = 0; t < n->nt; t++) {
        Ten *p = n->ts + t; float wd = p->decay ? c->wd : 0;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < p->n; i++) {
            float g = p->g[i] * clip;
            p->m[i] = .9f * p->m[i] + .1f * g; p->v[i] = .95f * p->v[i] + .05f * g * g;
            p->w[i] -= lr * ((p->m[i] / b1c) / (sqrtf(p->v[i] / b2c) + 1e-8f) + wd * p->w[i]);
        }
    }
    return norm;
}

/* ------------------------------------ tasks ------------------------------------ */
typedef struct { unsigned char *b; size_t n, ntrain; } Text;
static Text text_load(const char *path) {
    Text t = {0}; FILE *f = fopen(path, "rb"); if (!f) die("cannot open --data %s", path);
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    t.b = xalloc(len); if (fread(t.b, 1, len, f) != (size_t)len) die("read %s", path); fclose(f);
    t.n = len; t.ntrain = len * 9 / 10; return t;
}
/* fill ids / tgt / wt with batch `index` of stream `tag` */
static void batch(Net *n, const Text *tx, uint64_t tag, uint64_t index) {
    const Cfg *c = &n->c; int T = c->T;
    for (int b = 0; b < c->B; b++) {
        uint64_t r = index * c->B + b; int *id = n->ids + b * T, *tg = n->tgt + b * T; float *w = n->wt + b * T;
        if (c->task == 0) {       /* copy: x = s_1..s_h SEP s_1..s_h, h = T/2 */
            int h = T / 2, x[2 * 1024 + 2];
            for (int j = 0; j < h; j++) x[j] = x[h + 1 + j] = (int)(rnd(c->seed, tag, r * h + j) % 16);
            x[h] = 16;
            for (int t = 0; t < T; t++) { id[t] = x[t]; tg[t] = x[t + 1]; w[t] = t >= h; }
        } else {
            size_t lo = tag == TAG_TRAIN ? 0 : tx->ntrain, hi = tag == TAG_TRAIN ? tx->ntrain : tx->n;
            if (hi - lo < (size_t)T + 2) die("--data too short for --seq");
            size_t s = lo + rnd(c->seed, tag, r) % (hi - lo - T - 1);
            for (int t = 0; t < T; t++) { id[t] = tx->b[s + t]; tg[t] = tx->b[s + t + 1]; w[t] = 1; }
        }
    }
}
static double evaluate(Net *n, const Text *tx, double *acc) {
    double L = 0, A = 0, a;
    quantize_all(n);
    for (int e = 0; e < n->c.eval_batches; e++) { batch(n, tx, TAG_VAL, e); L += forward(n, &a); A += a; }
    *acc = A / n->c.eval_batches; return L / n->c.eval_batches;
}
static uint64_t weights_hash(Net *n) {
    uint64_t h = 1469598103934665603ull;
    for (int t = 0; t < n->nt; t++) {
        const unsigned char *p = (const unsigned char *)n->ts[t].w;
        for (size_t i = 0; i < n->ts[t].n * 4; i++) h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}
static void save(Net *n, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) die("cannot write %s", path);
    fwrite("BITNET01", 1, 8, f); fwrite(&n->c, sizeof(Cfg), 1, f);
    for (int t = 0; t < n->nt; t++) fwrite(n->ts[t].w, 4, n->ts[t].n, f);
    fclose(f);
}

/* ------------------------------------ CLI ------------------------------------ */
static Cfg defaults(void) {
    Cfg c = {0};
    c.V = 17; c.T = 64; c.d = 128; c.L = 2; c.H = 4; c.hid = 512; c.arch = 2; c.quant = 1; c.B = 32; c.steps = 300;
    c.warm = 20; c.eval_every = 50; c.eval_batches = 4; c.lr = 3e-3f; c.wd = .1f; c.clip = 1; c.seed = 1;
    c.threads = 0; return c;
}
static const char *ARCH[] = {"attn", "mlp", "full"};
static Cfg parse(int argc, char **argv) {
    Cfg c = defaults(); int vset = 0;
    for (int i = 2; i < argc; i++) {
        const char *k = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v) die("missing value for %s", k);
        i++;
        if (!strcmp(k, "--task")) c.task = !strcmp(v, "copy") ? 0 : !strcmp(v, "text") ? 1 : (die("--task copy|text"), 0);
        else if (!strcmp(k, "--arch")) { c.arch = -1; for (int a = 0; a < 3; a++) if (!strcmp(v, ARCH[a])) c.arch = a; if (c.arch < 0) die("--arch attn|mlp|full"); }
        else if (!strcmp(k, "--data")) c.data = v;
        else if (!strcmp(k, "--save")) c.save = v;
        else if (!strcmp(k, "--layers")) c.L = atoi(v);
        else if (!strcmp(k, "--dim")) c.d = atoi(v);
        else if (!strcmp(k, "--heads")) c.H = atoi(v);
        else if (!strcmp(k, "--hidden")) c.hid = atoi(v);
        else if (!strcmp(k, "--seq")) c.T = atoi(v);
        else if (!strcmp(k, "--batch")) c.B = atoi(v);
        else if (!strcmp(k, "--steps")) c.steps = atoi(v);
        else if (!strcmp(k, "--warmup")) c.warm = atoi(v);
        else if (!strcmp(k, "--eval-every")) c.eval_every = atoi(v);
        else if (!strcmp(k, "--eval-batches")) c.eval_batches = atoi(v);
        else if (!strcmp(k, "--lr")) c.lr = (float)atof(v);
        else if (!strcmp(k, "--wd")) c.wd = (float)atof(v);
        else if (!strcmp(k, "--clip")) c.clip = (float)atof(v);
        else if (!strcmp(k, "--seed")) c.seed = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--threads")) c.threads = atoi(v);
        else if (!strcmp(k, "--quant")) c.quant = atoi(v);
        else if (!strcmp(k, "--vocab")) { c.V = atoi(v); vset = 1; }
        else die("unknown option %s", k);
    }
    if (c.task == 1 && !vset) c.V = 256;
    if (c.task == 1 && !c.data) die("--task text needs --data FILE");
    if (c.task == 0 && (c.V < 17 || c.T > 2048)) die("copy needs --vocab >= 17 and --seq <= 2048");
    if (c.L < 1 || c.d < 1 || c.H < 1 || c.hid < 1 || c.T < 2 || c.B < 1 || c.steps < 1 || c.warm < 1) die("bad sizes");
    return c;
}
static int train(Cfg c, int quiet, uint64_t *hash, double *final) {
    if (c.threads > 0) omp_set_num_threads(c.threads);
    Text tx = {0}; if (c.task == 1) tx = text_load(c.data);
    Net *n = net_new(c);
    if (!quiet)
        printf("{\"arch\":\"%s\",\"task\":\"%s\",\"quant\":%d,\"layers\":%d,\"sublayers\":%d,\"dim\":%d,\"heads\":%d,"
               "\"hidden\":%d,\"seq\":%d,\"batch\":%d,\"vocab\":%d,\"params\":%zu,\"threads\":%d}\n",
               ARCH[c.arch], c.task ? "text" : "copy", c.quant, c.L, n->S, c.d, c.H, c.hid, c.T, c.B, c.V, nparams(n),
               omp_get_max_threads());
    double t0 = omp_get_wtime(), tl = t0, lsum = 0, vl = 0, va = 0; int ln = 0;
    for (int step = 0; step < c.steps; step++) {
        batch(n, &tx, TAG_TRAIN, step); quantize_all(n);
        double a, l = forward(n, &a); backward(n); double gn = adamw(n, step);
        lsum += l; ln++;
        if ((step + 1) % c.eval_every == 0 || step + 1 == c.steps) {
            double now = omp_get_wtime(), ms = 1e3 * (now - tl) / ln;
            vl = evaluate(n, &tx, &va);
            if (!quiet)
                printf("{\"step\":%d,\"train_loss\":%.4f,\"val_loss\":%.4f,\"val_acc\":%.4f,\"grad_norm\":%.3f,"
                       "\"ms_per_step\":%.1f,\"tokens_per_s\":%.0f,\"seconds\":%.2f}\n",
                       step + 1, lsum / ln, vl, va, gn, ms, 1e3 * n->N / ms, now - t0);
            fflush(stdout); lsum = 0; ln = 0; tl = omp_get_wtime();
        }
    }
    if (hash) *hash = weights_hash(n);
    if (final) *final = vl;
    if (c.save) save(n, c.save);
    net_free(n); free(tx.b);
    return 0;
}

/* gradient check in float mode (--quant 0): analytic vs central differences on sampled parameters */
static double gradcheck(int arch) {
    Cfg c = defaults(); c.arch = arch; c.quant = 0; c.L = 2; c.d = 16; c.H = 2; c.hid = 32; c.T = 8; c.B = 3; c.seed = 7;
    Net *n = net_new(c);
    for (int t = 0; t < n->nt; t++) for (size_t i = 0; i < n->ts[t].n; i++) n->ts[t].w[i] += .3f * gauss(9, t, i);
    batch(n, NULL, TAG_TRAIN, 0); quantize_all(n); forward(n, NULL); backward(n);
    double worst = 0;
    for (int t = 0; t < n->nt; t++)
        for (int j = 0; j < 4; j++) {
            Ten *p = n->ts + t; size_t i = rnd(3, t, j) % p->n; float e = 1e-2f;
            if (p == n->tok) i = (size_t)n->ids[j] * c.d + j;   /* a row that is used */
            float old = p->w[i];
            p->w[i] = old + e; quantize_all(n); double lp = forward(n, NULL);
            p->w[i] = old - e; quantize_all(n); double lm = forward(n, NULL);
            p->w[i] = old; double num = (lp - lm) / (2 * e), an = p->g[i];
            double rel = fabs(num - an) / fmax(fabs(num) + fabs(an), 1e-3);
            if (rel > worst) worst = rel;
        }
    net_free(n);
    return worst;
}
static int selftest(void) {
    int ok = 1;
    for (int a = 0; a < 3; a++) {
        double w = gradcheck(a); int pass = w < 2e-2; ok &= pass;
        printf("{\"check\":\"gradient\",\"arch\":\"%s\",\"max_relative_error\":%.2e,\"passed\":%s}\n", ARCH[a], w, pass ? "true" : "false");
    }
    for (int a = 0; a < 3; a++) {
        Cfg c = defaults(); c.arch = a; c.steps = 20; c.eval_every = 20; c.d = 64; c.hid = 256; c.B = 8;
        uint64_t h1, h2; double f1, f2;
        c.threads = 1; train(c, 1, &h1, &f1);
        c.threads = omp_get_num_procs(); train(c, 1, &h2, &f2);
        int pass = h1 == h2; ok &= pass;
        printf("{\"check\":\"threads_1_vs_%d\",\"arch\":\"%s\",\"hash\":\"%016llx\",\"passed\":%s}\n",
               omp_get_num_procs(), ARCH[a], (unsigned long long)h1, pass ? "true" : "false");
    }
    free(PA); free(PB); PA = PB = NULL; PAn = PBn = 0;
    printf("{\"selftest\":\"%s\"}\n", ok ? "passed" : "FAILED");
    return !ok;
}
int main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "help")) {
        puts("bitnet train [--task copy|text] [--data FILE] [--arch attn|mlp|full] [--layers 2] [--dim 128] [--heads 4]\n"
             "             [--hidden 512] [--seq 64] [--batch 32] [--steps 300] [--lr 3e-3] [--wd .1] [--warmup 20]\n"
             "             [--eval-every 50] [--eval-batches 4] [--quant 1] [--seed 1] [--threads N] [--save m.bin]\n"
             "bitnet selftest");
        return 0;
    }
    if (!strcmp(argv[1], "train")) { int r = train(parse(argc, argv), 0, NULL, NULL); free(PA); free(PB); return r; }
    if (!strcmp(argv[1], "selftest")) return selftest();
    die("unknown command %s (try help)", argv[1]);
}
