/* tln.c: run a threshold-logic network distilled from the MLP-only BitNet (stage_B/distill). Pure integer logic:
 * table lookup of input bits, AND + popcount per wire group, integer compares, integer vote sums.
 *
 *   gcc -O3 -march=native -fopenmp tln.c -lm -o tln
 *   ./tln verify M.tln M.votes        votes for all 256 x 64 (byte, position) inputs == the Python integer model
 *   ./tln eval   M.tln TEXT           next-byte CE / accuracy on the last 10% of TEXT (every 64-byte window)
 *   ./tln bench  M.tln                inputs per second (1 thread and all threads)
 *
 * Network: u = [token code(byte), position code(pos)] (bits); layer 1: unit j has wires A1[:, j] (integer weights
 * in [-qa, qa], stored as sign-split bit planes) and L thresholds, gate (j, l) = [u . A1[:, j] > t1[j, l]];
 * layer 2 the same over z1 = [u, r1]; output votes[v] = z . C[:, v] + bias[v] over z = [u, r1, r2].
 * Probabilities, where needed, are softmax(scale * votes); the argmax needs no scale.
 * File "TLN00001": int32 V T ntok npos H1 H2 L qa qc, float64 scale, uint8 tok[V][ntok], pos[T][npos],
 * int32 A1[nin][H1], t1[H1][L], A2[nin + H1 L][H2], t2[H2][L], C[nin + (H1 + H2) L][V], bias[V]. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *m) { fprintf(stderr, "tln: %s\n", m); exit(1); }
static void *xal(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) die("out of memory"); return p; }

typedef struct { int n, words, planes; uint64_t *P, *N; } Mat;   /* per output column: planes x (P, N) x words */
typedef struct {
    int V, T, ntok, npos, nin, H1, H2, L, qa, qc; double scale;
    uint8_t *tok, *pos; int32_t *t1, *t2, *bias; Mat a1, a2, c;
} Net;

static int nbits(int q) { int b = 0; while (q >> b) b++; return b ? b : 1; }
/* W is [rows][cols] int32 row-major: build sign-split bit planes per column */
static Mat mat(const int32_t *W, int rows, int cols, int q) {
    Mat m = {cols, (rows + 63) / 64, nbits(q), NULL, NULL};
    size_t per = (size_t)m.planes * m.words;
    m.P = xal((size_t)cols * per * 8); m.N = xal((size_t)cols * per * 8);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            int w = W[(size_t)i * cols + j], a = w < 0 ? -w : w;
            for (int p = 0; p < m.planes; p++)
                if (a >> p & 1) {
                    uint64_t *d = (w > 0 ? m.P : m.N) + (size_t)j * per + (size_t)p * m.words;
                    d[i >> 6] |= 1ull << (i & 63);
                }
        }
    return m;
}
static inline int64_t dot(const Mat *m, int j, const uint64_t *z) {
    size_t per = (size_t)m->planes * m->words; int64_t s = 0;
    for (int p = 0; p < m->planes; p++) {
        const uint64_t *P = m->P + (size_t)j * per + (size_t)p * m->words, *N = m->N + (size_t)j * per + (size_t)p * m->words;
        int64_t t = 0;
        for (int w = 0; w < m->words; w++) t += __builtin_popcountll(z[w] & P[w]) - __builtin_popcountll(z[w] & N[w]);
        s += t << p;
    }
    return s;
}
static Net *load(const char *path) {
    FILE *f = fopen(path, "rb"); char mg[8]; int32_t h[9]; if (!f) die("cannot open model");
    if (fread(mg, 1, 8, f) != 8 || memcmp(mg, "TLN00001", 8) || fread(h, 4, 9, f) != 9) die("not a TLN model");
    Net *n = xal(sizeof(Net));
    n->V = h[0]; n->T = h[1]; n->ntok = h[2]; n->npos = h[3]; n->H1 = h[4]; n->H2 = h[5]; n->L = h[6]; n->qa = h[7]; n->qc = h[8];
    n->nin = n->ntok + n->npos;
    if (fread(&n->scale, 8, 1, f) != 1) die("truncated");
    int V = n->V, T = n->T, nin = n->nin, G1 = n->H1 * n->L, G2 = n->H2 * n->L;
    n->tok = xal((size_t)V * n->ntok); n->pos = xal((size_t)T * n->npos);
#define RD(p, sz, cnt) if (fread(p, sz, cnt, f) != (size_t)(cnt)) die("truncated")
    RD(n->tok, 1, (size_t)V * n->ntok); RD(n->pos, 1, (size_t)T * n->npos);
    int32_t *A1 = xal((size_t)nin * n->H1 * 4), *A2 = xal((size_t)(nin + G1) * n->H2 * 4), *C = xal((size_t)(nin + G1 + G2) * V * 4);
    n->t1 = xal((size_t)G1 * 4); n->t2 = xal((size_t)G2 * 4); n->bias = xal((size_t)V * 4);
    RD(A1, 4, (size_t)nin * n->H1); RD(n->t1, 4, G1); RD(A2, 4, (size_t)(nin + G1) * n->H2); RD(n->t2, 4, G2);
    RD(C, 4, (size_t)(nin + G1 + G2) * V); RD(n->bias, 4, V);
    if (fgetc(f) != EOF) die("trailing bytes");
    fclose(f);
    n->a1 = mat(A1, nin, n->H1, n->qa); n->a2 = mat(A2, nin + G1, n->H2, n->qa); n->c = mat(C, nin + G1 + G2, V, n->qc);
    free(A1); free(A2); free(C);
    return n;
}
static inline void setbit(uint64_t *z, int i) { z[i >> 6] |= 1ull << (i & 63); }
/* votes[V] for input (byte, pos); z must hold c.words zeroed words */
static void run(const Net *n, int byte, int pos, uint64_t *z, int32_t *votes) {
    int nin = n->nin, G1 = n->H1 * n->L, L = n->L;
    memset(z, 0, (size_t)n->c.words * 8);
    for (int i = 0; i < n->ntok; i++) if (n->tok[(size_t)byte * n->ntok + i]) setbit(z, i);
    for (int i = 0; i < n->npos; i++) if (n->pos[(size_t)pos * n->npos + i]) setbit(z, n->ntok + i);
    for (int j = 0; j < n->H1; j++) {
        int64_t s = dot(&n->a1, j, z);
        for (int l = 0; l < L; l++) if (s > n->t1[j * L + l]) setbit(z, nin + j * L + l);
    }
    for (int j = 0; j < n->H2; j++) {
        int64_t s = dot(&n->a2, j, z);
        for (int l = 0; l < L; l++) if (s > n->t2[j * L + l]) setbit(z, nin + G1 + j * L + l);
    }
    for (int v = 0; v < n->V; v++) votes[v] = (int32_t)dot(&n->c, v, z) + n->bias[v];
}

int main(int argc, char **argv) {
    if (argc < 3) die("usage: tln verify M.tln M.votes | eval M.tln TEXT | bench M.tln");
    Net *n = load(argv[2]); int V = n->V, T = n->T, NI = 256 * T;
    int32_t *all = xal((size_t)NI * V * 4);
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        uint64_t *z = xal((size_t)n->c.words * 8);
        #pragma omp for schedule(static)
        for (int i = 0; i < NI; i++) run(n, i / T, i % T, z, all + (size_t)i * V);
        free(z);
    }
    double tall = omp_get_wtime() - t0;
    if (!strcmp(argv[1], "verify")) {
        if (argc < 4) die("verify needs M.votes");
        FILE *f = fopen(argv[3], "rb"); if (!f) die("cannot open votes");
        int32_t *ref = xal((size_t)NI * V * 4); if (fread(ref, 4, (size_t)NI * V, f) != (size_t)NI * V) die("votes size");
        fclose(f); long bad = 0;
        for (size_t i = 0; i < (size_t)NI * V; i++) bad += all[i] != ref[i];
        printf("{\"inputs\":%d,\"outputs\":%d,\"mismatched_votes\":%ld,\"verify\":\"%s\"}\n", NI, V, bad, bad ? "FAILED" : "passed");
        return bad != 0;
    }
    if (!strcmp(argv[1], "eval")) {
        if (argc < 4) die("eval needs TEXT");
        FILE *f = fopen(argv[3], "rb"); if (!f) die("cannot open text");
        fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
        unsigned char *b = xal(len); if (fread(b, 1, len, f) != (size_t)len) die("read text"); fclose(f);
        long lo = len * 9 / 10, nw = (len - lo - 1) / T; double ce = 0; long hit = 0, cnt = 0;
        for (long w = 0; w < nw; w++)
            for (int t = 0; t < T; t++) {
                long s = lo + w * T + t; const int32_t *v = all + ((size_t)b[s] * T + t) * V; int y = b[s + 1], am = 0;
                double mx = -1e300, z = 0;
                for (int k = 0; k < V; k++) { if (v[k] > v[am]) am = k; if (n->scale * v[k] > mx) mx = n->scale * v[k]; }
                for (int k = 0; k < V; k++) z += exp(n->scale * v[k] - mx);
                ce += mx + log(z) - n->scale * v[y]; hit += am == y; cnt++;
            }
        printf("{\"bytes\":%ld,\"ce\":%.6f,\"bits_per_byte\":%.6f,\"acc\":%.6f}\n", cnt, ce / cnt, ce / cnt / log(2), (double)hit / cnt);
        return 0;
    }
    if (!strcmp(argv[1], "bench")) {
        uint64_t *z = xal((size_t)n->c.words * 8); int32_t *v = xal(V * 4); long reps = 0; double t1 = omp_get_wtime();
        while (omp_get_wtime() - t1 < 1.0) { run(n, (int)(reps * 7 % 256), (int)(reps % T), z, v); reps++; }
        double one = (omp_get_wtime() - t1) / reps;
        printf("{\"us_per_input_1thread\":%.2f,\"inputs_per_s_all_threads\":%.0f,\"threads\":%d}\n", one * 1e6, NI / tall, omp_get_max_threads());
        return 0;
    }
    die("unknown command");
}
