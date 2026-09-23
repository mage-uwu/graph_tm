/*
 * gtm CLI
 *
 *   gtm init  --data D.gtmd --out M.gtmm [config]
 *   gtm train --data D.gtmd [--test T.gtmd] (--model M.gtmm | [config]) [--epochs E]
 *             [--steps N] [--threads P] [--save OUT.gtmm]
 *   gtm score --model M.gtmm --data D.gtmd [--threads P] [--sums F.i32] [--bits F.u64] [--reps R]
 *   gtm hash  --model M.gtmm
 *
 * config: --clauses --T --s (x or x,y,z per layer) --q --depth --msg-size --msg-bits
 *         --max-inc --state-bits --boost --neg --seed --rho
 *   --q -1    symmetric negative feedback (q = O-1); use for multi-output targets
 *   --rho r   automaton feedback budget; ~2/O for many outputs (weights still see everything)
 *   --senders all|pos   pos: only clauses with a positive layer-0 literal send messages
 *             (keeps bundles from saturating with large clause pools; not GraphTM semantics)
 */
#define _GNU_SOURCE
#include "gtm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct {
    const char *data, *test, *model, *out, *save, *sums, *bits;
    int epochs, threads, reps;
    long long steps;
    gtm_config cfg;
    int have_max_inc;
    const char *s_arg;
} args_t;

static void usage(void) {
    fprintf(stderr, "usage: gtm {init|train|score|hash} [options]  (see main.c header)\n");
    exit(2);
}

static void parse(int argc, char **argv, args_t *a) {
    memset(a, 0, sizeof *a);
    a->epochs = 1; a->threads = 1; a->steps = -1; a->reps = 1;
    a->cfg = (gtm_config){.C = 100, .D = 1, .MS = 256, .MB = 2, .B = 8, .boost = 1, .neg = 1, .T = 100, .q = 1.0, .seed = 42, .rho = 1.0};
    a->s_arg = "1.0";
    for (int i = 2; i < argc; i++) {
        const char *k = argv[i];
        if (i + 1 >= argc) usage();
        const char *v = argv[++i];
        if (!strcmp(k, "--data")) a->data = v;
        else if (!strcmp(k, "--test")) a->test = v;
        else if (!strcmp(k, "--model")) a->model = v;
        else if (!strcmp(k, "--out")) a->out = v;
        else if (!strcmp(k, "--save")) a->save = v;
        else if (!strcmp(k, "--sums")) a->sums = v;
        else if (!strcmp(k, "--bits")) a->bits = v;
        else if (!strcmp(k, "--epochs")) a->epochs = atoi(v);
        else if (!strcmp(k, "--threads")) a->threads = atoi(v);
        else if (!strcmp(k, "--steps")) a->steps = atoll(v);
        else if (!strcmp(k, "--reps")) a->reps = atoi(v);
        else if (!strcmp(k, "--clauses")) a->cfg.C = (uint32_t)atoi(v);
        else if (!strcmp(k, "--T")) a->cfg.T = atoi(v);
        else if (!strcmp(k, "--s")) a->s_arg = v;
        else if (!strcmp(k, "--q")) a->cfg.q = atof(v);
        else if (!strcmp(k, "--rho")) a->cfg.rho = atof(v);
        else if (!strcmp(k, "--senders")) a->cfg.senders = !strcmp(v, "pos");
        else if (!strcmp(k, "--depth")) a->cfg.D = (uint32_t)atoi(v);
        else if (!strcmp(k, "--msg-size")) a->cfg.MS = (uint32_t)atoi(v);
        else if (!strcmp(k, "--msg-bits")) a->cfg.MB = (uint32_t)atoi(v);
        else if (!strcmp(k, "--max-inc")) { a->cfg.max_inc = (uint32_t)atoi(v); a->have_max_inc = 1; }
        else if (!strcmp(k, "--state-bits")) a->cfg.B = (uint32_t)atoi(v);
        else if (!strcmp(k, "--boost")) a->cfg.boost = (uint32_t)atoi(v);
        else if (!strcmp(k, "--neg")) a->cfg.neg = (uint32_t)atoi(v);
        else if (!strcmp(k, "--seed")) a->cfg.seed = strtoull(v, NULL, 10);
        else { fprintf(stderr, "unknown option %s\n", k); usage(); }
    }
    /* per-layer s: "x" or "x,y,z" */
    char buf[256];
    snprintf(buf, sizeof buf, "%s", a->s_arg);
    int n = 0;
    for (char *tok = strtok(buf, ","); tok && n < GTM_MAX_DEPTH; tok = strtok(NULL, ",")) a->cfg.s[n++] = atof(tok);
    for (int i = n; i < GTM_MAX_DEPTH; i++) a->cfg.s[i] = a->cfg.s[n - 1];
}

static gtm_model *model_from(args_t *a, const gtm_data *d) {
    if (a->model) return gtm_load(a->model);
    a->cfg.H = d->H; a->cfg.NT = d->NT; a->cfg.O = d->O;
    if (!a->have_max_inc) a->cfg.max_inc = 2 * d->H;
    if (a->cfg.q < 0) a->cfg.q = d->O > 1 ? d->O - 1 : 1; /* --q -1: symmetric negative feedback */
    return gtm_new(&a->cfg);
}

static void report(const char *tag, const gtm_model *m, const gtm_data *d, int threads) {
    int32_t *sums = malloc((size_t)d->n_graphs * m->O * 4);
    double t0 = now();
    double fill = 0;
    gtm_score_ex(m, d, sums, NULL, threads, &fill);
    double dt = now() - t0;
    printf("  %s acc %.4f  (%.0f graphs/s)", tag, gtm_accuracy(d, sums), d->n_graphs / dt);
    if (m->D > 1) printf("  msg fill %.0f%%%s", 100 * fill, fill > 0.99 ? " SATURATED" : "");
    free(sums);
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    args_t a;
    parse(argc, argv, &a);
    const char *cmd = argv[1];

    if (!strcmp(cmd, "init")) {
        if (!a.data || !a.out) usage();
        gtm_data *d = gtm_data_load(a.data);
        gtm_model *m = d ? model_from(&a, d) : NULL;
        if (!m) return 1;
        gtm_save(m, a.out);
        printf("init: C=%u O=%u H=%u D=%u -> %s\n", m->C, m->O, m->H, m->D, a.out);
        gtm_free(m); gtm_data_free(d);
        return 0;
    }
    if (!strcmp(cmd, "train")) {
        if (!a.data) usage();
        gtm_data *d = gtm_data_load(a.data);
        gtm_data *te = a.test ? gtm_data_load(a.test) : NULL;
        gtm_model *m = d ? model_from(&a, d) : NULL;
        if (!m) return 1;
        printf("train: %u graphs, C=%u O=%u L=%u D=%u T=%d threads=%d\n", d->n_graphs, m->C, m->O, m->L, m->D, m->T, a.threads);
        if (a.steps >= 0) {
            double t0 = now();
            int ep = (int)((a.steps + d->n_graphs - 1) / d->n_graphs); /* --steps may span epochs */
            int64_t n = gtm_fit(m, d, ep > 0 ? ep : 1, a.steps, a.threads);
            printf("  steps %lld in %.3fs (%.0f ex/s)\n", (long long)n, now() - t0, n / (now() - t0));
        } else {
            for (int e = 0; e < a.epochs; e++) {
                double t0 = now();
                int64_t n = gtm_fit(m, d, 1, -1, a.threads);
                double dt = now() - t0;
                printf("epoch %2d  train %.2fs (%.0f ex/s)", e, dt, n / dt);
                if (te) report("test", m, te, a.threads);
                printf("\n");
                fflush(stdout);
            }
        }
        if (a.save) gtm_save(m, a.save);
        printf("  state hash %016llx\n", (unsigned long long)gtm_state_hash(m));
        gtm_free(m); gtm_data_free(d); gtm_data_free(te);
        return 0;
    }
    if (!strcmp(cmd, "score")) {
        if (!a.data || !a.model) usage();
        gtm_data *d = gtm_data_load(a.data);
        gtm_model *m = gtm_load(a.model);
        if (!d || !m) return 1;
        int32_t *sums = malloc((size_t)d->n_graphs * m->O * 4);
        uint64_t *bits = a.bits ? malloc((size_t)d->n_graphs * m->Cw * 8) : NULL;
        double best = 1e30, fill = 0;
        for (int r = 0; r < a.reps; r++) {
            double t0 = now();
            gtm_score_ex(m, d, sums, bits, a.threads, &fill);
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("score: %u graphs, acc %.4f, best %.4fs (%.0f graphs/s, %.2f us/graph) threads=%d\n",
               d->n_graphs, gtm_accuracy(d, sums), best, d->n_graphs / best, 1e6 * best / d->n_graphs, a.threads);
        if (m->D > 1)
            printf("  message fill %.1f%%%s\n", 100 * fill,
                   fill > 0.99 ? "  <-- SATURATED: messages are ~constant, so layers >= 1 learn nothing; try --senders pos" : "");
        if (a.sums) { FILE *f = fopen(a.sums, "wb"); fwrite(sums, 4, (size_t)d->n_graphs * m->O, f); fclose(f); }
        if (a.bits) { FILE *f = fopen(a.bits, "wb"); fwrite(bits, 8, (size_t)d->n_graphs * m->Cw, f); fclose(f); }
        free(sums); free(bits); gtm_free(m); gtm_data_free(d);
        return 0;
    }
    if (!strcmp(cmd, "hash")) {
        if (!a.model) usage();
        gtm_model *m = gtm_load(a.model);
        if (!m) return 1;
        printf("%016llx\n", (unsigned long long)gtm_state_hash(m));
        gtm_free(m);
        return 0;
    }
    usage();
    return 2;
}
