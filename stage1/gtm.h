/*
 * gtm.h - CPU-native Graph Tsetlin Machine.
 *
 * Semantics: cair/GraphTsetlinMachine (coalesced, multi-layer hypervector message
 * passing). RNG + file formats: stage0/gtmcore.py. Training is bit-exact with the
 * stage0 oracle and invariant to thread count.
 *
 * Layout choices (vs. the CUDA original):
 *   - 64-bit words everywhere; TA states bit-sliced as [clause][word][plane].
 *   - A mirrored include-mask array [clause][word] so evaluation never touches
 *     the state planes.
 *   - Clause outputs stored NODE-major ([node][clause_word]) so message passing
 *     can walk set bits per source node, and so clause-sharded threads write
 *     disjoint words.
 *   - Weights stored output-major ([output][clause], padded to 64) so weight updates and
 *     class sums are 16-lane masked vector ops over a clause word.
 */
#ifndef GTM_H
#define GTM_H

#include <stdint.h>
#include <stddef.h>

#define GTM_MAX_DEPTH 8

typedef struct {
    /* config */
    uint32_t C, O, H, L, NT, D, MS, M, MB, B, boost, neg, max_inc;
    int32_t T;
    double q;
    double s[GTM_MAX_DEPTH];
    double rho;         /* automaton feedback budget (1 = original semantics) */
    uint32_t senders;   /* 0: all clauses send messages; 1: only clauses with a positive layer-0 literal */
    uint32_t layered;   /* 1: a non-firing clause is forgotten only from the layer where it died onwards */
    uint64_t seed, step;
    /* training option, not saved: ranking output feedback against the best of rank_k draws from
     * the negative code table neg_codes [n_neg][O] (0/1 bytes); 0 = per-output feedback (CUDA) */
    uint32_t rank_k, n_neg;
    int64_t rank_margin;
    const uint8_t *neg_codes;

    /* derived */
    uint32_t Wl[GTM_MAX_DEPTH]; /* words per layer input (layer 0: L bits, others: M bits) */
    uint32_t Cw;                /* clause words */
    uint64_t thr_s[GTM_MAX_DEPTH];

    /* state */
    uint32_t *hv;                 /* [C][MB] clause message bits */
    int32_t *w;                   /* [O][Cw*64] output-major weights (padded) */
    uint64_t *ta[GTM_MAX_DEPTH];  /* [C][Wl][B] bit planes */
    uint64_t *inc[GTM_MAX_DEPTH]; /* [C][Wl] include masks (MSB plane mirror) */
    uint64_t *valid[GTM_MAX_DEPTH]; /* [Wl] valid-literal masks */
    uint32_t *ninc;               /* [C] total included literals across layers */
} gtm_model;

typedef struct {
    uint32_t n_graphs, H, W, NT, NET, O, kind;
    uint64_t total_nodes, total_edges;
    uint32_t *npg;       /* nodes per graph */
    uint64_t *node_off;  /* [n_graphs + 1] */
    uint32_t *ntype;     /* [total_nodes] */
    uint32_t *epn;       /* out-edges per node */
    uint64_t *edge_off;  /* [total_nodes + 1] */
    uint32_t *edges;     /* [total_edges][2] (dst local, type) */
    uint64_t *in_off;    /* [total_nodes + 1] inbound adjacency */
    uint32_t *in_edges;  /* [total_edges][2] (src local, type) */
    uint64_t *X;         /* [total_nodes][W] */
    int32_t *Y;          /* [n_graphs][O] 0/1 */
    uint32_t max_nodes;
    int comp;            /* every node's X is [x | not x]: enables the indexed evaluator at layer 0 */
    void *blob;
} gtm_data;

typedef struct {
    uint32_t C, O, H, NT, D, MS, MB, B, boost, neg, max_inc;
    int32_t T;
    double q;
    double s[GTM_MAX_DEPTH];
    uint64_t seed;
    double rho;         /* 0 is treated as 1 */
    uint32_t senders;
    uint32_t layered;
} gtm_config;

/* model lifecycle */
gtm_model *gtm_new(const gtm_config *cfg);
void gtm_free(gtm_model *m);
int gtm_save(const gtm_model *m, const char *path);
gtm_model *gtm_load(const char *path);

/* data */
gtm_data *gtm_data_load(const char *path);
void gtm_data_free(gtm_data *d);

/* training: `max_steps` < 0 means the whole dataset. Returns examples processed. */
int64_t gtm_fit(gtm_model *m, const gtm_data *d, int epochs, int64_t max_steps, int threads);

/* inference: sums [n_graphs][O]; bits (optional) [n_graphs][Cw] graph-level clause outputs */
void gtm_score(const gtm_model *m, const gtm_data *d, int32_t *sums, uint64_t *bits, int threads);
/* msg_fill (optional): fraction of message bits set, averaged over nodes and layers >= 1.
 * Near 1.0 means bundles are saturated and message layers carry little information. */
void gtm_score_ex(const gtm_model *m, const gtm_data *d, int32_t *sums, uint64_t *bits, int threads, double *msg_fill);
double gtm_accuracy(const gtm_data *d, const int32_t *sums);

/* introspection */
uint32_t gtm_ta_state(const gtm_model *m, int layer, uint32_t clause, uint32_t literal);
uint64_t gtm_state_hash(const gtm_model *m);

#endif
