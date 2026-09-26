/* fastlogic.c: LogicAE in one file. A text classifier made only of logic gates: train it, pretrain it,
 * adapt it, and run it as pure bit operations at ~0.05-0.09 ms per text on one CPU core. No dependencies
 * beyond a C compiler with OpenMP.
 *
 *   gcc -O3 -march=native -std=c11 -fopenmp fastlogic.c -lm -o fastlogic
 *   (-std=c11 on purpose: GNU mode lets gcc fuse a*b+c into FMA, which changes training bits)
 *
 * THE MODEL (defaults = the best recipe we have measured)
 *   tokens     bert-base-uncased WordPiece ids (30,522). Each token has a learned 128-bit code (float logits,
 *              sigmoid(z / temperature) in training, the sign bit when deployed).
 *   trunk      16 blocks x 1024 channels. Each channel is a depth-3 tree of 2-input gates (7 gates, 8 leaves).
 *              Each gate is a learned 4-entry truth table (4 float logits, sigmoid in training, a bit when deployed).
 *              Leaves are wired (fixed at init) to a channel of the previous block at a token offset in
 *              [-2,2] x dilation. Kernel 5, dilations 1,2,4,8 repeating. Every tree starts as a pass-through of
 *              its first leaf (logits +-3), which is what makes 16 logic layers trainable.
 *   head       OR-pool over position pairs, then 2 x 128 vote trees; class score = votes(1) - votes(0), a popcount.
 *              Binary output: a K-way question = one yes/no model per option, logits = vote differences.
 *   size       4.4M parameters (3.9M of them the token codes). The deployed .lth is integer-only.
 *   pairs      --match 1: one input bit per position, "this token also occurs in the other [SEP] segment".
 *              --global-every 4 --global-channels 128 --global-mean 1: every 4th block also sees majority-pooled
 *              channels (fires at more than half of the valid positions). QNLI 0.580 -> 0.710 with both.
 *
 * THE TRAINING RECIPE
 *   hard forward (straight-through), for train AND pretrain: the forward pass IS the deployed network (binary
 *              codes, binary gate outputs, OR / majority pooling of bits). The backward pass differentiates the
 *              gate interpolation at those binary points using the float master weights (the logits), and
 *              passes the thresholds straight through. Float masters + Adam state are kept in .ltc checkpoints.
 *              Why: a soft relaxation lets the model learn through sub-threshold values that hardening erases.
 *              SST-2, 600 steps, hardened test accuracy:
 *                  from scratch      soft 0.751 -> hard forward 0.764
 *                  from pretraining  soft 0.509 (stuck at a constant answer) -> hard forward 0.669
 *              and in masked-word pretraining it keeps the trunk alive (30% dead channels vs 71% soft) and
 *              is the first pretraining that transfers: SST-2 0.805 after pretraining vs 0.772 from scratch
 *              (soft pretraining: 0.766), although soft is the better masked-word model.
 *              `--soft-forward` restores the relaxation. `hardcheck` proves training forward == deployed
 *              network, record by record.
 *   optimiser  Adam, peak lr .025, cosine to 10%. Code temperature anneals 1.0 -> 0.2 over the first 80%.
 *              Batch 32 (supervised) / 64 (pretraining), seq 64.
 *   pretrain   masked-word objective (15% of positions, 80/10/10). Each masked position is decoded against
 *              all 30,519 token codes (tied Hamming decoder + per-word bias), capped at 512 targets per batch
 *              (--mlm-targets). Data: `-1 id id ...` lines, e.g. 64-token WikiText windows. Stream your corpus
 *              into that file; do not load it into memory.
 *              The tied decoder is kept on purpose: it is the best output layer we have measured. On the frozen
 *              30k-step model (WikiText-103 validation, masked-word CE, nats) the tied 128-bit decoder scores 5.63;
 *              untied softmax heads fitted on 200k targets score 5.77 on the same 128 projection bits, 6.32 on the
 *              1024 final-block channels, 5.93 on 5 x 1024 channels around the target (unigram 7.27). The limit
 *              is the trunk's features, not the decoder.
 *              How long: 3,000 steps (~12M tokens) is the recipe that transfers. 30,000 steps lowers masked-word
 *              CE (5.74 -> 5.51) but transfers worse (SST-2 0.743, QNLI 0.688 vs scratch 0.779 / 0.732) as the
 *              dead-channel share grows (30% -> 42%). Pick checkpoints by adaptation, not by masked-word CE.
 *   adapt      `train --load pretrained.ltc`: keeps the checkpoint's code temperature (the gates were trained
 *              on near-binary codes), fresh Adam, hard forward. `--load-part codes|gates` loads half.
 *              `revive` resets dead channels of a checkpoint to the pass-through init before adapting.
 *   speed      training, hard forward: each block's forward pass runs on packed bits (64 texts per word, every
 *              gate a mux, the inference engine's method); the backward pass takes the tree's node bits from the
 *              same words and uses the exact binary-input gate gradients (one corner gets u, d/da and d/db are
 *              corner differences). Deterministic parallel backward (phase 1 over outputs, phase 2 over input
 *              channels in the serial order). Masked-word output layer: tiles of 16 targets x 256 words with
 *              the +-code matrix precomputed once per step, so the 30,519-way logits, the projection gradient
 *              and the code gradient vectorise while every value keeps the same float operations in the same
 *              order. 4-core Xeon, 16x1024, seq 64: supervised batch 32 1.03 -> 0.69 s/step; pretraining
 *              batch 64 (512 targets) 4.19 -> 1.49 s/step. All byte-identical to the float reference.
 *              Where a pretraining step goes now (4 cores): trunk backward 58%, output layer 30%, trunk forward 9%,
 *              regularizer + Adam 3%.
 *              The binary sets OMP_WAIT_POLICY=passive itself: spinning OpenMP threads cost 5-10x on SMT machines.
 *              Results are byte-identical at any thread count; --resume is bit-exact.
 *
 * FAST INFERENCE (same scores as the reference engine, checked by `verify`)
 *   interpreter  one text per 64-bit word: bit t = token position t, one word per channel (T <= 512).
 *                A convolution tap is a shift. A gate is a branch-free mux from a 16-entry mask table.
 *                --lanes 1/2/4/8 runs texts side by side in one vector register.
 *   compiled     `fastlogic gen MODEL DATA SEQ model.inc` writes the model as straight-line C for that context length
 *                (each gate one bitwise op, each leaf a shift; over 64 tokens, a two-word shift across padded words,
 *                one word loop per block), then build this same file with it included:
 *                gcc -O2 -march=native -std=c11 -fopenmp -DGEN_INC='"model.inc"' fastlogic.c -lm -o fastlogic_m
 *                One build serves one words-per-channel count: SEQ 1-64, 65-128, ..., 449-512 (other lengths use the
 *                interpreter). Compile time ~2.5 min (64 tokens) to ~4-7 min (256-512) for 16x1024.
 *                4-core Xeon, 16x1024, compiled vs bert-tiny (PyTorch fp32), same machine, every score verified:
 *                  64 tokens:  1 text 83 us vs 2934 us (35x); batch 500, 4 threads 3.9 vs 417 us/text (107x)
 *                  256 tokens: 1 text 190 us vs 5228 us (27x); batch 256, 4 threads 18 vs 2218 us/text (123x)
 *                  512 tokens: 1 text 388 us vs 16382 us (42x); batch 256, 4 threads 39 vs 5394 us/text (138x)
 *                The compiled model is 8-17x faster than the interpreter at 256-512 tokens.
 *
 * DATA: `--format ids` lines `LABEL id id ...` (LABEL 0/1, or -1 unlabeled). Reserved ids: 0 PAD, 1 MASK,
 * 2 UNK; [SEP] = 102. Longer records are cut to --seq, shorter ones PAD-filled.
 *
 * KNOWN LIMITS: hard-forward pretraining (3000 steps, ~12M tokens) then adaptation: SST-2 0.805 vs scratch 0.772
 * (soft pretraining 0.766); with the pair options on, 0.792 vs 0.779. Longer pretraining kills channels and
 * transfers worse (see pretrain). Masked-word CE 5.51 at best (1.66 bits per byte) vs bert-tiny 3.96; the gap is
 * in the trunk (fixed random wiring, no content routing), not the decoder. Binary head. Inputs up to 512 tokens.
 *
 * FILE MAP: training engine (codes, gate trees, objectives, Adam, checkpoints, hardening, CLI) | fast inference
 * (interpreter, lanes, compiler) | checks and tools (hardcheck, revive) | main | compiled-model hook.
 */

/* =====================================================================================================
 * training engine: codes, gate trees, objectives, Adam, checkpoints, hardened network
 * ===================================================================================================== */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <float.h>
#include <stdarg.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define MAX_DEPTH 5
#define MAX_LEAVES 32
#define MAX_BLOCKS 32
#define PI 3.14159265358979323846
#define PAD 0
#define MASK 1
#define UNK 2
#define MAX_RECORD ((size_t)64*1024*1024)

static void die(const char *fmt, ...) {
    va_list ap; va_start(ap,fmt); fputs("error: ",stderr);
    vfprintf(stderr,fmt,ap); va_end(ap); fputc('\n',stderr); exit(EXIT_FAILURE);
}
static size_t mul(size_t a,size_t b) {
    if(b && a>SIZE_MAX/b) die("size overflow");
    return a*b;
}
static void *alloc(size_t n,size_t s) {
    size_t k=mul(n,s); void *p=calloc(k?k:1,1);
    if(!p) die("cannot allocate %zu bytes",k);
    return p;
}
static void *resize(void *p,size_t n,size_t s) {
    void *q=realloc(p,mul(n,s)); if(!q) die("reallocation failed"); return q;
}
static char *copystr(const char *s) {
    char *p=alloc(strlen(s)+1,1); strcpy(p,s); return p;
}
static double now(void) {
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC,&t)) die("monotonic clock failed");
    return t.tv_sec+1e-9*t.tv_nsec;
}
static void threads_set(int n) {
#ifdef _OPENMP
    omp_set_dynamic(0); omp_set_num_threads(n);
#else
    if(n!=1) fprintf(stderr,"note: compiled without OpenMP; using one thread\n");
#endif
}
static int thread_count(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
typedef struct { uint64_t s; } RNG;
static uint64_t ru(RNG *r) {
    uint64_t x=r->s; x^=x>>12; x^=x<<25; x^=x>>27; r->s=x;
    return x*UINT64_C(2685821657736338717);
}
static float uniform(RNG *r) { return (float)(ru(r)>>40)*(1.f/16777216.f); }
static int randint(RNG *r,int n) { return (int)(ru(r)%(uint64_t)n); }
static float normal(RNG *r) {
    return sqrtf(-2.f*logf(fmaxf(uniform(r),1e-7f)))*cosf((float)(2*PI)*uniform(r));
}
static float sigmoid(float x) {
    if(x>=0) return 1.f/(1.f+expf(-x));
    float e=expf(x); return e/(1.f+e);
}
static float corner(float x,int est) { return est?.5f+.5f*sinf(x):sigmoid(x); }
static float cderiv(float x,float p,int est) { return est?.5f*cosf(x):p*(1.f-p); }
static float gate(float a,float b,const float *p) {
    return (1-a)*((1-b)*p[0]+b*p[1])+a*((1-b)*p[2]+b*p[3]);
}
static uint64_t gate64(uint64_t a,uint64_t b,uint8_t f) {
    return (~a&~b&-(uint64_t)(f&1)) | (~a&b&-(uint64_t)((f>>1)&1)) |
           (a&~b&-(uint64_t)((f>>2)&1)) | (a&b&-(uint64_t)((f>>3)&1));
}
static int lowbit(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n=0; while(!(x&1)){x>>=1;n++;} return n;
#endif
}

typedef struct {
    int vocab,bits,width,blocks,depth,kernel,cycle,votes,est,q8;
    int seq,batch,steps,stage,max_targets;
    float peak_lr,sharp,clip,mask_prob,scale,mlm_scale;
    int gevery,gch;  /* global view: every gevery-th block input gets gch pooled channels */
    int gmean;       /* pooling: 0 = OR over positions (max), 1 = majority (mean > 1/2) */
    int match;       /* extra input bit: token also occurs in the other [SEP] segment */
} Config;
static Config defaults(void) {
    Config c={30522,128,1024,16,3,5,4,128,0,0,64,32,1000,0,512,  /* best recipe: see the header */
              .025f,.002f,5.f,.15f,16.f,16.f}; return c;
}
static void validate(Config c) {
    if(c.vocab<4 || c.vocab>1000000 || c.bits<2 || c.bits>1024 || c.bits%2 ||
       c.width<1 || c.width>65536 || c.blocks<1 || c.blocks>MAX_BLOCKS ||
       c.depth<1 || c.depth>MAX_DEPTH || c.kernel<1 || c.kernel>31 || !(c.kernel%2) ||
       c.cycle<1 || c.cycle>20 || c.votes<1 || c.votes>32768 ||
       c.est<0 || c.est>1 || c.q8<0 || c.q8>1 || c.seq<1 || c.seq>65536 ||
       c.batch<1 || c.batch>65536 || c.steps<1 || c.steps>1000000000 ||
       c.stage<0 || c.stage>1 || c.max_targets<0 ||
       !(c.peak_lr>0) || !isfinite(c.peak_lr) || !(c.sharp>=0) || !isfinite(c.sharp) ||
       !(c.clip>0) || !isfinite(c.clip) || !(c.mask_prob>0 && c.mask_prob<=1) ||
       !(c.scale>0) || !isfinite(c.scale) || !(c.mlm_scale>0) || !isfinite(c.mlm_scale) ||
       (int64_t)c.batch*c.seq>INT_MAX || (int64_t)c.seq*c.votes>INT_MAX/2 ||
       c.gevery<0 || c.gevery>MAX_BLOCKS || c.gch<0 || c.gch>c.width || (c.gevery>0)!=(c.gch>0) ||
       c.gmean<0 || c.gmean>1 || c.match<0 || c.match>1)
        die("invalid or excessive model/training dimensions");
}
static int gblock(const Config *c,int i){return c->gevery>0 && i>0 && i<c->blocks && i%c->gevery==0;}
static int in_ch(const Config *c,int i){return (i?c->width:c->bits+c->match)+(gblock(c,i)?c->gch:0);}
#define MATCH_SEP 102  /* bert-base-uncased [SEP] */
/* match bit per (text, position): the token also occurs in the other segment (split at the first [SEP]);
 * without a [SEP], at another position. PAD, special ids (< 3) and [SEP] itself get 0. */
static void match_bits(const uint32_t *ids,int B,int T,uint8_t *out) {
    for(int b=0;b<B;b++) {
        const uint32_t *x=ids+(size_t)b*T;int sep=-1;
        for(int t=0;t<T;t++)if(x[t]==MATCH_SEP){sep=t;break;}
        for(int t=0;t<T;t++) {
            uint8_t m=0;
            if(x[t]>=3 && x[t]!=MATCH_SEP)for(int u=0;u<T && !m;u++) {
                if(u==t || x[u]!=x[t])continue;
                m=sep<0 ? 1 : ((t<sep)!=(u<sep));
            }
            out[(size_t)b*T+t]=m;
        }
    }
}
typedef struct { size_t n; float *z,*g,*m,*v; uint64_t t; } Param;
static Param param_new(size_t n,int training) {
    Param p={0}; p.n=n; p.z=alloc(n,4);
    if(training){p.g=alloc(n,4);p.m=alloc(n,4);p.v=alloc(n,4);} return p;
}
static void param_free(Param *p) { free(p->z);free(p->g);free(p->m);free(p->v);memset(p,0,sizeof *p); }
typedef struct {
    int C,O,d,dilation,kernel,R,N;
    int32_t *ch,*off;
    Param p;
    float *corners;
} Layer;
static Layer layer_new(int C,int O,int d,int dilation,int kernel,int train) {
    Layer l={0}; l.C=C;l.O=O;l.d=d;l.dilation=dilation;l.kernel=kernel;
    l.R=1<<d;l.N=l.R-1;
    l.ch=alloc(mul(O,l.R),4);l.off=alloc(mul(O,l.R),4);
    l.p=param_new(mul(mul(O,l.N),4),train);l.corners=alloc(l.p.n,4);return l;
}
static void layer_init(Layer *l,int est,int anchor,RNG *r) {
    int rad=l->kernel/2;
    for(int o=0;o<l->O;o++) {
        int first=anchor?o%l->C:randint(r,l->C);
        int second=(first+1+randint(r,l->C>1?l->C-1:1))%l->C;
        for(int k=0;k<l->R;k++) {
            l->ch[o*l->R+k]=randint(r,2)?second:first;
            l->off[o*l->R+k]=randint(r,l->kernel)-rad;
        }
        l->ch[o*l->R]=first;l->off[o*l->R]=0;
        if(l->R>=4 && rad){l->off[o*l->R+1]=-rad;l->off[o*l->R+2]=rad;}
    }
    float mu=est?1.2f:3.f,sd=est?.25f:.5f;
    for(size_t k=0;k<l->p.n;k++) l->p.z[k]=((k%4)<2?-1.f:1.f)*(mu+sd*normal(r));
}
static void layer_free(Layer *l) {free(l->ch);free(l->off);free(l->corners);param_free(&l->p);}
typedef struct { int n; char **words; int32_t *slot; size_t cap; } Vocab;
static uint64_t hashword(const char *s) {
    uint64_t h=UINT64_C(1469598103934665603);
    for(;*s;s++){h^=(unsigned char)*s;h*=UINT64_C(1099511628211);}return h;
}
static void vocab_index(Vocab *v) {
    free(v->slot);v->cap=16;while(v->cap<(size_t)v->n*3)v->cap*=2;
    v->slot=alloc(v->cap,4);
    for(int i=0;i<v->n;i++) if(v->words[i] && *v->words[i]) {
        size_t j=hashword(v->words[i])&(v->cap-1);
        while(v->slot[j]) {
            if(!strcmp(v->words[v->slot[j]-1],v->words[i])) die("duplicate vocabulary word: %s",v->words[i]);
            j=(j+1)&(v->cap-1);
        }v->slot[j]=i+1;
    }
}
static int vocab_lookup(const Vocab *v,const char *s) {
    if(!v->n)return UNK;
    size_t j=hashword(s)&(v->cap-1);
    while(v->slot[j]){int i=v->slot[j]-1;if(!strcmp(s,v->words[i]))return i;j=(j+1)&(v->cap-1);}return UNK;
}
static void vocab_free(Vocab *v) {for(int i=0;i<v->n;i++)free(v->words[i]);free(v->words);free(v->slot);memset(v,0,sizeof *v);}
typedef struct {
    Config c; Layer *l; Param emb,bias; float *codes;
    float temperature,best_val; uint64_t step; RNG rng; Vocab vocab;
} Net;
static Net *net_new(Config c,int training,int initialize,uint64_t seed) {
    validate(c);Net *n=alloc(1,sizeof *n);n->c=c;n->rng.s=seed?seed:1;
    n->temperature=1;n->best_val=-1;
    n->emb=param_new(mul(c.vocab,c.bits),training);n->bias=param_new(c.vocab-3,training);
    n->codes=alloc(n->emb.n,4);n->l=alloc(c.blocks+2,sizeof(Layer));
    for(int i=0;i<c.blocks;i++)n->l[i]=layer_new(in_ch(&c,i),c.width,c.depth,1<<(i%c.cycle),c.kernel,training);
    n->l[c.blocks]=layer_new(c.width,c.bits,2,1,1,training);
    n->l[c.blocks+1]=layer_new(c.width,2*c.votes,c.depth,1,1,training);
    if(initialize) {
        int *perm=alloc(c.bits,sizeof(int));
        for(int v=0;v<c.vocab;v++) {
            for(int k=0;k<c.bits;k++){perm[k]=k;n->emb.z[(size_t)v*c.bits+k]=.8f;}
            for(int k=c.bits-1;k>0;k--){int j=randint(&n->rng,k+1),a=perm[k];perm[k]=perm[j];perm[j]=a;}
            for(int k=0;k<c.bits/2;k++)n->emb.z[(size_t)v*c.bits+perm[k]]=-.8f;
        }free(perm);memset(n->emb.z,0,c.bits*4);
        for(int i=0;i<c.blocks+2;i++)layer_init(&n->l[i],c.est,i!=c.blocks+1,&n->rng);
    }return n;
}
static void net_free(Net *n) {
    if(!n)return;
    for(int i=0;i<n->c.blocks+2;i++)layer_free(n->l+i);
    free(n->l);free(n->codes);param_free(&n->emb);param_free(&n->bias);vocab_free(&n->vocab);free(n);
}
static void refresh(Net *n) {
    float temp=fmaxf(.05f,n->temperature);
    #pragma omp parallel for schedule(static)
    for(size_t k=0;k<n->emb.n;k++)n->codes[k]=k<(size_t)n->c.bits?0:sigmoid(n->emb.z[k]/temp);
    for(int i=0;i<n->c.blocks+2;i++) {
        Layer *l=n->l+i;
        #pragma omp parallel for schedule(static)
        for(size_t k=0;k<l->p.n;k++)l->corners[k]=corner(l->p.z[k],n->c.est);
    }
}
/* Lane-major activations: X[((size_t)t*C+c)*B+b]. Batch lanes are contiguous, so
 * the inner loop over lanes vectorizes and the gather indices, which depend only
 * on (o,t), are computed once per tree instead of once per batch element.
 * Accumulation order per lane is unchanged, so results are bitwise identical. */
#define LANES 16
static void tree_leaves(const Layer *l,int T,int t,int o,int32_t *base) {
    for(int k=0;k<l->R;k++) {
        int s=t+l->off[o*l->R+k]*l->dilation;
        base[k]=(s<0||s>=T)?-1:(int32_t)((size_t)s*l->C+l->ch[o*l->R+k]);
    }
}
static int g_hard_fwd=0,g_fwd_opt=-1;  /* hard (straight-through) forward; g_fwd_opt: -1 default (hard), 0 --soft-forward, 1 --hard-forward */
static void tree_forward(const Layer *l,const float *x,const int32_t *base,const float *p,
                         int B,int b0,int nl,int q8,float v[][LANES]) {
    for(int k=0;k<l->R;k++) {
        float *dst=v[l->N+k];
        if(base[k]<0){for(int e=0;e<nl;e++)dst[e]=0.f;continue;}
        const float *src=x+(size_t)base[k]*B+b0;
        for(int e=0;e<nl;e++)dst[e]=src[e];
    }
    for(int j=l->N-1;j>=0;j--) {
        const float *q=p+4*j;const float *a=v[2*j+1],*bb=v[2*j+2];float *out=v[j];
        for(int e=0;e<nl;e++) {
            float r=(1-a[e])*((1-bb[e])*q[0]+bb[e]*q[1])+a[e]*((1-bb[e])*q[2]+bb[e]*q[3]);
            out[e]=g_hard_fwd?(r>.5f?1.f:0.f):q8?nearbyintf(r*255.f)/255.f:r;
        }
    }
}

/* ---- bitfwd_patch.py: bit-level kernels for the hard forward pass (exact; see the patch header) ---- */
static uint64_t *g_xbits=NULL;static size_t g_xbits_n=0;
static void bit_pack(const float *x,int C,int B,int T,int NWB) {  /* bits[(t*C+c)*NWB+wb], bit b%64 = lane b */
    size_t need=(size_t)T*C*NWB;
    if(need>g_xbits_n){free(g_xbits);g_xbits=alloc(need,8);g_xbits_n=need;}
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<(size_t)T*C;i++) {
        const float *s=x+i*B;
        for(int wb=0;wb<NWB;wb++){uint64_t w=0;int e0=wb*64,e1=B-e0<64?B:e0+64;
            for(int b=e0;b<e1;b++)w|=(uint64_t)(s[b]>.5f)<<(b-e0);
            g_xbits[i*NWB+wb]=w;}
    }
}
static inline uint64_t bit_gate(uint64_t a,uint64_t b,unsigned f) {  /* f bit 2a+b */
    uint64_t m0=-(uint64_t)(f&1),m1=-(uint64_t)((f>>1)&1),m2=-(uint64_t)((f>>2)&1),m3=-(uint64_t)((f>>3)&1);
    uint64_t t0=m0^(b&(m0^m1)),t1=m2^(b&(m2^m3));return t0^(a&(t0^t1));
}
static inline unsigned bit_tt(const float *q){return (q[0]>.5f)|((q[1]>.5f)<<1)|((q[2]>.5f)<<2)|((q[3]>.5f)<<3);}
/* node words of tree (o,t) for word wb: nw[node] */
static inline void bit_tree(const Layer *l,const int32_t *base,const unsigned char *tt,int NWB,int wb,uint64_t *nw) {
    for(int k=0;k<l->R;k++)nw[l->N+k]=base[k]<0?0:g_xbits[(size_t)base[k]*NWB+wb];
    for(int j=l->N-1;j>=0;j--)nw[j]=bit_gate(nw[2*j+1],nw[2*j+2],tt[j]);
}
static void bit_forward_layer(const Layer *l,const float *x,float *y,const uint8_t *valid,int B,int T) {
    int O=l->O,N=l->N,NWB=(B+63)/64;bit_pack(x,l->C,B,T,NWB);
    unsigned char *tt=alloc((size_t)O*N,1);
    for(int o=0;o<O;o++)for(int j=0;j<N;j++)tt[o*N+j]=(unsigned char)bit_tt(l->corners+((size_t)o*N+j)*4);
    #pragma omp parallel for collapse(2) schedule(static)
    for(int t=0;t<T;t++) for(int o=0;o<O;o++) {
        int32_t base[MAX_LEAVES];tree_leaves(l,T,t,o,base);uint64_t nw[2*MAX_LEAVES-1];
        float *dst=y+((size_t)t*O+o)*B;
        for(int wb=0;wb<NWB;wb++) {
            bit_tree(l,base,tt+(size_t)o*N,NWB,wb,nw);
            int e0=wb*64,e1=B-e0<64?B:e0+64;
            for(int b=e0;b<e1;b++)dst[b]=valid[(size_t)b*T+t]?(float)((nw[0]>>(b-e0))&1):0.f;
        }
    }
    free(tt);
}
/* phase 1 of backward_layer for the hard forward pass: partial (per-lane corner gradients) and G (leaf gradients) */
static void bit_backward_phase1(const Layer *l,const float *x,const float *dy,float *partial,float *G,
                                const uint8_t *valid,int B,int T) {
    int O=l->O,N=l->N,R=l->R,NWB=(B+63)/64;bit_pack(x,l->C,B,T,NWB);
    #pragma omp parallel for schedule(dynamic,4)
    for(int o=0;o<O;o++) {
        const float *p=l->corners+(size_t)o*N*4;unsigned char tt[MAX_LEAVES];float dA[MAX_LEAVES][2],dB[MAX_LEAVES][2];
        for(int j=0;j<N;j++){const float *q=p+4*j;tt[j]=(unsigned char)bit_tt(q);
            dA[j][0]=q[2]-q[0];dA[j][1]=q[3]-q[1];dB[j][0]=q[1]-q[0];dB[j][1]=q[3]-q[2];}
        for(int t=0;t<T;t++) {
            int32_t base[MAX_LEAVES];tree_leaves(l,T,t,o,base);
            uint64_t nw[8][2*MAX_LEAVES-1];  /* NWB <= 8: batch <= 512 */
            for(int wb=0;wb<NWB;wb++)bit_tree(l,base,tt,NWB,wb,nw[wb]);
            const float *up=dy+((size_t)t*O+o)*B;
            for(int b0=0;b0<B;b0+=LANES) {  /* LANES divides 64: a chunk's lanes share one word */
                int nl=B-b0<LANES?B-b0:LANES,sh=b0&63;float g[2*MAX_LEAVES-1][LANES],v[2*MAX_LEAVES-1][LANES];
                const uint64_t *w=nw[b0>>6];
                for(int n=0;n<2*N+1;n++){uint64_t x=w[n]>>sh;for(int e=0;e<nl;e++)v[n][e]=(float)((x>>e)&1);}
                for(int e=0;e<nl;e++)g[0][e]=valid[(size_t)(b0+e)*T+t]?up[b0+e]:0.f;
                for(int j=0;j<N;j++) {
                    float *d0=partial+(size_t)(((size_t)o*N+j)*4+0)*B+b0,*d1=partial+(size_t)(((size_t)o*N+j)*4+1)*B+b0;
                    float *d2=partial+(size_t)(((size_t)o*N+j)*4+2)*B+b0,*d3=partial+(size_t)(((size_t)o*N+j)*4+3)*B+b0;
                    const float *a=v[2*j+1],*bb=v[2*j+2],*u=g[j];float *ga=g[2*j+1],*gb=g[2*j+2];
                    const float a0=dA[j][0],a1=dA[j][1],b0f=dB[j][0],b1f=dB[j][1];
                    for(int e=0;e<nl;e++) {  /* a, b in {0,1}: one corner gets u, the others u*0 = +-0 (no-op) */
                        float m11=a[e]*bb[e],m10=a[e]-m11,m01=bb[e]-m11,m00=1.f-a[e]-bb[e]+m11;
                        d0[e]+=u[e]*m00;d1[e]+=u[e]*m01;d2[e]+=u[e]*m10;d3[e]+=u[e]*m11;
                        ga[e]=u[e]*(bb[e]!=0.f?a1:a0);gb[e]=u[e]*(a[e]!=0.f?b1f:b0f);
                    }
                }
                for(int k=0;k<R;k++) {
                    float *d=G+(((size_t)o*T+t)*R+k)*B+b0;const float *s=g[N+k];
                    for(int e=0;e<nl;e++)d[e]=s[e];
                }
            }
        }
    }
}
static void forward_layer(const Layer *l,const float *x,float *y,const uint8_t *valid,int B,int T,int q8) {
    if(g_hard_fwd && !q8 && B<=512){bit_forward_layer(l,x,y,valid,B,T);return;}
    int O=l->O;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int t=0;t<T;t++) for(int o=0;o<O;o++) {
        int32_t base[MAX_LEAVES];tree_leaves(l,T,t,o,base);
        const float *p=l->corners+(size_t)o*l->N*4;
        for(int b0=0;b0<B;b0+=LANES) {
            int nl=B-b0<LANES?B-b0:LANES;
            float v[2*MAX_LEAVES-1][LANES];
            tree_forward(l,x,base,p,B,b0,nl,q8,v);
            float *dst=y+((size_t)t*O+o)*B+b0;
            for(int e=0;e<nl;e++)dst[e]=valid[(size_t)(b0+e)*T+t]?v[0][e]:0.f;
        }
    }
}
static float *g_leafgrad=NULL;static size_t g_leafgrad_n=0;
static void backward_layer(Layer *l,const float *x,const float *dy,float *dx,float *partial,
                           const uint8_t *valid,int B,int T,int q8,int est) {
    /* fasttrain: phase 1 parallel over outputs, phase 2 parallel over input channels; the float
     * operations and their order per element are those of the original serial-over-outputs loop. */
    int O=l->O,N=l->N,R=l->R,C=l->C;
    memset(dx,0,mul(mul(B,T),l->C)*4);memset(partial,0,mul(B,l->p.n)*4);
    size_t need=mul(mul(mul(O,T),R),B);
    if(need>g_leafgrad_n){free(g_leafgrad);g_leafgrad=alloc(need,4);g_leafgrad_n=need;}
    float *G=g_leafgrad;
    if(g_hard_fwd && !q8 && B<=512){bit_backward_phase1(l,x,dy,partial,G,valid,B,T);}else{
    #pragma omp parallel for schedule(dynamic,4)
    for(int o=0;o<O;o++)for(int b0=0;b0<B;b0+=LANES)for(int t=0;t<T;t++) {
        int32_t base[MAX_LEAVES];tree_leaves(l,T,t,o,base);
        const float *p=l->corners+(size_t)o*l->N*4;
        int nl=B-b0<LANES?B-b0:LANES;
        float v[2*MAX_LEAVES-1][LANES],g[2*MAX_LEAVES-1][LANES];
        tree_forward(l,x,base,p,B,b0,nl,q8,v);
        const float *up=dy+((size_t)t*O+o)*B+b0;
        for(int e=0;e<nl;e++)g[0][e]=valid[(size_t)(b0+e)*T+t]?up[e]:0.f;
        for(int j=0;j<N;j++) {
            const float *q=p+4*j,*a=v[2*j+1],*bb=v[2*j+2],*u=g[j];
            float *d0=partial+(size_t)(((size_t)o*N+j)*4+0)*B+b0;
            float *d1=partial+(size_t)(((size_t)o*N+j)*4+1)*B+b0;
            float *d2=partial+(size_t)(((size_t)o*N+j)*4+2)*B+b0;
            float *d3=partial+(size_t)(((size_t)o*N+j)*4+3)*B+b0;
            float *ga=g[2*j+1],*gb=g[2*j+2];
            for(int e=0;e<nl;e++) {
                d0[e]+=u[e]*(1-a[e])*(1-bb[e]);d1[e]+=u[e]*(1-a[e])*bb[e];
                d2[e]+=u[e]*a[e]*(1-bb[e]);d3[e]+=u[e]*a[e]*bb[e];
                ga[e]=u[e]*((1-bb[e])*(q[2]-q[0])+bb[e]*(q[3]-q[1]));
                gb[e]=u[e]*((1-a[e])*(q[1]-q[0])+a[e]*(q[3]-q[2]));
            }
        }
        for(int k=0;k<R;k++) {
            float *d=G+(((size_t)o*T+t)*R+k)*B+b0;const float *s=g[N+k];
            for(int e=0;e<nl;e++)d[e]=s[e];
        }
    }
    }  /* bitfwd_patch.py: end of the float phase 1 */
    /* per input channel: (o,k) readers sorted by (o, -offset, k) = the original (o, t, k) order */
    int *cnt=alloc(C+1,sizeof(int)),*lo=alloc((size_t)O*R,sizeof(int)),*lk=alloc((size_t)O*R,sizeof(int));
    for(int i=0;i<O*R;i++)cnt[l->ch[i]+1]++;
    for(int c=0;c<C;c++)cnt[c+1]+=cnt[c];
    {
        int *fill=alloc(C,sizeof(int));memcpy(fill,cnt,C*sizeof(int));
        int ord[MAX_LEAVES];
        for(int o=0;o<O;o++) {
            for(int k=0;k<R;k++)ord[k]=k;
            for(int i=1;i<R;i++){int k=ord[i],j=i-1;  /* stable insertion sort by offset descending */
                while(j>=0 && l->off[o*R+ord[j]]<l->off[o*R+k]){ord[j+1]=ord[j];j--;}ord[j+1]=k;}
            for(int i=0;i<R;i++){int k=ord[i],c=l->ch[o*R+k];lo[fill[c]]=o;lk[fill[c]]=k;fill[c]++;}
        }
        free(fill);
    }
    #pragma omp parallel for schedule(dynamic,8)
    for(int c=0;c<C;c++)for(int s=0;s<T;s++) {
        float *d=dx+((size_t)s*C+c)*B;
        for(int i=cnt[c];i<cnt[c+1];i++) {
            int o=lo[i],k=lk[i],t=s-l->off[o*R+k]*l->dilation;
            if(t<0||t>=T)continue;
            const float *src=G+(((size_t)o*T+t)*R+k)*B;
            for(int e=0;e<B;e++)d[e]+=src[e];
        }
    }
    free(cnt);free(lo);free(lk);
    #pragma omp parallel for schedule(static)
    for(size_t k=0;k<l->p.n;k++) {
        double s=0;for(int b=0;b<B;b++)s+=partial[k*(size_t)B+b];
        l->p.g[k]=(float)s*cderiv(l->p.z[k],l->corners[k],est);
    }
}

typedef struct {
    int B,T,U,maxC;size_t max_param;
    float **a,*pool,*head,*proj,*g0,*g1,*partial,*code_grad,*logits;
    uint8_t *valid,*pv,*selected;uint32_t *ids,*targets;int *labels;
    float **aug,*gtmp;int32_t **amax;
} Work;
static Work *work_new(Net *n,int B,int T) {
    Work *w=alloc(1,sizeof *w);w->B=B;w->T=T;w->U=(T+1)/2;
    int K=n->c.blocks,C=n->c.bits,W=n->c.width;
    w->maxC=W>C?W:C;if(w->maxC<2*n->c.votes)w->maxC=2*n->c.votes;
    if(n->c.gevery && w->maxC<W+n->c.gch)w->maxC=W+n->c.gch;
    w->aug=alloc(K+1,sizeof(float*));w->amax=alloc(K+1,sizeof(int32_t*));
    for(int i=0;i<K;i++)if(gblock(&n->c,i)){w->aug[i]=alloc(mul(mul(B,T),W+n->c.gch),4);w->amax[i]=alloc(mul(B,n->c.gch),4);}
    if(n->c.gevery)w->gtmp=alloc(mul(mul(B,T),W),4);
    w->a=alloc(K+1,sizeof(float*));
    for(int i=0;i<=K;i++)w->a[i]=alloc(mul(mul(B,T),i?W:C+n->c.match),4);
    for(int i=0;i<K+2;i++)if(n->l[i].p.n>w->max_param)w->max_param=n->l[i].p.n;
    w->pool=alloc(mul(mul(B,w->U),W),4);w->head=alloc(mul(mul(B,w->U),2*n->c.votes),4);
    w->proj=alloc(mul(mul(B,T),C),4);w->g0=alloc(mul(mul(B,T),w->maxC),4);w->g1=alloc(mul(mul(B,T),w->maxC),4);
    w->partial=alloc(mul(B,w->max_param),4);w->code_grad=alloc(n->emb.n,4);w->logits=alloc(n->c.vocab,4);
    w->valid=alloc(mul(B,T),1);w->pv=alloc(mul(B,w->U),1);w->selected=alloc(mul(B,T),1);
    w->ids=alloc(mul(B,T),4);w->targets=alloc(mul(B,T),4);w->labels=alloc(B,sizeof(int));return w;
}
static void work_free(Work *w,int K) {
    for(int i=0;i<=K;i++)free(w->a[i]);
    free(w->a);free(w->pool);free(w->head);free(w->proj);
    free(w->g0);free(w->g1);free(w->partial);free(w->code_grad);free(w->logits);free(w->valid);free(w->pv);
    for(int i=0;i<=K;i++){free(w->aug[i]);free(w->amax[i]);}free(w->aug);free(w->amax);free(w->gtmp);
    free(w->selected);free(w->ids);free(w->targets);free(w->labels);free(w);
}
/* block i input: a[i], plus (global blocks) G channels = max over valid positions, broadcast */
static float *ginput(Net *n,Work *w,int i) {
    if(!gblock(&n->c,i))return w->a[i];
    int W=n->c.width,G=n->c.gch,B=w->B,T=w->T,CA=W+G;const float *x=w->a[i];float *y=w->aug[i];int32_t *am=w->amax[i];
    #pragma omp parallel for schedule(static)
    for(int t=0;t<T;t++)for(int c=0;c<W;c++)memcpy(y+((size_t)t*CA+c)*B,x+((size_t)t*W+c)*B,(size_t)B*4);
    #pragma omp parallel for schedule(static)
    for(int c=0;c<G;c++)for(int b=0;b<B;b++) {
        float m=0.f;int32_t at=-1;
        if(n->c.gmean){double s=0;int nv=0;for(int t=0;t<T;t++)if(w->valid[(size_t)b*T+t]){s+=x[((size_t)t*W+c)*B+b];nv++;}
            m=nv?(float)(s/nv):0.f;at=nv;if(g_hard_fwd)m=m>.5f?1.f:0.f;}  /* mean pooling: am holds the valid count */
        else for(int t=0;t<T;t++)if(w->valid[(size_t)b*T+t]){float v=x[((size_t)t*W+c)*B+b];if(at<0||v>m){m=v;at=t;}}
        am[(size_t)c*B+b]=at;
        for(int t=0;t<T;t++)y[((size_t)t*CA+W+c)*B+b]=w->valid[(size_t)b*T+t]?m:0.f;
    }
    return y;
}
/* gradient w.r.t. the augmented input (CA channels, in dx) -> w.r.t. a[i] (W channels, back into dx) */
static void gcollapse(Net *n,Work *w,int i,float *dx) {
    int W=n->c.width,G=n->c.gch,B=w->B,T=w->T,CA=W+G;float *o=w->gtmp;const int32_t *am=w->amax[i];
    #pragma omp parallel for schedule(static)
    for(int t=0;t<T;t++)for(int c=0;c<W;c++)memcpy(o+((size_t)t*W+c)*B,dx+((size_t)t*CA+c)*B,(size_t)B*4);
    #pragma omp parallel for schedule(static)
    for(int c=0;c<G;c++)for(int b=0;b<B;b++) {
        int32_t at=am[(size_t)c*B+b];if(at<=0 && (n->c.gmean || at<0))continue;
        float s=0.f;for(int t=0;t<T;t++)s+=dx[((size_t)t*CA+W+c)*B+b];
        if(n->c.gmean){float q=s/at;for(int t=0;t<T;t++)if(w->valid[(size_t)b*T+t])o[((size_t)t*W+c)*B+b]+=q;}
        else o[((size_t)at*W+c)*B+b]+=s;
    }
    memcpy(dx,o,(size_t)B*T*W*4);
}
static void encode(Net *n,Work *w) {
    int C=n->c.bits,C0=C+n->c.match;refresh(n);uint8_t *mb=NULL;
    if(n->c.match){mb=alloc(mul(w->B,w->T),1);match_bits(w->ids,w->B,w->T,mb);}
    #pragma omp parallel for schedule(static)
    for(size_t j=0;j<(size_t)w->B*w->T;j++) {
        uint32_t id=w->ids[j];if(id>=(uint32_t)n->c.vocab)die("token ID outside vocabulary");
        w->valid[j]=id!=PAD;
        int b=(int)(j/(size_t)w->T),t=(int)(j%(size_t)w->T);
        for(int k=0;k<C;k++) {
            float a=n->codes[(size_t)id*C+k];
            w->a[0][((size_t)t*C0+k)*w->B+b]=g_hard_fwd?(a>.5f?1.f:0.f):n->c.q8?nearbyintf(a*255.f)/255.f:a;
        }
        if(n->c.match)w->a[0][((size_t)t*C0+C)*w->B+b]=(float)mb[j];
    }
    free(mb);
    for(int i=0;i<n->c.blocks;i++)forward_layer(n->l+i,ginput(n,w,i),w->a[i+1],w->valid,w->B,w->T,n->c.q8);
}
static void pool_forward(Net *n,Work *w) {
    int C=n->c.width,T=w->T,U=w->U;
    const float *x=w->a[n->c.blocks];
    for(int b=0;b<w->B;b++)for(int u=0;u<U;u++) {
        int t=2*u;w->pv[b*U+u]=w->valid[b*T+t] | (t+1<T?w->valid[b*T+t+1]:0);
        for(int c=0;c<C;c++)w->pool[((size_t)u*C+c)*w->B+b]=
            fmaxf(x[((size_t)t*C+c)*w->B+b],t+1<T?x[((size_t)(t+1)*C+c)*w->B+b]:0.f);
    }
}
static void pool_backward(Net *n,Work *w,const float *dy,float *dx) {
    int C=n->c.width,T=w->T,U=w->U;const float *x=w->a[n->c.blocks];
    memset(dx,0,(size_t)w->B*T*C*4);
    for(int b=0;b<w->B;b++)for(int u=0;u<U;u++)for(int c=0;c<C;c++) {
        int t=2*u;size_t i=((size_t)t*C+c)*w->B+b,i2=((size_t)(t+1)*C+c)*w->B+b;
        float a=x[i],bb=t+1<T?x[i2]:0,g=dy[((size_t)u*C+c)*w->B+b];
        dx[i]=g*(a>bb?1.f:a==bb?.5f:0.f);
        if(t+1<T)dx[i2]=g*(bb>a?1.f:a==bb?.5f:0.f);
    }
}
static float supervised(Net *n,Work *w,int grad,int *correct) {
    encode(n,w);pool_forward(n,w);int O=2*n->c.votes;
    forward_layer(n->l+n->c.blocks+1,w->pool,w->head,w->pv,w->B,w->U,n->c.q8);
    double loss=0;int hit=0;
    for(int b=0;b<w->B;b++) {
        int count=0,y=w->labels[b];if(y<0||y>1)die("supervised training requires labels 0 or 1");
        double sum[2]={0,0};
        for(int u=0;u<w->U;u++) {
            count+=w->pv[b*w->U+u];
            for(int o=0;o<O;o++)sum[o/n->c.votes]+=w->head[((size_t)u*O+o)*w->B+b];
        }
        float scale=n->c.scale/((count?count:1)*n->c.votes);
        float d=(float)(sum[1]-sum[0])*scale,p=sigmoid(d);
        loss+=fmaxf(d,0)+log1pf(expf(-fabsf(d)))-y*d;hit+=(d>0)==y;
        if(grad)for(int u=0;u<w->U;u++)for(int o=0;o<O;o++)
            w->g0[((size_t)u*O+o)*w->B+b]=(o<n->c.votes?-1.f:1.f)*(p-y)*scale/w->B*w->pv[b*w->U+u];
    }
    if(correct)*correct=hit;
    if(grad) {
        backward_layer(n->l+n->c.blocks+1,w->pool,w->g0,w->g1,w->partial,w->pv,w->B,w->U,n->c.q8,n->c.est);
        pool_backward(n,w,w->g1,w->g0);
    }return (float)(loss/w->B);
}
static int corrupt(Net *n,Work *w) {
    size_t count=(size_t)w->B*w->T;memcpy(w->targets,w->ids,count*4);memset(w->selected,0,count);
    int total=0,eligible=0,last=-1;
    for(size_t i=0;i<count;i++)if(w->ids[i]>=3) {
        eligible++;if(randint(&n->rng,eligible)==0)last=(int)i;
        if(uniform(&n->rng)<n->c.mask_prob){w->selected[i]=1;total++;}
    }
    if(!eligible)die("MLM batch has no known real tokens (IDs >=3)");
    if(!total){w->selected[last]=1;total=1;}
    if(n->c.max_targets && total>n->c.max_targets) {
        int remaining=total,keep=n->c.max_targets;
        for(size_t i=0;i<count;i++)if(w->selected[i]) {
            if(randint(&n->rng,remaining)<keep)keep--;else w->selected[i]=0;
            remaining--;
        }
        total=n->c.max_targets;
    }
    for(size_t i=0;i<count;i++)if(w->selected[i]) {
        float u=uniform(&n->rng);
        if(u<.8f)w->ids[i]=MASK;else if(u<.9f)w->ids[i]=3+randint(&n->rng,n->c.vocab-3);
    }
    return total;
}
static float masked_loss(Net *n,Work *w,int grad) {
    encode(n,w);int C=n->c.bits,V=n->c.vocab,K=n->c.blocks;
    forward_layer(n->l+K,w->a[K],w->proj,w->valid,w->B,w->T,n->c.q8);
    size_t BT=(size_t)w->B*w->T;int count=0;
    for(size_t i=0;i<BT;i++)count+=w->selected[i];
    if(!count)die("no MLM targets");
    if(grad){memset(w->g0,0,BT*C*4);memset(n->bias.g,0,n->bias.n*4);}
    float scale=n->c.mlm_scale/C;double loss=0;
    /* Phase A (parallel over targets): logits, softmax numerators, denominators.
     * Phase B (parallel over targets): gradient into the projection output.
     * Phase C (parallel over vocabulary rows): bias and code gradients, summing
     * targets in ascending order. Same per-element accumulation order as the
     * serial loop, so results are bitwise identical at any thread count. */
    size_t *tix=alloc(count,sizeof(size_t));
    {int c=0;for(size_t i=0;i<BT;i++)if(w->selected[i])tix[c++]=i;}
    int Vr=V-3;float *ex=alloc(mul(count,Vr),4);double *den=alloc(count,8),*lt=alloc(count,8);
    /* mlmfast_patch.py: tiled, vectorisable layout; every element keeps the original's float operations and order */
    enum{MT=16,MV=256};
    float *hv=alloc(mul(count,C),4),*cm=alloc(mul(Vr,C),4),*ct=alloc(mul(Vr,C),4);
    #pragma omp parallel for schedule(static)
    for(int c=0;c<count;c++) {
        size_t i=tix[c];uint32_t target=w->targets[i];
        if(target<3||target>=(uint32_t)V)die("invalid MLM target");
        int bi=(int)(i/(size_t)w->T),ti=(int)(i%(size_t)w->T);
        const float *h=w->proj+(size_t)ti*C*w->B+bi;int hs=w->B;
        for(int k=0;k<C;k++)hv[(size_t)c*C+k]=2*h[(size_t)k*hs]-1;
    }
    #pragma omp parallel for schedule(static)
    for(int vr=0;vr<Vr;vr++)for(int k=0;k<C;k++){float x=2*n->codes[(size_t)(vr+3)*C+k]-1;cm[(size_t)vr*C+k]=x;ct[(size_t)k*Vr+vr]=x;}
    {
        int ntt=(count+MT-1)/MT,nvc=(Vr+MV-1)/MV;
        #pragma omp parallel for schedule(dynamic,4)
        for(int job=0;job<ntt*nvc;job++) {
            int c0=(job/nvc)*MT,v0=(job%nvc)*MV,nc=count-c0<MT?count-c0:MT,nv=Vr-v0<MV?Vr-v0:MV;
            float acc[MT][MV];
            for(int t=0;t<nc;t++)for(int j=0;j<nv;j++)acc[t][j]=0;
            for(int k=0;k<C;k++) {
                const float *row=ct+(size_t)k*Vr+v0;
                for(int t=0;t<nc;t++){const float hk=hv[(size_t)(c0+t)*C+k];float *a=acc[t];for(int j=0;j<nv;j++)a[j]+=hk*row[j];}
            }
            for(int t=0;t<nc;t++){float *e=ex+(size_t)(c0+t)*Vr+v0;for(int j=0;j<nv;j++)e[j]=scale*acc[t][j]+n->bias.z[v0+j];}
        }
    }
    #pragma omp parallel for schedule(static)
    for(int c=0;c<count;c++) {
        uint32_t target=w->targets[tix[c]];float *e=ex+(size_t)c*Vr,mx=-FLT_MAX;
        for(int v=3;v<V;v++)if(e[v-3]>mx)mx=e[v-3];
        float target_logit=e[target-3];double denom=0;
        for(int v=3;v<V;v++){e[v-3]=expf(e[v-3]-mx);denom+=e[v-3];}
        den[c]=denom;lt[c]=log(denom)+mx-target_logit;
    }
    for(int c=0;c<count;c++)loss+=lt[c];
    if(grad) {
        #pragma omp parallel for schedule(dynamic,1)
        for(int c0=0;c0<count;c0+=MT) {
            int nc=count-c0<MT?count-c0:MT;float *acc=alloc(mul(nc,C),4);
            for(int v0=0;v0<Vr;v0+=MV)for(int t=0;t<nc;t++) {
                int c=c0+t;uint32_t target=w->targets[tix[c]];const float *e=ex+(size_t)c*Vr;float *a=acc+(size_t)t*C;
                int v1=v0+MV<Vr?v0+MV:Vr;
                for(int vr=v0;vr<v1;vr++) {
                    int v=vr+3;float g=((float)(e[vr]/den[c])-(v==(int)target))/count;const float s=2*scale*g;
                    const float *row=cm+(size_t)vr*C;for(int k=0;k<C;k++)a[k]+=s*row[k];
                }
            }
            for(int t=0;t<nc;t++) {
                size_t i=tix[c0+t];int bi=(int)(i/(size_t)w->T),ti=(int)(i%(size_t)w->T);
                for(int k=0;k<C;k++)w->g0[((size_t)ti*C+k)*w->B+bi]+=acc[(size_t)t*C+k];
            }
            free(acc);
        }
        #pragma omp parallel for schedule(static)
        for(int v=3;v<V;v++) {
            float *cg=w->code_grad+(size_t)v*C;
            for(int c=0;c<count;c++) {
                uint32_t target=w->targets[tix[c]];
                float g=((float)(ex[(size_t)c*Vr+v-3]/den[c])-(v==(int)target))/count;
                n->bias.g[v-3]+=g;
                const float s=2*scale*g;const float *hc=hv+(size_t)c*C;
                for(int k=0;k<C;k++)cg[k]+=s*hc[k];
            }
        }
    }
    free(hv);free(cm);free(ct);
    free(tix);free(ex);free(den);free(lt);
    if(grad) {
        backward_layer(n->l+K,w->a[K],w->g0,w->g1,w->partial,w->valid,w->B,w->T,n->c.q8,n->c.est);
        memcpy(w->g0,w->g1,(size_t)w->B*w->T*n->c.width*4);
    }return (float)(loss/count);
}
static float regularizer(Net *n,Work *w,float sharp,int grad) {
    int C=n->c.bits,V=n->c.vocab,M=V-3;double row=0,col=0,binary=0;
    float *cm=alloc(C,4),*rm=alloc(M,4);
    for(int v=3;v<V;v++) {
        double s=0;for(int k=0;k<C;k++){float p=n->codes[(size_t)v*C+k];s+=p;cm[k]+=p;binary+=p*(1-p);}
        rm[v-3]=(float)(s/C)-.5f;row+=rm[v-3]*rm[v-3];
    }
    for(int k=0;k<C;k++){cm[k]=cm[k]/M-.5f;col+=cm[k]*cm[k];}
    if(grad)
    #pragma omp parallel for schedule(static)
    for(int v=3;v<V;v++)for(int k=0;k<C;k++) {
        size_t i=(size_t)v*C+k;float p=n->codes[i];
        w->code_grad[i]+=(.2f*(rm[v-3]+cm[k])+sharp*(1-2*p))/((float)M*C);
    }
    free(cm);free(rm);return (float)(.1*(row/M+col/C)+sharp*binary/((double)M*C));
}
static float objective(Net *n,Work *w,float sharp,int grad,float *ce) {
    if(grad)memset(w->code_grad,0,n->emb.n*4);
    *ce=n->c.stage?masked_loss(n,w,grad):supervised(n,w,grad,NULL);
    if(grad) {
        float *g=w->g0,*dx=w->g1;
        for(int i=n->c.blocks-1;i>=0;i--) {
            backward_layer(n->l+i,gblock(&n->c,i)?w->aug[i]:w->a[i],g,dx,w->partial,w->valid,w->B,w->T,n->c.q8,n->c.est);
            if(gblock(&n->c,i))gcollapse(n,w,i,dx);
            float *swap=g;g=dx;dx=swap;
        }
        for(size_t j=0;j<(size_t)w->B*w->T;j++)if(w->ids[j]!=PAD) {
            int b=(int)(j/(size_t)w->T),t=(int)(j%(size_t)w->T);
            for(int k=0;k<n->c.bits;k++)
                w->code_grad[(size_t)w->ids[j]*n->c.bits+k]+=g[((size_t)t*(n->c.bits+n->c.match)+k)*w->B+b];
        }
    }
    float reg=regularizer(n,w,sharp,grad);
    if(grad)
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n->emb.n;i++) {
        float p=n->codes[i];n->emb.g[i]=w->code_grad[i]*p*(1-p)/fmaxf(.05f,n->temperature);
    }
    return *ce+reg;
}
static int active_params(Net *n,Param **pp) {
    int k=0;pp[k++]=&n->emb;for(int i=0;i<n->c.blocks;i++)pp[k++]=&n->l[i].p;
    pp[k++]=&n->l[n->c.blocks+(n->c.stage?0:1)].p;
    if(n->c.stage)pp[k++]=&n->bias;
    return k;
}
static void adam(Net *n,float lr) {
    Param *ps[MAX_BLOCKS+3];int np=active_params(n,ps);double norm=0;
    for(int p=0;p<np;p++)for(size_t k=0;k<ps[p]->n;k++) {
        double g=ps[p]->g[k];if(!isfinite(g))die("nonfinite gradient; reduce learning rate");norm+=g*g;
    }
    float clip=(float)fmin(1.,n->c.clip/(sqrt(norm)+1e-6));
    for(int p=0;p<np;p++) {
        Param *a=ps[p];a->t++;double b1=1-pow(.9,(double)a->t),b2=1-pow(.999,(double)a->t);
        #pragma omp parallel for schedule(static)
        for(size_t k=0;k<a->n;k++) {
            float g=a->g[k]*clip;
            a->m[k]=.9f*a->m[k]+.1f*g;a->v[k]=.999f*a->v[k]+.001f*g*g;
            a->z[k]-=lr*(float)(a->m[k]/b1)/(sqrtf((float)(a->v[k]/b2))+1e-8f);
        }
    }
}

/* --------------------------- versioned checkpoints ------------------------ */
static FILE *open_file(const char *path,const char *mode) {
    FILE *f=fopen(path,mode);if(!f)die("%s: %s",path,strerror(errno));return f;
}
static void write_bytes(FILE *f,const void *p,size_t n) {if(n && fwrite(p,1,n,f)!=n)die("checkpoint write failed");}
static void read_bytes(FILE *f,void *p,size_t n) {if(n && fread(p,1,n,f)!=n)die("truncated file");}
static void put32(FILE *f,uint32_t x) {unsigned char a[4];for(int i=0;i<4;i++)a[i]=(unsigned char)(x>>(8*i));write_bytes(f,a,4);}
static uint32_t get32(FILE *f) {unsigned char a[4];read_bytes(f,a,4);uint32_t x=0;for(int i=0;i<4;i++)x|=(uint32_t)a[i]<<(8*i);return x;}
static void put64(FILE *f,uint64_t x) {put32(f,(uint32_t)x);put32(f,(uint32_t)(x>>32));}
static uint64_t get64(FILE *f) {uint64_t lo=get32(f);return lo|((uint64_t)get32(f)<<32);}
static void putfloat(FILE *f,float x) {uint32_t u;memcpy(&u,&x,4);put32(f,u);}
static float getfloat(FILE *f) {uint32_t u=get32(f);float x;memcpy(&x,&u,4);if(!isfinite(x))die("nonfinite checkpoint value");return x;}
static int little_endian(void) {uint32_t x=1;return *(unsigned char*)&x==1;}
static void float_write(FILE *f,const float *x,size_t n) {
    if(little_endian())write_bytes(f,x,mul(n,4));else for(size_t i=0;i<n;i++)putfloat(f,x[i]);
}
static void float_read(FILE *f,float *x,size_t n) {
    if(little_endian())read_bytes(f,x,mul(n,4));else for(size_t i=0;i<n;i++)x[i]=getfloat(f);
    for(size_t i=0;i<n;i++)if(!isfinite(x[i]))die("nonfinite checkpoint tensor");
}
static void config_write(FILE *f,Config c) {
    int a[]={c.vocab,c.bits,c.width,c.blocks,c.depth,c.kernel,c.cycle,c.votes,c.est,c.q8,
             c.seq,c.batch,c.steps,c.stage,c.max_targets};
    float b[]={c.peak_lr,c.sharp,c.clip,c.mask_prob,c.scale,c.mlm_scale};
    for(int i=0;i<15;i++)put32(f,(uint32_t)a[i]);
    for(int i=0;i<6;i++)putfloat(f,b[i]);
}
static Config config_read(FILE *f) {
    Config c={0};int *a[]={&c.vocab,&c.bits,&c.width,&c.blocks,&c.depth,&c.kernel,&c.cycle,&c.votes,&c.est,&c.q8,
                         &c.seq,&c.batch,&c.steps,&c.stage,&c.max_targets};
    float *b[]={&c.peak_lr,&c.sharp,&c.clip,&c.mask_prob,&c.scale,&c.mlm_scale};
    for(int i=0;i<15;i++){uint32_t v=get32(f);if(v>INT_MAX)die("invalid config field");*a[i]=(int)v;}
    for(int i=0;i<6;i++)*b[i]=getfloat(f);
    validate(c);return c;
}
static void vocab_write(FILE *f,const Vocab *v) {
    put32(f,(uint32_t)v->n);
    for(int i=0;i<v->n;i++){size_t len=strlen(v->words[i]);put32(f,(uint32_t)len);write_bytes(f,v->words[i],len);}
}
static Vocab vocab_read(FILE *f,int V) {
    Vocab v={0};uint32_t n=get32(f);if(n>(uint32_t)V || (n && n<3))die("invalid vocabulary size");
    v.n=(int)n;v.words=alloc(n,sizeof(char*));size_t total=0;
    for(int i=0;i<v.n;i++) {
        uint32_t len=get32(f);total+=len;if(!len || len>1048576 || total>MAX_RECORD)die("invalid vocabulary string length");
        v.words[i]=alloc((size_t)len+1,1);read_bytes(f,v.words[i],len);
        if(strlen(v.words[i])!=len)die("embedded NUL in vocabulary");
    }vocab_index(&v);return v;
}
static void param_write(FILE *f,Param p) {
    put64(f,p.n);put64(f,p.t);float_write(f,p.z,p.n);float_write(f,p.m,p.n);float_write(f,p.v,p.n);
}
static void param_read(FILE *f,Param *p) {
    if(get64(f)!=p->n)die("parameter shape mismatch");
    p->t=get64(f);if(p->t>1000000000)die("invalid Adam counter");
    float_read(f,p->z,p->n);float_read(f,p->m,p->n);float_read(f,p->v,p->n);
    for(size_t i=0;i<p->n;i++)if(p->v[i]<0)die("negative Adam second moment");
}
static char *temp_path(const char *path) {
    char *p=alloc(strlen(path)+5,1);sprintf(p,"%s.tmp",path);return p;
}
static void finish_save(FILE *f,char *tmp,const char *path) {
    if(fflush(f))die("flush failed: %s",tmp);
    if(fclose(f))die("close failed: %s",tmp);
    if(rename(tmp,path))die("rename %s -> %s: %s",tmp,path,strerror(errno));
    free(tmp);
}
static void save_net(Net *n,const char *path) {
    char *tmp=temp_path(path);FILE *f=open_file(tmp,"wb");write_bytes(f,(n->c.gevery||n->c.match)?"LTCKP002":"LTCKP001",8);config_write(f,n->c);
    if(n->c.gevery||n->c.match){put32(f,(uint32_t)n->c.gevery);put32(f,(uint32_t)n->c.gch);put32(f,(uint32_t)n->c.gmean);put32(f,(uint32_t)n->c.match);}
    put64(f,n->step);put64(f,n->rng.s);putfloat(f,n->temperature);putfloat(f,n->best_val);vocab_write(f,&n->vocab);
    param_write(f,n->emb);param_write(f,n->bias);
    for(int i=0;i<n->c.blocks+2;i++) {
        Layer *l=n->l+i;put32(f,l->C);put32(f,l->O);put32(f,l->d);put32(f,l->dilation);put32(f,l->kernel);
        for(int j=0;j<l->O*l->R;j++)put32(f,(uint32_t)l->ch[j]);
        for(int j=0;j<l->O*l->R;j++)put32(f,(uint32_t)l->off[j]);
        param_write(f,l->p);
    }finish_save(f,tmp,path);
}
static Net *load_net(const char *path) {
    FILE *f=open_file(path,"rb");char magic[8];read_bytes(f,magic,8);
    int v2=!memcmp(magic,"LTCKP002",8);
    if(memcmp(magic,"LTCKP001",8) && !v2)die("%s is not a training .ltc checkpoint",path);
    Config c=config_read(f);if(v2){c.gevery=(int)get32(f);c.gch=(int)get32(f);c.gmean=(int)get32(f);c.match=(int)get32(f);validate(c);}Net *n=net_new(c,1,0,1);n->step=get64(f);n->rng.s=get64(f);
    n->temperature=getfloat(f);n->best_val=getfloat(f);
    if(!n->rng.s || n->step>(uint64_t)c.steps || !(n->temperature>=.05f))die("invalid checkpoint state");
    n->vocab=vocab_read(f,c.vocab);param_read(f,&n->emb);param_read(f,&n->bias);
    for(int i=0;i<c.blocks+2;i++) {
        Layer *l=n->l+i;
        if(get32(f)!=(uint32_t)l->C || get32(f)!=(uint32_t)l->O || get32(f)!=(uint32_t)l->d ||
           get32(f)!=(uint32_t)l->dilation || get32(f)!=(uint32_t)l->kernel)die("checkpoint layer mismatch");
        for(int j=0;j<l->O*l->R;j++){uint32_t x=get32(f);if(x>=(uint32_t)l->C)die("channel out of bounds");l->ch[j]=(int32_t)x;}
        for(int j=0;j<l->O*l->R;j++){int32_t x=(int32_t)get32(f);if(x<-(l->kernel/2)||x>l->kernel/2)die("offset out of bounds");l->off[j]=x;}
        param_read(f,&l->p);
    }
    if(fgetc(f)!=EOF)die("trailing checkpoint bytes");
    fclose(f);refresh(n);return n;
}
static void reset_optimizer(Net *n) {
    n->step=0;n->best_val=-1;n->temperature=1;
    Param *a[MAX_BLOCKS+4];int k=0;a[k++]=&n->emb;a[k++]=&n->bias;
    for(int i=0;i<n->c.blocks+2;i++)a[k++]=&n->l[i].p;
    for(int i=0;i<k;i++){memset(a[i]->m,0,a[i]->n*4);memset(a[i]->v,0,a[i]->n*4);a[i]->t=0;}
}

/* ----------------------- float-free deployed model ----------------------- */
typedef struct {int C,O,d,dilation,kernel,R,N;int32_t *ch,*off;uint8_t *f;} HLayer;
typedef struct {Config c;uint64_t *codes;HLayer *l;Vocab vocab;} Hard;
static void hard_free(Hard *h) {
    if(!h)return;
    for(int i=0;i<h->c.blocks+1;i++){free(h->l[i].ch);free(h->l[i].off);free(h->l[i].f);}
    free(h->l);free(h->codes);vocab_free(&h->vocab);free(h);
}
static Hard *harden(Net *n) {
    Hard *h=alloc(1,sizeof *h);h->c=n->c;int Q=(n->c.bits+63)/64;
    h->codes=alloc(mul(n->c.vocab,Q),8);h->l=alloc(n->c.blocks+1,sizeof(HLayer));
    for(int v=1;v<n->c.vocab;v++)for(int k=0;k<n->c.bits;k++)
        if(sigmoid(n->emb.z[(size_t)v*n->c.bits+k]/fmaxf(.05f,n->temperature))>.5f)h->codes[(size_t)v*Q+k/64]|=UINT64_C(1)<<(k%64);
    for(int i=0;i<n->c.blocks+1;i++) {
        Layer *s=n->l+(i==n->c.blocks?i+1:i);HLayer *l=h->l+i;
        l->C=s->C;l->O=s->O;l->d=s->d;l->dilation=s->dilation;l->kernel=s->kernel;l->R=s->R;l->N=s->N;
        l->ch=alloc(mul(l->O,l->R),4);l->off=alloc(mul(l->O,l->R),4);l->f=alloc(mul(l->O,l->N),1);
        memcpy(l->ch,s->ch,(size_t)l->O*l->R*4);memcpy(l->off,s->off,(size_t)l->O*l->R*4);
        for(int j=0;j<l->O*l->N;j++)for(int k=0;k<4;k++)
            if(corner(s->p.z[4*j+k],n->c.est)>.5f)l->f[j]|=(uint8_t)(1u<<k);
    }
    h->vocab.n=n->vocab.n;h->vocab.words=alloc(h->vocab.n,sizeof(char*));
    for(int i=0;i<h->vocab.n;i++)h->vocab.words[i]=copystr(n->vocab.words[i]);
    vocab_index(&h->vocab);return h;
}
static void save_hard(Hard *h,const char *path) {
    char *tmp=temp_path(path);FILE *f=open_file(tmp,"wb");write_bytes(f,(h->c.gevery||h->c.match)?"LTHRD002":"LTHRD001",8);
    Config c=h->c;int a[]={c.vocab,c.bits,c.width,c.blocks,c.depth,c.kernel,c.cycle,c.votes,c.est,c.q8};
    for(int i=0;i<10;i++)put32(f,a[i]);
    if(c.gevery||c.match){put32(f,(uint32_t)c.gevery);put32(f,(uint32_t)c.gch);put32(f,(uint32_t)c.gmean);put32(f,(uint32_t)c.match);}
    vocab_write(f,&h->vocab);
    for(size_t i=0;i<(size_t)c.vocab*((c.bits+63)/64);i++)put64(f,h->codes[i]);
    for(int i=0;i<c.blocks+1;i++) {
        HLayer *l=h->l+i;put32(f,l->C);put32(f,l->O);put32(f,l->d);put32(f,l->dilation);put32(f,l->kernel);
        for(int j=0;j<l->O*l->R;j++)put32(f,(uint32_t)l->ch[j]);
        for(int j=0;j<l->O*l->R;j++)put32(f,(uint32_t)l->off[j]);
        write_bytes(f,l->f,(size_t)l->O*l->N);
    }finish_save(f,tmp,path);
}
static Hard *load_hard(const char *path) {
    FILE *f=open_file(path,"rb");char magic[8];read_bytes(f,magic,8);
    if(!memcmp(magic,"LTCKP001",8) || !memcmp(magic,"LTCKP002",8)) {fclose(f);Net *n=load_net(path);Hard *h=harden(n);net_free(n);return h;}
    int hv2=!memcmp(magic,"LTHRD002",8);
    if(memcmp(magic,"LTHRD001",8) && !hv2)die("not a .lth or .ltc checkpoint: %s",path);
    Hard *h=alloc(1,sizeof *h);Config c=defaults();
    int *a[]={&c.vocab,&c.bits,&c.width,&c.blocks,&c.depth,&c.kernel,&c.cycle,&c.votes,&c.est,&c.q8};
    for(int i=0;i<10;i++){uint32_t v=get32(f);if(v>INT_MAX)die("invalid hard config");*a[i]=(int)v;}
    if(hv2){c.gevery=(int)get32(f);c.gch=(int)get32(f);c.gmean=(int)get32(f);c.match=(int)get32(f);}validate(c);h->c=c;
    h->vocab=vocab_read(f,c.vocab);int Q=(c.bits+63)/64;h->codes=alloc(mul(c.vocab,Q),8);
    for(size_t i=0;i<(size_t)c.vocab*Q;i++)h->codes[i]=get64(f);
    for(int k=0;k<Q;k++)if(h->codes[k])die("PAD code must be all-zero");
    if(c.bits%64)for(int v=0;v<c.vocab;v++)if(h->codes[(size_t)v*Q+Q-1]>>(c.bits%64))die("codebook tail bits must be zero");
    h->l=alloc(c.blocks+1,sizeof(HLayer));
    for(int i=0;i<c.blocks+1;i++) {
        HLayer *l=h->l+i;l->C=(int)get32(f);l->O=(int)get32(f);l->d=(int)get32(f);l->dilation=(int)get32(f);l->kernel=(int)get32(f);
        int C=i==c.blocks?c.width:in_ch(&c,i),O=i==c.blocks?2*c.votes:c.width,dilation=i==c.blocks?1:1<<(i%c.cycle),kernel=i==c.blocks?1:c.kernel;
        if(l->C!=C||l->O!=O||l->d!=c.depth||l->dilation!=dilation||l->kernel!=kernel)die("invalid hard layer");
        l->R=1<<l->d;l->N=l->R-1;l->ch=alloc(mul(l->O,l->R),4);l->off=alloc(mul(l->O,l->R),4);l->f=alloc(mul(l->O,l->N),1);
        for(int j=0;j<l->O*l->R;j++){uint32_t v=get32(f);if(v>=(uint32_t)C)die("invalid hard channel");l->ch[j]=(int32_t)v;}
        for(int j=0;j<l->O*l->R;j++){int32_t v=(int32_t)get32(f);if(v<-(kernel/2)||v>kernel/2)die("invalid hard offset");l->off[j]=v;}
        read_bytes(f,l->f,(size_t)l->O*l->N);
        for(int j=0;j<l->O*l->N;j++)if(l->f[j]>15)die("invalid truth table");
    }
    if(fgetc(f)!=EOF)die("trailing hard checkpoint bytes");
    fclose(f);return h;
}
typedef struct {int B,T,P,maxC;uint64_t *a,*b,*mask,*pmask;int64_t *scores;uint64_t *g;} HWork;
static HWork *hwork_new(Hard *h,int B,int T) {
    Config c=h->c;c.batch=B;c.seq=T;validate(c);
    HWork *w=alloc(1,sizeof *w);w->B=B;w->T=T;w->P=(B+63)/64;
    w->maxC=c.width>c.bits?c.width:c.bits;if(w->maxC<2*c.votes)w->maxC=2*c.votes;
    if(c.gevery && w->maxC<c.width+c.gch)w->maxC=c.width+c.gch;
    size_t n=mul(mul(w->P,T),w->maxC);w->a=alloc(n,8);w->b=alloc(n,8);
    w->mask=alloc(mul(w->P,T),8);w->pmask=alloc(mul(w->P,(T+1)/2),8);w->scores=alloc(mul(B,2),8);return w;
}
static void hwork_free(HWork *w) {free(w->g);free(w->a);free(w->b);free(w->mask);free(w->pmask);free(w->scores);free(w);}
static void hard_layer(const HLayer *l,const uint64_t *x,uint64_t *y,const uint64_t *mask,int P,int T) {
    #pragma omp parallel for collapse(2) schedule(static)
    for(int p=0;p<P;p++)for(int o=0;o<l->O;o++)for(int t=0;t<T;t++) {
        uint64_t v[2*MAX_LEAVES-1];
        for(int k=0;k<l->R;k++) {
            int s=t+l->off[o*l->R+k]*l->dilation;
            v[l->N+k]=(s<0||s>=T)?0:x[((size_t)p*T+s)*l->C+l->ch[o*l->R+k]];
        }
        for(int j=l->N-1;j>=0;j--)v[j]=gate64(v[2*j+1],v[2*j+2],l->f[o*l->N+j]);
        y[((size_t)p*T+t)*l->O+o]=v[0]&mask[(size_t)p*T+t];
    }
}
static void hard_predict(Hard *h,HWork *w,const uint32_t *ids) {
    int C=h->c.bits,T=w->T,Q=(C+63)/64,W=h->c.width,U=(T+1)/2;
    uint64_t *cur=w->a,*next=w->b;
    for(size_t i=0;i<(size_t)w->B*T;i++)if(ids[i]>=(uint32_t)h->c.vocab)die("inference token outside vocabulary");
    uint8_t *mb=NULL;if(h->c.match){mb=alloc(mul(w->B,T),1);match_bits(ids,w->B,T,mb);}
    #pragma omp parallel for collapse(2) schedule(static)
    for(int p=0;p<w->P;p++)for(int t=0;t<T;t++) {
        uint64_t *row=cur+((size_t)p*T+t)*(C+h->c.match);memset(row,0,(C+h->c.match)*8);uint64_t mask=0;
        for(int k=0;k<64 && p*64+k<w->B;k++) {
            uint32_t id=ids[(size_t)(p*64+k)*T+t];uint64_t bit=UINT64_C(1)<<k;
            if(id)mask|=bit;
            if(mb && mb[(size_t)(p*64+k)*T+t])row[C]|=bit;
            for(int q=0;q<Q;q++) {
                uint64_t code=h->codes[(size_t)id*Q+q];
                while(code){int b=lowbit(code);int c=q*64+b;if(c<C)row[c]|=bit;code&=code-1;}
            }
        }w->mask[(size_t)p*T+t]=mask;
    }
    for(int i=0;i<h->c.blocks;i++) {
        if(i==0)free(mb);
        const uint64_t *in=cur;
        if(gblock(&h->c,i)) {  /* augmented input: W channels + G OR-pooled, broadcast to valid positions */
            int W=h->c.width,G=h->c.gch,CA=W+G;
            if(!w->g)w->g=alloc(mul(mul(w->P,T),CA),8);
            #pragma omp parallel for schedule(static)
            for(int p=0;p<w->P;p++) {
                for(int t=0;t<T;t++)memcpy(w->g+((size_t)p*T+t)*CA,cur+((size_t)p*T+t)*W,(size_t)W*8);
                for(int c=0;c<G;c++) {
                    uint64_t any=0;
                    if(h->c.gmean) {  /* majority: 2 x count > valid positions, per text lane */
                        for(int k=0;k<64;k++){int cnt=0,nv=0;for(int t=0;t<T;t++){cnt+=(int)((cur[((size_t)p*T+t)*W+c]>>k)&1);nv+=(int)((w->mask[(size_t)p*T+t]>>k)&1);}
                            if(2*cnt>nv)any|=UINT64_C(1)<<k;}
                    } else for(int t=0;t<T;t++)any|=cur[((size_t)p*T+t)*W+c];
                    for(int t=0;t<T;t++)w->g[((size_t)p*T+t)*CA+W+c]=any&w->mask[(size_t)p*T+t];
                }
            }
            in=w->g;
        }
        hard_layer(h->l+i,in,next,w->mask,w->P,T);uint64_t *tmp=cur;cur=next;next=tmp;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for(int p=0;p<w->P;p++)for(int u=0;u<U;u++) {
        int t=2*u;w->pmask[(size_t)p*U+u]=w->mask[(size_t)p*T+t] | (t+1<T?w->mask[(size_t)p*T+t+1]:0);
        for(int c=0;c<W;c++)next[((size_t)p*U+u)*W+c]=cur[((size_t)p*T+t)*W+c] | (t+1<T?cur[((size_t)p*T+t+1)*W+c]:0);
    }
    {uint64_t *tmp=cur;cur=next;next=tmp;}
    hard_layer(h->l+h->c.blocks,cur,next,w->pmask,w->P,U);
    int votes=h->c.votes,O=2*votes,bits=0;uint64_t maxcount=(uint64_t)U*votes;
    do{bits++;maxcount>>=1;}while(maxcount);
    #pragma omp parallel for collapse(2) schedule(static)
    for(int p=0;p<w->P;p++)for(int cls=0;cls<2;cls++) {
        uint64_t cnt[32]={0};
        for(int t=0;t<U;t++)for(int v=0;v<votes;v++) {
            uint64_t carry=next[((size_t)p*U+t)*O+cls*votes+v];
            for(int j=0;j<bits && carry;j++){uint64_t tmp=cnt[j]&carry;cnt[j]^=carry;carry=tmp;}
        }
        for(int k=0;k<64 && p*64+k<w->B;k++) {
            int64_t s=0;for(int j=0;j<bits;j++)s|=(int64_t)((cnt[j]>>k)&1)<<j;
            w->scores[(size_t)(p*64+k)*2+cls]=s;
        }
    }
}

/* ------------------------------ data / text ------------------------------ */
static char *line_read(FILE *f) {
    size_t cap=256,n=0;char *s=alloc(cap,1);int c;
    while((c=fgetc(f))!=EOF && c!='\n') {
        if(c==0)die("NUL in text input");
        if(n>=MAX_RECORD)die("input line exceeds 64 MiB");
        if(n+1>=cap){cap*=2;s=resize(s,cap,1);}s[n++]=(char)c;
    }
    if(c==EOF && !n){free(s);return NULL;}if(n && s[n-1]=='\r')n--;s[n]=0;return s;
}
static char *trim(char *s) {
    while(*s==' '||*s=='\t'||*s=='\r'||*s=='\n')s++;
    size_t n=strlen(s);while(n && (s[n-1]==' '||s[n-1]=='\t'||s[n-1]=='\r'||s[n-1]=='\n'))s[--n]=0;return s;
}
static int parse_label(const char *s) {
    if(!strcmp(s,"0")||!strcmp(s,"negative"))return 0;
    if(!strcmp(s,"1")||!strcmp(s,"positive"))return 1;
    if(!strcmp(s,"-1")||!*s)return -1;
    die("invalid label '%s'",s);return -1;
}
static Vocab vocab_text(const char *path,int V) {
    Vocab v={0};v.words=alloc(V,sizeof(char*));FILE *f=open_file(path,"rb");char *s;
    while((s=line_read(f))) {
        if(v.n==V)die("vocabulary exceeds --vocab-size");
        if(!*s)die("blank vocabulary entry");
        v.words[v.n++]=s;
    }fclose(f);
    if(v.n<3 || strcmp(v.words[0],"[PAD]") || strcmp(v.words[1],"[MASK]") || strcmp(v.words[2],"[UNK]"))
        die("vocabulary must start with [PAD], [MASK], [UNK]");
    vocab_index(&v);return v;
}
static int ascii_letter(int c) {return c>='a'&&c<='z';}
static int ascii_word(int c) {return ascii_letter(c)||(c>='0'&&c<='9');}
static char *normalize_text(const char *text) {
    size_t n=strlen(text);char *s=alloc(n+1,1);size_t w=0;
    for(size_t i=0;i<n;) {
        unsigned char c=(unsigned char)text[i++];
        if(c=='<') {const char *end=strchr(text+i,'>');if(end){i=(size_t)(end-text)+1;s[w++]=' ';continue;}}
        if(c=='&') {
            const char *end=strchr(text+i,';');
            if(end && end-(text+i)<16) {
                size_t len=(size_t)(end-(text+i));char buf[20];memcpy(buf,text+i,len);buf[len]=0;int decoded=-1;
                if(!strcmp(buf,"amp"))decoded='&';else if(!strcmp(buf,"lt"))decoded='<';else if(!strcmp(buf,"gt"))decoded='>';
                else if(!strcmp(buf,"quot"))decoded='"';else if(!strcmp(buf,"apos")||!strcmp(buf,"#39"))decoded='\'';
                else if(!strcmp(buf,"nbsp"))decoded=' ';
                else if(buf[0]=='#') {char *e;int hex=buf[1]=='x'||buf[1]=='X';long val=strtol(buf+1+hex,&e,hex?16:10);if(!*e && val>0 && val<128)decoded=(int)val;}
                if(decoded>=0){c=(unsigned char)decoded;i=(size_t)(end-text)+1;}
            }
        }
        if(c>='A'&&c<='Z')c=(unsigned char)(c+'a'-'A');
        s[w++]=(char)c;
    }s[w]=0;return s;
}
typedef void (*WordFn)(const char *,void *);
static void tokenize(const char *text,int limit,WordFn fn,void *ctx) {
    char *s=normalize_text(text);size_t i=0;int n=0;
    while(s[i] && n<limit) {
        size_t start=i;
        if(ascii_word((unsigned char)s[i])) {
            while(ascii_word((unsigned char)s[i]))i++;
            if(s[i]=='\'' && ascii_letter((unsigned char)s[i+1])) {i++;while(ascii_letter((unsigned char)s[i]))i++;}
        } else if(strchr("!?.,;:",s[i]))i++;
        else{i++;continue;}
        char end=s[i];s[i]=0;fn(s+start,ctx);s[i]=end;n++;
    }free(s);
}
static char **csv_row(FILE *f,int *ncol) {
    char **cols=alloc(64,sizeof(char*));size_t cap=256,n=0,total=0;char *s=alloc(cap,1);
    int quoted=0,at_start=1,after_quote=0,c,any=0;*ncol=0;
    for(;;) {
        c=fgetc(f);
        if(c==EOF && !any){free(s);free(cols);return NULL;}any=1;
        if(c==0)die("NUL in CSV");
        if(quoted) {
            if(c==EOF)die("unclosed CSV quote");
            if(c=='"') {int d=fgetc(f);if(d=='"')c='"';else{quoted=0;after_quote=1;if(d!=EOF)ungetc(d,f);continue;}}
        } else {
            if(at_start && c=='"'){quoted=1;at_start=0;continue;}
            if(c==','||c=='\n'||c=='\r'||c==EOF) {
                if(*ncol==64)die("CSV has more than 64 columns");
                s[n]=0;cols[(*ncol)++]=s;
                if(c!=',') {if(c=='\r'){int d=fgetc(f);if(d!=EOF && d!='\n')ungetc(d,f);}return cols;}
                cap=256;n=0;s=alloc(cap,1);at_start=1;after_quote=0;continue;
            }
            if(after_quote){if(c==' '||c=='\t')continue;die("characters after CSV closing quote");}
        }
        if(++total>MAX_RECORD)die("CSV record exceeds 64 MiB");
        if(n+1>=cap){cap*=2;s=resize(s,cap,1);}s[n++]=(char)c;at_start=0;
    }
}
static void csv_free(char **a,int n){for(int i=0;i<n;i++)free(a[i]);free(a);}
typedef void (*RecordFn)(int,const char *,void *);
static void text_records(const char *path,const char *fmt,RecordFn fn,void *ctx) {
    FILE *f=open_file(path,"rb");
    unsigned char bom[3];size_t got=fread(bom,1,3,f);if(got!=3||memcmp(bom,"\xef\xbb\xbf",3))rewind(f);
    if(!strcmp(fmt,"csv")) {
        int nc;char **header=csv_row(f,&nc);if(!header)die("empty CSV");int ti=-1,li=-1;
        for(int i=0;i<nc;i++){char *s=trim(header[i]);if(!strcmp(s,"review")||!strcmp(s,"text"))ti=i;if(!strcmp(s,"sentiment")||!strcmp(s,"label"))li=i;}
        if(ti<0)die("CSV needs a review or text header");
        csv_free(header,nc);
        char **row;
        while((row=csv_row(f,&nc))) {
            if(nc==1 && !*row[0]){csv_free(row,nc);continue;}
            if(ti>=nc||li>=nc)die("CSV row has missing columns");
            int label=li<0?-1:parse_label(trim(row[li]));fn(label,row[ti],ctx);csv_free(row,nc);
        }
    } else if(!strcmp(fmt,"tsv")||!strcmp(fmt,"text")) {
        char *s;while((s=line_read(f))) {
            if(!*s){free(s);continue;}int label=-1;char *text=s;
            if(!strcmp(fmt,"tsv")){char *tab=strchr(s,'\t');if(!tab)die("TSV expects label<TAB>text");*tab=0;label=parse_label(trim(s));text=tab+1;}
            fn(label,text,ctx);free(s);
        }
    }else die("unsupported text format: %s",fmt);
    if(ferror(f))die("read failed: %s",path);
    fclose(f);
}
typedef struct {int n,cap,T,V;uint32_t *x;int *y;const Vocab *vocab;} Data;
static void data_reserve(Data *d) {
    if(d->n<d->cap)return;
    int old=d->cap;
    if(old>INT_MAX/2)die("too many records");
    d->cap=old?old*2:128;
    d->x=resize(d->x,mul(d->cap,d->T),4);d->y=resize(d->y,d->cap,sizeof(int));
}
typedef struct {const Vocab *v;uint32_t *out;int n;} EncodeCtx;
static void encode_word(const char *s,void *ctx) {EncodeCtx *a=ctx;a->out[a->n++]=(uint32_t)vocab_lookup(a->v,s);}
static void data_text_record(int label,const char *s,void *ctx) {
    Data *d=ctx;data_reserve(d);uint32_t *x=d->x+(size_t)d->n*d->T;memset(x,0,d->T*4);
    EncodeCtx a={d->vocab,x,0};tokenize(s,d->T,encode_word,&a);d->y[d->n++]=label;
}
static long parse_long(const char *s,long lo,long hi) {
    char *e;errno=0;long v=strtol(s,&e,10);if(errno||!*s||*e||v<lo||v>hi)die("invalid integer: %s",s);return v;
}
static float parse_float(const char *s) {
    char *e;errno=0;float v=strtof(s,&e);if(errno||!*s||*e||!isfinite(v))die("invalid float: %s",s);return v;
}
static Data data_load(const char *path,const char *fmt,const Vocab *v,int V,int T) {
    Data d={0};d.T=T;d.V=V;d.vocab=v;
    if(!strcmp(fmt,"ids")) {
        FILE *f=open_file(path,"rb");char *s;
        while((s=line_read(f))) {
            char *p=trim(s);if(!*p){free(s);continue;}
            data_reserve(&d);uint32_t *x=d.x+(size_t)d.n*T;memset(x,0,T*4);
            char *e=p;errno=0;long lab=strtol(p,&e,10);if(errno||e==p||lab< -1||lab>1||(*e && *e!=' ' && *e!='\t'))die("IDS line needs label -1, 0 or 1");
            p=e;int k=0;
            while(*p) {
                while(*p==' '||*p=='\t')p++;
                if(!*p)break;
                if(*p<'0'||*p>'9')die("IDS tokens must be unsigned integers");
                errno=0;unsigned long id=strtoul(p,&e,10);
                if(errno||id>=(unsigned long)V||(*e && *e!=' ' && *e!='\t'))die("invalid token ID");
                if(k<T)x[k++]=(uint32_t)id;
                p=e;
            }
            d.y[d.n++]=(int)lab;free(s);
        }if(ferror(f))die("read failed: %s",path);fclose(f);
    } else {
        if(!v || !v->n)die("text input requires --vocab or vocabulary embedded in checkpoint");
        text_records(path,fmt,data_text_record,&d);
    }
    if(!d.n)die("dataset is empty: %s",path);
    return d;
}
static void data_free(Data *d){free(d->x);free(d->y);memset(d,0,sizeof *d);}
static void require_labels(Data *d){for(int i=0;i<d->n;i++)if(d->y[i]<0)die("supervised dataset contains an unlabeled record");}
typedef struct {Vocab v;uint64_t *count;int capacity,limit;} BuildVocab;
static void count_word(const char *s,void *ctx) {
    BuildVocab *b=ctx;int id=vocab_lookup(&b->v,s);
    if(id!=UNK){b->count[id]++;return;}
    if(b->v.n==b->capacity) {
        b->capacity*=2;b->v.words=resize(b->v.words,b->capacity,sizeof(char*));b->count=resize(b->count,b->capacity,8);
    }
    id=b->v.n++;b->v.words[id]=copystr(s);b->count[id]=1;
    if((size_t)b->v.n*2>b->v.cap){vocab_index(&b->v);return;}
    size_t i=hashword(s)&(b->v.cap-1);while(b->v.slot[i])i=(i+1)&(b->v.cap-1);b->v.slot[i]=id+1;
}
static void count_record(int label,const char *s,void *ctx){(void)label;BuildVocab *b=ctx;tokenize(s,b->limit,count_word,b);}
typedef struct {char *s;uint64_t n;} WordCount;
static int wordcmp(const void *a,const void *b) {
    const WordCount *x=a,*y=b;if(x->n!=y->n)return x->n>y->n?-1:1;return strcmp(x->s,y->s);
}
static void make_vocab(const char *data,const char *fmt,const char *out,int V,int T) {
    BuildVocab b={0};b.capacity=128;b.limit=T;b.v.n=3;b.v.words=alloc(b.capacity,sizeof(char*));b.count=alloc(b.capacity,8);
    b.v.words[0]=copystr("[PAD]");b.v.words[1]=copystr("[MASK]");b.v.words[2]=copystr("[UNK]");vocab_index(&b.v);
    text_records(data,fmt,count_record,&b);int words=b.v.n-3;
    WordCount *a=alloc(words,sizeof *a);for(int i=0;i<words;i++)a[i]=(WordCount){b.v.words[i+3],b.count[i+3]};qsort(a,words,sizeof *a,wordcmp);
    FILE *f=open_file(out,"wb");fputs("[PAD]\n[MASK]\n[UNK]\n",f);int kept=words<V-3?words:V-3;
    for(int i=0;i<kept;i++)fprintf(f,"%s\n",a[i].s);
    for(int i=kept+3;i<V;i++)fprintf(f,"[UNUSED_%d]\n",i);
    if(ferror(f)||fclose(f))die("vocabulary write failed");
    fprintf(stderr,"vocabulary: %d observed, %d retained, capacity %d -> %s\n",words,kept,V,out);
    free(a);free(b.count);vocab_free(&b.v);
}

/* --------------------------- training / evaluation ----------------------- */
static double eval_hard(Hard *h,Data *d,int B,int predictions) {
    HWork *w=hwork_new(h,B,d->T);uint32_t *tail=alloc(mul(B,d->T),4);int hit=0,labeled=0;
    for(int i=0;i<d->n;i+=B) {
        int take=d->n-i<B?d->n-i:B;const uint32_t *x=d->x+(size_t)i*d->T;
        if(take<B){memset(tail,0,(size_t)B*d->T*4);memcpy(tail,x,(size_t)take*d->T*4);x=tail;}
        hard_predict(h,w,x);
        for(int j=0;j<take;j++) {
            int pred=w->scores[2*j+1]>w->scores[2*j];
            if(d->y[i+j]>=0){labeled++;hit+=pred==d->y[i+j];}
            if(predictions)printf("{\"row\":%d,\"label\":%d,\"votes\":[%lld,%lld]}\n",i+j,pred,(long long)w->scores[2*j],(long long)w->scores[2*j+1]);
        }
    }
    hwork_free(w);free(tail);
    if(predictions)fprintf(stderr,"labeled=%d correct=%d accuracy=%.6f\n",labeled,hit,labeled?(double)hit/labeled:0.);
    return labeled?(double)hit/labeled:-1.;
}
static double eval_mlm(Net *n,Data *d) {
    RNG saved=n->rng;n->rng.s=UINT64_C(730921);double total=0;long nt=0;int B=n->c.batch;
    Work *w=work_new(n,B,d->T);
    for(int i=0;i<d->n;i+=B) {
        int take=d->n-i<B?d->n-i:B;Work *v=take==B?w:work_new(n,take,d->T);
        memcpy(v->ids,d->x+(size_t)i*d->T,(size_t)take*d->T*4);int nm=corrupt(n,v);
        total+=masked_loss(n,v,0)*nm;nt+=nm;if(v!=w)work_free(v,n->c.blocks);
    }work_free(w,n->c.blocks);n->rng=saved;return total/nt;
}
static size_t parameter_count(Net *n) {
    size_t p=n->emb.n+n->bias.n;for(int i=0;i<n->c.blocks+2;i++)p+=n->l[i].p.n;return p;
}
static char *suffix(const char *s,const char *tail) {char *out=alloc(strlen(s)+strlen(tail)+1,1);strcpy(out,s);strcat(out,tail);return out;}
static void export_net(Net *n,const char *path){Hard *h=harden(n);save_hard(h,path);hard_free(h);}
static float g_fixed_temp=0.f;  /* > 0: --keep-temperature */
static void train_net(Net *n,Data *d,Data *val,const char *save,const char *export_path,int stop,int every) {
    if(!n->c.stage)require_labels(d);
    if(val && !n->c.stage)require_labels(val);
    Work *w=work_new(n,n->c.batch,n->c.seq);double start=now(),compute=0;float lastce=0;
    uint64_t end=stop>0?(uint64_t)stop:(uint64_t)n->c.steps;if(end>(uint64_t)n->c.steps)end=n->c.steps;
    if(end<=n->step)die("stop point must be later than checkpoint step (%llu)",(unsigned long long)n->step);
    fprintf(stderr,"%s: records=%d params=%zu bits=%d width=%d blocks=%d batch=%d seq=%d threads=%d\n",
            n->c.stage?"MLM":"supervised",d->n,parameter_count(n),n->c.bits,n->c.width,n->c.blocks,w->B,w->T,thread_count());
    for(;n->step<end;) {
        double t=now();float progress=n->c.steps>1?(float)n->step/(n->c.steps-1):0;
        float lr=n->c.peak_lr*(.1f+.9f*.5f*(1.f+cosf((float)PI*progress)));
        n->temperature=g_fixed_temp>0.f?g_fixed_temp:1.f-.8f*fminf(1.f,progress/.8f);
        for(int b=0;b<w->B;b++) {
            int i=randint(&n->rng,d->n);memcpy(w->ids+(size_t)b*w->T,d->x+(size_t)i*d->T,w->T*4);w->labels[b]=d->y[i];
        }
        int targets=n->c.stage?corrupt(n,w):0;
        float loss=objective(n,w,n->c.sharp*progress,1,&lastce);
        if(!isfinite(loss))die("nonfinite training loss");
        adam(n,lr);n->step++;compute+=now()-t;
        if(n->step%(uint64_t)every==0 || n->step==end) {
            double metric=-1;int best=0;
            if(val) {
                if(n->c.stage)metric=-eval_mlm(n,val);
                else{Hard *h=harden(n);metric=eval_hard(h,val,n->c.batch,0);hard_free(h);}
                if((n->c.stage && (n->best_val==-1 || metric>n->best_val)) || (!n->c.stage && metric>n->best_val)) {
                    n->best_val=(float)metric;best=1;
                }
            }
            save_net(n,save);
            if(best) {
                char *p=suffix(save,".best");save_net(n,p);free(p);
                if(export_path){p=suffix(export_path,".best");export_net(n,p);free(p);}
            }
            fprintf(stderr,"{\"step\":%llu,\"ce\":%.7g,\"loss\":%.7g,\"lr\":%.7g,\"code_temperature\":%.5g,\"masked_targets\":%d,\"compute_seconds\":%.4f,\"wall_seconds\":%.4f",
                    (unsigned long long)n->step,lastce,loss,lr,n->temperature,targets,compute,now()-start);
            if(val)fprintf(stderr,n->c.stage?",\"val_mlm_ce\":%.7g":",\"val_hard_accuracy\":%.7g",n->c.stage?-metric:metric);
            fputs("}\n",stderr);fflush(stderr);
        }
    }
    if(export_path)export_net(n,export_path);
    work_free(w,n->c.blocks);
}
static int doublecmp(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static void bench(Hard *h,Data *d,int B,int repeats,int warmup) {
    HWork *w=hwork_new(h,B,d->T);uint32_t *x=alloc(mul(B,d->T),4);
    double *times=alloc(repeats,sizeof(double));volatile uint64_t checksum=0;long occupied=0;
    for(int b=0;b<B;b++)for(int t=0;t<d->T;t++) {uint32_t id=d->x[(size_t)(b%d->n)*d->T+t];x[(size_t)b*d->T+t]=id;occupied+=id!=0;}
    for(int r=-warmup;r<repeats;r++) {
        double t=now();hard_predict(h,w,x);
        for(int b=0;b<B;b++)checksum+=(uint64_t)w->scores[2*b]+(uint64_t)w->scores[2*b+1]+(w->scores[2*b+1]>w->scores[2*b]);
        double elapsed=now()-t;if(r>=0)times[r]=elapsed;
    }
    qsort(times,repeats,sizeof(double),doublecmp);double med=times[repeats/2];if(!(repeats%2))med=(med+times[repeats/2-1])*.5;
    printf("{\"batch\":%d,\"sequence_length\":%d,\"threads\":%d,\"median_batch_ms\":%.6f,\"reviews_per_second\":%.3f,\"positions_per_second\":%.3f,\"mean_nonpadding\":%.3f,\"source_records\":%d,\"repeats\":%d,\"checksum\":%llu,\"includes\":\"lookup,packing,gates,OR,integer_counts,argmax\",\"excludes\":\"tokenization,IO,load,workspace_allocation\"}\n",
           B,d->T,thread_count(),med*1000,B/med,(double)B*d->T/med,(double)occupied/B,d->n,repeats,(unsigned long long)checksum);
    free(times);free(x);hwork_free(w);
}

/* ---------------------- independent executable checks -------------------- */
static void scalar_hard(Hard *h,const uint32_t *ids,int B,int T,int64_t *scores) {
    int maxC=h->c.bits;if(maxC<h->c.width)maxC=h->c.width;if(maxC<2*h->c.votes)maxC=2*h->c.votes;
    uint8_t *a=alloc(mul(T,maxC),1),*b=alloc(mul(T,maxC),1);int Q=(h->c.bits+63)/64;
    for(int sample=0;sample<B;sample++) {
        int length=T,C=h->c.bits;uint8_t *cur=a,*nxt=b;
        for(int t=0;t<T;t++)for(int c=0;c<C;c++)cur[t*C+c]=(uint8_t)((h->codes[(size_t)ids[(size_t)sample*T+t]*Q+c/64]>>(c%64))&1);
        for(int i=0;i<=h->c.blocks;i++) {
            if(i==h->c.blocks) {
                int U=(T+1)/2;for(int u=0;u<U;u++)for(int c=0;c<C;c++)nxt[u*C+c]=cur[2*u*C+c] | (2*u+1<T?cur[(2*u+1)*C+c]:0);
                uint8_t *z=cur;cur=nxt;nxt=z;length=U;
            }
            HLayer *l=h->l+i;
            for(int t=0;t<length;t++)for(int o=0;o<l->O;o++) {
                uint8_t v[2*MAX_LEAVES-1];
                for(int k=0;k<l->R;k++){int s=t+l->off[o*l->R+k]*l->dilation;v[l->N+k]=(s<0||s>=length)?0:cur[s*C+l->ch[o*l->R+k]];}
                for(int j=l->N-1;j>=0;j--)v[j]=(uint8_t)((l->f[o*l->N+j]>>(2*v[2*j+1]+v[2*j+2]))&1);
                int valid=i==h->c.blocks?(ids[(size_t)sample*T+2*t]!=0 || (2*t+1<T && ids[(size_t)sample*T+2*t+1]!=0)):ids[(size_t)sample*T+t]!=0;
                nxt[t*l->O+o]=v[0] & valid;
            }
            uint8_t *z=cur;cur=nxt;nxt=z;C=l->O;
        }
        scores[2*sample]=scores[2*sample+1]=0;
        for(int t=0;t<length;t++)for(int o=0;o<C;o++)scores[2*sample+o/h->c.votes]+=cur[t*C+o];
    }free(a);free(b);
}
static void selftest(void) {
    for(int f=0;f<16;f++)for(int a=0;a<2;a++)for(int b=0;b<2;b++) {
        float p[4];for(int k=0;k<4;k++)p[k]=(float)((f>>k)&1);
        int expect=(f>>(2*a+b))&1;
        if(gate((float)a,(float)b,p)!=expect || (gate64(-(uint64_t)a,-(uint64_t)b,(uint8_t)f)&1)!=(uint64_t)expect)die("truth-table selftest failed");
    }
    Config c=defaults();c.vocab=16;c.bits=8;c.width=12;c.blocks=2;c.depth=2;c.cycle=2;c.votes=8;c.batch=4;c.seq=7;c.steps=80;
    Net *n=net_new(c,1,1,42);Work *w=work_new(n,4,7);RNG r={139};
    for(int i=0;i<28;i++){w->ids[i]=(uint32_t)(3+randint(&r,13));if(i%7>4)w->ids[i]=0;w->targets[i]=w->ids[i];w->selected[i]=i%7==2;}
    for(int i=0;i<4;i++)w->labels[i]=i%2;
    for(size_t i=c.bits;i<n->emb.n;i++)n->emb.z[i]+=.37f*normal(&r);
    for(int est=0;est<2;est++) {
        n->c.est=est;
        if(est)for(int l=0;l<n->c.blocks+2;l++)for(size_t k=0;k<n->l[l].p.n;k++)n->l[l].p.z[k]*=.4f;
        for(int stage=0;stage<2;stage++) {
            n->c.stage=stage;float ce;objective(n,w,.002f,1,&ce);Param *pp[MAX_BLOCKS+3];int np=active_params(n,pp);
            float worst=0,max_abs=0;
            for(int i=0;i<48;i++) {
                Param *p=pp[i%np];size_t j=(size_t)randint(&r,(int)p->n);float old=p->z[j],an=p->g[j],eps=.005f;
                p->z[j]=old+eps;float plus=objective(n,w,.002f,0,&ce);
                p->z[j]=old-eps;float minus=objective(n,w,.002f,0,&ce);p->z[j]=old;
                float num=(plus-minus)/(2*eps),abs=fabsf(an-num),rel=abs/fmaxf(.0005f,fabsf(an)+fabsf(num));
                if(rel>worst)worst=rel;
                if(abs>max_abs)max_abs=abs;
            }
            if(worst>.035f || max_abs>.0002f)die("gradient selftest failed: est=%d stage=%d rel=%g abs=%g",est,stage,worst,max_abs);
            printf("{\"check\":\"finite_difference\",\"estimator\":%d,\"stage\":%d,\"max_relative\":%.8g,\"max_absolute\":%.8g}\n",est,stage,worst,max_abs);
        }
    }
    n->c.stage=0;Hard *h=harden(n);int batches[]={1,4,65,128},lengths[]={1,7,32};
    for(int i=0;i<4;i++)for(int j=0;j<3;j++) {
        int B=batches[i],T=lengths[j];uint32_t *ids=alloc(mul(B,T),4);int64_t *ref=alloc(mul(B,2),8);HWork *hw=hwork_new(h,B,T);
        for(int b=0;b<B;b++)for(int t=0;t<T;t++)ids[(size_t)b*T+t]=(b%5 && t%7<5)?3+randint(&r,13):0;
        hard_predict(h,hw,ids);scalar_hard(h,ids,B,T,ref);
        if(memcmp(hw->scores,ref,(size_t)B*2*8))die("packed/scalar mismatch B=%d T=%d",B,T);
        free(ids);free(ref);hwork_free(hw);
    }
    char path[160],hp[160];unsigned long long tag=(unsigned long long)ru(&r)^(unsigned long long)(now()*1e9);
    snprintf(path,sizeof path,".logic_text_selftest_%llu.ltc",tag);snprintf(hp,sizeof hp,".logic_text_selftest_%llu.lth",tag);
    save_net(n,path);save_hard(h,hp);Net *copy=load_net(path);Hard *hc=load_hard(hp);
    if(memcmp(copy->emb.z,n->emb.z,n->emb.n*4))die("training checkpoint roundtrip failed");
    HWork *u=hwork_new(h,4,7),*v=hwork_new(hc,4,7);hard_predict(h,u,w->ids);hard_predict(hc,v,w->ids);
    if(memcmp(u->scores,v->scores,8*8))die("hard checkpoint roundtrip failed");
    hwork_free(u);hwork_free(v);hard_free(h);hard_free(hc);net_free(copy);remove(path);remove(hp);
    work_free(w,n->c.blocks);net_free(n);
    puts("{\"check\":\"truth_tables,packed_scalar,odd_padding,partial_packs,checkpoint_roundtrip\",\"passed\":true}");
}

/* --------------------------------- CLI ---------------------------------- */
typedef struct {
    Config c;uint64_t changed;
    const char *data,*val,*format,*vocab,*load,*resume,*save,*export_path,*out,*text;
    int threads,stop,every,repeats,warmup;uint64_t seed;int seed_set;
    int keep_temp;const char *load_part;
} Options;
static const char *USAGE=
"fastlogic: LogicAE in one file (see the header of fastlogic.c for the model and the recipe)\n"
"\n"
"train / adapt (hard forward by default; --soft-forward to opt out)\n"
"  fastlogic train --data train.ids --val dev.ids --steps 1500 --eval-every 250 --save m.ltc --export m.lth\n"
"  fastlogic train --load pre.ltc --data ... --save ft.ltc --export ft.lth   (keeps code temperature; --reset-temperature)\n"
"     options: --seq 64 --batch 32 --lr .025 --steps N --seed 17 --threads N --stop-after STEP --resume m.ltc\n"
"              --match 1 --global-every 4 --global-channels 128 --global-mean 1   (text pairs)\n"
"              --load-part codes|gates   --vocab-size --code-bits --width --blocks --kernel --cycle --votes\n"
"pretrain (masked words, hard forward; batch 64, 512 targets per batch)\n"
"  fastlogic pretrain --data windows.ids --val val.ids --steps N --eval-every 500 --save pre.ltc\n"
"fast inference (MODEL = .lth or .ltc; --lanes 8 --threads all by default)\n"
"  fastlogic predict MODEL DATA.ids SEQ          {\"row\",\"label\",\"votes\"} per record\n"
"  fastlogic verify  MODEL DATA.ids SEQ          every score == the reference engine, all lane widths\n"
"  fastlogic bench   MODEL DATA.ids SEQ --batch B [--lanes L] [--threads N] [--repeats R]\n"
"  fastlogic gen     MODEL DATA.ids SEQ model.inc   compile for SEQ (<= 512): gcc ... -DGEN_INC='\"model.inc\"' fastlogic.c\n"
"reference engine and tools\n"
"  fastlogic eval --load MODEL --data DATA.ids --seq 64      hardened accuracy\n"
"  fastlogic export --load m.ltc --out m.lth | info --load MODEL | predict-ref / bench-ref (reference engine)\n"
"  fastlogic hardcheck m.ltc DATA.ids SEQ      training forward == deployed network, record by record\n"
"  fastlogic revive m.ltc DATA.ids SEQ out.ltc reset dead channels to the pass-through init\n"
"  fastlogic selftest | vocab --data text.tsv --out vocab.txt\n"
;
static void usage(void) {
    fputs(USAGE,stdout);
}
static Options parse_options(int argc,char **argv) {
    Options o={0};o.c=defaults();o.keep_temp=1;o.threads=thread_count();o.every=100;o.repeats=40;o.warmup=5;o.seed=17;o.format="ids";
    for(int i=2;i<argc;i++) {
        const char *key=argv[i];
        if(!strcmp(key,"--keep-temperature")){o.keep_temp=1;continue;}
        if(!strcmp(key,"--reset-temperature")){o.keep_temp=0;continue;}
        if(!strcmp(key,"--hard-forward")){g_fwd_opt=1;continue;}
        if(!strcmp(key,"--soft-forward")){g_fwd_opt=0;continue;}
        if(!strcmp(key,"--q8")){o.c.q8=1;o.changed|=UINT64_C(1)<<9;continue;}
        if(!strcmp(key,"--help")){usage();exit(0);}
        if(i+1>=argc)die("missing value for %s",key);
        const char *s=argv[++i];
#define SI(name,field,bit,lo,hi) if(!strcmp(key,name)){o.c.field=(int)parse_long(s,lo,hi);o.changed|=UINT64_C(1)<<bit;continue;}
#define SF(name,field,bit) if(!strcmp(key,name)){o.c.field=parse_float(s);o.changed|=UINT64_C(1)<<bit;continue;}
#define SP(name,field) if(!strcmp(key,name)){o.field=s;continue;}
#define SO(name,field,lo,hi) if(!strcmp(key,name)){o.field=(int)parse_long(s,lo,hi);continue;}
        SI("--vocab-size",vocab,0,4,1000000) SI("--code-bits",bits,1,2,1024) SI("--width",width,2,1,65536)
        SI("--blocks",blocks,3,1,MAX_BLOCKS) SI("--depth",depth,4,1,MAX_DEPTH) SI("--kernel",kernel,5,1,31)
        SI("--cycle",cycle,6,1,20) SI("--votes",votes,7,1,32768) SI("--seq",seq,10,1,65536)
        SI("--batch",batch,11,1,65536) SI("--steps",steps,12,1,1000000000) SI("--mlm-targets",max_targets,14,0,INT_MAX)
        SI("--global-every",gevery,21,0,MAX_BLOCKS) SI("--global-channels",gch,22,0,65536)
        SI("--global-mean",gmean,23,0,1) SI("--match",match,24,0,1)
        SF("--lr",peak_lr,15) SF("--sharp",sharp,16) SF("--clip",clip,17) SF("--mask-prob",mask_prob,18)
        SF("--scale",scale,19) SF("--mlm-scale",mlm_scale,20)
        SP("--data",data) SP("--val",val) SP("--format",format) SP("--vocab",vocab) SP("--load",load)
        SP("--resume",resume) SP("--save",save) SP("--export",export_path) SP("--out",out) SP("--text",text)
        SO("--threads",threads,1,1024) SO("--eval-every",every,1,1000000000) SO("--stop-after",stop,1,1000000000)
        SO("--repeats",repeats,1,1000000) SO("--warmup",warmup,0,1000000)
        if(!strcmp(key,"--seed")){o.seed=(uint64_t)parse_long(s,1,LONG_MAX);o.seed_set=1;continue;}
        if(!strcmp(key,"--load-part")){if(strcmp(s,"codes")&&strcmp(s,"gates"))die("--load-part must be codes or gates");o.load_part=s;continue;}
        if(!strcmp(key,"--estimator")) {
            if(strcmp(s,"sigmoid")&&strcmp(s,"sin"))die("estimator must be sigmoid or sin");
            o.c.est=!strcmp(s,"sin");o.changed|=UINT64_C(1)<<8;continue;
        }
        die("unknown option %s",key);
#undef SI
#undef SF
#undef SP
#undef SO
    }
    if(o.load && o.resume)die("use either --load or --resume, not both");
    validate(o.c);return o;
}
static void overlay(Net *n,Options *o,int resume) {
    Config *c=&n->c,*s=&o->c;
    int *a[]={&c->vocab,&c->bits,&c->width,&c->blocks,&c->depth,&c->kernel,&c->cycle,&c->votes,&c->est,&c->q8,&c->seq,&c->batch,&c->steps,&c->stage,&c->max_targets};
    int b[]={s->vocab,s->bits,s->width,s->blocks,s->depth,s->kernel,s->cycle,s->votes,s->est,s->q8,s->seq,s->batch,s->steps,s->stage,s->max_targets};
    float *af[]={&c->peak_lr,&c->sharp,&c->clip,&c->mask_prob,&c->scale,&c->mlm_scale};
    float bf[]={s->peak_lr,s->sharp,s->clip,s->mask_prob,s->scale,s->mlm_scale};
    for(int i=0;i<15;i++)if(o->changed&(UINT64_C(1)<<i)) {
        if((resume||i<10) && *a[i]!=b[i])die("checkpoint architecture/schedule mismatch for option %d",i);
        *a[i]=b[i];
    }
    for(int i=0;i<6;i++)if(o->changed&(UINT64_C(1)<<(i+15))) {
        if(resume && *af[i]!=bf[i])die("resume cannot change schedule; use --load instead");
        *af[i]=bf[i];
    }
    if(o->changed&(UINT64_C(15)<<21)) {
        if(c->gevery!=s->gevery || c->gch!=s->gch || c->gmean!=s->gmean || c->match!=s->match)
            die("--global-*/--match differ from the checkpoint");
    }
    validate(*c);
}
static int lt_main(int argc,char **argv) {
    if(sizeof(float)!=4 || FLT_RADIX!=2 || FLT_MANT_DIG!=24)die("requires IEEE-754 binary32 floats");
    if(argc<2 || !strcmp(argv[1],"--help") || !strcmp(argv[1],"help")){usage();return 0;}
    Options o=parse_options(argc,argv);threads_set(o.threads);const char *cmd=argv[1];
    if(!strcmp(cmd,"selftest")){selftest();return 0;}
    if(!strcmp(cmd,"vocab")) {
        if(!o.data||!o.out)die("vocab requires --data and --out");
        make_vocab(o.data,o.format,o.out,o.c.vocab,o.c.seq);return 0;
    }
    if(!strcmp(cmd,"train")||!strcmp(cmd,"pretrain")) {
        if(!o.data||!o.save)die("training requires --data and --save");
        int stage=!strcmp(cmd,"pretrain");Net *n;
        g_hard_fwd=g_fwd_opt>=0?g_fwd_opt:1;
        fprintf(stderr,"forward: %s\n",g_hard_fwd?"hard (straight-through: trains the deployed network)":"soft relaxation");
        if(o.resume||o.load) {
            n=load_net(o.resume?o.resume:o.load);overlay(n,&o,o.resume!=NULL);
            if(o.resume){if(n->c.stage!=stage || o.seed_set)die("resume preserves stage and RNG; use --load for a new run");}
            else{
                float t0=n->temperature;
                if(o.load_part){
                    Net *f=net_new(n->c,1,1,o.seed);f->vocab=n->vocab;memset(&n->vocab,0,sizeof n->vocab);
                    if(!strcmp(o.load_part,"codes")){
                        memcpy(f->emb.z,n->emb.z,n->emb.n*4);memcpy(f->bias.z,n->bias.z,n->bias.n*4);
                    }else for(int i=0;i<n->c.blocks;i++){
                        Layer *a=f->l+i,*b=n->l+i;
                        memcpy(a->ch,b->ch,(size_t)a->O*a->R*4);memcpy(a->off,b->off,(size_t)a->O*a->R*4);
                        memcpy(a->p.z,b->p.z,a->p.n*4);
                    }
                    f->temperature=n->temperature;net_free(n);n=f;
                    fprintf(stderr,"load-part %s: rest freshly initialized (seed %llu)\n",o.load_part,(unsigned long long)o.seed);
                }
                reset_optimizer(n);n->c.stage=stage;n->rng.s=o.seed;
                if(o.keep_temp){g_fixed_temp=fmaxf(.05f,t0);n->temperature=g_fixed_temp;
                    fprintf(stderr,"keep-temperature: %.4f (checkpoint) instead of the 1.0 -> 0.2 anneal\n",g_fixed_temp);}
            }
        }else{if(stage && !(o.changed&(UINT64_C(1)<<11)))o.c.batch=64;n=net_new(o.c,1,1,o.seed);n->c.stage=stage;}
        if(o.vocab) {
            Vocab v=vocab_text(o.vocab,n->c.vocab);
            if(n->vocab.n){if(n->vocab.n!=v.n)die("checkpoint vocabulary mismatch");for(int i=0;i<v.n;i++)if(strcmp(v.words[i],n->vocab.words[i]))die("checkpoint vocabulary mismatch");vocab_free(&v);}
            else n->vocab=v;
        }
        Data d=data_load(o.data,o.format,&n->vocab,n->c.vocab,n->c.seq),val={0};
        if(o.val)val=data_load(o.val,o.format,&n->vocab,n->c.vocab,n->c.seq);
        train_net(n,&d,o.val?&val:NULL,o.save,o.export_path,o.stop,o.every);
        data_free(&d);data_free(&val);net_free(n);return 0;
    }
    if(!strcmp(cmd,"export")) {
        if(!o.load||!o.out)die("export needs --load and --out");
        Hard *h=load_hard(o.load);save_hard(h,o.out);hard_free(h);return 0;
    }
    if(!strcmp(cmd,"predict")||!strcmp(cmd,"eval")||!strcmp(cmd,"bench")||!strcmp(cmd,"info")) {
        if(!o.load)die("--load is required");
        Hard *h=load_hard(o.load);
        if(o.vocab && !h->vocab.n)h->vocab=vocab_text(o.vocab,h->c.vocab);
        if(!strcmp(cmd,"info")) {
            size_t gates=0;for(int i=0;i<=h->c.blocks;i++)gates+=(size_t)h->l[i].O*h->l[i].N;
            printf("{\"vocab_size\":%d,\"code_bits\":%d,\"width\":%d,\"blocks\":%d,\"tree_depth\":%d,\"votes_per_class\":%d,\"deployed_gate_parameters\":%zu,\"has_vocabulary\":%s}\n",
                   h->c.vocab,h->c.bits,h->c.width,h->c.blocks,h->c.depth,h->c.votes,gates,h->vocab.n?"true":"false");hard_free(h);return 0;
        }
        Data d={0};
        if(o.text) {
            if(!h->vocab.n)die("--text needs an embedded vocabulary");
            d.T=o.c.seq;d.V=h->c.vocab;d.vocab=&h->vocab;data_text_record(-1,o.text,&d);
        }else {if(!o.data)die("--data or --text is required");d=data_load(o.data,o.format,&h->vocab,h->c.vocab,o.c.seq);}
        if(!strcmp(cmd,"bench"))bench(h,&d,o.c.batch,o.repeats,o.warmup);
        else if(!strcmp(cmd,"eval")){require_labels(&d);double acc=eval_hard(h,&d,o.c.batch,0);printf("{\"records\":%d,\"hard_accuracy\":%.8f}\n",d.n,acc);}
        else eval_hard(h,&d,o.text?1:o.c.batch,1);
        data_free(&d);hard_free(h);return 0;
    }
    die("unknown command: %s",cmd);return 1;
}


/* =====================================================================================================
 * fast inference: bit-sliced interpreter (bit t = token position t), lanes, compiler to C
 * ===================================================================================================== */


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
/* global view (global_patch.py): channels W..W+G-1 = "channel c fired anywhere", on valid positions */   \
static inline void global_##L(const Hard *h, V *x, const V *valid, int NW) {                                \
    int W = h->c.width, G = h->c.gch;                                                                        \
    for (int c = 0; c < G; c++) {                                                                            \
        V m;                                                                                                 \
        if (h->c.gmean) {  /* majority: 2 x fired positions > valid positions, per lane */                   \
            m = (V){0};                                                                                      \
            for (int e = 0; e < L; e++) {                                                                    \
                int cnt = 0, nv = 0;                                                                         \
                for (int w = 0; w < NW; w++) { cnt += __builtin_popcountll(x[(size_t)c * NW + w][e]); nv += __builtin_popcountll(valid[w][e]); } \
                m[e] = 2 * cnt > nv ? ~UINT64_C(0) : 0;                                                      \
            }                                                                                                \
        } else {                                                                                             \
            V nz = (V){0};                                                                                   \
            for (int w = 0; w < NW; w++) nz |= x[(size_t)c * NW + w];                                        \
            m = (V)(nz != (V){0});                                                                           \
        }                                                                                                    \
        for (int w = 0; w < NW; w++) x[(size_t)(W + c) * NW + w] = m & valid[w];                             \
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
    memset(a, 0, sizeof(V) * (size_t)(C + h->c.match) * NW);                                                  \
    uint8_t mb[8 * 64 * MAXNW];  /* match bits (global_patch.py --match), L x T */                              \
    if (h->c.match) { memset(mb, 0, sizeof mb); match_bits(ids, n, T, mb); }                                 \
    memset(valid, 0, sizeof valid);                                                                          \
    for (int e = 0; e < n; e++) for (int t = 0; t < T; t++) {                                               \
        uint32_t id = ids[(size_t)e * T + t];                                                                \
        if (id >= (uint32_t)h->c.vocab) die("inference token outside vocabulary");                          \
        if (!id) continue;                                                                                   \
        uint64_t bit = UINT64_C(1) << (t % 64);                                                              \
        valid[t / 64][e] |= bit;                                                                             \
        if (h->c.match && mb[(size_t)e * T + t]) a[(size_t)C * NW + t / 64][e] |= bit;                       \
        for (int q = 0; q < Q; q++) {                                                                        \
            uint64_t code = h->codes[(size_t)id * Q + q];                                                    \
            while (code) { int c = q * 64 + lowbit(code); if (c < C) a[(size_t)c * NW + t / 64][e] |= bit; code &= code - 1; } \
        }                                                                                                    \
    }                                                                                                        \
    V *cur = a, *nxt = b;                                                                                    \
    if (GEN_OK(NW)) GEN_BLOCKS(L)(cur, nxt, valid);  /* compiled model: result left in cur */            \
    else for (int i = 0; i < h->c.blocks; i++) {                                                            \
        if (gblock(&h->c, i)) global_##L(h, cur, valid, NW);                                                 \
        layer_##L(F, h->l + i, cur, nxt, valid, NW); V *z = cur; cur = nxt; nxt = z;                         \
    }                                                                                                        \
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
    if (F->h->c.gevery && maxC < F->h->c.width + F->h->c.gch) maxC = F->h->c.width + F->h->c.gch;             \
    if (maxC < F->h->c.bits + F->h->c.match) maxC = F->h->c.bits + F->h->c.match;                             \
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
/* Built with -DGEN_INC='"model.inc"' (from `fastlogic gen`), the blocks of that one model run as straight-line
 * word operations for T <= 64 (see the end of the file). Without it the interpreter runs every layer. */
#ifdef GEN_INC
static int gen_nwords(void);  /* words per channel the model was compiled for (T <= 64 x that) */
#define GEN_OK(NW) ((NW) == gen_nwords())
#define GEN_BLOCKS(L) gen_blocks_##L
#define GEN_DECL(L, V) static void gen_blocks_##L(V *a, V *b, const V *valid);
GEN_DECL(1, v1) GEN_DECL(2, v2) GEN_DECL(4, v4) GEN_DECL(8, v8)
#else
#define GEN_OK(NW) 0
#define GEN_BLOCKS(L) gen_none
#define gen_none(a, b, v) ((void)0)
#define gen_nwords() 0
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

static int fast_main(int argc, char **argv) {
    if (argc < 5) die("usage: fastlogic predict|verify|bench|gen MODEL.lth DATA.ids SEQ [...] (fastlogic help)");
    const char *cmd = argv[1];
    int T = (int)parse_long(argv[4], 1, 65536);
    int threads = opt_int(argc, argv, "--threads", thread_count()), lanes = opt_int(argc, argv, "--lanes", 8);
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
    if (!strcmp(cmd, "gen")) {  /* gen MODEL DATA SEQ OUT.inc: the blocks as straight-line C for this SEQ */
        const char *ops[16] = {"Z", "~(A|B)", "(~A&B)", "~A", "(A&~B)", "~B", "(A^B)", "~(A&B)",
                               "(A&B)", "~(A^B)", "B", "(~A|B)", "A", "(A|~B)", "(A|B)", "~Z"};
        int NWg = F->NW, PW = NWg == 1 ? 1 : NWg + 2, P0 = NWg == 1 ? 0 : 1;  /* multi-word: 1 zero pad word each side */
        int W = h->c.width, C0 = h->c.bits + h->c.match, MC = W + (h->c.gevery ? h->c.gch : 0);
        if (MC < C0) MC = C0;
        FILE *o = open_file(argv[5], "w");
        fprintf(o, "/* generated by fastlogic gen from %s: %d blocks x %d outputs, T <= %d (%d word%s per channel) */\n",
                argv[2], h->c.blocks, W, 64 * NWg, NWg, NWg > 1 ? "s" : "");
        fprintf(o, "#define GEN_NWORDS %d\n", NWg);
        for (int i = 0; i < h->c.blocks; i++) {
            HLayer *l = h->l + i;
            fprintf(o, "static void GEN_L(blk%d)(const V *restrict x, V *restrict y, const V *valid) {\n  const V Z = {0}; (void)Z;\n", i);
            if (NWg > 1) fprintf(o, "  for (int w = 0; w < %d; w++) {  /* one word loop per block: straight-line outputs inside */\n", NWg);
            for (int oo = 0; oo < l->O; oo++) {
                fprintf(o, "  { ");
                for (int k = 0; k < l->R; k++) {
                    int c = l->ch[(size_t)oo * l->R + k], dd = l->off[(size_t)oo * l->R + k] * l->dilation, base = c * PW + P0;
                    if (NWg == 1) {
                        if (dd == 0) fprintf(o, "V g%d = x[%d]; ", l->N + k, c);
                        else if (dd >= 64 || dd <= -64) fprintf(o, "V g%d = Z; ", l->N + k);
                        else fprintf(o, "V g%d = x[%d] %s %d; ", l->N + k, c, dd > 0 ? ">>" : "<<", dd > 0 ? dd : -dd);
                    } else {  /* position t reads t + dd: word w from words w + floor(dd/64) and the next one */
                        if (dd >= 64 || dd <= -64) die("gen: tap offset %d needs |offset| < 64 for multi-word models", dd);
                        if (dd == 0) fprintf(o, "V g%d = x[%d + w]; ", l->N + k, base);
                        else if (dd > 0) fprintf(o, "V g%d = (x[%d + w] >> %d) | (x[%d + w] << %d); ", l->N + k, base, dd, base + 1, 64 - dd);
                        else fprintf(o, "V g%d = (x[%d + w] >> %d) | (x[%d + w] << %d); ", l->N + k, base - 1, 64 + dd, base, -dd);
                    }
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
                if (NWg == 1) fprintf(o, "y[%d] = g0 & valid[0]; }\n", oo);
                else fprintf(o, "y[%d + w] = g0 & valid[w]; }\n", oo * PW + P0);
            }
            if (NWg > 1) fprintf(o, "  }\n");
            fprintf(o, "}\n");
        }
        /* driver: a = input (interpreter layout c*NW+w), b = scratch; the result is left in a */
        fprintf(o, "static void GEN_L(blocks)(V *a, V *b, const V *valid) {\n");
        if (NWg == 1) fprintf(o, "  V *A = a, *B = b;\n");
        else {
            fprintf(o, "  static _Thread_local V *A = 0, *B = 0;\n"
                       "  if (!A) { A = aligned_alloc(64, sizeof(V) * %d); B = aligned_alloc(64, sizeof(V) * %d);\n"
                       "    memset(A, 0, sizeof(V) * %d); memset(B, 0, sizeof(V) * %d); }\n",
                    MC * PW, MC * PW, MC * PW, MC * PW);
            fprintf(o, "  for (int c = 0; c < %d; c++) for (int w = 0; w < %d; w++) A[c * %d + %d + w] = a[c * %d + w];\n",
                    C0, NWg, PW, P0, NWg);
        }
        for (int i = 0; i < h->c.blocks; i++) {
            const char *src = i % 2 ? "B" : "A";
            if (gblock(&h->c, i)) {  /* global view: pool the first G channels into W..W+G-1 (as global_L) */
                if (h->c.gmean)
                    fprintf(o, "  for (int c = 0; c < %d; c++) { V m = {0}; for (int e = 0; e < (int)(sizeof(V) / 8); e++) {"
                               " int cnt = 0, nv = 0; for (int w = 0; w < %d; w++) { cnt += __builtin_popcountll(%s[c * %d + %d + w][e]);"
                               " nv += __builtin_popcountll(valid[w][e]); } m[e] = 2 * cnt > nv ? ~0ULL : 0; }"
                               " for (int w = 0; w < %d; w++) %s[(%d + c) * %d + %d + w] = m & valid[w]; }\n",
                            h->c.gch, NWg, src, PW, P0, NWg, src, W, PW, P0);
                else
                    fprintf(o, "  for (int c = 0; c < %d; c++) { const V Z = {0}; V nz = Z; for (int w = 0; w < %d; w++) nz |= %s[c * %d + %d + w];"
                               " V m = (V)(nz != Z); for (int w = 0; w < %d; w++) %s[(%d + c) * %d + %d + w] = m & valid[w]; }\n",
                            h->c.gch, NWg, src, PW, P0, NWg, src, W, PW, P0);
            }
            fprintf(o, "  GEN_L(blk%d)(%s, %s, valid);\n", i, src, i % 2 ? "A" : "B");
        }
        const char *fin = h->c.blocks % 2 ? "B" : "A";
        if (NWg == 1) { if (h->c.blocks % 2) fprintf(o, "  memcpy(a, b, sizeof(V) * %d);\n", W); }
        else fprintf(o, "  for (int c = 0; c < %d; c++) for (int w = 0; w < %d; w++) a[c * %d + w] = %s[c * %d + %d + w];\n",
                     W, NWg, NWg, fin, PW, P0);
        fprintf(o, "}\n");
        fclose(o);
        printf("{\"generated\":\"%s\",\"blocks\":%d,\"outputs\":%d,\"seq_max\":%d,\"words\":%d}\n", argv[5], h->c.blocks,
               h->c.blocks * W, 64 * NWg, NWg);
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

/* =====================================================================================================
 * checks and tools
 * ===================================================================================================== */
/* hardcheck: the training forward pass (hard, what train/pretrain optimize) must compute exactly the deployed
 * network: for every record, the class-vote difference from supervised() == hard_predict's integer votes. */
static int hardcheck_main(int argc,char **argv){
    if(argc<5)die("usage: fastlogic hardcheck MODEL.ltc DATA.ids SEQ");
    int T=(int)parse_long(argv[4],1,65536),B=32;threads_set(thread_count());
    Net *n=load_net(argv[2]);n->c.stage=0;g_hard_fwd=1;refresh(n);
    Data d=data_load(argv[3],"ids",&n->vocab,n->c.vocab,T);
    Hard *h=harden(n);HWork *hw=hwork_new(h,B,T);Work *w=work_new(n,B,T);
    int bad=0,N=0,O=2*n->c.votes;
    for(int i=0;i+B<=d.n;i+=B){
        memcpy(w->ids,d.x+(size_t)i*T,(size_t)B*T*4);
        for(int b=0;b<B;b++)w->labels[b]=d.y[i+b]<0?0:d.y[i+b];
        supervised(n,w,0,NULL);hard_predict(h,hw,d.x+(size_t)i*T);
        for(int b=0;b<B;b++){
            double s[2]={0,0};for(int u=0;u<w->U;u++)for(int o=0;o<O;o++)s[o/n->c.votes]+=w->head[((size_t)u*O+o)*B+b];
            bad+=llround(s[1]-s[0])!=(long long)(hw->scores[2*b+1]-hw->scores[2*b]);N++;
        }
    }
    printf("{\"records\":%d,\"vote_mismatches\":%d}\n",N,bad);
    return bad!=0;
}
/* revive: gates whose output channel is constant (hard forward, over the valid positions of DATA) are reset to
 * the scratch pass-through init (+-3). Dead, saturated channels are why masked-word pretraining did not transfer. */
static int revive_main(int argc,char **argv){
    if(argc<6)die("usage: fastlogic revive MODEL.ltc DATA.ids SEQ OUT.ltc");
    int T=(int)parse_long(argv[4],1,65536),B=64;threads_set(thread_count());
    Net *n=load_net(argv[2]);g_hard_fwd=1;refresh(n);int K=n->c.blocks,W=n->c.width;
    Data d=data_load(argv[3],"ids",&n->vocab,n->c.vocab,T);Work *w=work_new(n,B,T);
    uint8_t *s0=alloc((size_t)K*W,1),*s1=alloc((size_t)K*W,1);
    for(int i=0;i<d.n;i+=B){
        int take=d.n-i<B?d.n-i:B;memset(w->ids,0,(size_t)B*T*4);
        for(int b=0;b<take;b++)memcpy(w->ids+(size_t)b*T,d.x+(size_t)(i+b)*T,T*4);
        encode(n,w);
        #pragma omp parallel for schedule(static)
        for(int li=0;li<K;li++)for(int c=0;c<W;c++)for(int t=0;t<T;t++)for(int b=0;b<take;b++)
            if(w->valid[(size_t)b*T+t]){if(w->a[li+1][((size_t)t*W+c)*B+b]>.5f)s1[li*W+c]=1;else s0[li*W+c]=1;}
    }
    int tot=0;
    for(int li=0;li<K;li++){
        Layer *l=n->l+li;int dead=0;
        for(int o=0;o<W;o++)if(!(s0[li*W+o]&&s1[li*W+o])){dead++;for(int k=0;k<l->N*4;k++)l->p.z[(size_t)o*l->N*4+k]=(k%4)<2?-3.f:3.f;}
        fprintf(stderr,"block %d: %d/%d dead\n",li,dead,W);tot+=dead;
    }
    save_net(n,argv[5]);printf("{\"reset_channels\":%d,\"of\":%d}\n",tot,K*W);return 0;
}

/* =====================================================================================================
 * main
 * ===================================================================================================== */
int main(int argc,char **argv){
    if(!getenv("OMP_WAIT_POLICY")){setenv("OMP_WAIT_POLICY","passive",1);execv("/proc/self/exe",argv);}  /* before OpenMP starts */
    if(sizeof(float)!=4 || FLT_RADIX!=2 || FLT_MANT_DIG!=24)die("requires IEEE-754 binary32 floats");
    if(argc<2 || !strcmp(argv[1],"help") || !strcmp(argv[1],"--help")){usage();return 0;}
    const char *c=argv[1];
    if(!strcmp(c,"predict")||!strcmp(c,"verify")||!strcmp(c,"bench")||!strcmp(c,"gen"))return fast_main(argc,argv);
    if(!strcmp(c,"hardcheck"))return hardcheck_main(argc,argv);
    if(!strcmp(c,"revive"))return revive_main(argc,argv);
    if(!strcmp(c,"predict-ref"))argv[1]=(char*)"predict";
    else if(!strcmp(c,"bench-ref"))argv[1]=(char*)"bench";
    return lt_main(argc,argv);
}

/* =====================================================================================================
 * compiled model (optional): gcc ... -DGEN_INC='"model.inc"' fastlogic.c   (model.inc from `fastlogic gen`)
 * ===================================================================================================== */
#ifdef GEN_INC
#define V v1
#define GEN_L(n) n##_1
#include GEN_INC
#undef V
#undef GEN_L
#define V v2
#define GEN_L(n) n##_2
#include GEN_INC
#undef V
#undef GEN_L
#define V v4
#define GEN_L(n) n##_4
#include GEN_INC
#undef V
#undef GEN_L
#define V v8
#define GEN_L(n) n##_8
#include GEN_INC
#undef V
#undef GEN_L
static void gen_blocks_1(v1 *a, v1 *b, const v1 *valid) { blocks_1(a, b, valid); }
static void gen_blocks_2(v2 *a, v2 *b, const v2 *valid) { blocks_2(a, b, valid); }
static void gen_blocks_4(v4 *a, v4 *b, const v4 *valid) { blocks_4(a, b, valid); }
static void gen_blocks_8(v8 *a, v8 *b, const v8 *valid) { blocks_8(a, b, valid); }
static int gen_nwords(void) { return GEN_NWORDS; }
#endif
