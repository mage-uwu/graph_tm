/* mlmrank: rank the true centre token of fixed windows with a logic-bert MLM checkpoint, the
 * same way the GraphTM is evaluated (stage2/eval_mlm.py): rank among a candidate list, ties
 * count half; acc@1, acc@10, MRR. Soft (relaxed) forward pass -- the MLM decoder is training-only.
 * usage: mlmrank MODEL.ltc WINDOWS.ids TRUE.txt CANDIDATES.txt CENTRE THREADS
 * Build: gcc -O3 -march=native -std=c11 -fopenmp -I<logic-bert>/src mlmrank.c -lm */
#define LOGIC_NO_MAIN
#include "logic_text.c"
static int *read_ints(const char *path,int *n){
    FILE *f=open_file(path,"r");int cap=1024,k=0;int *a=alloc(cap,sizeof(int));long v;
    while(fscanf(f,"%ld",&v)==1){if(k==cap){cap*=2;a=resize(a,cap,sizeof(int));}a[k++]=(int)v;}
    fclose(f);*n=k;return a;
}
int main(int argc,char **argv){
    if(argc<7)die("usage: mlmrank MODEL.ltc WINDOWS.ids TRUE.txt CANDIDATES.txt CENTRE THREADS");
    int centre=(int)parse_long(argv[5],0,65535);threads_set((int)parse_long(argv[6],1,1024));
    Net *n=load_net(argv[1]);n->c.stage=1;int C=n->c.bits,K=n->c.blocks,T=centre*2+1;
    Data d=data_load(argv[2],"ids",NULL,n->c.vocab,T);
    int nt,nc;int *truth=read_ints(argv[3],&nt),*cand=read_ints(argv[4],&nc);
    if(nt!=d.n)die("TRUE.txt has %d ids for %d windows",nt,d.n);
    float scale=n->c.mlm_scale/C;int B=256;double r1=0,r10=0,mrr=0;
    float *sc=alloc(mul(B,nc),4);
    for(int i=0;i<d.n;i+=B){
        int take=d.n-i<B?d.n-i:B;Work *w=work_new(n,take,T);
        memcpy(w->ids,d.x+(size_t)i*T,(size_t)take*T*4);
        encode(n,w);forward_layer(n->l+K,w->a[K],w->proj,w->valid,take,T,n->c.q8);
        #pragma omp parallel for schedule(static) reduction(+:r1,r10,mrr)
        for(int b=0;b<take;b++){
            const float *h=w->proj+(size_t)centre*C*take+b;float *s=sc+(size_t)b*nc;
            int tv=truth[i+b];float st=0;
            for(int j=0;j<nc;j++){
                int v=cand[j];const float *cv=n->codes+(size_t)v*C;float x=0;
                for(int k=0;k<C;k++)x+=(2*h[(size_t)k*take]-1)*(2*cv[k]-1);
                s[j]=scale*x+n->bias.z[v-3];if(v==tv)st=s[j];
            }
            double gt=0,eq=0;for(int j=0;j<nc;j++){gt+=s[j]>st;eq+=s[j]==st;}
            double r=gt+.5*(eq-1);r1+=r<1;r10+=r<10;mrr+=1./(1.+r);
        }
        work_free(w,K);
    }
    printf("{\"n\":%d,\"acc@1\":%.6f,\"acc@10\":%.6f,\"mrr\":%.6f,\"params\":%zu}\n",d.n,r1/d.n,r10/d.n,mrr/d.n,parameter_count(n));
    return 0;
}
