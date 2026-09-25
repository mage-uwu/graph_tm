/* diag: why does LogicAE pretraining not transfer? Read-only probes of a training checkpoint (.ltc),
 * soft (float) network at the checkpoint's code temperature unless --temp is given.
 *   diag mlm  CKPT DATA.ids TRUE.txt T [--temp X]        masked-token accuracy at the centre position
 *                                                       ((T-1)/2) over the 30,519 real ids: acc@1, acc@10, CE
 *   diag feat CKPT DATA.ids T OUT.f32 [--fresh] [--temp X]  per record, per layer in {0,2,4,8,12,16}: the
 *                                                       fraction of valid positions where each channel > .5
 *                                                       (mean pooling; OR pooling saturates)
 *   diag grad CKPT DATA.ids T [--fresh] [--temp X]       supervised fine-tuning gradients at step 0 (the
 *                                                       never-trained vote head): per-layer mean |grad|, gate
 *                                                       saturation, codes' mean |grad|
 * --fresh: the checkpoint's config, freshly initialized (seed 17): what scratch fine-tuning starts from.
 * Build: gcc -O3 -march=native -std=c11 -fopenmp -I<logic-bert>/src diag.c -lm -o diag */
#define LOGIC_NO_MAIN
#include "logic_text.c"

static const int FL[]={0,2,4,8,12,16};
static Net *get(int argc,char **argv,int stage){
    Net *n=load_net(argv[2]);float t=n->temperature;
    for(int i=3;i<argc;i++)if(!strcmp(argv[i],"--fresh")){
        Config c=n->c;Vocab v=n->vocab;memset(&n->vocab,0,sizeof n->vocab);net_free(n);
        n=net_new(c,1,1,17);n->vocab=v;n->temperature=1.f;t=1.f;
    }
    for(int i=3;i+1<argc;i++)if(!strcmp(argv[i],"--temp"))t=parse_float(argv[i+1]);
    n->temperature=t;n->c.stage=stage;refresh(n);
    fprintf(stderr,"diag: %s%s temperature %.3f blocks %d width %d bits %d\n",argv[2],
            n->c.vocab?"":"",t,n->c.blocks,n->c.width,n->c.bits);
    return n;
}
int main(int argc,char **argv){
    if(argc<5)die("usage: see diag.c");
    threads_set(thread_count());
    if(!strcmp(argv[1],"mlm")){
        Net *n=get(argc,argv,1);int T=(int)parse_long(argv[5],1,4096),c0=(T-1)/2,C=n->c.bits,V=n->c.vocab,K=n->c.blocks;
        Data d=data_load(argv[3],"ids",NULL,V,T);FILE *ft=open_file(argv[4],"r");
        int *tru=alloc(d.n,sizeof(int));for(int i=0;i<d.n;i++)if(fscanf(ft,"%d",tru+i)!=1)die("true ids");
        int B=64;Work *w=work_new(n,B,T);double ce=0;int a1=0,a10=0,N=0;float scale=n->c.mlm_scale/C;
        for(int i=0;i<d.n;i+=B){
            int take=d.n-i<B?d.n-i:B;memset(w->ids,0,(size_t)B*T*4);
            for(int b=0;b<take;b++)memcpy(w->ids+(size_t)b*T,d.x+(size_t)(i+b)*T,T*4);
            encode(n,w);forward_layer(n->l+K,w->a[K],w->proj,w->valid,B,T,0);
            #pragma omp parallel for reduction(+:ce,a1,a10) schedule(static)
            for(int b=0;b<take;b++){
                const float *h=w->proj+(size_t)c0*C*B+b;float mx=-FLT_MAX,*e=alloc(V,4);
                for(int v=3;v<V;v++){float s=0;const float *cv=n->codes+(size_t)v*C;
                    for(int k=0;k<C;k++)s+=(2*h[(size_t)k*B]-1)*(2*cv[k]-1);
                    e[v]=scale*s+n->bias.z[v-3];if(e[v]>mx)mx=e[v];}
                double den=0;int rank=0;float te=e[tru[i+b]];
                for(int v=3;v<V;v++){den+=exp(e[v]-mx);rank+=e[v]>te;}
                ce+=log(den)+mx-te;a1+=rank<1;a10+=rank<10;free(e);
            }N+=take;
        }
        printf("{\"mode\":\"mlm\",\"data\":\"%s\",\"T\":%d,\"n\":%d,\"acc1\":%.4f,\"acc10\":%.4f,\"ce\":%.4f}\n",argv[3],T,N,(double)a1/N,(double)a10/N,ce/N);
        return 0;
    }
    if(!strcmp(argv[1],"feat")){
        Net *n=get(argc,argv,1);int T=(int)parse_long(argv[4],1,4096),K=n->c.blocks,W=n->c.width,C=n->c.bits;
        Data d=data_load(argv[3],"ids",NULL,n->c.vocab,T);int B=64,L=sizeof FL/sizeof *FL,D=0;
        for(int j=0;j<L;j++)if(FL[j]<=K)D+=FL[j]?W:C;
        Work *w=work_new(n,B,T);float *row=alloc((size_t)B*D,4);FILE *out=open_file(argv[5],"wb");
        for(int i=0;i<d.n;i+=B){
            int take=d.n-i<B?d.n-i:B;memset(w->ids,0,(size_t)B*T*4);
            for(int b=0;b<take;b++)memcpy(w->ids+(size_t)b*T,d.x+(size_t)(i+b)*T,T*4);
            encode(n,w);memset(row,0,(size_t)B*D*4);
            #pragma omp parallel for schedule(static)
            for(int b=0;b<take;b++){
                int cnt=0;for(int t=0;t<T;t++)cnt+=w->valid[(size_t)b*T+t];
                int o=0;for(int j=0;j<L;j++){if(FL[j]>K)continue;int CC=FL[j]?W:C;const float *x=w->a[FL[j]];
                    for(int c=0;c<CC;c++){int s=0;for(int t=0;t<T;t++)s+=w->valid[(size_t)b*T+t]&&x[((size_t)t*CC+c)*B+b]>.5f;
                        row[(size_t)b*D+o+c]=cnt?(float)s/cnt:0.f;}o+=CC;}
            }
            write_bytes(out,row,(size_t)take*D*4);
        }
        fclose(out);printf("{\"mode\":\"feat\",\"n\":%d,\"dim\":%d}\n",d.n,D);return 0;
    }
    if(!strcmp(argv[1],"grad")){
        Net *n=get(argc,argv,0);int T=(int)parse_long(argv[4],1,4096),K=n->c.blocks;
        Data d=data_load(argv[3],"ids",NULL,n->c.vocab,T);int B=64;Work *w=work_new(n,B,T);
        for(int b=0;b<B;b++){memcpy(w->ids+(size_t)b*T,d.x+(size_t)b*T,T*4);w->labels[b]=d.y[b];}
        float ce;objective(n,w,0.f,1,&ce);
        printf("{\"mode\":\"grad\",\"ce\":%.4f,\"layers\":[",ce);
        for(int i=0;i<K+2;i++){Layer *l=n->l+i;double g=0,sat=0;
            for(size_t k=0;k<l->p.n;k++){g+=fabsf(l->p.g[k]);float c=l->corners[k];sat+=c<.02f||c>.98f;}
            printf("%s{\"layer\":%d,\"mean_abs_grad\":%.3e,\"saturated\":%.3f}",i?",":"",i,g/l->p.n,sat/l->p.n);}
        double ge=0;size_t ne=0;for(size_t k=0;k<n->emb.n;k++)if(n->emb.g[k]!=0){ge+=fabsf(n->emb.g[k]);ne++;}
        printf("],\"codes_mean_abs_grad\":%.3e}\n",ne?ge/ne:0);return 0;
    }
    if(!strcmp(argv[1],"wires")){  /* diag wires CKPT x x x: channels of the last block read by the MLM projection / vote head */
        Net *n=load_net(argv[2]);int K=n->c.blocks;
        for(int j=K;j<=K+1;j++){Layer *l=n->l+j;printf("%s",j==K?"proj":"head");
            for(int k=0;k<l->O*l->R;k++)printf(" %d",l->ch[k]);printf("\n");}
        return 0;
    }
    if(!strcmp(argv[1],"codes")){  /* diag codes CKPT OUT.u8 x x: hard code bits (z > 0) per vocab id, V x bits bytes */
        Net *n=load_net(argv[2]);FILE *o=open_file(argv[3],"wb");
        for(size_t k=0;k<n->emb.n;k++){unsigned char b=n->emb.z[k]>0;write_bytes(o,&b,1);}
        fclose(o);printf("{\"mode\":\"codes\",\"V\":%d,\"bits\":%d}\n",n->c.vocab,n->c.bits);return 0;
    }
    if(!strcmp(argv[1],"corners")){  /* diag corners CKPT x x x: per layer, share of gate corners undecided (.2-.8) / saturated */
        Net *n=load_net(argv[2]);printf("{\"mode\":\"corners\",\"layers\":[");
        for(int i=0;i<n->c.blocks+2;i++){Layer *l=n->l+i;double u=0,s=0;
            for(size_t k=0;k<l->p.n;k++){float c=l->corners[k];u+=c>.2f&&c<.8f;s+=c<.02f||c>.98f;}
            printf("%s[%.3f,%.3f]",i?",":"",u/l->p.n,s/l->p.n);}
        printf("]}\n");return 0;
    }
    if(!strcmp(argv[1],"revive")){  /* diag revive CKPT DATA.ids T OUT.ltc: gates whose output channel is constant on DATA
                                       (hard, over valid positions) are reset to the scratch pass-through init (+-3, no noise) */
        Net *n=get(argc,argv,1);int T=(int)parse_long(argv[4],1,4096),K=n->c.blocks,W=n->c.width,B=64;
        Data d=data_load(argv[3],"ids",NULL,n->c.vocab,T);Work *w=work_new(n,B,T);
        uint8_t *s0=alloc((size_t)K*W,1),*s1=alloc((size_t)K*W,1);
        for(int i=0;i<d.n;i+=B){int take=d.n-i<B?d.n-i:B;memset(w->ids,0,(size_t)B*T*4);
            for(int b=0;b<take;b++)memcpy(w->ids+(size_t)b*T,d.x+(size_t)(i+b)*T,T*4);
            encode(n,w);
            #pragma omp parallel for schedule(static)
            for(int li=0;li<K;li++)for(int c=0;c<W;c++)for(int t=0;t<T;t++)for(int b=0;b<take;b++)
                if(w->valid[(size_t)b*T+t]){float x=w->a[li+1][((size_t)t*W+c)*B+b];if(x>.5f)s1[li*W+c]=1;else s0[li*W+c]=1;}
        }
        int tot=0;for(int li=0;li<K;li++){Layer *l=n->l+li;int dead=0;
            for(int o=0;o<W;o++)if(!(s0[li*W+o]&&s1[li*W+o])){dead++;
                for(int k=0;k<l->N*4;k++)l->p.z[(size_t)o*l->N*4+k]=(k%4)<2?-3.f:3.f;}
            fprintf(stderr,"L%d dead %d/%d\n",li,dead,W);tot+=dead;}
        save_net(n,argv[5]);printf("{\"mode\":\"revive\",\"reset_channels\":%d}\n",tot);return 0;
    }
    die("unknown mode");
}
