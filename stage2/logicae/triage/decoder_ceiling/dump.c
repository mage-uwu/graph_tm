/* dump: hard forward of a checkpoint on masked-centre windows; per record writes packed bits
 * [final block, positions c0-2..c0+2, W each][projection at c0, C bits] and the tied decoder's CE (float) */
#define LOGIC_NO_MAIN
#include "logic_text.c"
int main(int argc,char **argv){
    threads_set(thread_count());g_hard_fwd=1;
    Net *n=load_net(argv[1]);n->c.stage=1;refresh(n);
    int T=65,c0=32,C=n->c.bits,V=n->c.vocab,K=n->c.blocks,W=n->c.width,NB=5*W+C;
    Data d=data_load(argv[2],"ids",NULL,V,T);FILE *ft=open_file(argv[3],"r");
    int *tru=alloc(d.n,sizeof(int));for(int i=0;i<d.n;i++)if(fscanf(ft,"%d",tru+i)!=1)die("true ids");
    FILE *fb=open_file(argv[4],"wb"),*fc=open_file(argv[5],"wb");
    int B=64;Work *w=work_new(n,B,T);float scale=n->c.mlm_scale/C;
    uint8_t *pk=alloc((size_t)B*(NB/8),1);float *ce=alloc(B,4);double tot=0;
    for(int i=0;i<d.n;i+=B){
        int take=d.n-i<B?d.n-i:B;memset(w->ids,0,(size_t)B*T*4);
        for(int b=0;b<take;b++)memcpy(w->ids+(size_t)b*T,d.x+(size_t)(i+b)*T,T*4);
        encode(n,w);forward_layer(n->l+K,w->a[K],w->proj,w->valid,B,T,0);
        memset(pk,0,(size_t)B*(NB/8));
        #pragma omp parallel for schedule(static)
        for(int b=0;b<take;b++){
            uint8_t *p=pk+(size_t)b*(NB/8);int o=0;
            for(int dt=-2;dt<=2;dt++)for(int c=0;c<W;c++,o++)if(w->a[K][((size_t)(c0+dt)*W+c)*B+b]>.5f)p[o>>3]|=1<<(o&7);
            const float *h=w->proj+(size_t)c0*C*B+b;
            for(int k=0;k<C;k++,o++)if(h[(size_t)k*B]>.5f)p[o>>3]|=1<<(o&7);
            float mx=-FLT_MAX,*e=alloc(V,4);
            for(int v=3;v<V;v++){float s=0;const float *cv=n->codes+(size_t)v*C;
                for(int k=0;k<C;k++)s+=(2*h[(size_t)k*B]-1)*(2*cv[k]-1);
                e[v]=scale*s+n->bias.z[v-3];if(e[v]>mx)mx=e[v];}
            double den=0;for(int v=3;v<V;v++)den+=exp(e[v]-mx);
            ce[b]=(float)(log(den)+mx-e[tru[i+b]]);free(e);
        }
        fwrite(pk,NB/8,take,fb);fwrite(ce,4,take,fc);for(int b=0;b<take;b++)tot+=ce[b];
    }
    fclose(fb);fclose(fc);printf("n %d tied_ce %.4f bytes/rec %d\n",d.n,tot/d.n,NB/8);return 0;
}
