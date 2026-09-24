/* lfeat: frozen LogicAE (DDLGN) features for the System-1 shootout. Runs a hardened logic-bert
 * network (.lth: bit-parallel gates, the deployable CPU path) up to its last block and ORs each
 * gate output over the valid positions of the sequence -> WIDTH bits per record, the same pooling
 * as the GraphTM features (clause fired anywhere). Output: records x WIDTH/8 bytes, bit c of a
 * record at byte c/8, bit c%8 (numpy unpackbits bitorder="little").
 * usage: lfeat MODEL.lth DATA.ids SEQ THREADS OUT.bin
 * Build: gcc -O3 -march=native -std=c11 -fopenmp -I<logic-bert>/src lfeat.c -lm -o lfeat */
#define LOGIC_NO_MAIN
#include "logic_text.c"

int main(int argc,char **argv){
    if(argc<6)die("usage: lfeat MODEL.lth DATA.ids SEQ THREADS OUT.bin");
    int T=(int)parse_long(argv[3],1,65536);threads_set((int)parse_long(argv[4],1,1024));
    Hard *h=load_hard(argv[1]);Data d=data_load(argv[2],"ids",NULL,h->c.vocab,T);
    int C=h->c.bits,Q=(C+63)/64,W=h->c.width,B=1024;
    HWork *w=hwork_new(h,B,T);uint32_t *x=alloc(mul(B,T),4);uint64_t *acc=alloc(mul(B/64,W),8);
    unsigned char *row=alloc((size_t)(W+7)/8,1);FILE *out=open_file(argv[5],"wb");
    for(int i=0;i<d.n;i+=B){
        int take=d.n-i<B?d.n-i:B;memset(x,0,(size_t)B*T*4);memcpy(x,d.x+(size_t)i*T,(size_t)take*T*4);
        uint64_t *cur=w->a,*next=w->b;
        #pragma omp parallel for collapse(2) schedule(static)
        for(int p=0;p<w->P;p++)for(int t=0;t<T;t++){  /* same packing as hard_predict */
            uint64_t *r=cur+((size_t)p*T+t)*C;memset(r,0,C*8);uint64_t mask=0;
            for(int k=0;k<64 && p*64+k<B;k++){
                uint32_t id=x[(size_t)(p*64+k)*T+t];uint64_t bit=UINT64_C(1)<<k;
                if(id)mask|=bit;
                for(int q=0;q<Q;q++){uint64_t code=h->codes[(size_t)id*Q+q];
                    while(code){int b=lowbit(code);int c=q*64+b;if(c<C)r[c]|=bit;code&=code-1;}}
            }w->mask[(size_t)p*T+t]=mask;
        }
        for(int l=0;l<h->c.blocks;l++){hard_layer(h->l+l,cur,next,w->mask,w->P,T);uint64_t *tmp=cur;cur=next;next=tmp;}
        memset(acc,0,(size_t)(B/64)*W*8);
        #pragma omp parallel for schedule(static)
        for(int p=0;p<w->P;p++)for(int t=0;t<T;t++)for(int c=0;c<W;c++)acc[(size_t)p*W+c]|=cur[((size_t)p*T+t)*W+c];
        for(int j=0;j<take;j++){
            int p=j/64,k=j%64;memset(row,0,(size_t)(W+7)/8);
            for(int c=0;c<W;c++)if((acc[(size_t)p*W+c]>>k)&1)row[c/8]|=(unsigned char)(1u<<(c%8));
            write_bytes(out,row,(size_t)(W+7)/8);
        }
    }
    fclose(out);fprintf(stderr,"lfeat: %d records x %d bits\n",d.n,W);return 0;
}
