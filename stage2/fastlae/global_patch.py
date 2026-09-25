"""global view: give LogicAE's convolutional blocks a whole-text view, still pure logic.

Every block of logic-bert sees kernel x dilation neighbours, so relating distant tokens (QNLI, a
question vs an option, sentence-level nuance) needs many blocks. With --global-every K and
--global-channels G, the input of every block i with i % K == 0 (0 < i < blocks) gets G extra
channels: channel W+c = "channel c fired anywhere in the text" (OR over valid positions; max in the
float relaxation, gradient to the first arg-max position), broadcast to every valid position. The
gates of that block can wire to them like any other channel (offsets are irrelevant for a constant).
Parameters are unchanged (gate count depends on outputs, not inputs); only wiring choices grow.

Hardened, it is one OR-reduction per global channel per text: hard_predict, and stage2/fastlae
(fastlae.c interpreter and fastgen compiled network) implement it; scores must stay identical.

v2 options (after the OR version saturated: channels fire at ~49% of positions, so OR over a text is 1
for ~98% of texts x channels and carries no information):
  --global-mean 1   majority pooling: channel = fires at more than half the valid positions (mean in
                    training, 2 x popcount > valid count hardened)
  --match 1         one extra input bit per position: the token also occurs in the other [SEP] segment
                    (id 102; without a [SEP], at another position): the exact-match signal attention gets.

Checkpoints: global models are written as LTCKP002 / LTHRD002 (the two ints appended to the config);
models without it keep LTCKP001 / LTHRD001 byte-for-byte, so with the options off the patched binary
trains and saves exactly like the unpatched one.

usage: python3 global_patch.py IN/logic_text.c OUT/logic_text.c   (apply after transfer_patch, before fasttrain_patch)
"""
import sys

EDITS = [
    # config
    ("    float peak_lr,sharp,clip,mask_prob,scale,mlm_scale;\n} Config;",
     "    float peak_lr,sharp,clip,mask_prob,scale,mlm_scale;\n"
     "    int gevery,gch;  /* global view: every gevery-th block input gets gch pooled channels */\n"
     "    int gmean;       /* pooling: 0 = OR over positions (max), 1 = majority (mean > 1/2) */\n"
     "    int match;       /* extra input bit: token also occurs in the other [SEP] segment */\n} Config;"),
    ("       (int64_t)c.batch*c.seq>INT_MAX || (int64_t)c.seq*c.votes>INT_MAX/2)",
     "       (int64_t)c.batch*c.seq>INT_MAX || (int64_t)c.seq*c.votes>INT_MAX/2 ||\n"
     "       c.gevery<0 || c.gevery>MAX_BLOCKS || c.gch<0 || c.gch>c.width || (c.gevery>0)!=(c.gch>0) ||\n"
     "       c.gmean<0 || c.gmean>1 || c.match<0 || c.match>1)"),
    ("typedef struct { size_t n; float *z,*g,*m,*v; uint64_t t; } Param;",
     "static int gblock(const Config *c,int i){return c->gevery>0 && i>0 && i<c->blocks && i%c->gevery==0;}\n"
     "static int in_ch(const Config *c,int i){return (i?c->width:c->bits+c->match)+(gblock(c,i)?c->gch:0);}\n"
     "#define MATCH_SEP 102  /* bert-base-uncased [SEP] */\n"
     "/* match bit per (text, position): the token also occurs in the other segment (split at the first [SEP]);\n"
     " * without a [SEP], at another position. PAD, special ids (< 3) and [SEP] itself get 0. */\n"
     "static void match_bits(const uint32_t *ids,int B,int T,uint8_t *out) {\n"
     "    for(int b=0;b<B;b++) {\n"
     "        const uint32_t *x=ids+(size_t)b*T;int sep=-1;\n"
     "        for(int t=0;t<T;t++)if(x[t]==MATCH_SEP){sep=t;break;}\n"
     "        for(int t=0;t<T;t++) {\n"
     "            uint8_t m=0;\n"
     "            if(x[t]>=3 && x[t]!=MATCH_SEP)for(int u=0;u<T && !m;u++) {\n"
     "                if(u==t || x[u]!=x[t])continue;\n"
     "                m=sep<0 ? 1 : ((t<sep)!=(u<sep));\n"
     "            }\n"
     "            out[(size_t)b*T+t]=m;\n"
     "        }\n"
     "    }\n"
     "}\n"
     "typedef struct { size_t n; float *z,*g,*m,*v; uint64_t t; } Param;"),
    ("    for(int i=0;i<c.blocks;i++)n->l[i]=layer_new(i?c.width:c.bits,c.width,c.depth,1<<(i%c.cycle),c.kernel,training);",
     "    for(int i=0;i<c.blocks;i++)n->l[i]=layer_new(in_ch(&c,i),c.width,c.depth,1<<(i%c.cycle),c.kernel,training);"),
    # work buffers
    ("    for(int i=0;i<=K;i++)w->a[i]=alloc(mul(mul(B,T),i?W:C),4);",
     "    for(int i=0;i<=K;i++)w->a[i]=alloc(mul(mul(B,T),i?W:C+n->c.match),4);"),
    ("    uint8_t *valid,*pv,*selected;uint32_t *ids,*targets;int *labels;\n} Work;",
     "    uint8_t *valid,*pv,*selected;uint32_t *ids,*targets;int *labels;\n    float **aug,*gtmp;int32_t **amax;\n} Work;"),
    ("    w->maxC=W>C?W:C;if(w->maxC<2*n->c.votes)w->maxC=2*n->c.votes;",
     "    w->maxC=W>C?W:C;if(w->maxC<2*n->c.votes)w->maxC=2*n->c.votes;\n"
     "    if(n->c.gevery && w->maxC<W+n->c.gch)w->maxC=W+n->c.gch;\n"
     "    w->aug=alloc(K+1,sizeof(float*));w->amax=alloc(K+1,sizeof(int32_t*));\n"
     "    for(int i=0;i<K;i++)if(gblock(&n->c,i)){w->aug[i]=alloc(mul(mul(B,T),W+n->c.gch),4);w->amax[i]=alloc(mul(B,n->c.gch),4);}\n"
     "    if(n->c.gevery)w->gtmp=alloc(mul(mul(B,T),W),4);"),
    ("    free(w->selected);free(w->ids);free(w->targets);free(w->labels);free(w);",
     "    for(int i=0;i<=K;i++){free(w->aug[i]);free(w->amax[i]);}free(w->aug);free(w->amax);free(w->gtmp);\n"
     "    free(w->selected);free(w->ids);free(w->targets);free(w->labels);free(w);"),
    # forward
    ("    for(int i=0;i<n->c.blocks;i++)forward_layer(n->l+i,w->a[i],w->a[i+1],w->valid,w->B,w->T,n->c.q8);",
     "    for(int i=0;i<n->c.blocks;i++)forward_layer(n->l+i,ginput(n,w,i),w->a[i+1],w->valid,w->B,w->T,n->c.q8);"),
    ("    int C=n->c.bits;refresh(n);\n    #pragma omp parallel for schedule(static)\n    for(size_t j=0;j<(size_t)w->B*w->T;j++) {",
     "    int C=n->c.bits,C0=C+n->c.match;refresh(n);uint8_t *mb=NULL;\n"
     "    if(n->c.match){mb=alloc(mul(w->B,w->T),1);match_bits(w->ids,w->B,w->T,mb);}\n"
     "    #pragma omp parallel for schedule(static)\n    for(size_t j=0;j<(size_t)w->B*w->T;j++) {"),
    ("            w->a[0][((size_t)t*C+k)*w->B+b]=n->c.q8?nearbyintf(a*255.f)/255.f:a;\n        }\n    }",
     "            w->a[0][((size_t)t*C0+k)*w->B+b]=n->c.q8?nearbyintf(a*255.f)/255.f:a;\n        }\n"
     "        if(n->c.match)w->a[0][((size_t)t*C0+C)*w->B+b]=(float)mb[j];\n    }\n    free(mb);"),
    ("static void encode(Net *n,Work *w) {",
     "/* block i input: a[i], plus (global blocks) G channels = max over valid positions, broadcast */\n"
     "static float *ginput(Net *n,Work *w,int i) {\n"
     "    if(!gblock(&n->c,i))return w->a[i];\n"
     "    int W=n->c.width,G=n->c.gch,B=w->B,T=w->T,CA=W+G;const float *x=w->a[i];float *y=w->aug[i];int32_t *am=w->amax[i];\n"
     "    #pragma omp parallel for schedule(static)\n"
     "    for(int t=0;t<T;t++)for(int c=0;c<W;c++)memcpy(y+((size_t)t*CA+c)*B,x+((size_t)t*W+c)*B,(size_t)B*4);\n"
     "    #pragma omp parallel for schedule(static)\n"
     "    for(int c=0;c<G;c++)for(int b=0;b<B;b++) {\n"
     "        float m=0.f;int32_t at=-1;\n"
     "        if(n->c.gmean){double s=0;int nv=0;for(int t=0;t<T;t++)if(w->valid[(size_t)b*T+t]){s+=x[((size_t)t*W+c)*B+b];nv++;}\n"
     "            m=nv?(float)(s/nv):0.f;at=nv;}  /* mean pooling: am holds the valid count */\n"
     "        else for(int t=0;t<T;t++)if(w->valid[(size_t)b*T+t]){float v=x[((size_t)t*W+c)*B+b];if(at<0||v>m){m=v;at=t;}}\n"
     "        am[(size_t)c*B+b]=at;\n"
     "        for(int t=0;t<T;t++)y[((size_t)t*CA+W+c)*B+b]=w->valid[(size_t)b*T+t]?m:0.f;\n"
     "    }\n"
     "    return y;\n"
     "}\n"
     "/* gradient w.r.t. the augmented input (CA channels, in dx) -> w.r.t. a[i] (W channels, back into dx) */\n"
     "static void gcollapse(Net *n,Work *w,int i,float *dx) {\n"
     "    int W=n->c.width,G=n->c.gch,B=w->B,T=w->T,CA=W+G;float *o=w->gtmp;const int32_t *am=w->amax[i];\n"
     "    #pragma omp parallel for schedule(static)\n"
     "    for(int t=0;t<T;t++)for(int c=0;c<W;c++)memcpy(o+((size_t)t*W+c)*B,dx+((size_t)t*CA+c)*B,(size_t)B*4);\n"
     "    #pragma omp parallel for schedule(static)\n"
     "    for(int c=0;c<G;c++)for(int b=0;b<B;b++) {\n"
     "        int32_t at=am[(size_t)c*B+b];if(at<=0 && (n->c.gmean || at<0))continue;\n"
     "        float s=0.f;for(int t=0;t<T;t++)s+=dx[((size_t)t*CA+W+c)*B+b];\n"
     "        if(n->c.gmean){float q=s/at;for(int t=0;t<T;t++)if(w->valid[(size_t)b*T+t])o[((size_t)t*W+c)*B+b]+=q;}\n"
     "        else o[((size_t)at*W+c)*B+b]+=s;\n"
     "    }\n"
     "    memcpy(dx,o,(size_t)B*T*W*4);\n"
     "}\n"
     "static void encode(Net *n,Work *w) {"),
    # backward
    ("            backward_layer(n->l+i,w->a[i],g,dx,w->partial,w->valid,w->B,w->T,n->c.q8,n->c.est);",
     "            backward_layer(n->l+i,gblock(&n->c,i)?w->aug[i]:w->a[i],g,dx,w->partial,w->valid,w->B,w->T,n->c.q8,n->c.est);\n"
     "            if(gblock(&n->c,i))gcollapse(n,w,i,dx);"),
    ("                w->code_grad[(size_t)w->ids[j]*n->c.bits+k]+=g[((size_t)t*n->c.bits+k)*w->B+b];",
     "                w->code_grad[(size_t)w->ids[j]*n->c.bits+k]+=g[((size_t)t*(n->c.bits+n->c.match)+k)*w->B+b];"),
    # checkpoints
    ('    char *tmp=temp_path(path);FILE *f=open_file(tmp,"wb");write_bytes(f,"LTCKP001",8);config_write(f,n->c);',
     '    char *tmp=temp_path(path);FILE *f=open_file(tmp,"wb");write_bytes(f,(n->c.gevery||n->c.match)?"LTCKP002":"LTCKP001",8);config_write(f,n->c);\n'
     '    if(n->c.gevery||n->c.match){put32(f,(uint32_t)n->c.gevery);put32(f,(uint32_t)n->c.gch);put32(f,(uint32_t)n->c.gmean);put32(f,(uint32_t)n->c.match);}'),
    ('    if(memcmp(magic,"LTCKP001",8))die("%s is not a training .ltc checkpoint",path);\n    Config c=config_read(f);Net *n=net_new(c,1,0,1);',
     '    int v2=!memcmp(magic,"LTCKP002",8);\n    if(memcmp(magic,"LTCKP001",8) && !v2)die("%s is not a training .ltc checkpoint",path);\n'
     '    Config c=config_read(f);if(v2){c.gevery=(int)get32(f);c.gch=(int)get32(f);c.gmean=(int)get32(f);c.match=(int)get32(f);validate(c);}Net *n=net_new(c,1,0,1);'),
    ('    char *tmp=temp_path(path);FILE *f=open_file(tmp,"wb");write_bytes(f,"LTHRD001",8);',
     '    char *tmp=temp_path(path);FILE *f=open_file(tmp,"wb");write_bytes(f,(h->c.gevery||h->c.match)?"LTHRD002":"LTHRD001",8);'),
    ("    for(int i=0;i<10;i++)put32(f,a[i]);",
     "    for(int i=0;i<10;i++)put32(f,a[i]);\n    if(c.gevery||c.match){put32(f,(uint32_t)c.gevery);put32(f,(uint32_t)c.gch);put32(f,(uint32_t)c.gmean);put32(f,(uint32_t)c.match);}"),
    ('    if(!memcmp(magic,"LTCKP001",8)) {fclose(f);Net *n=load_net(path);Hard *h=harden(n);net_free(n);return h;}\n'
     '    if(memcmp(magic,"LTHRD001",8))die("not a .lth or .ltc checkpoint: %s",path);',
     '    if(!memcmp(magic,"LTCKP001",8) || !memcmp(magic,"LTCKP002",8)) {fclose(f);Net *n=load_net(path);Hard *h=harden(n);net_free(n);return h;}\n'
     '    int hv2=!memcmp(magic,"LTHRD002",8);\n'
     '    if(memcmp(magic,"LTHRD001",8) && !hv2)die("not a .lth or .ltc checkpoint: %s",path);'),
    ("    for(int i=0;i<10;i++){uint32_t v=get32(f);if(v>INT_MAX)die(\"invalid hard config\");*a[i]=(int)v;}validate(c);h->c=c;",
     "    for(int i=0;i<10;i++){uint32_t v=get32(f);if(v>INT_MAX)die(\"invalid hard config\");*a[i]=(int)v;}\n"
     "    if(hv2){c.gevery=(int)get32(f);c.gch=(int)get32(f);c.gmean=(int)get32(f);c.match=(int)get32(f);}validate(c);h->c=c;"),
    ("        int C=i?c.width:c.bits,O=i==c.blocks?2*c.votes:c.width,",
     "        int C=i==c.blocks?c.width:in_ch(&c,i),O=i==c.blocks?2*c.votes:c.width,"),
    # hard inference
    ("        uint64_t *row=cur+((size_t)p*T+t)*C;memset(row,0,C*8);uint64_t mask=0;",
     "        uint64_t *row=cur+((size_t)p*T+t)*(C+h->c.match);memset(row,0,(C+h->c.match)*8);uint64_t mask=0;"),
    ("            if(id)mask|=bit;\n            for(int q=0;q<Q;q++) {\n                uint64_t code=h->codes[(size_t)id*Q+q];\n                while(code){int b=lowbit(code);int c=q*64+b;if(c<C)row[c]|=bit;code&=code-1;}\n            }\n        }w->mask[(size_t)p*T+t]=mask;",
     "            if(id)mask|=bit;\n            if(mb && mb[(size_t)(p*64+k)*T+t])row[C]|=bit;\n            for(int q=0;q<Q;q++) {\n                uint64_t code=h->codes[(size_t)id*Q+q];\n                while(code){int b=lowbit(code);int c=q*64+b;if(c<C)row[c]|=bit;code&=code-1;}\n            }\n        }w->mask[(size_t)p*T+t]=mask;"),
    ("    for(size_t i=0;i<(size_t)w->B*T;i++)if(ids[i]>=(uint32_t)h->c.vocab)die(\"inference token outside vocabulary\");\n    #pragma omp parallel for collapse(2) schedule(static)\n    for(int p=0;p<w->P;p++)for(int t=0;t<T;t++) {\n        uint64_t *row",
     "    for(size_t i=0;i<(size_t)w->B*T;i++)if(ids[i]>=(uint32_t)h->c.vocab)die(\"inference token outside vocabulary\");\n"
     "    uint8_t *mb=NULL;if(h->c.match){mb=alloc(mul(w->B,T),1);match_bits(ids,w->B,T,mb);}\n"
     "    #pragma omp parallel for collapse(2) schedule(static)\n    for(int p=0;p<w->P;p++)for(int t=0;t<T;t++) {\n        uint64_t *row"),
    ("typedef struct {int B,T,P,maxC;uint64_t *a,*b,*mask,*pmask;int64_t *scores;} HWork;",
     "typedef struct {int B,T,P,maxC;uint64_t *a,*b,*mask,*pmask;int64_t *scores;uint64_t *g;} HWork;"),
    ("    w->maxC=c.width>c.bits?c.width:c.bits;if(w->maxC<2*c.votes)w->maxC=2*c.votes;",
     "    w->maxC=c.width>c.bits?c.width:c.bits;if(w->maxC<2*c.votes)w->maxC=2*c.votes;\n"
     "    if(c.gevery && w->maxC<c.width+c.gch)w->maxC=c.width+c.gch;"),
    ("static void hwork_free(HWork *w) {free(w->a);free(w->b);",
     "static void hwork_free(HWork *w) {free(w->g);free(w->a);free(w->b);"),
    ("        hard_layer(h->l+i,cur,next,w->mask,w->P,T);uint64_t *tmp=cur;cur=next;next=tmp;",
     "        if(i==0)free(mb);\n"
     "        const uint64_t *in=cur;\n"
     "        if(gblock(&h->c,i)) {  /* augmented input: W channels + G OR-pooled, broadcast to valid positions */\n"
     "            int W=h->c.width,G=h->c.gch,CA=W+G;\n"
     "            if(!w->g)w->g=alloc(mul(mul(w->P,T),CA),8);\n"
     "            #pragma omp parallel for schedule(static)\n"
     "            for(int p=0;p<w->P;p++) {\n"
     "                for(int t=0;t<T;t++)memcpy(w->g+((size_t)p*T+t)*CA,cur+((size_t)p*T+t)*W,(size_t)W*8);\n"
     "                for(int c=0;c<G;c++) {\n"
     "                    uint64_t any=0;\n"
     "                    if(h->c.gmean) {  /* majority: 2 x count > valid positions, per text lane */\n"
     "                        for(int k=0;k<64;k++){int cnt=0,nv=0;for(int t=0;t<T;t++){cnt+=(int)((cur[((size_t)p*T+t)*W+c]>>k)&1);nv+=(int)((w->mask[(size_t)p*T+t]>>k)&1);}\n"
     "                            if(2*cnt>nv)any|=UINT64_C(1)<<k;}\n"
     "                    } else for(int t=0;t<T;t++)any|=cur[((size_t)p*T+t)*W+c];\n"
     "                    for(int t=0;t<T;t++)w->g[((size_t)p*T+t)*CA+W+c]=any&w->mask[(size_t)p*T+t];\n"
     "                }\n"
     "            }\n"
     "            in=w->g;\n"
     "        }\n"
     "        hard_layer(h->l+i,in,next,w->mask,w->P,T);uint64_t *tmp=cur;cur=next;next=tmp;"),
    # options
    ('        SI("--batch",batch,11,1,65536) SI("--steps",steps,12,1,1000000000) SI("--mlm-targets",max_targets,14,0,INT_MAX)',
     '        SI("--batch",batch,11,1,65536) SI("--steps",steps,12,1,1000000000) SI("--mlm-targets",max_targets,14,0,INT_MAX)\n'
     '        SI("--global-every",gevery,21,0,MAX_BLOCKS) SI("--global-channels",gch,22,0,65536)\n'
     '        SI("--global-mean",gmean,23,0,1) SI("--match",match,24,0,1)'),
    ("    validate(*c);\n}\n#ifndef LOGIC_NO_MAIN",
     "    if(o->changed&(UINT64_C(15)<<21)) {\n"
     "        if(c->gevery!=s->gevery || c->gch!=s->gch || c->gmean!=s->gmean || c->match!=s->match)\n"
     "            die(\"--global-*/--match differ from the checkpoint\");\n"
     "    }\n"
     "    validate(*c);\n}\n#ifndef LOGIC_NO_MAIN"),
]


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit(f"global_patch: anchor not found exactly once: {old[:80]!r}")
        s = s.replace(old, new)
    open(out, "w").write(s)
    print(f"global_patch: {len(EDITS)} edits -> {out}")


if __name__ == "__main__":
    main()
