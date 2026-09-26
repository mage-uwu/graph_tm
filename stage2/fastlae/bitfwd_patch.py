"""Bit-level training kernels for the hard (straight-through) forward pass. Apply after hard_patch.py and
fasttrain_patch.py. Active only when g_hard_fwd && !q8; otherwise the float kernels run unchanged.

With --hard-forward every activation is exactly 0 or 1, so:
  forward   each block is evaluated like the inference engine: the batch's 0/1 activations are packed 64 texts
            per word, every gate is a branch-free mux on words (truth table bit k = corner[k] > .5, the same test
            the float path applies to r = q[2a+b]), and the result is written back as 0.0 / 1.0 floats.
  backward  the tree's node values come from the same word evaluation instead of a float recompute, and the
            gate gradients use their exact binary-input forms:
              corner 2a+b gets += u, the others += u*0 = +-0, a no-op (partial sums start at +0 and can never
              become -0), computed with 0/1 float masks so the lane loop vectorizes,
              d/da = u*(b ? q3-q1 : q2-q0),  d/db = u*(a ? q3-q2 : q1-q0)   (bitwise the float code's products).
            Loops run o -> t -> lanes (was o -> lanes -> t): every partial / leaf-gradient element still receives
            its additions in increasing t, so all sums are unchanged.
Checkpoints are byte-identical to the float kernels (checked by the caller: hash of trained checkpoints).
usage: python3 bitfwd_patch.py IN/logic_text.c OUT/logic_text.c
"""
import sys

KERNELS = r'''
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
'''

EDITS = [
    # kernels after the float forward_layer's helpers (before forward_layer itself)
    ("static void forward_layer(const Layer *l,const float *x,float *y,const uint8_t *valid,int B,int T,int q8) {\n    int O=l->O;\n",
     KERNELS + "static void forward_layer(const Layer *l,const float *x,float *y,const uint8_t *valid,int B,int T,int q8) {\n"
     "    if(g_hard_fwd && !q8 && B<=512){bit_forward_layer(l,x,y,valid,B,T);return;}\n    int O=l->O;\n"),
    ("    float *G=g_leafgrad;\n    #pragma omp parallel for schedule(dynamic,4)\n"
     "    for(int o=0;o<O;o++)for(int b0=0;b0<B;b0+=LANES)for(int t=0;t<T;t++) {\n",
     "    float *G=g_leafgrad;\n    if(g_hard_fwd && !q8 && B<=512){bit_backward_phase1(l,x,dy,partial,G,valid,B,T);}else{\n"
     "    #pragma omp parallel for schedule(dynamic,4)\n"
     "    for(int o=0;o<O;o++)for(int b0=0;b0<B;b0+=LANES)for(int t=0;t<T;t++) {\n"),
    ("    }\n    /* per input channel: (o,k) readers sorted by (o, -offset, k) = the original (o, t, k) order */\n",
     "    }\n    }  /* bitfwd_patch.py: end of the float phase 1 */\n"
     "    /* per input channel: (o,k) readers sorted by (o, -offset, k) = the original (o, t, k) order */\n"),
]


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    if "g_hard_fwd" not in s or "g_leafgrad" not in s:
        sys.exit("bitfwd_patch: apply after hard_patch.py and fasttrain_patch.py")
    for a, b in EDITS:
        if s.count(a) != 1:
            sys.exit(f"bitfwd_patch: anchor found {s.count(a)}x: {a[:80]!r}")
        s = s.replace(a, b)
    open(out, "w").write(s)
    print(f"bitfwd_patch: {len(EDITS)} edits -> {out}")


if __name__ == "__main__":
    main()
