"""fasttrain: make logic-bert training use every core, with bit-identical results.

logic_text.c's backward_layer parallelizes over batch chunks of LANES=16, so a batch of 32 keeps at
most 2 threads busy (measured: 1.33 s/step at 1 thread, 1.02 s/step at 4, seq 48, batch 32).
It cannot simply loop over outputs in parallel, because different outputs add into the same input
gradients (dx) and float addition order would change.

The patch splits it in two phases that keep every float operation and its order:
  1. parallel over outputs o: recompute the tree, accumulate the gate-parameter partials exactly as
     before (t ascending within each (o, lane chunk)), and store each leaf's gradient in G[o][t][k][b]
     instead of adding it into dx.
  2. parallel over input channels c: for every position s, add into dx[s][c][b] the leaf gradients
     that read (s, c), in the original order (o ascending, then t ascending, then k ascending): per
     channel a static list of (o, k) sorted by (o, -offset, k), since t = s - offset * dilation.
forward_layer also parallelizes over (t, o) instead of t alone (each output is independent there).
Checked: saved checkpoints after N steps are byte-identical to the unpatched binary at any thread count.

usage: python3 fasttrain_patch.py IN/logic_text.c OUT/logic_text.c   (original or transfer_patch output)
"""
import sys

NEW_BACKWARD = r'''static float *g_leafgrad=NULL;static size_t g_leafgrad_n=0;
static void backward_layer(Layer *l,const float *x,const float *dy,float *dx,float *partial,
                           const uint8_t *valid,int B,int T,int q8,int est) {
    /* fasttrain: phase 1 parallel over outputs, phase 2 parallel over input channels; the float
     * operations and their order per element are those of the original serial-over-outputs loop. */
    int O=l->O,N=l->N,R=l->R,C=l->C;
    memset(dx,0,mul(mul(B,T),l->C)*4);memset(partial,0,mul(B,l->p.n)*4);
    size_t need=mul(mul(mul(O,T),R),B);
    if(need>g_leafgrad_n){free(g_leafgrad);g_leafgrad=alloc(need,4);g_leafgrad_n=need;}
    float *G=g_leafgrad;
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
'''


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    a = s.index("static void backward_layer(Layer *l,")
    b = s.index("typedef struct {\n    int B,T,U,maxC;", a)
    s = s[:a] + NEW_BACKWARD + "\n" + s[b:]
    old = "    #pragma omp parallel for schedule(static)\n    for(int t=0;t<T;t++) for(int o=0;o<O;o++) {\n        int32_t base[MAX_LEAVES];tree_leaves(l,T,t,o,base);"
    if s.count(old) != 1:
        sys.exit("fasttrain_patch: forward_layer anchor not found")
    s = s.replace(old, old.replace("parallel for schedule(static)", "parallel for collapse(2) schedule(static)"))
    # elementwise loops over the embedding table: same arithmetic per element, now parallel
    for a, b in [
        ("    if(grad)for(size_t i=0;i<n->emb.n;i++) {\n        float p=n->codes[i];n->emb.g[i]",
         "    if(grad)\n    #pragma omp parallel for schedule(static)\n    for(size_t i=0;i<n->emb.n;i++) {\n        float p=n->codes[i];n->emb.g[i]"),
        ("    if(grad)for(int v=3;v<V;v++)for(int k=0;k<C;k++) {\n        size_t i=(size_t)v*C+k;",
         "    if(grad)\n    #pragma omp parallel for schedule(static)\n    for(int v=3;v<V;v++)for(int k=0;k<C;k++) {\n        size_t i=(size_t)v*C+k;"),
        ("        for(size_t k=0;k<l->p.n;k++)l->corners[k]=corner(l->p.z[k],n->c.est);",
         "        #pragma omp parallel for schedule(static)\n        for(size_t k=0;k<l->p.n;k++)l->corners[k]=corner(l->p.z[k],n->c.est);"),
    ]:
        if s.count(a) != 1:
            sys.exit(f"fasttrain_patch: anchor not found: {a[:60]!r}")
        s = s.replace(a, b)
    open(out, "w").write(s)
    print(f"fasttrain_patch: backward_layer, forward_layer, embedding loops -> {out}")


if __name__ == "__main__":
    main()
