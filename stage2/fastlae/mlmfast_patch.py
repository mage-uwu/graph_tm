"""Faster masked-word output layer (the 30k-way softmax) with identical float math. Apply after fasttrain_patch.py
(and bitfwd_patch.py if used).

The original computes, per target c and word v, sc = sum_k (2h_k - 1)(2code_vk - 1) with h read at stride B and the
k-sum as a scalar reduction (which the compiler may not vectorise without reassociating it). Here:
  prep     hv[c][k] = 2h_k - 1 (contiguous per target), cm[v][k] = ct[k][v] = 2code_vk - 1: the same float values the
           original computes inline.
  phase A  tiles of 16 targets x 256 words; for k ascending, acc[t][j] += hv[t][k] * ct[k][j]. Every logit is still
           the sum of the same products in increasing k starting from +0; the vector lanes run across words.
           Max, exp and the softmax denominator stay the original per-target sequential loops.
  phase B  per tile of 16 targets, words in ascending order: acc[t][k] += (2*scale*g) * cm[v][k]; stored into the
           projection gradient (which the original accumulates from +0 in the same order).
  phase C  unchanged order (per word row, targets ascending), reading hv contiguously so the k loop vectorises.
Checkpoints and logged losses are byte-identical to the original (checked by the caller).
usage: python3 mlmfast_patch.py IN/logic_text.c OUT/logic_text.c
"""
import sys

START = "    int Vr=V-3;float *ex=alloc(mul(count,Vr),4);double *den=alloc(count,8),*lt=alloc(count,8);\n"
END = "    free(tix);free(ex);free(den);free(lt);\n"

NEW = r'''    int Vr=V-3;float *ex=alloc(mul(count,Vr),4);double *den=alloc(count,8),*lt=alloc(count,8);
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
'''


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    if s.count(START) != 1 or s.count(END) != 1:
        sys.exit("mlmfast_patch: masked_loss anchors not found exactly once")
    a = s.index(START)
    b = s.index(END)
    s = s[:a] + NEW + s[b:]
    open(out, "w").write(s)
    print(f"mlmfast_patch: masked_loss output layer -> {out}")


if __name__ == "__main__":
    main()
