/* hardcheck: the training engine's hard forward pass (--hard-forward, what `train` optimizes by default) must
 * compute exactly the deployed hardened network. For every record, compares the class-vote difference from the
 * training path (supervised(), g_hard_fwd=1, float master weights) with hard_predict's integer votes (.lth path).
 *   hardcheck MODEL.ltc DATA.ids SEQ      -> {"records":N,"vote_mismatches":0,...}; exit 1 on any mismatch */
#define LOGIC_NO_MAIN
#include "logic_text.c"
int main(int argc,char **argv){
    if(argc<4)die("usage: hardcheck MODEL.ltc DATA.ids SEQ");
    int T=(int)parse_long(argv[3],1,65536),B=32;threads_set(thread_count());
    Net *n=load_net(argv[1]);n->c.stage=0;g_hard_fwd=1;refresh(n);
    Data d=data_load(argv[2],"ids",&n->vocab,n->c.vocab,T);
    Hard *h=harden(n);HWork *hw=hwork_new(h,B,T);Work *w=work_new(n,B,T);
    int bad=0,N=0,O=2*n->c.votes;
    for(int i=0;i+B<=d.n;i+=B){
        memcpy(w->ids,d.x+(size_t)i*T,(size_t)B*T*4);
        for(int b=0;b<B;b++)w->labels[b]=d.y[i+b]<0?0:d.y[i+b];
        supervised(n,w,0,NULL);hard_predict(h,hw,d.x+(size_t)i*T);
        for(int b=0;b<B;b++){
            double s[2]={0,0};for(int u=0;u<w->U;u++)for(int o=0;o<O;o++)s[o/n->c.votes]+=w->head[((size_t)u*O+o)*B+b];
            long long soft=llround(s[1]-s[0]),hard=(long long)(hw->scores[2*b+1]-hw->scores[2*b]);
            bad+=soft!=hard;N++;
        }
    }
    printf("{\"records\":%d,\"vote_mismatches\":%d,\"match\":%d,\"global_every\":%d,\"global_mean\":%d}\n",N,bad,n->c.match,n->c.gevery,n->c.gmean);
    return bad!=0;
}
