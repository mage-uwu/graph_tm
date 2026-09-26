/* slim: a checkpoint for adaptation only: same weights, Adam moments zeroed (--load resets the optimizer anyway).
 * The zeros compress away: gzip the result to ship it. Not for --resume (that needs the optimizer state).
 *   slim IN.pt OUT.pt */
#define LOGIC_NO_MAIN
#include "logic_text.c"
int main(int argc,char **argv){
    if(argc<3)die("usage: slim IN.pt OUT.pt");
    Net *n=load_net(argv[1]);Param *a[MAX_BLOCKS+4];int k=0;a[k++]=&n->emb;a[k++]=&n->bias;
    for(int i=0;i<n->c.blocks+2;i++)a[k++]=&n->l[i].p;
    for(int i=0;i<k;i++){memset(a[i]->m,0,a[i]->n*4);memset(a[i]->v,0,a[i]->n*4);a[i]->t=0;}
    save_net(n,argv[2]);printf("{\"slim\":\"%s\"}\n",argv[2]);return 0;
}
