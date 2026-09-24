"""Patch logic-bert's logic_text.c for the pretraining-transfer diagnostics (stage2/logicae/transfer.sh).

Adds two opt-in options to `lt train` / `lt pretrain` with --load (defaults are unchanged: a
patched binary without them trains bit-identically to the original; transfer.sh checks this):
  --keep-temperature     keep the checkpoint's code temperature (0.2 after run 1's pretraining),
                         constant, instead of reset_optimizer's reset to 1.0 + the 1.0 -> 0.2 anneal.
                         Codes are sigmoid(z / temperature): the reset turns the near-binary codes
                         the pretrained gates were trained on into soft ones (z = 1: 0.99 -> 0.73).
  --load-part codes|gates  take only part of the checkpoint into a freshly initialized network:
                         codes = token codes (embedding + MLM bias), gates = the 16 blocks' gate
                         parameters and their wiring (channels / offsets). Everything else fresh.
usage: python3 transfer_patch.py SRC/logic_text.c OUT/logic_text.c
"""
import hashlib
import sys

SHA16 = "28d24cdb55f7f312"  # the logic_text.c run 1 used (run.sh checks the same prefix)

EDITS = [
    # 1. options
    ("    int threads,stop,every,repeats,warmup;uint64_t seed;int seed_set;\n} Options;",
     "    int threads,stop,every,repeats,warmup;uint64_t seed;int seed_set;\n    int keep_temp;const char *load_part;\n} Options;"),
    ('        if(!strcmp(key,"--seed")){o.seed=(uint64_t)parse_long(s,1,LONG_MAX);o.seed_set=1;continue;}',
     '        if(!strcmp(key,"--seed")){o.seed=(uint64_t)parse_long(s,1,LONG_MAX);o.seed_set=1;continue;}\n'
     '        if(!strcmp(key,"--load-part")){if(strcmp(s,"codes")&&strcmp(s,"gates"))die("--load-part must be codes or gates");o.load_part=s;continue;}'),
    ('        const char *key=argv[i];\n',
     '        const char *key=argv[i];\n        if(!strcmp(key,"--keep-temperature")){o.keep_temp=1;continue;}\n'),
    # 2. fixed temperature in the training loop
    ("static void train_net(Net *n,Data *d,Data *val,const char *save,const char *export_path,int stop,int every) {",
     "static float g_fixed_temp=0.f;  /* > 0: --keep-temperature */\n"
     "static void train_net(Net *n,Data *d,Data *val,const char *save,const char *export_path,int stop,int every) {"),
    ("        n->temperature=1.f-.8f*fminf(1.f,progress/.8f);",
     "        n->temperature=g_fixed_temp>0.f?g_fixed_temp:1.f-.8f*fminf(1.f,progress/.8f);"),
    # 3. partial load + keep temperature at load time
    ("            else{reset_optimizer(n);n->c.stage=stage;n->rng.s=o.seed;}",
     "            else{\n"
     "                float t0=n->temperature;\n"
     "                if(o.load_part){\n"
     "                    Net *f=net_new(n->c,1,1,o.seed);f->vocab=n->vocab;memset(&n->vocab,0,sizeof n->vocab);\n"
     "                    if(!strcmp(o.load_part,\"codes\")){\n"
     "                        memcpy(f->emb.z,n->emb.z,n->emb.n*4);memcpy(f->bias.z,n->bias.z,n->bias.n*4);\n"
     "                    }else for(int i=0;i<n->c.blocks;i++){\n"
     "                        Layer *a=f->l+i,*b=n->l+i;\n"
     "                        memcpy(a->ch,b->ch,(size_t)a->O*a->R*4);memcpy(a->off,b->off,(size_t)a->O*a->R*4);\n"
     "                        memcpy(a->p.z,b->p.z,a->p.n*4);\n"
     "                    }\n"
     "                    f->temperature=n->temperature;net_free(n);n=f;\n"
     "                    fprintf(stderr,\"load-part %s: rest freshly initialized (seed %llu)\\n\",o.load_part,(unsigned long long)o.seed);\n"
     "                }\n"
     "                reset_optimizer(n);n->c.stage=stage;n->rng.s=o.seed;\n"
     "                if(o.keep_temp){g_fixed_temp=fmaxf(.05f,t0);n->temperature=g_fixed_temp;\n"
     "                    fprintf(stderr,\"keep-temperature: %.4f (checkpoint) instead of the 1.0 -> 0.2 anneal\\n\",g_fixed_temp);}\n"
     "            }"),
]


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    h = hashlib.sha256(s.encode()).hexdigest()[:16]
    if h != SHA16:
        sys.exit(f"transfer_patch: unexpected logic_text.c (sha256 {h}..., want {SHA16}...)")
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit(f"transfer_patch: anchor not found exactly once: {old[:70]!r}")
        s = s.replace(old, new)
    open(out, "w").write(s)
    print(f"transfer_patch: {len(EDITS)} edits -> {out}")


if __name__ == "__main__":
    main()
