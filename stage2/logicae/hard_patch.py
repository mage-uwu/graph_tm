"""Patch logic-bert's logic_text.c (after transfer_patch.py, before fasttrain_patch.py) with an opt-in
straight-through training mode:

  --hard-forward   the forward pass is the hardened network: codes binarized (code > .5, i.e. z > 0) and
                   every gate output binarized (> .5), so training sees exactly what the deployed .lth
                   computes. The backward pass is unchanged code: it recomputes the tree from the (now
                   binary) node values, so each gate's gradient is the multilinear derivative at binary
                   inputs with the float corner values (d out/d a = q[2]-q[0] or q[3]-q[1], one-hot corner
                   gradients), and the thresholds pass gradients straight through. Float master weights
                   and Adam are kept as before.
Without the flag the patched binary is unchanged (the branch is on a global that stays 0).
usage: python3 hard_patch.py IN/logic_text.c OUT/logic_text.c
"""
import sys

EDITS = [
    ("static void tree_forward(const Layer *l,const float *x,const int32_t *base,const float *p,",
     "static int g_hard_fwd=0;  /* --hard-forward: straight-through (hard_patch.py) */\n"
     "static void tree_forward(const Layer *l,const float *x,const int32_t *base,const float *p,"),
    ("            out[e]=q8?nearbyintf(r*255.f)/255.f:r;",
     "            out[e]=g_hard_fwd?(r>.5f?1.f:0.f):q8?nearbyintf(r*255.f)/255.f:r;"),
    ("            w->a[0][((size_t)t*C+k)*w->B+b]=n->c.q8?nearbyintf(a*255.f)/255.f:a;",
     "            w->a[0][((size_t)t*C+k)*w->B+b]=g_hard_fwd?(a>.5f?1.f:0.f):n->c.q8?nearbyintf(a*255.f)/255.f:a;"),
    ('        if(!strcmp(key,"--keep-temperature")){o.keep_temp=1;continue;}\n',
     '        if(!strcmp(key,"--keep-temperature")){o.keep_temp=1;continue;}\n'
     '        if(!strcmp(key,"--hard-forward")){g_hard_fwd=1;continue;}\n'),
]


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    for a, b in EDITS:
        if s.count(a) != 1:
            sys.exit(f"hard_patch: anchor not found exactly once: {a[:70]!r}")
        s = s.replace(a, b)
    open(out, "w").write(s)
    print(f"hard_patch: {len(EDITS)} edits -> {out}")


if __name__ == "__main__":
    main()
