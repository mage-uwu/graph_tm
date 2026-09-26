/* fastgen: fastlae with one model's blocks compiled to straight-line C for one context length (T <= 64 x words;
 * other lengths fall back to the interpreter). Same commands and outputs as fastlae; verify still checks every score.
 *   fastlae gen MODEL.lth DATA.ids SEQ model.inc
 *   gcc -O2 -march=native -std=gnu11 -fopenmp -I<logic-bert>/src -DGEN_INC='"model.inc"' fastgen.c -lm -o fastgen
 * The binary only runs the model it was generated from (fastgen checks the layer shapes, not the
 * truth tables: run verify once after building). */
#define FASTLAE_GEN
#include "fastlae.c"
#define V v1
#define GEN_L(n) n##_1
#include GEN_INC
#undef V
#undef GEN_L
#define V v2
#define GEN_L(n) n##_2
#include GEN_INC
#undef V
#undef GEN_L
#define V v4
#define GEN_L(n) n##_4
#include GEN_INC
#undef V
#undef GEN_L
#define V v8
#define GEN_L(n) n##_8
#include GEN_INC
#undef V
#undef GEN_L
static void gen_blocks_1(v1 *a, v1 *b, const v1 *valid) { blocks_1(a, b, valid); }
static void gen_blocks_2(v2 *a, v2 *b, const v2 *valid) { blocks_2(a, b, valid); }
static void gen_blocks_4(v4 *a, v4 *b, const v4 *valid) { blocks_4(a, b, valid); }
static void gen_blocks_8(v8 *a, v8 *b, const v8 *valid) { blocks_8(a, b, valid); }
static int gen_nwords(void) { return GEN_NWORDS; }
