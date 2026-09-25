# fastlae: fast LogicAE inference and training (same model, same results)

LogicAE (logic-bert DDLGN) hardened models are pure boolean circuits: 16 blocks x 1024 outputs, each
a depth-3 tree of 2-input gates over 8 wired leaves (channel, offset x dilation), then an OR pool
over position pairs and a vote head (popcounts). The reference engine (`logic_text.c`
`hard_predict`) packs 64 *texts* per 64-bit word, so one text costs as much as 64. Training
parallelizes its backward pass over batch chunks of 16, so batch 32 keeps 2 threads busy.

| piece | what changes | check |
|---|---|---|
| `fastlae.c` (interpreter) | one text per word: bit t = token position t, one word per channel (T <= 64; NW words up to 512). A conv tap is a shift. Gates branch-free via a 16-entry mask table. `--lanes 1/2/4/8` runs texts side by side in vector registers (GCC vector extensions) | `verify`: every score == `hard_predict`, all lane widths |
| `fastgen.c` + `fastlae gen` | the network compiled to straight-line C: each gate is its one native bitwise op, each leaf a shift (T <= 64; longer inputs fall back to the interpreter) | same `verify` |
| `fasttrain_patch.py` | `backward_layer` in two phases: parallel over outputs (leaf gradients to a buffer), then parallel over input channels (gather in the original (o, t, k) order); `forward_layer` over (t, o); parallel elementwise embedding loops | saved checkpoints byte-identical to the unpatched engine at any thread count (supervised and MLM) |

## Local results (4-core Xeon 2.1 GHz, 1 thread, 64-token inputs, run 1's `sst2_scratch.lth`)

| texts per call | reference | fastlae | fastgen | bert-tiny (PyTorch) |
|---|---|---|---|---|
| 1 | 31.4 ms | 0.86 ms | **0.055 ms** | ~1.4 ms |
| 64 (8 lanes) | 29.3 ms | ~7.7 ms | **0.80 ms** (80k texts/s) | ~30 ms |
| 512 (8 lanes) | 274 ms | ~61 ms | **6.9 ms** (74k texts/s) | ~370 ms |

Training (seq 48, batch 32): 0.85 s/step -> 0.39 s/step at 4 threads, byte-identical checkpoints.
`bench.sh` measures all of it at 16 threads on a pod, with bert-tiny on the same CPU.

Build: `gcc -O3 -march=native -std=gnu11 -fopenmp -I<logic-bert>/src fastlae.c -lm -o fastlae`;
`fastlae gen MODEL.lth DATA.ids 64 model.inc` then
`gcc -O2 -march=native -std=gnu11 -fopenmp -I<logic-bert>/src -DGEN_INC='"model.inc"' fastgen.c -lm -o fastgen`
(about 1.5 min for a 16 x 1024 model); `python3 fasttrain_patch.py logic_text.c lt_fast.c`.
