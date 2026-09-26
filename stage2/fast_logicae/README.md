# fast_logicae: standalone LogicAE (pure-logic text classifier), training + fast inference

**`fastlogic.c` is the whole thing in one file** (nanoGPT style): training, masked-word pretraining,
adaptation, fast bit-sliced inference, the model-to-C compiler, `hardcheck` and `revive`. Its header is
the spec of the best recipe, and `fastlogic help` lists the commands.

```
gcc -O3 -march=native -std=c11 -fopenmp fastlogic.c -lm -o fastlogic
./fastlogic train --data train.ids --val dev.ids --steps 1500 --eval-every 250 --save m.ltc --export m.lth
./fastlogic predict m.lth test.ids 64
```

Defaults in `fastlogic.c` are the best recipe:
- **Architecture:** 16x1024, 128-bit codes, kernel 5, cycle 4; seq 64, batch 32 (64 for pretraining),
  512 masked targets per batch.
- **Training:** hard forward for train *and* pretrain.
- **Adaptation:** `--load` keeps the checkpoint's code temperature.

The split files below (`logic_text.c`, `fastlae.c`, `fastgen.c`, `tools/`) are the pieces it is assembled from
(`build_fastlogic.py`). Checked on assembly:
- `fastlogic.c` checkpoints are byte-identical to the reference engine (soft mode) and to the patched engines
  (hard mode, pretraining, adaptation), at any thread count.
- `verify` passes, and `hardcheck` shows 0 mismatches.

The remainder of this file documents the split tools; `lt` below behaves like `fastlogic`, except that pretraining
there stays soft unless `--hard-forward`.

Self-contained C: `logic_text.c` (training engine, generated from logic-bert + our patches), `fastlae.c` /
`fastgen.c` (inference), `tools/hardcheck.c`. Needs only gcc with OpenMP.

```
make            # lt, fastlae, hardcheck
make check      # selftest
```

## What the model is

- Tokens → learned 128-bit codes.
- 16 blocks × 1024 channels: each channel is a depth-3 tree of 2-input gates (8 wired leaves: channel, offset ×
  dilation; kernel 5; dilations 1, 2, 4, 8 cycle).
- OR pool over position pairs, then a vote head (popcounts); binary output (labels 0/1).
- 4.4M parameters. The deployed `.lth` is integer-only: bit ops plus popcounts.
- Training runs a float relaxation with float master weights and Adam (`.ltc`). `export` hardens to `.lth`.

## Training against the hard network (built in, on by default for `train`)

`lt train` uses straight-through training by default:

- **Forward:** the forward pass is exactly the deployed hard network (binary codes, binary gate outputs, OR and
  majority pooling on bits).
- **Backward:** the gate interpolation derivative at those binary points, from the float masters; the thresholds
  pass gradients straight through.
- **Why:** soft training lets a model learn through sub-threshold soft values that hardening erases. We measured it
  (SST-2, 600 steps, hardened test accuracy):

| start | soft training | hard forward (default) |
|---|---|---|
| scratch | 0.751 | **0.764** |
| pretrained checkpoint | 0.509 (stuck at a constant answer) | **0.669** |

- `--soft-forward` restores the old behaviour.
- `lt pretrain` stays soft unless `--hard-forward` is given. Which is better for pretraining is being tested.
- `hardcheck MODEL.ltc DATA.ids SEQ` confirms, record by record, that the training forward pass gives the same
  integer votes as the exported model (0 mismatches on plain, match-bit and majority-pooling models).

## Data format

- `--format ids`, one record per line: `LABEL id id id ...`.
- `LABEL` is 0/1, or -1 for unlabeled/pretraining.
- Ids are bert-base-uncased WordPiece ids (vocab 30522). Reserved: 0 = PAD, 1 = MASK, 2 = UNK. `[SEP]` = 102 (used by `--match`).
- Records longer than `--seq` are cut; shorter ones are PAD-filled.
- K-way questions: the head is binary, so train one yes/no model per option. K-way logits = the K models' vote
  differences.
- Word-level text: `--format tsv --vocab vocab.txt` (see `lt vocab`).

## Recipes (the settings behind our best numbers)

```
ARCH="--vocab-size 30522 --code-bits 128 --width 1024 --blocks 16 --kernel 5 --cycle 4"

# single-text task from scratch (hard forward is the default)
./lt train $ARCH --data train.ids --val dev.ids --format ids --seq 64 --batch 32 --steps 1500 \
   --eval-every 250 --seed 17 --save m.ltc --export m.lth

# text pairs (question [SEP] passage, record [SEP] option): add the token-match bit + majority pooling
./lt train $ARCH --match 1 --global-every 4 --global-channels 128 --global-mean 1 ...   # QNLI 0.580 -> 0.710

# fine-tune from a checkpoint: keep its code temperature
./lt train --load pretrained.ltc --keep-temperature --data ... --seq 64 --batch 32 --steps 1000 --save ft.ltc --export ft.lth

# evaluate / predict (hardened)
./lt eval --load m.lth --data test.ids --format ids --seq 64
./lt predict --load m.lth --data x.ids --format ids --seq 64      # {"row","label","votes"} per line
```

Notes:
- `--threads` defaults to all cores.
- The binary sets `OMP_WAIT_POLICY=passive` unless you set it: spinning OpenMP threads cost 5–10× on SMT/shared
  machines. On a dedicated machine without SMT you can try `OMP_WAIT_POLICY=active`.
- Training is deterministic: the same checkpoint at any thread count. `--resume` continues bit-exactly.
- `--stop-after N` is an absolute step.
- Reference speed: ~0.4 s/step at 16 threads (8 cores), 16×1024, seq 64, batch 32; ~1 s/step at 4 cores.
- Pair-task options: `--match 1` adds one input bit per position ("this token also occurs in the other [SEP]
  segment"). `--global-mean 1` lets every 4th block see majority-pooled channels (fires at > half the positions).
  OR pooling (`--global-mean 0`) saturates; don't use it.
- `--load-part codes|gates` loads only the pretrained token codes or only the pretrained gates.

## Fast inference

```
./fastlae verify m.lth data.ids 64                 # every score == the reference engine
./fastlae predict m.lth data.ids 64 --lanes 8      # same output as lt predict
make gen MODEL=m.lth DATA=data.ids SEQ=64          # fastgen: the model compiled to straight-line C for SEQ (<= 512)
./fastgen bench m.lth data.ids 64 --batch 1 --lanes 1 --threads 1
```

Measured on a 4-core Xeon, 64-token inputs, 16×1024 model, 0 mismatched scores:

| engine | per text, 1 thread | throughput |
|---|---|---|
| reference (`lt`) | 44.5 ms | 3.9k texts/s (4 threads) |
| `fastgen` | **0.075 ms** | **266k texts/s** (4 threads, 8 lanes) |

bert-tiny on the same CPU: ~1.4 ms per text.

## Known limits (see stage2/logicae/triage/TRIAGE.md)

- Pretraining transfers at 3,000 hard-forward steps (SST-2 0.805 vs scratch 0.772). Longer pretraining lowers
  masked-word CE but transfers worse, as the dead-channel share grows (30% at 3k steps, 42% at 30k).
- The output layer is the tied 128-bit Hamming decoder, kept on purpose: untied softmax heads fitted on the frozen
  30k-step model do no better (5.77 / 6.32 / 5.93 vs tied 5.63 masked-word CE). The quality limit is the trunk.
- Pretraining step time (4 cores): trunk backward 58%, output layer 30%, trunk forward 9%.
- Binary head only (K-way = one model per option).
- `fastgen` is built for one context length band (1-64, 65-128, ..., 449-512 tokens); `fastlae` handles any length up to 512.

## Rebuilding logic_text.c

`python3 make_source.py` regenerates it from `logic-bert.tar.gz` and the patches, in order: transfer,
global, hard, fasttrain, plus the standalone defaults (see its header). With `--soft-forward` and no new options,
checkpoints are byte-identical to the unpatched logic-bert engine.
