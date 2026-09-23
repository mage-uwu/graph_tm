# lgtm — logic Graph Tsetlin Machine, CPU-native

A logic-based System-1 encoder: a CPU-native rebuild of
[cair/GraphTsetlinMachine](https://github.com/cair/GraphTsetlinMachine), with a numpy oracle
that the C engine matches bit for bit, including during training.

```
./run_all.sh          # build, generate tasks, parity suite, train all tasks (~2 min, 1 core)
THREADS=32 ./run_all.sh
./bench_scaling.sh    # thread-scaling sweep: run this on the 32/64 vCPU box
```

**Recommended settings beyond the GraphTM defaults:**

| Situation | Flags | Why |
|---|---|---|
| Many output bits (teacher-LSH targets) | `--q -1 --rho 2/O` | defaults collapse to chance at 256 outputs |
| Large clause pools (≳1k clauses) | `--senders pos` | default bundles saturate; a 4k-clause model can't learn at all |

Defaults reproduce the CUDA original's semantics.

## Layout

```
handoff.md                 working context for the next developer / agent (start here)
CLAUDE.md                  auto-loaded by Claude Code: rules + pointer to handoff.md
stage0/gtmcore.py          shared spec: counter-based RNG + .gtmd/.gtmm file formats
stage0/gtm_oracle.py       numpy oracle, transcribed from the CUDA kernels
stage0/datasets.py         ground-truth tasks (2 repo demos + masked-token MLM target, --codes N)
stage0/eval_codes.py       per-bit / nearest-code token accuracy for code targets
stage0/cross_check_cuda.py bridge to the ORIGINAL CUDA code (needs a GPU)
stage1/gtm.{h,c}           the engine (C11, pthreads, AVX-512 with scalar fallback)
stage1/main.c              CLI: init / train / score / hash
tests/test_parity.py       oracle <-> C parity suite
bench/bench.sh             fingerprinted single-core benchmark (every speedup must keep results identical)
bench_scaling.sh           threads x clauses sweep with a results-don't-depend-on-threads check
```

## Verification

| Claim | Status |
|---|---|
| Oracle reproduces the repo's published behavior (NoisyXOR, sequence demo) | verified |
| C inference == oracle on random sparse models (1–3 node types, 1–4 edge types, self-loops, multi-edges, depth 1–3) | verified: sums + per-clause bits, via both the indexed and the reference evaluator |
| C **training** == oracle, bit-exact automata + weights over multi-epoch runs | verified on 9 configs: s=1 and hashed feedback, per-layer s, boost off, no negative clauses, q≠1, 5-bit states, include caps, C not a multiple of 64, 80–100 outputs, decoupled feedback, positive senders, non-`[x∣¬x]` features |
| Training results independent of thread count (1/2/3/5/8/16) and ISA (AVX-512 vs scalar) | verified |
| ThreadSanitizer: no data races (training + inference, pinning on) | verified |
| ASan + UBSan clean | verified |
| C inference == original CUDA `tm.score()` | **untested** (no GPU): `cross_check_cuda.py`; its graph and bit-plane conversions are verified against the repo's real `graphs.py` |

The oracle stores automata as plain integers, so it independently checks the engine's
bit-sliced arithmetic. Every random draw is a pure function of
`(seed, tag, step, clause, output, layer, index)`, which is what makes training bit-exact and
thread-invariant, and lets the engine skip draws that can't matter without shifting others.

## Spec (v5), relative to the CUDA original

All of these are mirrored in the oracle and pass parity.

- **Counter-based RNG** instead of curand. CUDA *training* can only be compared statistically.
- **Update selection:** one Bernoulli(p_k) draw per (clause, output), with
  p_k = err_k/2T · q'_k. Same law as CUDA's two-draw test, since both factors depend only on the
  output; it vectorizes 64 clauses per output.
- **Decoupled feedback (ρ):** automaton feedback is the subset of weight-selected pairs whose
  draw is also below p_k·ρ (same uniform, second threshold). ρ = 1 is the original.
- **16-bit coin flips**, two per 32-bit hash: probabilities quantized to 1/65536.
- **`--senders pos`** (opt-in): only clauses with a positive layer-0 literal send messages.
- Weight rule written as `w += target` (algebraically identical to CUDA's sign-dependent rule).

## Ground-truth tasks

| Task | What it tests | Result |
|---|---|---|
| `noisy_xor` (repo demo) | 2 nodes, depth 2 | 100% test (40 clauses) |
| `sequence` (repo demo, published settings) | run-length counting, depth 3 | 100% test after 1 epoch |
| `masked` | predict a masked token; one neighbor is ambiguous, so clauses must AND both neighbors' messages | depth-1 control at chance (12.7%), depth 2 → 99–100% |
| `masked --codes 256` | same, with a random 256-bit code per token (worst-case teacher-LSH shape) | defaults: chance; `--q -1 --rho 2/256`: 99.5% per-bit, 99.7% token recovery |

The depth-1 control matters: graph output is an OR over all nodes, so if the masked token can
be inferred from the *bag* of node symbols (e.g. it's the one missing), depth 1 solves it.
A first version of `masked` leaked this way (73% at depth 1).

## Findings that change how to configure this

**1. Many outputs.** On `masked --codes 256` the CUDA defaults collapse:

| Feedback scheme | Per-bit | Token |
|---|---|---|
| Original (q=1, ρ=1) | 52.8% | chance (predicts 99% ones) |
| Symmetric q, budget on weights *and* automata | ~79% | ~65% plateau |
| **Symmetric q + decoupled budget** (`--q -1 --rho 2/O`) | **99.5%** | **99.7%** |

The CUDA code scales negative-target feedback by q/(O−1), right for multiclass but
crippling for multi-output. With symmetric q, each clause gets ~O·err/2T Type I updates per
step and memorizes single nodes. Subsampling everything starves the per-(clause, output)
weights. So weights see every selected pair; automata see a ρ-fraction.

**2. Message saturation.** A node's message is an OR of 2 hypervector bits per firing clause.
About half of all clauses are pure "NOT symbol" clauses that fire on almost every node, so
bundles fill up. The engine now reports this (`score` and `train` print message fill).

| Clauses | Senders | Msg size | Test acc | Train ex/s | Fill |
|---|---|---|---|---|---|
| 400 | all | 256 | 97.5% @12k steps | 15.5k | 94% |
| 400 | all | 1024 | 71.7% @12k steps | 5.9k | 71% |
| 400 | **pos** | 256 | **99.1%** @12k steps | 18.2k | 9% |
| 4096 | all | 256 or 2048 | **chance** @40k steps | 1.3k | 100% |
| 4096 | **pos** | 256 | **100%** @40k steps | 4.2k | 37% |

94% fill is fine: the zero bits carry the signal, and bigger messages just add literals to learn.
Total saturation is fatal: messages become constant and depth 2 degenerates to depth 1. Growing
the message doesn't rescue 4k clauses (99.5% fill at 2048). Letting only clauses with a positive
literal send fixes it, and trains 3x faster because informative messages let clauses specialize.

## Performance (single core: Xeon @ 2.1 GHz, AVX-512; sandbox has 1 vCPU)

From `bench/bench.sh` (fixed checkpoints; inference outputs identical before and after):

| Benchmark | Start of optimization round | Now | Speedup |
|---|---|---|---|
| Inference, 400 clauses | 22k graphs/s | 219k | 9.9x |
| Inference, 256-bit codes | 21k | 163k | 7.7x |
| Inference, 4,096 clauses | 1.6k | 82k | 50x |
| Training, 400 clauses | 15.8k ex/s | 49k | 3.1x |
| Training, 256-bit codes | 5.1k | 18.7k | 3.7x |
| Training, 4,096 clauses (early) | 556 | 1.2k | 2.1x |

What did it, in order of impact:

- **Inverted literal index.** Literal vectors are always `[x | ¬x]`. For a sparse node
  (2 of 128 bits set), a clause is false if it includes ¬x_i for a set bit (row OR), true if it
  has no positive literals, else its lowest positive literal must be set (a "witness"), then
  full check. Dense nodes (saturated messages) use the mirror image over zero bits. Cost is
  min(|set|, |zero|) row ORs per node instead of one check per clause. Maintained incrementally
  on include flips; each thread touches only its own clause words.
- **Inference memo.** A node's layer-0 outputs and bundled message depend only on its features:
  a per-thread direct-mapped cache (full key, no false hits) makes repeated tokens a copy.
- **Bundling.** Byte-indexed LUT of pre-ORed clause hypervectors (8 clauses per lookup, when it
  fits in ~1 MB; nibble LUT otherwise), accumulated in registers, stopping at saturation.
- **Output-major weights.** `w += target` is clause-independent, so weight updates are 16-lane
  masked adds; clause votes are masked adds (training) or a clause-major mirror (inference).
- **Plane-major automata.** State planes stored `[plane][word]` so each carry step covers the
  whole layer in one vector op; include flips found by comparing one plane with its mirror.
- Node selection via a per-clause node bitmask + `pdep`; witness updates only when needed.

Measured, not assumed: gprof with inlining disabled blamed tiny hash helpers; cycle counters in
the real build showed the actual cost was Type I "forget" feedback on non-firing clauses. Two
ideas were dropped after measurement (skipping zero-state automata: only 0–14% sit at zero).
Large-model early training is partly memory-bound (4k clauses = 3 MB of automata vs 2 MB L2).

## Threads

- **Inference:** example-parallel, no synchronization. Should scale ~linearly.
- **Training:** clause-sharded. Threads write disjoint 64-clause words of shared arrays; ordering
  comes only from barriers (3 per message layer + 1 per example). The barrier is a dissemination
  barrier: log₂P rounds of single-writer release stores, no shared hot cache line. Scaling needs
  enough clauses per thread to amortize barrier latency; `bench_scaling.sh` finds that crossover.
- `GTM_PIN=1` pins threads. On two-socket machines use `numactl --interleave=all`.
- Threads > cores works (spin ~20 µs, then yield) but costs a context switch per barrier.

## Limitations

- `msg_size` a multiple of 64; H, msg_size ≤ 8192; depth ≤ 8; state bits ≤ 16; outputs ≤ 32768.
- Datasets are loaded whole into memory.
- The epoch-to-epoch accuracy oscillation in the repo demos is reproduced exactly by the oracle:
  it's the algorithm, not the port.
- The CUDA bridge's GPU path is untested.

## Next

1. **Run `bench_scaling.sh` on the 32/64 vCPU box.** The training crossover sets the minimum
   model size worth running there.
2. **Real text.** Token graphs (chain + offset-typed skip edges), a marked MASK node, teacher
   LSH bits (`--q -1 --rho 2/O --senders pos`), plus the depth-1 leak control.
3. Training-time memo within a graph (repeated tokens in a sentence) and NUMA-aware allocation
   of automaton state, if the box shows them mattering.
