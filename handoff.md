# Handoff: lgtm — logic Graph Tsetlin Machine (CPU-native)

Read this file fully before changing anything. `README.md` is the user-facing summary; this
document is the working context: what exists, why it is the way it is, the rules that keep it
correct, and what to do next.

## 1. Goal and context

The long-term goal is a **logic-based alternative to Jev** (TypeSafe AI's "System One" model:
unstructured state in, typed decisions with calibrated probabilities out, no text generation).
Jev's core is BERT-like; the idea here is to build the encoder from logic instead
(Tsetlin machines / differentiable logic gate networks), so the runtime is CPU-native and very
cheap.

Plan, as agreed with the user:

1. Represent text as **graphs** (tokens as nodes, typed edges) and use the **Graph Tsetlin
   Machine** (GraphTM, Granmo et al., [cair/GraphTsetlinMachine](https://github.com/cair/GraphTsetlinMachine))
   as the encoder.
2. Pretrain it with an **MLM-style objective**: mask a node, mark it with a MASK symbol, and
   predict the hidden token. Targets are **teacher LSH bits**: sign bits of random hyperplane
   projections of a strong transformer's (e.g. ModernBERT) embedding of the token. Every target
   bit is a binary output, which is what a coalesced TM handles natively.
3. The set of clauses that fire becomes a binary representation; small TM/DDLGN heads on top
   answer Jev-style typed questions.

Stages so far:

- **Stage 0 (ground truth):** a numpy oracle transcribed from the CUDA kernels, a shared
  counter-based RNG spec, and ground-truth tasks.
- **Stage 1 (runtime):** a C11/pthreads/AVX-512 engine that is **bit-exact with the oracle,
  including training**, at any thread count, then heavily optimized.
- **Stage 2 (in progress):** scaling measurements on the user's 32/64 vCPU box, then real text.

The user's style: they want things fast, measured, and honest. Report negative results.
Measure before optimizing. Don't claim what wasn't verified.

## 2. Current state

Everything below is done and verified unless marked otherwise.

| Area | State |
|---|---|
| Oracle vs C training parity | Bit-exact on 9 configurations (see §5) |
| Thread-count invariance | Identical results at 1/2/3/5/8/16 threads and for the scalar (non-AVX-512) build |
| Sanitizers | ThreadSanitizer: no races. ASan + UBSan: clean |
| CUDA cross-check | **Untested** (no GPU was available). `stage0/cross_check_cuda.py` is written; its data and bit-plane conversions are verified against the repo's real `graphs.py` |
| Multi-core scaling | **Unmeasured** (dev sandbox had 1 vCPU). `bench_scaling.sh` is ready; the user will run it |
| Real text | Not started |

Single-core numbers (Xeon @ 2.1 GHz with AVX-512), from `bench/bench.sh`:

| Benchmark | Inference | Training |
|---|---|---|
| masked task, 400 clauses, depth 2 | ~219k graphs/s | ~49k ex/s |
| 256-bit code targets, 400 clauses | ~163k graphs/s | ~19k ex/s |
| 4,096 clauses (early training) | ~82k graphs/s | ~1.2k ex/s |

## 3. Repository map

```
handoff.md                 this file
README.md                  user-facing summary, findings, performance
run_all.sh                 build + generate data + parity + train every task (~2 min, 1 core)
bench_scaling.sh           threads x clauses sweep (for the multi-core box)
bench/bench.sh             fingerprinted single-core benchmark (creates its own checkpoints)
bench/summary.txt          numbers from the last optimization round
stage0/gtmcore.py          THE SPEC: counter-based RNG + file formats (.gtmd, .gtmm)
stage0/gtm_oracle.py       slow, obviously-correct oracle (plain integer automata)
stage0/datasets.py         task generators: noisy_xor, sequence, masked [--bits | --codes N]
stage0/eval_codes.py       per-bit + nearest-code token accuracy for code targets
stage0/cross_check_cuda.py bridge to the original CUDA implementation (GPU needed)
stage1/gtm.h, gtm.c        the engine
stage1/main.c              CLI: init / train / score / hash
stage1/Makefile            gtm (native), gtm_portable (no AVX-512), gtm_asan
tests/test_parity.py       oracle <-> C parity suite
data/                      generated datasets (empty in the package; run_all.sh fills it)
```

Requirements: gcc (AVX-512 for the fast path; the portable build works without it), Python 3
with numpy. `cross_check_cuda.py` additionally needs pycuda, the original package, and a GPU.

## 4. Commands

```sh
make -C stage1                                  # builds stage1/gtm and stage1/gtm_portable
python3 tests/test_parity.py [--quick]          # needs data/: run ./run_all.sh once first
bench/bench.sh                                  # timings + result fingerprints
./run_all.sh                                    # everything, end to end
./bench_scaling.sh [max_threads]                # scaling sweep; CLAUSES="..." to override sizes

# CLI
stage1/gtm train --data D.gtmd [--test T.gtmd] [config] [--epochs E | --steps N] [--threads P] [--save M.gtmm]
stage1/gtm score --model M.gtmm --data D.gtmd [--threads P] [--sums F.i32] [--bits F.u64] [--reps R]
stage1/gtm init  --data D.gtmd --out M.gtmm [config]
stage1/gtm hash  --model M.gtmm
# config: --clauses --T --s (x or x,y,z per layer) --q (-1 = O-1) --rho --senders all|pos
#         --depth --msg-size --msg-bits --max-inc --state-bits --boost --neg --seed
```

Environment variables: `GTM_DENSE=1` forces the reference evaluator and bundler (A/B testing);
`GTM_LUT8=0|1` forces the byte bundling LUT off/on; `GTM_PIN=1` pins threads to CPUs.

## 5. Non-negotiable invariants

These are what make the project trustworthy. Break one and every later result is suspect.

1. **The oracle defines the semantics.** Any change to what training computes (RNG, feedback
   rules, message passing, file format) goes into `stage0/gtmcore.py` / `gtm_oracle.py` first,
   then the C engine, and `tests/test_parity.py` must pass (full, not just `--quick`) before
   the change counts as done.
2. **Performance work must not change results.** Before optimizing, run `bench/bench.sh` and
   save the output. After, inference `sums` fingerprints and training `hash` values must be
   identical. If a speedup requires a spec change, it is a spec change (rule 1), and you say
   so. Record fresh baselines on each machine: fingerprints depend on generated data and
   checkpoints, which may differ across numpy versions.
3. **Thread-count invariance.** Training results must not depend on thread count or ISA. The
   parity suite checks 1/2/3/5/8/16 threads plus the portable build; `bench_scaling.sh`
   checks it again on the real box and flags mismatches.
4. **Counter-based randomness only.** Every random draw is a pure function of
   `(seed, tag, step, clause, output, layer, index)`. Never introduce sequential RNG state,
   and never make a draw depend on thread scheduling. This property is also what makes
   skipping unnecessary draws exact.
5. **Defaults reproduce the CUDA original's semantics.** Improvements that change behavior
   (`--q -1`, `--rho`, `--senders pos`) are opt-in flags, documented in the README.
6. **Memory safety and races.** After touching threading or shared arrays, rerun
   ThreadSanitizer: `gcc -O1 -g -march=native -std=c11 -fsanitize=thread -o /tmp/gtm_tsan stage1/gtm.c stage1/main.c -lpthread -lm`,
   then train with 3–5 threads. Same for `make -C stage1 gtm_asan`.

Parity suite configurations (`tests/test_parity.py`): `xor`, `seq` (repo demos, s=1),
`maskbits` (hashed feedback, per-layer s), `weird` (3 node/edge types, boost off, no negative
clauses, q≠1, 5-bit states, include cap, C not a multiple of 64), `wide_out` (100 outputs),
`decoupled` (ρ<1, depth 3), `senders` (positive senders, depth 3), `noncomp` (features not of
the form `[x | ¬x]`), `multi_out`. Plus init parity, inference parity on random sparse models,
and indexed-vs-dense agreement on every trained model.

## 6. The spec in brief

Full text lives in the `gtmcore.py` docstring. Relative to the CUDA original:

- Counter-based RNG (splitmix64 keys, `lowbias32` per-lane hashing) instead of curand.
- Update selection: one Bernoulli(p_k) per (clause, output), p_k = err_k/2T · q'_k. Same law
  as CUDA's two-draw test.
- Decoupled feedback: automaton feedback pairs are the subset whose same draw is also below
  p_k·ρ. ρ = 1 is the original.
- Draws are 16-bit (two per 32-bit hash), so probabilities are quantized to 1/65536.
- Weight rule written as `w += target` (algebraically identical).
- `senders` model flag: 0 = all firing clauses send messages (GraphTM), 1 = only clauses with
  a positive layer-0 literal.
- `.gtmm` is v2 (adds ρ as a double after `s[]`; the header's last u32 is `senders`). v1 loads.

GraphTM semantics to keep in mind: node features are literal vectors `[x | ¬x]`; a clause is
one deep conjunction across layers (`out_d = out_{d-1} ∧ C_d(msg_d)`); messages at layer d come
from `out_{d-1}`; each firing clause sends its hypervector bits rotated by edge type, OR-bundled
at the destination; graph output is an OR over nodes; a clause only fires on nodes whose type
equals `clause % n_node_types`. Learning picks one random true node per clause, and every
layer's automata update against that node's input.

## 7. Engine walkthrough (`stage1/gtm.c`)

Data layout:

- **Automata:** bit-sliced, 64 per word, **plane-major** per clause and layer:
  `ta[l][c*B*W + b*W + w]`. One carry step covers the whole layer (`layer_inc` / `layer_dec`,
  specialized at compile time for W = 1/2/4/8/16).
- **Include masks:** mirrored MSB plane, `inc[l][c*W + w]` (compact, for evaluation).
- **Clause outputs:** node-major `out[node][clause_word]`, so clause-sharded threads write
  disjoint 64-clause words.
- **Weights:** output-major and padded, `w[k*Cw*64 + c]`. Inference builds a clause-major mirror.
- **Literal index** (`lit_index`, per layer): rows P/N (clauses including x_i / ¬x_i), witness
  rows WP/WN (clauses whose lowest positive/negative literal is i), and NP0/NN0 (clauses with no
  positive/negative literals). Maintained incrementally by `refresh_words` on include flips.

Key functions, in pipeline order:

| Function | Role |
|---|---|
| `eval_layer` → `eval_layer_indexed` | Per node: if the positive half is sparse, OR rows N[i] and WP[i] over set bits; if dense, OR P[i] and WN[i] over zero bits; resolve NP0/NN0 clauses directly and full-check witness candidates. Falls back per node to dense checks when that is cheaper, and to `eval_layer_dense` for non-complementary data or `GTM_DENSE=1` |
| `bundle_sources` | OR of fired clauses' hypervectors per node via a byte (or nibble) LUT, register-accumulated, stops at saturation. With `senders`, masks fired clauses by `~NP0` of layer 0 |
| `build_messages` + `rotate_or` | Rotate each source bundle by edge type and OR into the destination; rotation distributes over OR, so it's once per edge, not per clause |
| `clause_votes` / `clause_votes_rows` | Graph-level clause outputs and class sums (masked vector adds / clause-major rows) |
| `output_feedback_plan` + `sel_word` | Per output: target, both thresholds, key; 64 clauses of selection per hash |
| `update_weights` | 16-lane masked `w += target` with the INT_MAX guard and no-negative clamp |
| `update_clause` | Node selection (`select_bit` + `pdep` on a transposed node mask), then Type I (grow or forget) and Type II per layer, then `refresh_words` for flipped include bits |
| `train_worker` / `gtm_fit` | Clause-sharded training; see threading below |
| `score_worker` / `gtm_score_ex` | Example-parallel inference with a per-thread layer-0 memo (`memo_t`); reports message fill |

Threading model for training: thread t owns clause words from `clause_range` (512-clause
blocks when there are enough, else single words). Per example: eval layer 0 on own words →
for each message layer: barrier, bundle own node slice, barrier, build messages for own node
slice, barrier, eval own words → partial class sums into a double-buffered slot → barrier →
every thread reduces all partial sums redundantly → updates for own clauses. The double buffer
removes the between-example barrier. The barrier is a **dissemination barrier** (log₂P rounds
of single-writer release stores on per-thread cache lines), spinning ~20 µs before yielding.

## 8. Findings (with evidence) and recommended configs

**Many outputs need `--q -1 --rho 2/O`.** On `masked --codes 256`, CUDA defaults predict 99%
ones: per-bit 52.8%, token recovery at chance. Negative targets are scaled by q/(O−1), which is
right for multiclass and crippling for multi-output. Symmetric q alone floods clauses with
~O·err/2T Type I updates per step. Subsampling weights too plateaus at ~65% token. Decoupled
(weights see everything, automata a ρ-fraction): 99.5% per-bit, 99.7% token.

**Large clause pools need `--senders pos`.** About half of all clauses are "NOT symbol X"
clauses that fire on nearly every node, so message bundles saturate. At 4,096 clauses fill is
100%, messages are constant, and the model sits at chance even after 40k steps, at any T and
even with 2048-bit messages. With positive senders: 100% test accuracy, fill 37%, 3.3x faster
training. At 400 clauses it also helps (99.1% vs 97.5% at 12k steps). Fill around 94% is fine
(the zero bits carry the signal); only total saturation is fatal. `score`/`train` print fill and
warn above 99%.

**Bigger messages are not the fix.** At 400 clauses, 256 → 512 → 1024 message bits gave 97.5% →
87.9% → 71.7% at 12k steps: more message literals for every clause to learn, and slower steps.

**Depth-1 leak control.** Graph output is an OR over nodes, so a depth-1 model can exploit the
bag of node symbols. An early version of the masked task leaked (73% at depth 1, chance is
12.5%). Every new MLM-style task needs a depth-1 control that sits at chance.

**Epoch-to-epoch accuracy oscillation** (e.g. the sequence demo dipping to ~89% for an epoch)
is reproduced exactly by the oracle: it's the algorithm, not the port.

Recommended starting point for pretraining-style runs:
`--depth 2 --msg-size 256 --senders pos --q -1 --rho 2/O --s 5 --T ~C/4`.

## 9. Pitfalls learned the hard way

- **gprof with `-fno-inline` lies** about tiny inlined helpers (it blamed the hash function
  heavily). Use cycle counters (`__rdtsc`) around phases in the real `-O3` build. The real cost
  in training turned out to be Type I "forget" feedback on clauses that didn't fire.
- **Measure ideas before building them.** Two were dropped: skipping zero-state automata (only
  0–14% sit at zero) and batching feedback (only ~1 update per clause per step at q=1).
- Early training of large models is partly **memory-bound**: 4,096 clauses = 3 MB of automata
  against 2 MB L2, and ~70% of clauses update per step early on.
- Inference with the memo is only exact because layer-0 outputs are a pure function of
  (features, node type) with a fixed model. Never enable it during training.
- Anything read across threads in training (out words, message buffers, bundles, NP0 for
  senders) must be written before a barrier and read after one. TSan verifies this.
- The dev sandbox's `/bin/sh` is dash (no `<(...)`), and its output mount strips executable
  bits; neither applies on a normal machine.

## 10. Next work, in priority order

1. **Interpret the scaling run.** The user will run `bench_scaling.sh` on 32/64 vCPUs and paste
   the output. Find the training crossover (clauses per thread where speedup flattens) and
   check inference is near-linear. If training scales poorly below ~16k clauses, candidate
   fixes: fewer barriers per layer (e.g. fuse bundling into message building), NUMA-aware
   allocation of automaton state (first-touch by the owning thread), larger clause blocks per
   thread. Acceptance: state hashes identical across all thread counts; speedups reported.
2. **Real text pipeline.** Tokenizer → graphs (chain with Left/Right edges plus offset-typed
   skip edges, e.g. ±2, ±4), one masked node per graph marked with a MASK symbol, targets =
   sign bits of random hyperplane projections (256–1024) of a teacher's token embedding.
   Deliverable: a `datasets.py` task (or a separate converter writing `.gtmd`) plus an
   evaluation that reports per-bit accuracy and teacher-neighbor retrieval (Hamming distance
   of predicted bits vs cosine similarity of teacher embeddings). Acceptance: the depth-1
   control sits near its floor, and depth 2 clearly beats it.
3. **CUDA cross-check** on any GPU machine: `cd stage0 && python3 cross_check_cuda.py --data ../data/seq.test.gtmd --model <C-trained .gtmm>`.
   Note the CUDA classes hard-code negative clauses and don't know about ρ/senders, so use a
   model trained with defaults.
4. **Downstream heads**: transform graphs into clause-output bits (`gtm score --bits`) and train
   small heads for Jev-style typed questions; add calibration (temperature / isotonic on vote
   sums).
5. Lower priority: a training-time memo within a graph (repeated tokens in a sentence),
   sparse candidate pruning at layer ≥1, NUMA work if the box shows it matters.

## 11. Glossary

- **Clause:** a conjunction of literals; here one deep conjunction across message layers.
- **Literal:** a feature bit x_i or its negation ¬x_i. Layer inputs are `[x | ¬x]`.
- **TA (Tsetlin automaton):** per-literal counter; MSB = include decision.
- **Type I feedback:** reinforces a clause toward the selected node's pattern (grow), or, if
  the clause didn't fire, decays all its automata with probability 1/s (forget).
- **Type II feedback:** includes currently-excluded literals that are false in the input, to
  make the clause reject it.
- **Coalesced:** one shared clause pool with per-output signed weights.
- **T:** vote clamp / target margin. **s:** specificity (feedback probability 1/s).
- **q:** negative-target feedback scale. **ρ:** automaton feedback budget.
- **Message fill:** fraction of message bits set at a node; near 100% means no information.
