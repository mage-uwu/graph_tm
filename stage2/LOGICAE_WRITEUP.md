# LogicAE: status writeup (2026-09-25)

**Goal:** a CPU-native, pure-logic text classifier that can take Jev/Laya-style decisions
(choice, score, yes/no) about as well as BERT, and much faster.

**Where it stands:** at 4.4M parameters LogicAE is within 2-4 points of bert-tiny on 5 of 6 decision
tasks. The compiled engine runs one decision in 0.05-0.09 ms, 15-25x faster than bert-tiny on one CPU
core, from a 1.7 MB integer-only model. Two hurdles remain: fine-grained emotion (-19) and
pretraining, which still hurts.

## Performance

Test accuracy. bert-tiny is fine-tuned per task; LogicAE is trained from scratch per task, one yes/no
model per option. Raw results are in `stage2/system1/results/`.

| task (type) | bert-tiny | best LogicAE | gap | LogicAE configuration |
|---|---|---|---|---|
| jailbreak (yes/no) | 0.970 | 0.943 | -2.7 | one yes/no model |
| agnews (4 topics) | 0.906 | 0.883 | -2.3 | one yes/no model per topic |
| sst2 (sentiment) | 0.803 | 0.780 | -2.3 | 1500 steps |
| sst5 (5-level score) | 0.430 | 0.398 | -3.2 | one yes/no model per level |
| QNLI (question vs sentence) | 0.750 | 0.710 | -4.0 | + token-match bit + majority pooling |
| emotion (6-way) | 0.896 | 0.708 | -18.8 | + token-match bit |

Calibration after temperature fitting is comparable to bert-tiny (ECE 0.01-0.10 vs 0.02-0.08).

| speed, 64-token text, 1 CPU thread | per decision | throughput |
|---|---|---|
| bert-tiny (PyTorch fp32) | ~1.4 ms | ~2.4k texts/s |
| LogicAE, original engine | 31-46 ms | 1.4-2.9k texts/s at batch 64 |
| LogicAE, fastgen (compiled, same scores) | **0.05-0.09 ms** | **35k texts/s** (317k texts/s at 12 threads) |

## Key improvements, in order of impact

1. **Compiled inference engine (`stage2/fastlae`), 500x faster than the original.**
   - One text's token positions are packed into each 64-bit word, so convolution taps become bit shifts.
   - The network is compiled to straight-line C: each gate is one CPU instruction, with SIMD lanes across texts.
   - Scores are verified identical to the original engine for every model and input length tested.
2. **One yes/no model per option ("one-vs-rest") instead of reading text + option jointly.**
   - The joint form collapsed to a near-constant answer (emotion 0.29-0.34).
   - One-vs-rest: emotion 0.66-0.71, agnews 0.88.
3. **Token-match input bit** ("this token also occurs in the other [SEP] segment").
   - One bit per position; it gives the logic gates the exact-match signal attention gets for free.
   - QNLI gap closed from -17 to -5; majority pooling brings it to -4.
4. **Training about 10x faster, with bit-identical results.**
   - The backward pass was capped at 2 threads; it now runs parallel, with byte-identical checkpoints.
   - Spinning OpenMP threads were starving shared 8-core pods; fixed with `OMP_WAIT_POLICY=passive`.
   - One LogicAE model: ~64 min -> ~2-8 min.
5. **Negative results, recorded so they are not re-run:**
   - OR-pooled global channels saturate: a channel fires at ~49% of positions, so "fired anywhere"
     is 1 for 97.7% of texts x channels.
   - Keeping the code temperature after pretraining helps only a little (+5 pts on emotion; still below scratch).

## Remaining hurdles

1. **Pretraining does not transfer.** It hurts on every task (-2 to -24 vs scratch). Likely causes:
   - 17-token pretraining windows vs 48-96-token tasks.
   - Small pretraining data: 3.2M windows.
   - Unstable fine-tuning: some per-option models collapse, e.g. val 0.16.
2. **Fine-grained emotion (-19).** Six overlapping classes from short texts. Local patterns and
   exact-match overlap do not cover it; this needs a richer word representation, which is the job of
   pretraining.
3. **K models per K-way question.** One-vs-rest costs K trainings and K inferences; the output head is binary-only.
4. **Pretraining speed.** Masked-word pretraining is dominated by the 30k-way output softmax
   (4.4 s/step), which the training speedups do not touch.
5. **Scale is untested.** Every result is at 16 blocks x 1024 wide, single seed, 1000-1500 steps
   per model; differences under ~1.5 points are within noise.

## Remaining work, highest expected value first

1. **Native K-way vote head**: one model per question instead of K; cheaper, and probably more
   accurate on emotion and sst5. Small engine change.
2. **Pretraining that transfers**:
   - 64-128-token windows, with the match bit and majority pooling in.
   - Sampled or two-level softmax for 5-10x faster pretraining steps.
   - More data.
   - Then re-test pretrained vs scratch fine-tuning.
3. **Longer training and seeds**: 3000+ steps; emotion was still climbing (0.659 -> 0.694 from 1000 to 1500 steps). 3 seeds for error bars.
4. **Width/depth sweep** on the fast engine (now cheap), plus the 2x-scale run.
5. **Ship the fast path**:
   - fastgen in the Jev/System-1 harness for latency numbers.
   - `lfeat`/shootout on the new engine.
   - An end-to-end CPU serving demo (batching, tokenizer cost included).
6. **Housekeeping**:
   - The logicAE pod's shootout / 2x-scale results are still on the pod (logs unreadable from here).
   - ENGINEER_SQUAD and FAST_AE are idle and can be stopped.

## Where things are

- Engines:
  - `stage2/fastlae/`: `fastlae.c`, `fastgen.c`, `fasttrain_patch.py`, `global_patch.py`.
  - `stage2/logicae/transfer_patch.py`.
- Harness: `stage2/system1/jevft.py`, `jev.sh`; `stage2/fastlae/global.sh`.
- Results:
  - `stage2/system1/results/jev/`, `global_v1/`, `global_v2/`.
  - Tables in `stage2/README.md` and `stage2/fastlae/README.md`.
