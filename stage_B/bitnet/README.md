# stage_B / bitnet: a small BitNet b1.58 transformer that trains on the CPU

`bitnet.c` is one file: model, training, built-in toy tasks, and a selftest. It is built to train toy tasks in
seconds; it is not meant for full-size runs.

```
gcc -O3 -march=native -fopenmp bitnet.c -lm -o bitnet
./bitnet selftest
./bitnet train --task copy --arch full --steps 200
./bitnet train --task text --data input.txt --arch full --steps 2000
```

## Model

- **Layout:** float token embedding + learned positions → residual sublayers (pre-norm RMSNorm) → RMSNorm → float LM head.
- **Attention sublayer:** BitLinear qkv (d→3d) → causal softmax attention (float scores) → BitLinear o (d→d).
- **MLP sublayer:** BitLinear up (d→hidden) → ReLU² → BitLinear down (hidden→d).
- **Three ways to train (`--arch`):**
  - `full`: layers × (attention, mlp).
  - `attn`: layers × attention, no MLP layers.
  - `mlp`: layers × mlp, no attention, so no mixing across positions.
- **BitLinear (b1.58):**
  - Weights are ternary {-1, 0, +1} × γ, where γ = mean |W| over the matrix.
  - Inputs are int8 per token (absmax → 127).
  - The product is an integer matrix product, computed as float arithmetic on integer values. That is exact, so
    the training forward pass is the deployed integer model.
- **Training:** float latent weights; AdamW (β 0.9/0.95, weight decay 0.1 on matrices); warmup then cosine to
  10%; gradient clipping at 1. The straight-through estimator is used for both quantizers.
- `--quant 0` turns the quantizers off, giving a plain float transformer (used by the gradient check).

## Speed

- **GEMM:** one register-blocked kernel handles every matrix product.
  - BLAS-style packing: A into 4-row panels, B into zero-padded 32-column panels.
  - A 4×32 accumulator tile using gcc vector extensions; OpenMP over tiles.
  - The weight gradients (Xᵀ·dY) read A transposed in place, with no transpose pass.
  - 175–310 GFLOPS on the training shapes (4-core Xeon, AVX-512).
- **Quantization:** activations are quantized with explicit vectors; weights are quantized once per step.
- **Attention:** parallel over (sequence, head), on contiguous Q / Kᵀ / Vᵀ copies.

## Determinism

- Randomness is counter-based.
- Every parallel loop writes disjoint outputs, and every sum has a fixed order.
- Training is bit-identical at any thread count (`selftest` checks 1 vs all threads).

## Vetting (4-core Xeon @ 2.1 GHz; d 128, 2 layers, 4 heads, hidden 512, seq 64, batch 32)

**Selftest:**
- Float-mode gradient check against central differences: max relative error 4e-3 (attn, mlp, full).
- Weights bit-identical at 1 and 4 threads.
- ASan + UBSan: clean on ragged sizes (d 48, 3 heads, hidden 80, seq 13, vocab 17/19/256) and on the selftest.
- ThreadSanitizer can't check this binary: stock libgomp is not instrumented, so every report is a write before
  an OpenMP implicit barrier read after it. The thread-count bit-identity check stands in for it.

**Copy task** (32 random symbols, separator, the same 32 again; chance 2.773 nats):

| arch | params | ms/step | solved (val acc 1.0) |
|---|---|---|---|
| attn | 144k | 13 | step 75, 1.1 s |
| mlp | 275k | 18 | never: 2.772 at step 200 (no token mixing, as expected) |
| full | 406k | 33 | step 75, 2.7 s |

**Byte-level text** (WikiText-103 validation text, 1.1 MB; 90/10 split). Validation loss in nats per byte:

| arch | 600 steps | time |
|---|---|---|
| attn | 2.284 | 10 s |
| mlp | 2.476 (bigram plateau) | 13 s |
| full | 2.065 | 22 s |

- **Longer run:** full, 2,000 steps (74 s): 2.296 → 2.130 → 1.996 → 1.888, smooth, gradient norm 1.0–1.5.
- **Learning-rate sweep** (full, 300 steps; BitNet / float):

| lr | BitNet | float |
|---|---|---|
| 1e-3 | 2.430 | 2.341 |
| 3e-3 (default) | 2.324 | 2.194 |
| 1e-2 | 2.502 | 2.488 |
| 3e-2 | 2.555 | 2.597 |

  No run diverged or produced a non-finite value.
