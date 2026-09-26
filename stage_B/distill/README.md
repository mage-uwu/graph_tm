# stage_B / distill: the MLP-only BitNet → pure threshold logic

## Result

Teacher: `models/bitnet/bitnet_mlp.bin`, scored on the full WikiText-103 validation text.
- Teacher: CE 2.4595 nats/byte, next-byte accuracy 0.2865.
- Unigram baseline: 3.1696.
- Preservation = the share of the teacher's gain over unigram that is kept. The target, 0.97, means CE ≤ 2.4808.

| student | CE | acc | gain kept | acc kept |
|---|---|---|---|---|
| weights → gates, then hard-forward distillation (`models/bitnet/tln_mlp.*`) | 2.4715 | 0.2846 | **0.983** | **0.993** |
| same network, random init (weights shuffled), same training | 2.4805 | 0.2846 | 0.970 | 0.994 |
| conversion only, no training | 4.7617 | 0.1864 | < 0 | 0.651 |

`tln verify` compared all 16,384 inputs × 256 outputs: 0 mismatched votes between the C engine and the Python integer
model. `tln eval` scores the text directly and reproduces the table.

## Method (`convert.py`, `student.py`)

1. **Hidden unit → threshold gates.**
   - A ReLU unit fires iff its pre-activation is > 0. RMSNorm and int8 input scaling are positive per token, so
     they don't change that sign.
   - A unit's ternary weight row is therefore the wiring and polarity of a threshold gate `[Σ w·x > t]`.
   - One bit per unit keeps only 56% of the gain. L gates per unit, with the same wiring and L thresholds (a
     thermometer of ReLU²), keep 97.2% at L = 4 and 98.9% at L = 8, in the float teacher.
2. **Residual stream → wiring.** Down-projections and the head are linear, so layer 2 and the output read input bits,
   layer-1 gates and layer-2 gates directly. Each gate's contribution is the teacher's mean ReLU² step at that level.
3. **Embedding → bits.** A thermometer code of every token and position embedding dimension (3 levels).
4. **Correction.** Hard forward (the integer network) with straight-through backward, loss KL(teacher ‖ student).
5. **Deployed network.** Everything is integer: ternary wires (A1, A2, C), integer thresholds, and an integer vote
   head. That is popcounts and compares; `tln.c` runs it.

## Did it preserve the MLP or fit its own solution? (`preserve.py`)

| | converted, step 0 | weight init, trained | random init, trained |
|---|---|---|---|
| layer-1 wires identical to the conversion (chance 0.34) | 1.000 | **0.908** | 0.335 |
| layer-2 / output wires kept with their sign | 1.000 / 1.000 | **0.992 / 0.967** | 0.164 / 0.166 |
| layer-1 gate vs the SAME teacher unit-level bit, median phi | 0.50 | **0.43** | 0.00 |
| layer-1 gate vs the best-matching teacher bit, median phi | 0.69 | 0.64 | 0.51 |
| live layer-2 gates (of 2,048) | 105 | 139 | **0** |
| output layer fed the teacher's exact layer-1 bits: gain kept | < 0 | 0.40 | 0.24 |

**Verdict:** the network keeps the MLP's **wiring and layer structure**, and its **units only partly**.

- Training kept 91–99% of the converted wires. The random run's wires are at chance level.
- Each gate still tracks its own teacher unit (phi 0.43 vs 0.00 for random). That fidelity was set by the
  conversion (0.50 at step 0): training eroded it only slightly.
- It is not a drop-in copy of the units. The output layer co-adapted to its own versions of them, so the teacher's
  exact bits transplanted in keep only 40%.
- Random init fits a different, shallower solution that reaches 0.970. Its second layer is completely dead, so it
  is effectively one hidden layer.
- Starting from the weights buys +1.3 points and a working 2-layer circuit. On this toy it is not necessary for the
  target: the model is a byte bigram, and any small network can fit it.

Engine: 26 µs per input on one thread (dense ternary wiring, ~826k wires, popcount interpreter). No speed work has
been done.
