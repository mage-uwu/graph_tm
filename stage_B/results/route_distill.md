# Routing distillation: supervise the gate field with the teacher's attention

Setup: `distill/lgfield.py --frac 8 --steps 1500`, 8 trees × depth 3 per offset over q/k sign and top bits,
window 16, quantized weighted-mean retrieval (Q = 8), 1,024 full-validation windows. `--route-w 1` adds
KL(teacher attention restricted to the window ‖ log_softmax(beta·tree scores)) per head, with a learned beta.
The only difference between each pair is `--route-w`.

| layer | route-w 0 (control) | route-w 1 | final route KL (nats) | reference |
|---|---|---|---|---|
| 0 | **0.620** | 0.526 | 0.415 | byte-pair ROM 0.978 |
| 1 | **0.841** | 0.804 | 0.618 | BLT hash emb 0.980; naive n-gram 0.648 |

(gain = fraction of the teacher's nats/byte gain over unigram that the student keeps; the other layer is exact softmax)

Result: negative. Telling the trees what the routing is makes the result worse on both layers.
The route KL plateaus early (L0 0.48 → 0.41, L1 0.70 → 0.62) and never approaches 0, so this tree field cannot
represent the teacher's pattern. The KL then pulls the trees toward a compromise that fits the
teacher's distribution less badly but serves the next-byte loss worse than the solution the trees find alone.
The bottleneck is the capacity of the tree field over these bits, not the supervision signal. Same q/k bits through
the plane-pair gate match (`gatefield.py`, pass-through = q8·k8) keep 0.994.

Logs: `lgfield_rd_l{0,1}.log` (route-w 1), `lgfield_rd0_l{0,1}.log` (control).
