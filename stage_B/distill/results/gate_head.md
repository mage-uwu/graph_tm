# Can a sparse gate circuit replace layer 1's W_o + the head? (gate_head.py, sparse_readout.py)

Site: logits = head(q16(gf o (x1 + W_o1 oq1 s))), inputs as bits: xq1 and oq1 in positive/negative bit-planes
(2 x 1,280) + an 8-bit relative-scale thermometer. Everything else is the exact LGN engine. 1,024 validation windows;
gain = share of the float BitNet's gain over unigram. Criterion: >= 0.97 at a gate count whose bit-sliced cost
undercuts the batched VNNI site (~0.7 us/token).

| replacement | gain |
|---|---|
| exact site (ceiling) | 0.984 |
| exact linear readout from the same bits (control) | 0.983 |
| DDLGN circuit 16k gates (k = 32, 1 hidden layer 8192), wiring from weights, 600 steps | 0.302 (peak 0.323) |
| same, random wiring | 0.321 (nothing learned for the first 150 steps) |
| DDLGN circuit 65k gates (k = 64, hidden 32768, 16384), wiring from weights, 600 steps | 0.462 (peak 0.559 at step 450) |

Magnitude pruning of the exact readout per class (no retraining), terms per class of 256 (x units + o units):
256 0.983 | 128 0.951 | 64 0.845 | 32 0.567 | 16 0.115 | 8 -0.40.
Head only (W_o1 exact), of 128: 128 0.983 | 64 0.951 | 32 0.816 | 16 0.490.

Result: negative. The bit encoding costs nothing, but the readout is dense (distributed residual code: each class
needs most of the dimensions), and unweighted group-sum popcounts of 2-input gates are the wrong primitive for a
dense weighted sum; a VNNI instruction does 64 weighted terms at once. The circuits plateau within a few hundred
steps, far below 0.97.
