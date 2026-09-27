Layer-0 routing without q.k or softmax (attention-only BitNet; layer 1 unchanged softmax; rest frozen float).
Retrieval in every row: value dims as 7-threshold thermometers, output bit = fraction of selection weight whose value
bit is set, quantized with Q = 8 integer comparisons (a quantized weighted mean). Full WikiText-103 validation.

selection                                                           gain kept
teacher attention weights, round(16 p), weighted MEDIAN (majority)     ~0.84  (1024-window subset)
teacher attention weights, round(16 p), quantized weighted mean Q=8     0.964 (subset; ceiling of this value coding)
learned gate trees on q/k sign + top bits, 8 trees/offset, 3000 steps   0.620 (subset; plateau, 32 trees: 0.615)
byte-pair x offset table E[h, qbyte, kbyte, delta], untrained           0.946 (full)
same table, fine-tuned (600 steps, hard forward + recovery loss)        0.978 (full)  acc 0.4273 vs 0.4353
