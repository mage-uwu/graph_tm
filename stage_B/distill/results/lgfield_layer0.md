Layer-0 routing without q.k or softmax (attention-only BitNet; layer 1 unchanged softmax; rest frozen float).
Retrieval in every row: value dims as 7-threshold thermometers, output bit = fraction of selection weight whose value
bit is set, quantized with Q = 8 integer comparisons (a quantized weighted mean). Full WikiText-103 validation.

selection                                                           gain kept
teacher attention weights, round(16 p), weighted MEDIAN (majority)     ~0.84  (1024-window subset)
teacher attention weights, round(16 p), quantized weighted mean Q=8     0.964 (subset; ceiling of this value coding)
learned gate trees on q/k sign + top bits, 8 trees/offset, 3000 steps   0.620 (subset; plateau, 32 trees: 0.615)
byte-pair x offset table E[h, qbyte, kbyte, delta], untrained           0.946 (full)
same table, fine-tuned (600 steps, hard forward + recovery loss)        0.978 (full)  acc 0.4273 vs 0.4353

Layer 1 (lg1.py; layer 0 exact softmax; first 1024 validation windows). Routing from lookups, BLT-style:
routing code per token and head (256 k-means codes of the teacher's layer-1 q and k), score = code-pair table.
assignment                                         untrained   pair table fine-tuned (300 steps)
codes from hashed byte n-grams (n <= 8, backoff)     0.613       0.648
codes from the true q / k (VQ bound)                 0.854       0.880
BLT hash n-gram embeddings (lg1b.py: byte emb + sum over n = 3..8 of learned tables, 65,536 buckets per n, q and k
sides; stage 1 regress teacher q/k, stage 2 routing loss). Layer 0 exact; first 1024 validation windows.
  lr .05, 1500 + 500 steps                    0.894
  lr .01, 1000 + 2000 steps                   0.980 (0.981 best at step 2000)
  lr .01, routing loss only (no stage 1)      0.681 (stuck: zero-initialised tables give flat scores)
