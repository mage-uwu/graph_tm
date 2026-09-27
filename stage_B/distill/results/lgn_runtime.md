LGN-attention BitNet runtime (lgn.c), attention-only model, 4-core Xeon @ 2.1 GHz (AVX-512 VNNI)

accuracy (full WikiText-103 validation, every 64-byte window): CE 1.974832, acc 0.4295 -> 98.30% of the float
BitNet's gain over unigram (float BitNet: CE 1.954176, acc 0.4353). Python reference == C: 0 mismatched integer
logits (64 windows x 64 x 256), on every path (VNNI / scalar / bit-plane popcount matmuls, layer-0 table on / off,
pass-through / general gate path).

speed, full 64-byte windows (prefill), ns per token:
  engine                                  1 thread    4 threads
  bitnet.c float (eval, batch 32)          10910        3464
  lgn.c, scalar int64 (first version)      65069       15937
  lgn.c, head/mixing restructured          17878        4462
  lgn.c, + AVX-512 VNNI matmuls / head      7099        1981
per-token breakdown (1 thread, VNNI): layer inputs 1.85 us, attention 2.92 us, W_o + residual 1.21 us, head 1.59 us
