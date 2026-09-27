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
  lgn.c, + batched matmuls (--batch 1)      4900        1300   (interleaved 3x: before 6712-7424 / 1733-1881, after 4801-5036 / 1233-1398)
per-token breakdown (1 thread, VNNI, per token): layer inputs 1.85 us, attention 2.92 us, W_o + residual 1.21 us, head 1.59 us

batched matmuls: the 64 tokens of a window go through each matmul site at once (q/k/v of layer 1, W_o after each
layer's attention - x is not read by that layer's attention -, head), register-blocked 4 tokens x 4 column blocks of 16
(16 independent VNNI accumulators instead of one dependent chain per column block). Same integers: verify 0 mismatches
(VNNI / general gates / --table 0 / build without VNNI), full-validation logit_hash 58860df24d5bb502 unchanged on every
path and at 1 and 4 threads (lgn_tr.bin: 32776cb5589bc7f0 unchanged; its .ref predates the int16 head, so verify fails
on it before and after).
per-token breakdown after batching (1 thread, timer build):
  site                         kernel     bookkeeping (quantize, norm, rescale, residual)
  layer-1 q/k/v (128->384)     262 ns     831 ns
  W_o, both layers (128->128)  160 ns     118 ns
  head (128->256, int16)       310 ns     256 ns
  attention (gate match + retrieval)      2966 ns
matmul floor: 0.73 us/token of 4.9 (the head kernel runs at about one VNNI instruction per cycle).
