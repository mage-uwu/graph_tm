#!/bin/sh
# Reproduce Stage 0 + Stage 1 end to end.
set -e
cd "$(dirname "$0")"
make -C stage1 -s
mkdir -p data
(cd stage0 && python3 datasets.py noisy_xor ../data/xor \
           && python3 datasets.py sequence  ../data/seq \
           && python3 datasets.py masked    ../data/mask \
           && python3 datasets.py masked    ../data/maskbits --bits \
           && python3 datasets.py masked    ../data/codes256 --codes 256)
python3 tests/test_parity.py
G=./stage1/gtm
T=${THREADS:-$(nproc)}
echo; echo "== NoisyXOR (repo demo settings, 40 clauses)"
$G train --data data/xor.train.gtmd --test data/xor.test.gtmd --clauses 40 --T 100 --s 1.0 --depth 2 --max-inc 4 --epochs 5 --threads $T
echo; echo "== Sequence (repo demo settings)"
$G train --data data/seq.train.gtmd --test data/seq.test.gtmd --clauses 60 --T 600 --s 1.0 --depth 3 --max-inc 4 --epochs 5 --threads $T
echo; echo "== Masked token, depth-1 control (should be ~12.5% = chance)"
$G train --data data/mask.train.gtmd --test data/mask.test.gtmd --clauses 400 --T 300 --s 5.0 --depth 1 --epochs 2 --threads $T
echo; echo "== Masked token, depth 2"
$G train --data data/mask.train.gtmd --test data/mask.test.gtmd --clauses 400 --T 300 --s 5.0 --depth 2 --epochs 4 --threads $T --save data/mask.model.gtmm
echo; echo "== 256-bit codes per token (teacher-LSH shape), ORIGINAL feedback (q=1, rho=1): collapses"
$G train --data data/codes256.train.gtmd --clauses 400 --T 300 --s 5.0 --depth 2 --epochs 2 --threads $T --save /tmp/c256a.gtmm | grep epoch
$G score --model /tmp/c256a.gtmm --data data/codes256.test.gtmd --threads $T --sums /tmp/c256a.i32 >/dev/null
(cd stage0 && python3 eval_codes.py ../data/codes256.test.gtmd /tmp/c256a.i32)
echo; echo "== 256-bit codes, symmetric q + decoupled feedback (--q -1 --rho 2/256)"
$G train --data data/codes256.train.gtmd --clauses 400 --T 300 --s 5.0 --depth 2 --epochs 5 --q -1 --rho 0.0078 --threads $T --save /tmp/c256b.gtmm | grep epoch
$G score --model /tmp/c256b.gtmm --data data/codes256.test.gtmd --threads $T --sums /tmp/c256b.i32 >/dev/null
(cd stage0 && python3 eval_codes.py ../data/codes256.test.gtmd /tmp/c256b.i32)

echo; echo "== 4096 clauses: default senders saturate (fill 100%, chance) vs --senders pos"
$G train --data data/mask.train.gtmd --test data/mask.test.gtmd --clauses 4096 --T 1000 --s 5.0 --depth 2 --steps 20000 --threads $T --save /tmp/s_all.gtmm | grep steps
$G score --model /tmp/s_all.gtmm --data data/mask.test.gtmd --threads $T | cut -c1-40
$G train --data data/mask.train.gtmd --clauses 4096 --T 1000 --s 5.0 --depth 2 --senders pos --steps 20000 --threads $T --save /tmp/s_pos.gtmm | grep steps
$G score --model /tmp/s_pos.gtmm --data data/mask.test.gtmd --threads $T | cut -c1-40

echo; echo "== Inference throughput"
$G score --model data/mask.model.gtmm --data data/mask.test.gtmd --threads $T --reps 3
