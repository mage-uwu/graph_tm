#!/bin/sh
# Fixed benchmark: timings + result fingerprints. Every optimization must keep fingerprints identical.
cd "$(dirname "$0")/.."
G=${G:-./stage1/gtm}
[ -x $G ] || make -C stage1 -s
for t in mask codes256 seq; do
  [ -f data/$t.train.gtmd ] || (cd stage0 && case $t in mask) python3 datasets.py masked ../data/mask;;
    codes256) python3 datasets.py masked ../data/codes256 --codes 256;; seq) python3 datasets.py sequence ../data/seq;; esac >/dev/null)
done
# fixed checkpoints (deterministic, so identical on every machine for the same spec)
[ -f bench/mask400.ckpt ] || $G train --data data/mask.train.gtmd --clauses 400 --T 300 --s 5.0 --depth 2 --epochs 1 --save bench/mask400.ckpt >/dev/null
[ -f bench/seq60.ckpt ]   || $G train --data data/seq.train.gtmd --clauses 60 --T 600 --s 1.0 --depth 3 --max-inc 4 --epochs 1 --save bench/seq60.ckpt >/dev/null
[ -f bench/codes.ckpt ]   || $G train --data data/codes256.train.gtmd --clauses 400 --T 300 --s 5.0 --depth 2 --q -1 --rho 0.0078 --epochs 1 --save bench/codes.ckpt >/dev/null
[ -f bench/mask4k.ckpt ]  || $G train --data data/mask.train.gtmd --clauses 4096 --T 1000 --s 5.0 --depth 2 --steps 6000 --save bench/mask4k.ckpt >/dev/null
tr() { # name data ckpt steps
  $G train --data data/$2 --model bench/$3 --steps $4 --save /tmp/bench_out.gtmm > /tmp/bench_tr.txt
  printf "%-22s %10s ex/s    hash %s\n" "train $1" "$(awk '/steps/ {gsub(/[()]/,"",$5); print $5}' /tmp/bench_tr.txt)" "$(awk '/state hash/ {print $3}' /tmp/bench_tr.txt)"
}
sc() { # name data ckpt
  $G score --model bench/$3 --data data/$2 --reps 3 --sums /tmp/bench_sums.i32 > /tmp/bench_sc.txt
  printf "%-22s %10s graphs/s  sums %s\n" "infer $1" "$(awk '/^score/ {for(i=1;i<=NF;i++) if($i=="graphs/s,"){gsub(/[(]/,"",$(i-1)); print $(i-1)}}' /tmp/bench_sc.txt)" "$(md5sum /tmp/bench_sums.i32 | cut -c1-16)"
}
sc seq60      seq.test.gtmd      seq60.ckpt
sc mask400    mask.test.gtmd     mask400.ckpt
sc codes256   codes256.test.gtmd codes.ckpt
sc mask4k     mask.test.gtmd     mask4k.ckpt
tr seq60      seq.train.gtmd     seq60.ckpt   40000
tr mask400    mask.train.gtmd    mask400.ckpt 20000
tr codes256   codes256.train.gtmd codes.ckpt  20000
tr mask4k     mask.train.gtmd    mask4k.ckpt  3000
