#!/bin/sh
# Thread-scaling benchmark. Run on the target box (32/64 vCPU):
#
#   ./bench_scaling.sh             # threads 1,2,4,... up to nproc
#   ./bench_scaling.sh 64          # cap
#   GTM_PIN=0 ./bench_scaling.sh   # without pinning (pinned by default here)
#   CLAUSES="4096 16384" ./bench_scaling.sh
#
# Two-socket boxes: run under `numactl --interleave=all` (automaton state is allocated by the
# main thread, so without interleaving every page lands on one node).
#
# Per clause count: warm a model up (using all threads; training is thread-count invariant, so
# the checkpoint is identical to a 1-thread warm-up), then from that SAME checkpoint train a
# fixed number of steps at each thread count (speedup + state hashes must be identical), then
# score a held-out set (example-parallel inference). Uses --senders pos: with default GraphTM
# senders, bundles saturate at these sizes and the model cannot learn (see README).
#
# Expectation: inference ~linear. Training is clause-sharded with 3 barriers per message layer
# + 1 per example (dissemination barrier, log2(P) rounds); it needs enough clauses per thread
# before barrier latency stops dominating. That crossover is the number we want.
set -e
cd "$(dirname "$0")"
G=./stage1/gtm
MAXT=${1:-$(nproc)}
CLAUSES=${CLAUSES:-"1024 4096 16384"}
export GTM_PIN=${GTM_PIN:-1}
[ -x $G ] || make -C stage1 -s
[ -f data/mask.train.gtmd ] || (cd stage0 && python3 datasets.py masked ../data/mask)
[ -f data/maskbench.test.gtmd ] || (cd stage0 && python3 datasets.py masked ../data/maskbench --n 3000 --seed 7 >/dev/null)

THREADS=""
t=1
while [ $t -le $MAXT ]; do THREADS="$THREADS $t"; t=$((t * 2)); done
[ $((t / 2)) -ne $MAXT ] && THREADS="$THREADS $MAXT"

echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')  nproc=$(nproc)  pin=$GTM_PIN"
for C in $CLAUSES; do
    T=$((C / 4)); [ $T -lt 300 ] && T=300
    WARM=$((20000 * 1024 / C)); [ $WARM -lt 3000 ] && WARM=3000
    STEPS=$((8000 * 1024 / C)); [ $STEPS -lt 1000 ] && STEPS=1000
    CK=/tmp/bench_warm_$C.gtmm
    echo
    echo "== clauses=$C  T=$T  (warm-up $WARM steps on $MAXT threads, then $STEPS timed steps; masked task, depth 2, --senders pos)"
    $G train --data data/mask.train.gtmd --clauses $C --T $T --s 5.0 --depth 2 --senders pos \
        --steps $WARM --threads $MAXT --save $CK | awk '/steps/ {print "   warm-up:", $2, "steps", $5}' | tr -d '()'
    $G score --model $CK --data data/maskbench.test.gtmd --threads $MAXT | awk '/score/ {printf "   warm model: test acc %s", $5} /fill/ {print "  message fill", $3}' | tr -d ','
    printf "%8s %12s %9s %14s %9s  %s\n" threads train_ex/s speedup infer_graph/s speedup state_hash
    BASE_T=""; BASE_I=""; HASH0=""
    for t in $THREADS; do
        OUT=$($G train --data data/mask.train.gtmd --model $CK --steps $STEPS --threads $t --save /tmp/bench_$C.gtmm)
        EXS=$(echo "$OUT" | awk '/steps/ {gsub(/[()]/, "", $5); print $5}')
        HASH=$(echo "$OUT" | awk '/state hash/ {print $3}')
        GPS=$($G score --model $CK --data data/maskbench.test.gtmd --threads $t --reps 3 | awk '/score/ {for (i = 1; i <= NF; i++) if ($i == "graphs/s,") {gsub(/[(]/, "", $(i-1)); print $(i-1)}}')
        [ -z "$BASE_T" ] && BASE_T=$EXS && BASE_I=$GPS && HASH0=$HASH
        FLAG=""; [ "$HASH" != "$HASH0" ] && FLAG="  <-- MISMATCH (bug: results depend on thread count)"
        printf "%8s %12s %9s %14s %9s  %s%s\n" $t $EXS $(awk "BEGIN {printf \"%.2fx\", $EXS / $BASE_T}") \
            $GPS $(awk "BEGIN {printf \"%.2fx\", $GPS / $BASE_I}") $HASH "$FLAG"
    done
done
