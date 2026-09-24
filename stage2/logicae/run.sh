#!/bin/bash
# LogicAE run: logic-bert (DDLGN) at our scale on our task, on a pod. The logic-bert source is
# not in this repo: after data prep the script waits until $LB/src/logic_text.c exists (copy it
# onto the pod), then builds and runs. Prints a HB line every 30 s and
# RESULT {json} lines; everything also goes to $W/logicae.log.
#   masked-token pretraining on WikiText-103 +-8-token windows (bert WordPiece, V=30522),
#   4.40M params (128-bit word codes 3.91M + 16 blocks x width 1024 gate trees + bias),
#   time-budgeted steps; then our 20k-window test ranking (same windows / candidates as the
#   GraphTM and bert-tiny), then SST-2 and QNLI fine-tuning, pretrained vs scratch.
# Checkpoints: pt.ltc (latest) and pt.ltc.best (best validation CE) are overwritten at every
# evaluation; the heartbeat also keeps the last 2 step-numbered copies (ckpt_<step>.ltc).
# At the end, logicae/export.py serves the models and log over HTTP on :8888 for EXPORT_HOURS
# (sha256 manifest, token in the "LAE-EXPORT serving" log line; 0 = no export).
# Knobs (env): PT_HOURS (2.5), PT_BATCH (256), MLM_CAP (512), FT_STEPS (2000), THREADS (nproc), EXPORT_HOURS (3)
set -uo pipefail
cd "$(dirname "$0")/.."
LB=${LB:-/root/logic-bert}; W=${W:-/root/lae}; mkdir -p $W
T=${THREADS:-$(nproc)}; PT_HOURS=${PT_HOURS:-2.5}; PT_BATCH=${PT_BATCH:-256}; MLM_CAP=${MLM_CAP:-512}
FT_STEPS=${FT_STEPS:-2000}; EXPORT_HOURS=${EXPORT_HOURS:-3}; EXPORT_PY=$PWD/logicae/export.py
ARCH="--vocab-size 30522 --code-bits 128 --width 1024 --blocks 16 --kernel 5 --cycle 4"
exec > >(tee -a $W/logicae.log) 2>&1
phase(){ echo "$*" > $W/phase; echo "== [$(date -u +%H:%M:%S)] $*"; }
( while sleep 30; do  # heartbeat: phase, last training log line, disk, container memory
    mem=$(cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null | head -1 | awk '{printf "%.1fG", $1/2^30}')
    lim=$(cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null | head -1 | awk '{if ($1=="max" || $1>2^50) print "?"; else printf "%.0fG", $1/2^30}')
    if [ -s $W/pt.ltc ] && [ $W/pt.ltc -nt $W/ckpt.stamp ]; then  # rotating snapshots of the latest checkpoint (keep 2)
      s=$(grep -o '"step":[0-9]*' $W/current.log | tail -1 | cut -d: -f2)
      cp $W/pt.ltc $W/ckpt_${s:-x}.ltc.tmp && mv $W/ckpt_${s:-x}.ltc.tmp $W/ckpt_${s:-x}.ltc && touch $W/ckpt.stamp
      ls -t $W/ckpt_*.ltc 2>/dev/null | tail -n +3 | xargs -r rm -f
    fi
    last=$(tail -n 50 $W/current.log 2>/dev/null | grep '^{"step"' | tail -1 | cut -c1-200)
    echo "HB [$(date -u +%H:%M:%S)] $(cat $W/phase 2>/dev/null) | disk free $(df -h $W | awk 'NR==2{print $4}') | mem $mem/$lim | $last"
  done ) &
HB=$!; trap 'kill $HB 2>/dev/null' EXIT
phase "setup"
python3 -c "import tokenizers, pyarrow, safetensors, huggingface_hub" 2>/dev/null ||
  python3 -m pip install -q --break-system-packages tokenizers pyarrow safetensors huggingface_hub
phase "data prep"
python3 pretrain_data.py prep
[ -f ../data/stage2/test20k.gtmd.npz ] || python3 pretrain_data.py evalset --split test --n 20000 --out ../data/stage2/test20k.gtmd
[ -f $W/pre_train.ids ] || python3 logicae/prep.py $W --n ${PREP_N:-3000000}
phase "waiting for the logic-bert source at $LB/src/logic_text.c (copy it onto the pod)"
until [ -s $LB/src/logic_text.c ]; do sleep 20; done; sleep 5
phase "build"
gcc -O3 -march=native -std=c11 -fopenmp -Wno-unknown-pragmas $LB/src/logic_text.c -lm -o $W/lt
gcc -O3 -march=native -std=c11 -fopenmp -Wno-unknown-pragmas -I$LB/src logicae/mlmrank.c -lm -o $W/mlmrank
OMP_NUM_THREADS=$T $W/lt selftest | tail -1
cd $W
P="--threads $T --data pre_train.ids --val pre_val.ids --format ids $ARCH --seq 17 --batch $PT_BATCH --mlm-targets $MLM_CAP --seed 17"
phase "throughput probe"
rm -f probe.ltc; ./lt pretrain $P --steps 1000 --stop-after 4 --eval-every 4 --save probe.ltc 2> current.log
sps=$(grep -o '"compute_seconds":[0-9.]*' current.log | tail -1 | cut -d: -f2); sps=$(awk -v s=$sps 'BEGIN{print s/4}')
STEPS=$(awk -v h=$PT_HOURS -v s=$sps 'BEGIN{printf "%d", h*3600/s}'); EV=$(( STEPS/20 > 0 ? STEPS/20 : 1 ))
echo "RESULT {\"probe_seconds_per_step\":$sps,\"pretrain_steps\":$STEPS,\"windows\":$((STEPS*PT_BATCH)),\"masked_targets\":$((STEPS*MLM_CAP))}"
phase "pretrain $STEPS steps"
./lt pretrain $P --steps $STEPS --eval-every $EV --save pt.ltc 2> current.log
grep -o '"step":[0-9]*\|"val_mlm_ce":[0-9.]*' current.log | paste - - | sed 's/^/curve /'
for m in pt.ltc pt.ltc.best; do
  phase "masked-token test ranking ($m)"
  echo "RESULT {\"model\":\"logicae-$m\",\"test20k\":$(./mlmrank $m test_win.ids test_true.txt candidates.txt 8 $T)}"
done
ft(){ # name task seq extra...
  local n=$1 task=$2 seq=$3; shift 3
  phase "fine-tune $n"
  ./lt train --threads $T --data ${task}_train.ids --val ${task}_dev.ids --format ids --seq $seq --batch 32 \
     --steps $FT_STEPS --eval-every 250 --seed 17 --save $n.ltc --export $n.lth "$@" 2> current.log
  acc=$(./lt eval --load $n.lth --data ${task}_test.ids --format ids --seq $seq --batch 64 | grep -o '"hard_accuracy":[0-9.]*' | cut -d: -f2)
  best=$(grep -o '"val_hard_accuracy":[0-9.]*' current.log | cut -d: -f2 | sort -n | tail -1)
  echo "RESULT {\"model\":\"$n\",\"task\":\"$task\",\"test_final\":$acc,\"dev_best\":$best}"
}
ft sst2_pretrained sst2 64 --load pt.ltc
ft sst2_scratch sst2 64 $ARCH
ft qnli_pretrained qnli 128 --load pt.ltc
ft qnli_scratch qnli 128 $ARCH
phase "done"
[ "$EXPORT_HOURS" != 0 ] && python3 $EXPORT_PY --dir $W --hours $EXPORT_HOURS
