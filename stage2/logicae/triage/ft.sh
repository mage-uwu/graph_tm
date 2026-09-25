#!/bin/bash
# small SST-2 fine-tunes: scratch vs pretrained variants; train-fit + dev + test hard accuracy
# setup: $S/lt/lt = logic_text.c + transfer_patch.py + fasttrain_patch.py (c11); $S/pt_revived.ltc = diag revive pt.ltc $D/sst2_tr3k.ids 64 OUT;
# $D = data/diag (sst2_{train,dev500,tr500,test}.ids: GLUE SST-2, bert WordPiece ids). 4 threads, ~1 s/step, 600 steps ~11 min per run.
S=${S:-/tmp/lae_triage}; D=${D:-$(cd "$(dirname "$0")/../../.." && pwd)/data/diag}; PT=${PT:-$(cd "$(dirname "$0")/../../.." && pwd)/models/logicae/pt.ltc}
cd $S/lt; export OMP_WAIT_POLICY=passive; STEPS=${STEPS:-600}
ARCH="--vocab-size 30522 --code-bits 128 --width 1024 --blocks 16 --kernel 5 --cycle 4"
run(){ name=$1; shift
  ./lt train --threads 4 --data $D/sst2_train.ids --val $D/sst2_dev500.ids --format ids --seq 64 --batch 32 --steps $STEPS \
     --eval-every 100 --seed 17 --save $name.ltc --export $name.lth "$@" 2> $name.log
  r=""; for s in tr500 dev500 test; do a=$(./lt eval --load $name.lth --data $D/sst2_$s.ids --format ids --seq 64 --threads 4 2>&1 | grep -o '"hard_accuracy":[0-9.]*\|"accuracy":[0-9.]*' | head -1 | cut -d: -f2); r="$r \"$s\":$a,"; done
  curve=$(grep -o '"step":[0-9]*\|"val_hard_accuracy":[0-9.]*' $name.log | cut -d: -f2 | paste -sd' ')
  echo "{\"run\":\"$name\",$r \"curve\":\"$curve\"}" | tee -a $S/ft_results.jsonl
}
for v in ${VARIANTS:-scratch pt keep codes gates revive}; do case $v in
  scratch) run scratch $ARCH;;
  pt)      run pt --load $PT;;
  keep)    run keep --load $PT --keep-temperature;;
  codes)   run codes --load $PT --load-part codes --keep-temperature;;
  gates)   run gates --load $PT --load-part gates --keep-temperature;;
  revive)  run revive --load $S/pt_revived.ltc --keep-temperature;;
  revive_t1) run revive_t1 --load $S/pt_revived.ltc;;
esac; done
