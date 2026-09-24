#!/bin/bash
# Stage 2 end to end: real-text pretraining of a ~4.4M-parameter GraphTM, then frozen-feature
# adapters on SST-2 / QNLI / CoNLL-2003 against bert-tiny. Every result line starts with RESULT
# (JSON) so the log is the record. Knobs (env): NS shards of SHARD windows, the model config,
# STEPS_CTL windows for the depth-1 control, SKIP_PRETRAIN=1 to reuse $D/gtm_final.gtmm.
set -eo pipefail
cd "$(dirname "$0")"
P=${THREADS:-$(nproc)}
D=${GTM_STAGE2_DATA:-../data/stage2}
NS=${NS:-10}; SHARD=${SHARD:-1000000}
CFG=${CFG:---clauses 2880 --T 50000 --s 5.0 --depth 3 --msg-size 256 --senders pos --q -1 --rho 0.0078}
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=true
G=../stage1/gtm
mkdir -p $D
stamp() { echo "== [$(date -u +%H:%M:%S)] $*"; }

stamp "setup (threads=$P)"
python3 -c "import torch, transformers, tokenizers, pyarrow, sklearn, safetensors" 2>/dev/null || {
  python3 -m pip install -q --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch ||
  python3 -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch
  python3 -m pip install -q --break-system-packages transformers tokenizers pyarrow scikit-learn safetensors huggingface_hub ||
  python3 -m pip install -q transformers tokenizers pyarrow scikit-learn safetensors huggingface_hub; }
make -C ../stage1 -s
stamp "data prep"
python3 pretrain_data.py prep
[ -f $D/val10k.gtmd ] || python3 pretrain_data.py evalset --split validation --n 10000 --out $D/val10k.gtmd
[ -f $D/test20k.gtmd ] || python3 pretrain_data.py evalset --split test --n 20000 --out $D/test20k.gtmd
python3 eval_mlm.py --evalset $D/test20k.gtmd --baselines
stamp "bert-tiny MLM on the same windows"
python3 bert_baselines.py mlm --evalset $D/test20k.gtmd

if [ -z "$SKIP_PRETRAIN" ]; then
  stamp "pretraining: $NS x $SHARD windows, config: $CFG"
  python3 pretrain_data.py shard --split train --n 1000 --seed 999 --out $D/_init.gtmd >/dev/null
  $G init --data $D/_init.gtmd --out $D/gtm.gtmm $CFG
  for k in $(seq 1 $NS); do
    python3 pretrain_data.py shard --split train --n $SHARD --seed $k --out $D/shard.gtmd
    $G train --data $D/shard.gtmd --model $D/gtm.gtmm --epochs 1 --threads $P --save $D/gtm.gtmm | grep -E "epoch|hash"
    cp $D/gtm.gtmm $D/gtm_ckpt$k.gtmm
    python3 eval_mlm.py --model $D/gtm.gtmm --evalset $D/val10k.gtmd --threads $P --tag "val_after_${k}x${SHARD}"
  done
  cp $D/gtm.gtmm $D/gtm_final.gtmm
  stamp "depth-1 control (bag of context, no message passing): ${STEPS_CTL:=2000000} windows"
  $G init --data $D/_init.gtmd --out $D/gtm_d1.gtmm $(echo $CFG | sed 's/--depth [0-9]*/--depth 1/')
  python3 pretrain_data.py shard --split train --n $STEPS_CTL --seed 500 --out $D/shard.gtmd
  $G train --data $D/shard.gtmd --model $D/gtm_d1.gtmm --epochs 1 --threads $P --save $D/gtm_d1.gtmm | grep -E "epoch|hash"
  rm -f $D/shard.gtmd $D/_init.gtmd
fi
stamp "final masked-token test (20k WikiText-103 test windows)"
python3 eval_mlm.py --model $D/gtm_final.gtmm --evalset $D/test20k.gtmd --threads $P --tag gtm-depth3-test
python3 eval_mlm.py --model $D/gtm_d1.gtmm --evalset $D/test20k.gtmd --threads $P --tag gtm-depth1-control-test

for task in sst2 qnli conll; do
  stamp "adapters: $task"
  python3 adapters.py --task $task --features gtm --model $D/gtm_final.gtmm --tag depth3 --threads $P
  python3 adapters.py --task $task --features bow --tag lexical --threads $P
  python3 adapters.py --task $task --features gtm+bow --model $D/gtm_final.gtmm --tag depth3 --threads $P
  python3 adapters.py --task $task --features gtm --model $D/gtm_d1.gtmm --tag depth1-control --heads logreg --threads $P
  python3 bert_baselines.py task --task $task --mode frozen
  python3 bert_baselines.py task --task $task --mode finetune
done
stamp "done"
