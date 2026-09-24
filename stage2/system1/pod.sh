#!/bin/bash
# System-1 shootout on the LogicAE pod, then the LogicAE scale-up (sweep + long run).
# Paste-once, from the pod's web terminal:
#   rm -rf /root/s1repo && git clone -q -b claude/lucid-ritchie-qtpddj https://github.com/mage-uwu/graph_tm /root/s1repo && setsid nohup bash /root/s1repo/stage2/system1/pod.sh > /root/s1pod.out 2>&1 < /dev/null &
# Order: wait for run 1 ("done") -> export run 1 (4 h) -> deps (sklearn, scipy, torch cpu,
# transformers) -> GraphTM engine + cached dist codes -> shootout.py (all backends, THREADS=nproc)
# -> export the shootout results (6 h) -> stage2/logicae/scale2x.py. Run 1's files are only read.
# Every step logs "S1POD ..." to the container log; shootout.py logs "S1 ..." lines.
set -uo pipefail
R=/root/s1repo; L=/root/lae; D=/root/graph_tm/data/stage2; O=/root/s1out
log(){ echo "S1POD [$(date -u +%H:%M:%S)] $*" | tee -a /root/s1pod.log > /proc/1/fd/1; }
log "started ($(git -C $R log --oneline -1)); waiting for run 1 ($L/phase == done)"
until [ "$(cat $L/phase 2>/dev/null)" = "done" ]; do sleep 60; done
log "run 1 done; exporting it"
python3 $R/stage2/logicae/export.py --dir $L --hours 4 || log "run 1 export FAILED"
log "installing deps"
python3 -m pip install -q --break-system-packages numpy scipy scikit-learn tokenizers pyarrow huggingface_hub safetensors 2>&1 | tail -1
python3 -c "import torch" 2>/dev/null || python3 -m pip install -q --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
python3 -m pip install -q --break-system-packages "transformers>=4.48" 2>&1 | tail -1
python3 -c "import torch, transformers, sklearn, scipy; print(torch.__version__, transformers.__version__)" 2>&1 | while read l; do log "deps: $l"; done
(cd $R/stage1 && make -s) || log "gtm build FAILED"
[ -f $D/dist_emb_40000000_4096_128.npy ] || cp $R/models/graphtm/dist_emb_40000000_4096_128.npy $D/
mkdir -p $O
log "shootout start (threads $(nproc), mem $(free -g | awk 'NR==2{print $7}')G available)"
(cd $R/stage2/system1 && GTM_STAGE2_DATA=$D S1_OUT=$O THREADS=$(nproc) LAE_DIR=$L LB=/root/logic-bert/src \
  python3 shootout.py > $O/shootout.out 2>&1) || log "shootout exited non-zero"
log "shootout finished; exporting results"
python3 $R/stage2/logicae/export.py --dir $O --hours 6 || log "shootout export FAILED"
log "starting the LogicAE scale-up (sweep + long run)"
cd $R/stage2/logicae && RUN1=$L python3 scale2x.py
log "all done"
