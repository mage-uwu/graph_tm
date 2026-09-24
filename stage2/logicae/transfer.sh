#!/bin/bash
# LogicAE transfer diagnostics on the playground pod (AUTIST_ENGINEER_PLAYGROUND). Paste-once:
#   (command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)) && rm -rf /root/txrepo && git clone -q -b claude/lucid-ritchie-qtpddj https://github.com/mage-uwu/graph_tm /root/txrepo && setsid nohup bash /root/txrepo/stage2/logicae/transfer.sh > /root/tx.out 2>&1 < /dev/null &
# Installs build tools + python deps if missing, runs transfer.py (waits for models/logicae/pt.ltc
# on the branch), then exports results.jsonl, transfer.log and the hardened models (6 h on :8888).
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); W=/root/tx
log(){ echo "TXPOD [$(date -u +%H:%M:%S)] $*" | tee -a /root/txpod.log > /proc/1/fd/1; }
log "started ($(git -C $R log --oneline -1)); $(nproc) vCPU, $(free -g | awk 'NR==2{print $7}')G RAM available, $(df -h /root | awk 'NR==2{print $4}') disk free"
export DEBIAN_FRONTEND=noninteractive
if ! command -v gcc >/dev/null || ! command -v python3 >/dev/null || ! python3 -m pip --version >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq build-essential python3 python3-pip >/dev/null 2>&1 || log "apt-get FAILED"
fi
python3 -m pip install -q --break-system-packages numpy tokenizers pyarrow huggingface_hub 2>/dev/null \
  || python3 -m pip install -q numpy tokenizers pyarrow huggingface_hub || log "pip FAILED"
log "$(python3 --version 2>&1), $(gcc --version | head -1)"
mkdir -p $W
(cd $R/stage2/logicae && W=$W GTM_STAGE2_DATA=$W/data python3 transfer.py > $W/transfer.out 2>&1) || log "transfer.py exited non-zero: $(tail -3 $W/transfer.out | tr '\n' ' ')"
python3 $R/stage2/logicae/export.py --dir $W --hours 6 || log "export FAILED"
log "all done"
