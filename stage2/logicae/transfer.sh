#!/bin/bash
# LogicAE transfer diagnostics on the playground pod (AUTIST_ENGINEER_PLAYGROUND). Paste-once:
#   (command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)) && rm -rf /root/txrepo && git clone -q --depth 1 -b claude/lucid-ritchie-qtpddj https://github.com/mage-uwu/graph_tm /root/txrepo && setsid nohup bash /root/txrepo/stage2/logicae/transfer.sh > /root/tx.out 2>&1 < /dev/null &
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
# Wheels only (the image's Python may be 3.8 with an old pip that falls back to source builds):
# upgrade pip, then pin the last releases with Python 3.8 wheels; each package on its own so one
# failure is named instead of blocking the rest.
python3 -m pip install -q -U pip >/dev/null 2>&1 || log "pip self-upgrade FAILED"
BSP=$(python3 -m pip install --help 2>/dev/null | grep -q break-system-packages && echo --break-system-packages)
for pkg in "numpy<1.25" "pyarrow<18" "tokenizers<0.21" "huggingface_hub<0.26"; do
  python3 -m pip install -q $BSP --only-binary=:all: "$pkg" > /root/pip.err 2>&1 || log "pip FAILED for $pkg: $(tail -2 /root/pip.err | tr '\n' ' ')"
done
python3 -c "import numpy, pyarrow, tokenizers, huggingface_hub" 2>/dev/null && log "python deps ok" || log "python deps MISSING"
# vCPUs actually granted (cgroup quota), not the host's core count
Q=$(awk '$1!="max"{printf "%d", $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null)
[ -z "$Q" ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && Q=$(awk -v p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) '$1>0{printf "%d", $1/p}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
[ -z "$Q" ] || [ "$Q" -lt 1 ] && Q=$(nproc)
[ "$Q" -gt 32 ] && Q=16
export THREADS=${THREADS:-$Q}
log "threads $THREADS"
log "$(python3 --version 2>&1), $(gcc --version | head -1)"
mkdir -p $W
(cd $R/stage2/logicae && W=$W GTM_STAGE2_DATA=$W/data python3 transfer.py > $W/transfer.out 2>&1) || log "transfer.py exited non-zero: $(tail -3 $W/transfer.out | tr '\n' ' ')"
python3 $R/stage2/logicae/export.py --dir $W --hours 6 || log "export FAILED"
log "all done"
