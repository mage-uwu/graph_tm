#!/bin/bash
# Pod launcher for vet.py (LogicAE pretraining vetting). Deps, thread count, results served read-only at
# :8888/<VET_TOKEN>/ (vet.log, results.jsonl, <arm>.ltc) via export.py. Data is streamed (vet.py).
#   git clone -q --depth 1 -b claude/lucid-ritchie-qtpddj https://github.com/mage-uwu/graph_tm /root/vrepo \
#     && VET_TOKEN=<24 hex> setsid nohup bash /root/vrepo/stage2/logicae/vet.sh > /root/vet.out 2>&1 < /dev/null &
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=${VET_OUT:-/root/vet}; mkdir -p $OUT
exec 9> $OUT/.lock; flock -n 9 || { echo "vet.sh already running"; exit 1; }
export DEBIAN_FRONTEND=noninteractive HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1 OMP_WAIT_POLICY=passive
log(){ echo "VETPOD [$(date -u +%H:%M:%S)] $*" | tee -a $OUT/vet.log > /proc/1/fd/1; }
log "started ($(git -C $R log --oneline -1)); $(nproc) vCPU, $(df -h /root | awk 'NR==2{print $4}') disk free"
if ! command -v gcc >/dev/null || ! python3 -m pip --version >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq build-essential python3 python3-pip >/dev/null 2>&1 || log "apt-get FAILED"
fi
BSP=$(python3 -m pip install --help 2>/dev/null | grep -q break-system-packages && echo --break-system-packages)
for pkg in numpy pyarrow tokenizers huggingface_hub; do
  python3 -c "import $pkg" 2>/dev/null || python3 -m pip install -q $BSP --only-binary=:all: "$pkg" > /root/pip.err 2>&1 \
    || log "pip FAILED for $pkg: $(tail -2 /root/pip.err | tr '\n' ' ')"
done
rm -rf /root/.cache/pip
Q=$(awk '$1!="max"{printf "%d", $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null)
[ -z "$Q" ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && Q=$(awk -v p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) '$1>0{printf "%d", $1/p}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
[ -z "$Q" ] || [ "$Q" -lt 1 ] && Q=$(nproc)
[ "$Q" -gt 32 ] && Q=16
export THREADS=${THREADS:-$Q} VET_OUT=$OUT
if [ -n "${VET_TOKEN:-}" ]; then
  mkdir -p /root/vet_exports/$VET_TOKEN
  setsid timeout 72h python3 $R/stage2/logicae/export.py --serve /root/vet_exports --port 8888 > /dev/null 2>&1 < /dev/null &
  log "serving /root/vet_exports on :8888 (token set)"
fi
log "threads $THREADS, $(python3 --version 2>&1), $(gcc --version | head -1)"
(cd $R/stage2/logicae && python3 vet.py > $OUT/vet.out 2>&1) || log "vet.py exited non-zero: $(grep -v Warning $OUT/vet.out | tail -3 | tr '\n' ' ')"
