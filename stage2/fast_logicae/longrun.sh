#!/bin/bash
# Pod launcher for longrun.py (the long LogicAE pretraining run). Deps, thread count, results served read-only at
# :8888/<LR_TOKEN>/ (run.log, perf.log, results.jsonl, saved.pt, saved_q1.pt, saved_fastae.pt) via export.py. Data is streamed (longrun.py).
#   git clone -q --depth 1 -b claude/lucid-ritchie-qtpddj https://github.com/mage-uwu/graph_tm /root/vrepo \
#     && LR_TOKEN=<24 hex> setsid nohup bash /root/vrepo/stage2/fast_logicae/longrun.sh > /root/lr.out 2>&1 < /dev/null &
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=${LR_OUT:-/root/lr}; mkdir -p $OUT
exec 9> $OUT/.lock; flock -n 9 || { echo "longrun.sh already running"; exit 1; }
export DEBIAN_FRONTEND=noninteractive HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1 OMP_WAIT_POLICY=passive
log(){ echo "LRPOD [$(date -u +%H:%M:%S)] $*" | tee -a $OUT/run.log > /proc/1/fd/1; }
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
export THREADS=${THREADS:-$Q} LR_OUT=$OUT
if [ -n "${LR_TOKEN:-}" ]; then
  mkdir -p /root/lr_exports/$LR_TOKEN
  setsid timeout 72h python3 $R/stage2/logicae/export.py --serve /root/lr_exports --port 8888 > /dev/null 2>&1 < /dev/null &
  log "serving /root/lr_exports on :8888 (token set)"
fi
log "threads $THREADS, $(python3 --version 2>&1), $(gcc --version | head -1)"
S=${LR_SCRIPT:-longrun.py}  # ab.py: the 3k-step A/B follow-up
(cd $R/stage2/fast_logicae && python3 $S > $OUT/longrun.out 2>&1) || log "$S exited non-zero: $(grep -v Warning $OUT/longrun.out | tail -3 | tr '\n' ' ')"
