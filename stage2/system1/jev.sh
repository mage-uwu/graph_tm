#!/bin/bash
# Jev-style adaptation experiment (stage2/system1/jevft.py) on a test pod. Boot / paste-once:
#   git clone -q --depth 1 --filter=blob:none --sparse -b claude/lucid-ritchie-qtpddj https://github.com/mage-uwu/graph_tm /root/jevrepo \
#     && git -C /root/jevrepo sparse-checkout set --no-cone /stage2/ /logic-bert.tar.gz /models/logicae/pt.ltc /models/logicae/MANIFEST.txt \
#     && JEV_TOKEN=<24 hex> setsid nohup bash /root/jevrepo/stage2/system1/jev.sh > /root/jev.out 2>&1 < /dev/null &
# Installs build tools + CPU torch / transformers (Python 3.8 wheels only), checks pt.ltc against the
# manifest, serves /root/jev_exports/<JEV_TOKEN>/ (results.jsonl, summary.md, jev.log, the hardened
# LogicAE models) read-only on :8888 from the start, then runs jevft.py.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=/root/jev
mkdir -p $OUT
log(){ echo "JEVPOD [$(date -u +%H:%M:%S)] $*" | tee -a $OUT/jev.log > /proc/1/fd/1; }
export DEBIAN_FRONTEND=noninteractive HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
log "started ($(git -C $R log --oneline -1)); $(nproc) vCPU, $(df -h /root | awk 'NR==2{print $4}') disk free"
if ! command -v gcc >/dev/null || ! python3 -m pip --version >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq build-essential python3 python3-pip >/dev/null 2>&1 || log "apt-get FAILED"
fi
python3 -m pip install -q -U pip >/dev/null 2>&1 || log "pip self-upgrade FAILED"
BSP=$(python3 -m pip install --help 2>/dev/null | grep -q break-system-packages && echo --break-system-packages)
PV=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PV" -le 8 ]; then TORCH="torch==2.4.1"; TF="transformers==4.46.3"; else TORCH="torch"; TF="transformers"; fi
python3 -m pip install -q $BSP --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu "$TORCH" > /root/pip.err 2>&1 \
  || log "pip FAILED for $TORCH: $(tail -2 /root/pip.err | tr '\n' ' ')"
for pkg in "numpy<1.25" "pyarrow<18" "tokenizers<0.21" "huggingface_hub<0.26" "$TF"; do
  python3 -m pip install -q $BSP --only-binary=:all: "$pkg" > /root/pip.err 2>&1 || log "pip FAILED for $pkg: $(tail -2 /root/pip.err | tr '\n' ' ')"
done
rm -rf /root/.cache/pip
python3 -c "import numpy, pyarrow, tokenizers, torch, transformers" 2>/dev/null && log "python deps ok" || log "python deps MISSING"
Q=$(awk '$1!="max"{printf "%d", $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null)
[ -z "$Q" ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && Q=$(awk -v p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) '$1>0{printf "%d", $1/p}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
[ -z "$Q" ] || [ "$Q" -lt 1 ] && Q=$(nproc)
[ "$Q" -gt 32 ] && Q=16
export THREADS=${THREADS:-$Q}
want=$(awk '$3=="pt.ltc"{print $1}' $R/models/logicae/MANIFEST.txt)
got=$(sha256sum $R/models/logicae/pt.ltc | cut -d' ' -f1)
[ "$want" = "$got" ] && log "pt.ltc sha256 ok" || log "pt.ltc sha256 MISMATCH ($got, want $want)"
if [ -n "${JEV_TOKEN:-}" ]; then
  mkdir -p /root/jev_exports/$JEV_TOKEN
  setsid timeout 48h python3 $R/stage2/logicae/export.py --serve /root/jev_exports --port 8888 > /dev/null 2>&1 < /dev/null &
  log "serving /root/jev_exports on :8888 (token set)"
fi
log "threads $THREADS, $(python3 --version 2>&1), $(gcc --version | head -1)"
(cd $R/stage2/system1 && GTM_STAGE2_DATA=$OUT/data JEV_OUT=$OUT python3 jevft.py > $OUT/jevft.out 2>&1) \
  || log "jevft.py exited non-zero: $(grep -v Warning $OUT/jevft.out | tail -3 | tr '\n' ' ')"
log "all done"
