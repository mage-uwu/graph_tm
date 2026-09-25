#!/bin/bash
# FAST_AE thread diagnosis: why 16 OpenMP threads run slower than 1 on the pod. Sweeps threads x
# OMP_WAIT_POLICY (active spin vs passive sleep) for fast training and compiled inference, logging the
# cgroup CPU quota and throttling counters; then retries bert-tiny with its errors logged.
# Same serving as bench.sh (:8888/<FAST_TOKEN>/, results.jsonl + fast.log). Run by fastboot (FAST_SCRIPT).
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=/root/fast; H=$R/stage2/fastlae
mkdir -p $OUT; exec 9> $OUT/.lock; flock -n 9 || { echo "already running"; exit 1; }
EXP=/root/fast_exports/${FAST_TOKEN:-none}; mkdir -p $EXP
log(){ echo "FAST [$(date -u +%H:%M:%S)] $*" | tee -a $OUT/fast.log > /proc/1/fd/1; cp $OUT/fast.log $EXP/ 2>/dev/null; [ -f $OUT/results.jsonl ] && cp $OUT/results.jsonl $EXP/; }
res(){ echo "$1" >> $OUT/results.jsonl; log "RESULT $1"; }
export DEBIAN_FRONTEND=noninteractive HF_HUB_ENABLE_HF_TRANSFER=0
[ -n "${FAST_TOKEN:-}" ] && { setsid timeout 48h python3 $R/stage2/logicae/export.py --serve /root/fast_exports --port 8888 >/dev/null 2>&1 < /dev/null & }
thr(){ cat /sys/fs/cgroup/cpu.stat 2>/dev/null | awk '/nr_throttled|throttled_usec/{printf "%s=%s ", $1, $2}'; cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null | awk '{printf "%s=%s ", $1, $2}'; }
log "started ($(git -C $R log --oneline -1)); nproc $(nproc); cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)/$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)"); cpuset: $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null); $(thr)"
command -v gcc >/dev/null || { apt-get update -qq && apt-get install -y -qq build-essential >/dev/null 2>&1; }
cd $OUT && rm -rf lb && mkdir lb && tar xzf $R/logic-bert.tar.gz -C lb; LB=$OUT/lb/logic-bert/src
F="-O3 -march=native -std=gnu11 -fopenmp -Wno-unknown-pragmas"
python3 $H/fasttrain_patch.py $LB/logic_text.c lt_fast.c > /dev/null && gcc $F lt_fast.c -lm -o lt_fast && gcc $F -I$LB $H/fastlae.c -lm -o fastlae || { log "build FAILED"; exit 1; }
M=$R/models/logicae/sst2_scratch.lth
python3 - <<'PY'
import random
random.seed(7)
for name, n, lo, hi, lab in (("bench64.ids", 1024, 8, 64, False), ("train64.ids", 4000, 8, 64, True)):
    with open(f"/root/fast/{name}", "w") as f:
        for i in range(n):
            L = random.randint(lo, hi)
            f.write(f"{random.randint(0,1) if lab else -1} " + " ".join(str(random.randint(3, 30521)) for _ in range(L)) + "\n")
PY
./fastlae gen $M bench64.ids 64 $OUT/sst2.inc > /dev/null && gcc -O2 -march=native -std=gnu11 -fopenmp -Wno-unknown-pragmas -I$LB -DGEN_INC="\"$OUT/sst2.inc\"" $H/fastgen.c -lm -o fastgen 2> gen.err \
  || { log "fastgen build FAILED: $(head -c 400 gen.err | tr '\n' ' ')"; exit 1; }
log "builds ok"
A="--data train64.ids --format ids --vocab-size 30522 --code-bits 128 --width 1024 --blocks 16 --kernel 5 --cycle 4 --seed 17 --eval-every 1000 --seq 64 --batch 32"
for pol in passive active; do for th in 1 2 4 8 12 16; do
  before=$(thr)
  s=$(OMP_WAIT_POLICY=$pol ./lt_fast train $A --steps 10 --threads $th --save t.ltc 2>&1 | grep -o '"compute_seconds":[0-9.]*' | tail -1 | cut -d: -f2)
  i=$(OMP_WAIT_POLICY=$pol ./fastgen bench $M bench64.ids 64 --batch 4096 --lanes 8 --threads $th --repeats 10 | tail -1)
  res "{\"bench\":\"threads\",\"omp_wait_policy\":\"$pol\",\"threads\":$th,\"train_seconds_per_step\":$(python3 -c "print(round($s/10,4))"),\"fastgen_4096\":$i,\"cgroup_before\":\"$before\",\"cgroup_after\":\"$(thr)\"}"
done; done
# bert-tiny, errors logged this time
python3 -m pip install -q -U pip > pip.log 2>&1
BSP=$(python3 -m pip install --help 2>/dev/null | grep -q break-system-packages && echo --break-system-packages)
PV=$(python3 -c 'import sys; print(sys.version_info[1])'); if [ "$PV" -le 8 ]; then TORCH="torch==2.4.1"; TF="transformers==4.46.3"; else TORCH="torch"; TF="transformers"; fi
python3 -m pip install -q $BSP --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu "$TORCH" >> pip.log 2>&1 || log "pip torch FAILED: $(tail -2 pip.log | tr '\n' ' ')"
python3 -m pip install -q $BSP --only-binary=:all: "numpy<1.25" "tokenizers<0.21" "huggingface_hub<0.26" "$TF" >> pip.log 2>&1 || log "pip transformers FAILED: $(tail -2 pip.log | tr '\n' ' ')"
rm -rf /root/.cache/pip
for pol in passive; do
OMP_WAIT_POLICY=$pol python3 - <<'PY' > bert.jsonl 2> bert.err || log "bert FAILED: $(tail -3 bert.err | tr '\n' ' ')"
import json, time, torch
from transformers import BertModel
m = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2").eval()
for th in (1, 4, 16):
    torch.set_num_threads(th)
    for B in (1, 6, 64, 512, 4096):
        ids = torch.randint(1000, 20000, (B, 64)); am = torch.ones_like(ids)
        with torch.no_grad():
            for _ in range(2): m(input_ids=ids, attention_mask=am)
            ts = []
            for _ in range(10 if B < 4096 else 3):
                t = time.perf_counter(); m(input_ids=ids, attention_mask=am); ts.append(time.perf_counter() - t)
        ms = sorted(ts)[len(ts) // 2] * 1000
        print(json.dumps({"bench": "inference", "engine": "bert-tiny (PyTorch fp32)", "r": {"batch": B, "seq": 64, "threads": th, "ms": round(ms, 4), "texts_per_s": round(B / ms * 1000)}}), flush=True)
PY
done
while read -r l; do res "$l"; done < bert.jsonl
log "all done"
