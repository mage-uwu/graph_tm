#!/bin/bash
# FAST_AE benchmark: fast LogicAE inference (fastlae interpreter, fastgen compiled network) and fast
# training (fasttrain_patch) vs the reference logic-bert engine, plus bert-tiny (PyTorch) on the same
# CPU. Every fast result is checked for equality with the reference (scores / checkpoint bytes).
# Results: $OUT/results.jsonl, served read-only at :8888/<FAST_TOKEN>/ (stage2/logicae/export.py).
#   git clone -q --depth 1 -b claude/lucid-ritchie-qtpddj https://github.com/mage-uwu/graph_tm /root/fastrepo \
#     && FAST_TOKEN=<24 hex> setsid nohup bash /root/fastrepo/stage2/fastlae/bench.sh > /root/fast.out 2>&1 < /dev/null &
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=/root/fast; H=$R/stage2/fastlae
mkdir -p $OUT; exec 9> $OUT/.lock; flock -n 9 || { echo "bench.sh already running"; exit 1; }
EXP=/root/fast_exports/${FAST_TOKEN:-none}; mkdir -p $EXP
log(){ echo "FAST [$(date -u +%H:%M:%S)] $*" | tee -a $OUT/fast.log > /proc/1/fd/1; cp $OUT/fast.log $EXP/ 2>/dev/null; [ -f $OUT/results.jsonl ] && cp $OUT/results.jsonl $EXP/; }
res(){ echo "$1" >> $OUT/results.jsonl; log "RESULT $1"; }
export DEBIAN_FRONTEND=noninteractive HF_HUB_ENABLE_HF_TRANSFER=0
[ -n "${FAST_TOKEN:-}" ] && { setsid timeout 48h python3 $R/stage2/logicae/export.py --serve /root/fast_exports --port 8888 >/dev/null 2>&1 < /dev/null & }
log "started ($(git -C $R log --oneline -1)); $(nproc) vCPU; $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
command -v gcc >/dev/null || { apt-get update -qq && apt-get install -y -qq build-essential >/dev/null 2>&1; }
T=${THREADS:-$(nproc)}; [ "$T" -gt 16 ] && T=16
# --- builds
cd $OUT && rm -rf lb && mkdir lb && tar xzf $R/logic-bert.tar.gz -C lb; LB=$OUT/lb/logic-bert/src
F="-O3 -march=native -std=gnu11 -fopenmp -Wno-unknown-pragmas"
gcc $F $LB/logic_text.c -lm -o lt_ref && python3 $H/fasttrain_patch.py $LB/logic_text.c lt_fast.c > /dev/null && gcc $F lt_fast.c -lm -o lt_fast \
  && gcc $F -I$LB $H/fastlae.c -lm -o fastlae 2> build.err || { log "build FAILED: $(head -c 400 build.err | tr '\n' ' ')"; exit 1; }
M=$R/models/logicae/sst2_scratch.lth   # run 1's SST-2 model (seq 64)
# synthetic inputs: token ids / lengths only matter for timing; labels for training
python3 - <<'PY'
import random
random.seed(7)
for name, n, lo, hi, lab in (("bench64.ids", 1024, 8, 64, False), ("train64.ids", 4000, 8, 64, True), ("mlm17.ids", 20000, 17, 17, False)):
    with open(f"/root/fast/{name}", "w") as f:
        for i in range(n):
            L = random.randint(lo, hi)
            f.write(f"{random.randint(0,1) if lab else -1} " + " ".join(str(random.randint(3, 30521)) for _ in range(L)) + "\n")
PY
# (absolute GEN_INC: a quoted #include searches the including file's directory, not the cwd)
./fastlae gen $M bench64.ids 64 $OUT/sst2.inc > /dev/null && gcc -O2 -march=native -std=gnu11 -fopenmp -Wno-unknown-pragmas -I$LB -DGEN_INC="\"$OUT/sst2.inc\"" $H/fastgen.c -lm -o fastgen 2> gen.err \
  || { log "fastgen build FAILED: $(head -c 400 gen.err | tr '\n' ' ')"; exit 1; }
log "builds ok, threads $T"
# --- inference: correctness, then speed
for b in fastlae fastgen; do
  v=$(./$b verify $M bench64.ids 64 --threads $T | tail -1); res "{\"check\":\"$b scores == reference\",\"seq\":64,\"records\":1024,\"result\":$v}"
done
for th in 1 $T; do for B in 1 6 64 512 4096; do
  L=1; [ $B -ge 64 ] && L=8
  for b in fastlae fastgen; do
    r=$(./$b bench $M bench64.ids 64 --batch $B --lanes $L --threads $th --repeats 20 | tail -1); res "{\"bench\":\"inference\",\"engine\":\"$b\",\"model\":\"sst2_scratch.lth\",\"r\":$r}"
  done
done; done
# --- bert-tiny on the same CPU (PyTorch, fp32), 64-token inputs
python3 -m pip install -q -U pip >/dev/null 2>&1
BSP=$(python3 -m pip install --help 2>/dev/null | grep -q break-system-packages && echo --break-system-packages)
PV=$(python3 -c 'import sys; print(sys.version_info[1])'); if [ "$PV" -le 8 ]; then TORCH="torch==2.4.1"; TF="transformers==4.46.3"; else TORCH="torch"; TF="transformers"; fi
python3 -m pip install -q $BSP --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu "$TORCH" >/dev/null 2>&1
python3 -m pip install -q $BSP --only-binary=:all: "numpy<1.25" "tokenizers<0.21" "huggingface_hub<0.26" "$TF" >/dev/null 2>&1; rm -rf /root/.cache/pip
python3 - "$T" <<'PY' >> $OUT/bert.jsonl 2>/dev/null || true
import json, sys, time, torch
from transformers import BertModel
m = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2").eval()
for th in (1, int(sys.argv[1])):
    torch.set_num_threads(th)
    for B in (1, 6, 64, 512, 4096):
        ids = torch.randint(1000, 20000, (B, 64)); am = torch.ones_like(ids)
        with torch.no_grad():
            for _ in range(3): m(input_ids=ids, attention_mask=am)
            ts = []
            for _ in range(10 if B < 4096 else 3):
                t = time.perf_counter(); m(input_ids=ids, attention_mask=am); ts.append(time.perf_counter() - t)
        ms = sorted(ts)[len(ts) // 2] * 1000
        print(json.dumps({"bench": "inference", "engine": "bert-tiny (PyTorch fp32)", "r": {"batch": B, "seq": 64, "threads": th, "ms": round(ms, 4), "texts_per_s": round(B / ms * 1000)}}), flush=True)
PY
while read -r l; do res "$l"; done < $OUT/bert.jsonl
# --- training: reference vs fasttrain, same seed -> checkpoints must be byte-identical
A="--data train64.ids --format ids --vocab-size 30522 --code-bits 128 --width 1024 --blocks 16 --kernel 5 --cycle 4 --seed 17 --eval-every 1000"
for th in $T 1; do
  for e in lt_ref lt_fast; do
    [ $th = 1 ] && [ $e = lt_ref ] && continue
    s=$(./$e train $A --seq 64 --batch 32 --steps 20 --threads $th --save sup_$e.ltc 2>&1 | grep -o '"compute_seconds":[0-9.]*' | tail -1 | cut -d: -f2)
    res "{\"bench\":\"train\",\"mode\":\"supervised seq 64 batch 32\",\"engine\":\"$e\",\"threads\":$th,\"steps\":20,\"seconds_per_step\":$(python3 -c "print(round($s/20,4))")}"
  done
done
res "{\"check\":\"supervised checkpoints identical\",\"result\":\"$(cmp -s sup_lt_ref.ltc sup_lt_fast.ltc && echo passed || echo FAILED)\"}"
P="--data mlm17.ids --format ids --vocab-size 30522 --code-bits 128 --width 1024 --blocks 16 --kernel 5 --cycle 4 --seed 17 --eval-every 1000 --seq 17 --batch 256 --mlm-targets 512"
for e in lt_ref lt_fast; do
  s=$(./$e pretrain $P --steps 5 --threads $T --save mlm_$e.ltc 2>&1 | grep -o '"compute_seconds":[0-9.]*' | tail -1 | cut -d: -f2)
  res "{\"bench\":\"train\",\"mode\":\"MLM pretrain seq 17 batch 256 (run 1)\",\"engine\":\"$e\",\"threads\":$T,\"steps\":5,\"seconds_per_step\":$(python3 -c "print(round($s/5,4))")}"
done
res "{\"check\":\"MLM checkpoints identical\",\"result\":\"$(cmp -s mlm_lt_ref.ltc mlm_lt_fast.ltc && echo passed || echo FAILED)\"}"
log "all done"
