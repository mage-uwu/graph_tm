#!/bin/bash
# Pod-side testbed runner. Sets up once, then keeps the checkout synced to the branch (new
# experiments and code arrive by git push) and runs SLOTS workers that each claim the next
# pending experiment from stage2/experiments.tsv (see testbed.py). Logs go to stdout.
cd "$(dirname "$0")/.."
BR=${BR:-claude/clever-ramanujan-958lkj}
SLOTS=${SLOTS:-2}
P=$(nproc)
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
python3 -c "import tokenizers, pyarrow, safetensors, huggingface_hub" 2>/dev/null ||
  python3 -m pip install -q --break-system-packages tokenizers pyarrow safetensors huggingface_hub
make -C stage1 -s
(cd stage2 && python3 pretrain_data.py prep)
echo "== [$(date -u +%H:%M:%S)] testbed up: $SLOTS slots x $((P / SLOTS)) threads, branch $BR"
( while true; do
    if git fetch -q origin "$BR" && [ "$(git rev-parse HEAD)" != "$(git rev-parse FETCH_HEAD)" ]; then
      git reset -q --hard FETCH_HEAD && make -C stage1 -s && echo "== [$(date -u +%H:%M:%S)] synced to $(git log --oneline -1)"
    fi
    sleep 60
  done ) &
for s in $(seq 1 $SLOTS); do
  ( sleep $((s * 5)); while true; do
      python3 stage2/testbed.py next --threads $((P / SLOTS)); [ $? -eq 3 ] && sleep 30
    done ) &
done
wait
