#!/bin/bash
# FAST_AE: no training. Bundles what the pod kept under /root/jev: every hardened LogicAE model (.lth / .lth.best),
# its training log, and the run records (results.jsonl, summary.md, jev.log), then scores the stopped typed-decisions
# lae_gm arm on the questions it finished (stage2/system1/td_partial.py). Served at :8888/<FAST_TOKEN>/:
#   bundle.log, files.txt, MANIFEST.sha256 (sha256  size  path), fast_ae_models.tar.gz, td_partial.json
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=/root/jev; EXP=/root/jev_exports/${FAST_TOKEN:?FAST_TOKEN unset}
mkdir -p $EXP
export OMP_WAIT_POLICY=passive PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
log(){ echo "BUNDLE [$(date -u +%H:%M:%S)] $*" | tee -a $EXP/bundle.log > /proc/1/fd/1; }
setsid timeout 24h python3 $R/stage2/logicae/export.py --serve /root/jev_exports --port 8888 > /dev/null 2>&1 < /dev/null &
log "started ($(git -C $R log --oneline -1)); $(df -h /root | awk 'NR==2{print $4}') disk free; $OUT: $(du -sh $OUT 2>/dev/null | cut -f1)"
[ -d $OUT ] || { log "no $OUT on this disk: nothing to bundle"; exit 0; }
cd $OUT
find . -type f \( -name '*.lth' -o -name '*.lth.best' -o -name '*.log' -o -name 'results.jsonl' -o -name 'summary.md' \) \
  ! -path './data/*' ! -path './lb/*' | sort > $EXP/files.txt
log "$(wc -l < $EXP/files.txt) files, $(grep -c '\.lth' $EXP/files.txt) models"
while read -r f; do printf '%s  %s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" "$f"; done < $EXP/files.txt > $EXP/MANIFEST.sha256
tar czf $EXP/fast_ae_models.tar.gz.tmp -T $EXP/files.txt && mv $EXP/fast_ae_models.tar.gz.tmp $EXP/fast_ae_models.tar.gz \
  && log "fast_ae_models.tar.gz $(du -h $EXP/fast_ae_models.tar.gz | cut -f1)" || log "tar FAILED"
if [ -x $OUT/ltx ] && [ -d $OUT/td ]; then
  (cd $R/stage2/system1 && THREADS=$(nproc) GTM_STAGE2_DATA=$OUT/data JEV_OUT=$OUT JEV_TOKEN= python3 td_partial.py \
     > $EXP/td_partial.json 2> $EXP/td_partial.err) && log "td_partial done: $(tail -1 $EXP/td_partial.json | cut -c1-300)" \
    || log "td_partial FAILED: $(grep -v Warning $EXP/td_partial.err | tail -3 | tr '\n' ' ')"
else
  log "no engine or typed-decisions dir: skipping td_partial"
fi
log "done"
