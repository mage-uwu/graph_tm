#!/bin/bash
# FAST_AE: does the global view (global_patch.py) fix LogicAE's relational failures? Runs the Jev
# harness (stage2/system1/jevft.py via jev.sh) on QNLI (question vs sentence), SST-2 and emotion with
# one-vs-rest LogicAE from scratch: no global view vs global view every 4th block (128 channels) vs every
# 2nd (256), all JEV_OVR_STEPS steps per model, plus bert-tiny on the shared objective for reference.
# Results (results.jsonl, summary.md, jev.log) at :8888/<FAST_TOKEN>/ as for bench.sh.
export JEV_TOKEN=${FAST_TOKEN:-}  # jev.sh serves /root/jev_exports/<token>/ on :8888
export JEV_TASKS=${JEV_TASKS:-qnli,sst2,emotion} JEV_OVR_STEPS=${JEV_OVR_STEPS:-1500}
export JEV_ARMS=${JEV_ARMS:-bert_pair,lae_ovr_scratch,lae_ovr_scratch_g4,lae_ovr_scratch_g2}
exec bash "$(dirname "$0")/../system1/jev.sh"
