#!/bin/bash
# FAST_AE: LogicAE vs bert-tiny / bag of words on Laya's own benchmark (LocalLLaMA/typed-decisions),
# stage2/system1/laya_td.py through jev.sh (deps, engine build, serving at :8888/<FAST_TOKEN>/).
export JEV_TOKEN=${FAST_TOKEN:-} JEV_SCRIPT=laya_td.py
exec bash "$(dirname "$0")/../system1/jev.sh"
