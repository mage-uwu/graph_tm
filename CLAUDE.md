# CLAUDE.md

Read `handoff.md` before making changes. It has the goal, current state, engine walkthrough,
findings, and prioritized next work.

Rules (details in handoff.md §5):

1. `stage0/gtmcore.py` + `stage0/gtm_oracle.py` define the semantics. Behavior changes go there
   first; then `python3 tests/test_parity.py` (full) must pass. The C engine is bit-exact with
   the oracle, including training.
2. Performance changes must not change results: run `bench/bench.sh` before and after;
   inference `sums` and training `hash` fingerprints must be identical.
3. Training results must not depend on thread count or ISA (the parity suite checks this).
4. Randomness is counter-based only: every draw a pure function of (seed, tag, indices).
5. Defaults reproduce the CUDA GraphTM semantics; improvements are opt-in flags.
6. After touching threads or shared arrays, run ThreadSanitizer (command in handoff.md §5).

Quick start: `./run_all.sh` (builds, generates data, runs parity, trains every task).
