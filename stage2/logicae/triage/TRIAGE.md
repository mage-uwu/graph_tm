# Why LogicAE pretraining does not transfer (triage, 2026-09-25)

Checkpoint: `models/logicae/pt.ltc` (masked-word pretraining, 16 blocks x 1024, 128-bit codes, 17-token
windows, 3.2M windows). Tools: `../diag.c` (read-only probes of a .ltc), `ft.sh` (SST-2 fine-tunes, 600
steps, batch 32, seq 64, seed 17, 4 threads). Raw results: `ft_results.jsonl`. Accuracy = hardened model.

## Answer

Pretraining collapses most of the trunk into **dead, saturated gates**: 68% of trunk channels are constant
on SST-2 text (a fresh network: 28%), including most early and middle channels. Fine-tuning cannot revive
them, and the fixed random wiring sends most of the answer head into them (142 of 256 head trees read only
dead channels). The pretrained model then learns only through the soft residue of saturated gates, which
hardening erases: hard accuracy stays at the constant answer. Resetting only the dead channels to the
scratch pass-through init removes the failure (test 0.510 -> 0.653). It is neither a length problem,
a temperature problem, nor gradient starvation.

## Evidence

| check | result |
|---|---|
| masked-word acc@1 on the same 2,000 targets, context 17 / 33 / 65 / 129 tokens | 0.226 / 0.197 / 0.194 / 0.194: mild, plateaus (not the cause) |
| step-0 fine-tuning gradients, pretrained vs fresh | same order of magnitude in every block and the codes (not starvation) |
| saturated gate corners (<.02 or >.98), blocks 0-15 | pretrained 16-28%, fresh 4% |
| dead channels (never both 0 and 1 on 3,000 SST-2 texts), block 2 / 4 / 8 / 15 | pretrained 529 / 686 / 810 / 779 of 1024; fresh 1 / 50 / 354 / 579; scratch after fine-tuning 72 / 144 / 291 / 468 |
| frozen linear probe, mean-pooled bits (8k train, SST-2 test), best block | pretrained 0.634, fresh 0.631: the frozen trunk adds nothing linear for sentiment |
| pretrained codes: Hamming good-bad / best-worst / random pairs | 43 / 37 / 51 of 128: distributional, antonyms pulled together |
| gate statistics before vs after fine-tuning from pretrained | unchanged: the trunk barely moves; only head + codes adapt |

SST-2 fine-tunes, 600 steps (scratch-init network = pass-through trees):

| start | train (hard) | dev500 hard / soft | test (872) | dev curve (every 100 steps) |
|---|---|---|---|---|
| scratch | 0.780 | 0.748 / 0.731 | **0.751** | .554 .650 .698 .712 .736 .748 |
| pretrained (run 1 recipe) | 0.568 | 0.544 / 0.681 | 0.510 | .532 x5 then .544 (constant answer) |
| pretrained, keep temperature | 0.560 | 0.534 / 0.675 | 0.509 | .532 x4 .534 .534 |
| pretrained codes, fresh trunk | 0.702 | 0.696 / 0.581 | 0.702 | **.632** .644 .684 .690 .700 .696 |
| pretrained trunk, fresh codes | 0.684 | 0.662 / **0.788** | 0.619 | .532 .534 .540 .628 .652 .662 |
| pretrained, dead channels reset | 0.686 | 0.708 / 0.673 | 0.653 | .588 .640 .646 .708 .696 .708 |

Reading:
- The harm is in the trunk (gates), not the codes. Pretrained codes on a fresh trunk start faster than
  scratch (0.632 vs 0.554 at step 100) and finish lower (stiff at temperature 0.2).
- Pretrained trunk + fresh codes reaches soft 0.788, above scratch: the pretrained gates do compute something
  useful, but through sub-threshold soft values that do not survive hardening.
- Reviving the dead channels removes the stuck phase and the hard/soft gap. The remaining gap to scratch
  (0.653 vs 0.751) is the live 32% of pretrained gates plus stiff codes.

## Next: a pretraining recipe that does not kill the trunk
Why channels die is untested. Candidates, each a small pretraining run measured by dead-channel
fraction + this fine-tune: (1) the masked-word readout reads only 223 of 1024 last-block channels and
~12% of positions, so nothing keeps the rest informative: a per-position readout over all channels
(e.g. replaced-token detection); (2) a balance penalty on gate outputs like the codes' row/column balance;
(3) periodic re-initialization of dead channels during pretraining; (4) skip channels copied through
every block.

## Straight-through fine-tuning (`../hard_patch.py`, `--hard-forward`)

Forward = the hardened network exactly (binary codes, binary gate outputs; checked: training-mode accuracy
equals the deployed .lth). Backward = the multilinear gate derivative at those binary points with the float
corner values, thresholds passed straight through. Float master weights and Adam kept. Flag off: checkpoints
byte-identical to the unpatched build. Same SST-2 setup (600 steps, hard accuracy):

| start | soft training: test | straight-through: train / dev500 / **test** | straight-through dev curve |
|---|---|---|---|
| scratch | 0.751 | 0.792 / 0.756 / **0.764** | .514 .544 .628 .732 .748 .756 |
| pretrained (keep temperature) | 0.509 | 0.678 / 0.674 / **0.669** | .538 .552 .592 .584 .624 .674 |
| pretrained, dead channels reset | 0.653 | 0.740 / 0.726 / **0.688** | .598 .646 .646 .702 .716 .726 |

- Training against the hard network removes the constant-answer failure: pretrained 0.509 -> 0.669 (+16).
- It is also better from scratch (0.751 -> 0.764): adaptation should always train the network that is deployed.
- Pretrained is still below scratch at 600 steps (0.688 vs 0.764 best), but the gap shrank from 24 to 8 points,
  and pretrained starts faster (dev 0.598 vs 0.514 at step 100). What remains is the dead / saturated trunk
  that pretraining leaves: fix it in pretraining (see Next), and pretrain with --hard-forward as well.

## Vetting: soft vs hard-forward pretraining, then adaptation (`../vet.py`, logicAE pod, 2026-09-25)

Same 4.4M model, 64-token WikiText-103 windows (streamed), batch 64, 512 masked words/batch, 3000 steps each
(~12M tokens; a fifth of run 1's masked words), three arms in parallel on 16 vCPU. Adaptation: straight-through,
1000 steps, batch 32, seed 17, final-step hardened accuracy. Raw: `../vet_results/`.

| pretraining | masked-word acc@1 / acc@10 (65-token context) | dead channels | SST-2 test | QNLI test |
|---|---|---|---|---|
| none (scratch) | - | - | 0.772 | 0.588 |
| run 1 (soft, 17-token windows) | 0.194 / 0.421 | 68% | 0.729 | 0.588 |
| soft64 | **0.186 / 0.395** | 70.7% | 0.766 | 0.586 |
| **hard64** (straight-through) | 0.141 / 0.327 | 29.9% | **0.805** | 0.588 |
| hardrev64 (+ dead-channel reset every 500 steps) | 0.118 / 0.342 | 2.3% | 0.797 | 0.588 |

SST-2 dev (1k) curves, every 250 steps: scratch .662 .787 .819 .841; hard64 .750 .810 .855 .862;
hardrev64 .765 .816 .840 .850; soft64 .655 .748 .796 .812.

Verdict:
- Pretraining against the hard network is the first LogicAE pretraining that transfers. SST-2: +3.3 over scratch
  (0.805 vs 0.772; test n=872, ~1.4 pt standard error, one seed), and much faster early (0.750 vs 0.662 dev
  at step 250). Soft pretraining at the same budget does not (0.766).
- Masked-word accuracy does not predict transfer: soft is the best masked-word model and the worst transfer.
  What predicts it is a live trunk trained on the deployed (hard) network.
- QNLI: every arm lands at 0.586-0.588, pretrained or not. Without `--match` the architecture cannot relate
  question and sentence tokens (known: 0.580 -> 0.710 with match + majority pooling), so this test says nothing
  about transfer. Next: re-test QNLI with the pair options on, and a longer hard64 pretraining run (5-10x).
- Models: `models/logicae/pt_hard64.ltc`, `pt_hardrev64.ltc` (MANIFEST.txt). fastlogic.c defaults pretraining to
  the hard forward pass.

## Long run (`stage2/fast_logicae/longrun.py`, 2026-09-26): pretraining improved, transfer did not

fastlogic.c, hard forward, pair options (--match 1, majority pooling every 4th block, 128 channels), 30k steps
x 64 x 64 = 123M tokens of WikiText-103 streamed in article order (12 chunks of 160k windows), 16 vCPU, 4.0 h.
Final: validation masked-word CE 5.51; on the fixed 2,000-target probe acc@1 0.204 / acc@10 0.416 (bert-tiny on
the same targets: CE 3.96, 0.314 / 0.596); dead channels 42% (flat after step 10k). Raw: `stage2/fast_logicae/longrun_results/`.

Adaptation (straight-through, 1000 steps, batch 32, seed 17, final-step hardened test accuracy):

| start | SST-2 | QNLI |
|---|---|---|
| scratch, with pair options (pod) | 0.779 | **0.732** |
| long run final, 30k steps (pod) | 0.743 | 0.688 |
| long run 25%, 7.5k steps (local) | 0.751 (dev1k .641 .739 .782 .802) | - |
| vetting hard64, 3k steps, no pair options (local re-run; reproduces exactly) | **0.805** (dev1k .750 .810 .855 .862) | - |

The long run's checkpoints adapt worse than scratch on both tasks, already at 25%, while the short vetting model
transfers (+2.6 over this scratch). Candidates (untested): (1) the pair options in pretraining; (2) article-order
streamed chunks (non-iid: each chunk holds ~160k windows from consecutive articles; vetting sampled windows
uniformly over the corpus); (3) the longer schedule. Next: 3k-step hard pretraining with pairs on iid windows vs
without pairs on stream-order chunks, then the same SST-2 fine-tune.
