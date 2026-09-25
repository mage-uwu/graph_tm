# Jev-style decisions: pretrained LogicAE vs bert-tiny, adapted on one shared objective

LogicAE arms: 2000 steps, batch 32. Test metrics after one dev-fitted temperature per question. Latency: one decision (state x all options), CPU, 1 thread.

## qnli (noul, K=2, test n=2000)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | lae_ovr_scratch_gm4 | 0.7095 | 0.0358 | 0.5754 | 0.3887 | 0.8120 | - | 2.0579 |
| 2 | lae_ovr_scratch_m | 0.6990 | 0.0341 | 0.5844 | 0.3971 | 0.8080 | - | 2.0879 |
| 3 | lae_ovr_scratch | 0.5795 | 0.0136 | 0.6743 | 0.4815 | 0.6230 | - | 2.0229 |

## sst2 (choice, K=2, test n=872)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | lae_ovr_scratch | 0.7798 | 0.0969 | 0.5775 | 0.3402 | 0.8693 | - | 1.1071 |
| 2 | lae_ovr_scratch_m | 0.7787 | 0.0896 | 0.5740 | 0.3387 | 0.8807 | - | 1.0766 |
| 3 | lae_ovr_scratch_gm4 | 0.7741 | 0.0906 | 0.5712 | 0.3402 | 0.8670 | - | 1.0518 |

## emotion (choice, K=6, test n=2000)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | lae_ovr_scratch_m | 0.7075 | 0.0545 | 0.8905 | 0.4294 | 0.8360 | - | 6.8538 |
| 2 | lae_ovr_scratch | 0.6935 | 0.0498 | 0.9169 | 0.4438 | 0.8280 | - | 6.591 |
| 3 | lae_ovr_scratch_gm4 | 0.6915 | 0.0628 | 0.9223 | 0.4473 | 0.8230 | - | 6.2648 |

## Mean test accuracy over the tasks every arm finished

| # | arm | mean acc |
|---|---|---|
| 1 | lae_ovr_scratch_m | 0.7284 |
| 2 | lae_ovr_scratch_gm4 | 0.7250 |
| 3 | lae_ovr_scratch | 0.6843 |
