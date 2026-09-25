# Jev-style decisions: pretrained LogicAE vs bert-tiny, adapted on one shared objective

LogicAE arms: 2000 steps, batch 32. Test metrics after one dev-fitted temperature per question. Latency: one decision (state x all options), CPU, 1 thread.

## qnli (noul, K=2, test n=2000)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | bert_pair | 0.7495 | 0.0262 | 0.5148 | 0.3416 | 0.8640 | - | 1.264 |
| 2 | lae_ovr_scratch_g2 | 0.5840 | 0.0225 | 0.6712 | 0.4785 | 0.6310 | - | 1.8763 |
| 3 | lae_ovr_scratch_g4 | 0.5820 | 0.0120 | 0.6740 | 0.4812 | 0.6230 | - | 1.87 |
| 4 | lae_ovr_scratch | 0.5795 | 0.0136 | 0.6743 | 0.4815 | 0.6230 | - | 1.8684 |

## sst2 (choice, K=2, test n=872)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | bert_pair | 0.7982 | 0.0773 | 0.4919 | 0.3093 | 0.9037 | - | 1.198 |
| 2 | lae_ovr_scratch | 0.7798 | 0.0969 | 0.5775 | 0.3402 | 0.8693 | - | 0.9674 |
| 3 | lae_ovr_scratch_g4 | 0.7775 | 0.1020 | 0.5766 | 0.3397 | 0.8647 | - | 0.9774 |
| 4 | lae_ovr_scratch_g2 | 0.7752 | 0.0966 | 0.5698 | 0.3374 | 0.8853 | - | 1.0251 |

## emotion (choice, K=6, test n=2000)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | bert_pair | 0.8960 | 0.0225 | 0.2523 | 0.1380 | 0.9930 | - | 2.514 |

## Mean test accuracy over the tasks every arm finished

| # | arm | mean acc |
|---|---|---|
| 1 | bert_pair | 0.8146 |
