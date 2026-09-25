# Jev-style decisions: pretrained LogicAE vs bert-tiny, adapted on one shared objective

LogicAE arms: 2000 steps, batch 32. Test metrics after one dev-fitted temperature per question. Latency: one decision (state x all options), CPU, 1 thread.

## emotion (choice, K=6, test n=2000)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | bert_pair | 0.8960 | 0.0225 | 0.2523 | 0.1380 | 0.9930 | - | 1.441 |
| 2 | bert_head | 0.8915 | 0.0306 | 0.2617 | 0.1411 | 0.9960 | - | - |
| 3 | lae_ovr_scratch | 0.6590 | 0.0664 | 1.0305 | 0.4972 | 0.7800 | - | 3.5585 |
| 4 | lae_ovr_pt_keep | 0.5145 | 0.1541 | 1.4623 | 0.6970 | 0.5470 | - | 3.4883 |
| 5 | lae_pt_keep | 0.3410 | 0.0293 | 1.5834 | 0.7616 | 0.3450 | - | 19.417 |
| 6 | lae_pt_keep | 0.3400 | 0.0371 | 1.5819 | 0.7615 | 0.3360 | - | 21.151 |
| 7 | lae_pt | 0.2905 | 0.0208 | 1.5819 | 0.7616 | 0.2870 | - | 21.612 |
| 8 | lae_pt | 0.2905 | 0.0207 | 1.5814 | 0.7611 | 0.2840 | - | 21.451 |
| 9 | lae_scratch | 0.2905 | 0.0248 | 1.5637 | 0.7559 | 0.3080 | - | 29.405 |
| 10 | lae_scratch | 0.2905 | 0.0248 | 1.5637 | 0.7559 | 0.3080 | - | 19.773 |

## agnews (choice, K=4, test n=2000)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | bert_head | 0.9055 | 0.0227 | 0.2767 | 0.1465 | 0.9900 | - | - |
| 2 | bert_pair | 0.8900 | 0.0222 | 0.3082 | 0.1584 | 0.9820 | - | 1.906 |
| 3 | lae_ovr_scratch | 0.8825 | 0.0239 | 0.3646 | 0.1826 | 0.9770 | - | 2.3153 |
| 4 | lae_ovr_pt_keep | 0.8205 | 0.0223 | 0.5045 | 0.2593 | 0.9640 | - | 2.5898 |

## sst5 (score, K=5, test n=2000)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | bert_head | 0.4295 | 0.0354 | 1.2760 | 0.6745 | 0.5010 | 0.7707 | - |
| 2 | lae_ovr_scratch | 0.3975 | 0.0427 | 1.4036 | 0.7189 | 0.4540 | 0.9076 | 2.8945 |
| 3 | lae_ovr_pt_keep | 0.3640 | 0.0595 | 1.4732 | 0.7459 | 0.3980 | 1.0153 | 2.8879 |
| 4 | bert_pair | 0.2305 | 0.0600 | 1.5601 | 0.7823 | 0.2210 | 1.1338 | 1.138 |

## sst2 (choice, K=2, test n=872)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | bert_head | 0.8028 | 0.0736 | 0.4558 | 0.2928 | 0.9197 | - | - |
| 2 | bert_pair | 0.7982 | 0.0773 | 0.4919 | 0.3093 | 0.9037 | - | 0.599 |
| 3 | lae_ovr_scratch | 0.7683 | 0.0844 | 0.5654 | 0.3430 | 0.8601 | - | 0.5878 |
| 4 | lae_ovr_pt_keep | 0.5287 | 0.1537 | 0.7158 | 0.5188 | 0.6170 | - | 0.5794 |

## jailbreak (noul, K=2, test n=262)

| # | arm | acc | ECE | NLL | Brier | acc@50% cov | MAE | ms/decision |
|---|---|---|---|---|---|---|---|---|
| 1 | bert_head | 0.9695 | 0.0229 | 0.1080 | 0.0618 | 1.0000 | - | - |
| 2 | bert_pair | 0.9656 | 0.0315 | 0.0909 | 0.0540 | 1.0000 | - | 1.023 |
| 3 | lae_ovr_scratch | 0.9427 | 0.0210 | 0.1658 | 0.0872 | 0.9924 | - | 0.6371 |
| 4 | lae_ovr_pt_keep | 0.9198 | 0.0488 | 0.2464 | 0.1372 | 0.9924 | - | 0.613 |

## Mean test accuracy over the tasks every arm finished

| # | arm | mean acc |
|---|---|---|
| 1 | bert_head | 0.7998 |
| 2 | bert_pair | 0.7561 |
| 3 | lae_ovr_scratch | 0.7300 |
| 4 | lae_ovr_pt_keep | 0.6295 |
