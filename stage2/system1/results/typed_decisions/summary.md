# LocalLLaMA/typed-decisions (Laya's benchmark): 2,000 test decisions

| system | accuracy | soft acc | Brier | ECE | score MAE | ms / case (CPU, 1 thread) |
|---|---|---|---|---|---|---|
| Laya fine-tuned (421M, GPU; published) | 0.766 | - | - | - | - | - |
| teacher self-agreement (published) | 0.735 | - | - | - | - | - |
| TypeSafe Jev 1.13.0 (published) | 0.727 | - | - | - | - | - |
| bow | 0.681 | 0.5066 | 0.1811 | 0.0574 | 0.3798 | - |
| berttiny | 0.661 | 0.4982 | 0.1598 | 0.0447 | 0.4113 | - |
| majority | 0.483 | 0.392 | 0.2011 | 0.0534 | 0.5733 | - |
| majority class (published) | 0.461 | - | - | - | - | - |
| Laya zero-shot (published) | 0.362 | - | - | - | - | - |

| system | agent_trace_observability | customer_service | invoice_processing | security_incidents | choice | noul | score |
|---|---|---|---|---|---|---|---|
| majority | 0.362 | 0.452 | 0.518 | 0.602 | 0.4183 | 0.6417 | 0.4138 |
| bow | 0.674 | 0.706 | 0.664 | 0.682 | 0.6433 | 0.7883 | 0.63 |
| berttiny | 0.646 | 0.684 | 0.642 | 0.672 | 0.63 | 0.765 | 0.6062 |
