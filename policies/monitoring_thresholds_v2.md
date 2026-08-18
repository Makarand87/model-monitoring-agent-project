# Initial Monitoring Thresholds

These provisional rules give the project a transparent policy baseline and are implemented by the deterministic monitoring application.

| Metric | Green | Amber | Red |
|---|---|---|---|
| PSI | `<= 0.10` | `> 0.10 to <= 0.25` | `> 0.25` |
| AUC deterioration from previous period | `<= 0.03` | `> 0.03 to <= 0.05` | `> 0.05` |
| Bad-rate relative increase | `< 10%` | `10% to < 20%` | `>= 20%` |
| Approval-rate absolute change | `< 0.03` | `0.03 to < 0.05` | `>= 0.05` |

Thresholds must be reviewed and approved by Model Risk Management and the model owner before production use.
