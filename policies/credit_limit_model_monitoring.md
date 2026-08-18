---
document_type: product_monitoring_standard
product: credit_limit_model
model_id: LIM_005
owner: Account Management Risk
version: 1.0
---

# Credit Limit Model Monitoring Standard

## Scope
This standard applies to models supporting credit-line increase, decrease, or limit-management decisions.

## Required monitoring
Review PSI, AUC/Gini where applicable, utilization, delinquency after limit action, exposure movement, acceptance rate, override rate, and performance by risk tier.

## Stability interpretation
Population stability must be reviewed alongside strategy changes because eligibility rules can materially change the monitored population even when the model itself is unchanged.

## Escalation
Any RED metric requires escalation under the enterprise Monitoring Policy. Two or more AMBER metrics require investigation and documented commentary.

## Governance
Limit changes affecting customer exposure are material actions and require approved decision rules and human governance. The monitoring system may recommend investigation but may not autonomously alter production limit strategy.