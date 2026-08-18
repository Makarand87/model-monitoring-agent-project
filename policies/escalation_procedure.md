---
document_type: escalation_procedure
scope: model_monitoring
owner: Model Risk Management
version: 1.0
effective_date: 2026-08-01
---

# Monitoring Escalation Procedure

## Trigger
Escalation is mandatory when any monitoring metric is classified RED. Investigation is mandatory when two or more metrics are classified AMBER during the same monitoring period.

## Required steps for RED
1. Record the breached metric, observed value, threshold, model, and monitoring period.
2. Notify the model owner and Model Risk Management.
3. Perform segment-level and historical analysis to identify likely drivers.
4. Assess whether the issue is caused by data, population, business strategy, implementation, or model deterioration.
5. Document immediate controls and recommended remediation.
6. Determine whether targeted revalidation or increased monitoring frequency is required.

## PSI example
When PSI is 0.25 or higher, the PSI status is RED. The required action is escalation to the model owner and MRM, followed by investigation of the population segments driving the shift.

## Human decision
The monitoring process may recommend actions, but material remediation decisions such as model recalibration, cutoff changes, redevelopment, or model suspension require authorized human approval.