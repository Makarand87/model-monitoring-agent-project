# Model Monitoring Decision Policy

## Purpose

This policy converts the monitoring classifications for one model and one monitoring period into a deterministic action.

## Valid inputs

Each monitored metric must have exactly one status: `GREEN`, `AMBER`, or `RED`.

The decision uses:

- `red_count`: number of metrics classified as `RED`.
- `amber_count`: number of metrics classified as `AMBER`.

## Decision rules

Rules are evaluated in the following order. The first matching rule determines the action.

| Priority | Condition | Decision | Required action |
|---|---|---|---|
| 1 | `red_count >= 1` | `ESCALATE` | Escalate the monitoring result to Model Risk Management and the model owner. |
| 2 | `red_count == 0` and `amber_count >= 2` | `INVESTIGATE` | Open an investigation and document the cause, impact, and proposed response. |
| 3 | `red_count == 0` and `amber_count < 2` | `CONTINUE_MONITORING` | Record the result and continue routine monitoring. |

## Deterministic precedence

A `RED` breach always takes priority. For example, one `RED` and two `AMBER` breaches produce `ESCALATE`, not `INVESTIGATE`.

Equivalent pseudocode:

```python
if red_count >= 1:
    decision = "ESCALATE"
elif amber_count >= 2:
    decision = "INVESTIGATE"
else:
    decision = "CONTINUE_MONITORING"
```

## Examples

| GREEN | AMBER | RED | Decision |
|---:|---:|---:|---|
| 2 | 0 | 0 | `CONTINUE_MONITORING` |
| 1 | 1 | 0 | `CONTINUE_MONITORING` |
| 0 | 2 | 0 | `INVESTIGATE` |
| 0 | 3 | 0 | `INVESTIGATE` |
| 1 | 0 | 1 | `ESCALATE` |
| 0 | 2 | 1 | `ESCALATE` |

## Input-control rule

If a metric has a missing or unrecognised status, return `DATA_QUALITY_ERROR`. Do not infer a status or make an automated monitoring decision until the input is corrected.

## Scope

This policy defines decision logic only. It does not assign investigation timelines, escalation recipients beyond the named governance roles, or remediation actions.
