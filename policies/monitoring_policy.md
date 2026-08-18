---
document_type: monitoring_policy
scope: all_credit_risk_models
owner: Model Risk Management
version: 1.0
effective_date: 2026-08-01
---

# Model Performance Monitoring Policy

## Purpose
This policy defines the minimum monitoring requirements for production credit-risk and decision models.

## Core metrics
Models must be reviewed for population stability, discriminatory performance, outcome movement, approval-rate movement, missing-data changes, and material segment deterioration.

## PSI thresholds
Population Stability Index (PSI) is classified as follows:

- PSI < 0.10: GREEN.
- 0.10 <= PSI < 0.25: AMBER.
- PSI >= 0.25: RED.

A RED PSI breach requires escalation to the model owner and Model Risk Management (MRM). The monitoring analyst must investigate the main population segments contributing to the shift and document whether the movement is caused by business mix, data changes, strategy changes, or potential model deterioration.

## AUC thresholds
AUC deterioration from the approved baseline is classified as:

- decrease < 0.03: GREEN.
- decrease from 0.03 to < 0.05: AMBER.
- decrease >= 0.05: RED.

## Overall action rule
Any RED monitoring breach requires escalation. Two or more AMBER breaches in the same monitoring period require investigation and documented management commentary.

## Evidence
Every monitoring conclusion must record the metric value, threshold used, monitoring period, affected model, source data, and reviewer.