---
document_type: model_governance_policy
scope: enterprise_models
owner: Model Governance
version: 1.0
effective_date: 2026-08-01
---

# Model Governance Policy

## Governance requirements
Every production model must have an approved owner, documented purpose, risk tier, monitoring frequency, validation status, known limitations, approved thresholds, and escalation route.

## Accountability
The model owner is accountable for first-line monitoring and remediation. Model Risk Management provides independent challenge and validation. Material model-risk decisions must be documented and approved by the designated governance forum or authority.

## Change management
Material changes to model methodology, features, segmentation, calibration, implementation, or approved use must be assessed for validation impact before production release.

## Monitoring governance
Monitoring evidence must be reproducible and retained. RED breaches cannot be closed solely through management commentary; supporting analysis and an approved disposition are required.

## Automation controls
Automated monitoring systems may calculate metrics, retrieve policy, classify deterministic thresholds, and prepare recommendations. They must not independently approve material model changes or risk acceptance.