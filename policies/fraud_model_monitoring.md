---
document_type: product_monitoring_standard
product: fraud_model
model_id: FRD_004
owner: Fraud Risk
version: 1.0
---

# Fraud Model Monitoring Standard

## Scope
This standard applies to fraud-detection and fraud-prioritization models.

## Required monitoring
Review score distribution, PSI, precision, recall, false-positive rate, fraud capture, alert volume, investigation conversion, and performance by channel and fraud typology.

## Risk considerations
Fraud patterns can change quickly. A stable aggregate metric does not remove the need to review newly emerging fraud types and channel-level deterioration.

## Escalation
Any RED monitoring breach requires escalation. A material increase in false positives affecting legitimate customers must also be investigated even where aggregate fraud capture remains stable.

## Response
Monitoring actions may recommend threshold review, feature investigation, or targeted revalidation, but production threshold changes require authorized approval.