# Evaluations

This folder now contains the first retrieval-focused evaluation dataset for the policy RAG module.

## `rag_retrieval_dataset.jsonl`

The dataset contains 30 hand-authored golden questions grounded in the current `policies/` corpus.

Each record has:

- `question`: query presented to the retriever.
- `expected_document`: policy document expected to contain the best supporting passage; `null` means the corpus does not contain an authoritative answer.
- `expected_passage/topic`: concise description of the evidence or topic that should be retrieved.
- `risk_level`: LOW, MEDIUM, HIGH, or CRITICAL based on the consequence of incorrect retrieval.
- `difficulty`: EASY, MEDIUM, or HARD.

Coverage includes PSI and AUC thresholds, two-AMBER logic, escalation, revalidation, governance restrictions, application scorecards, behaviour scorecards, collections, fraud, credit-limit models, cross-document questions, overlapping guidance, and deliberately unanswerable questions.

This dataset is intentionally retrieval-focused. It does not yet score generated answers or agent behaviour. A later evaluation runner can use it for metrics such as Hit@K, source accuracy, MRR, and unanswerable-query handling.
