Reserved for future evaluation datasets and evaluation code. No agent, RAG, or MCP evaluations are implemented in this initial phase.

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

This dataset is intentionally retrieval-focused. It does not score generated answers or agent behaviour.

## Run the retrieval evaluation

From the repository root (with the project installed), run:

```bash
python evals/evaluate_retrieval.py --output .rag/retrieval_report.json
```

The runner indexes the policy corpus once, evaluates every one of the 30 cases,
prints a JSON report, and optionally saves the same report with `--output`. The
report contains ranked passages and scores for each case, plus overall and
`risk_level`/`difficulty` slices for Hit@1, Hit@3, mean reciprocal rank (MRR),
context precision, context recall, and normalized discounted cumulative gain
(NDCG). `--top-k` controls the context cutoff and defaults to 3 (and cannot be
less than 3 because Hit@3 is always reported).

A retrieved chunk is considered relevant when its source filename equals the
case's `expected_document`. Context precision is the relevant fraction of the
top-k context; context recall is the fraction of all relevant chunks from the
expected document retrieved in that context; and NDCG uses binary chunk
relevance. The three deliberately unanswerable cases remain visible in the
per-case results but are excluded from these ranking metrics because the current
retriever always returns passages and has no abstention signal.


## Run the generation evaluation

The same golden questions and expected topics can also evaluate the answer layer:

```bash
python evals/evaluate_generation.py --output .rag/generation_report.json
```

The report includes four deterministic, zero-dependency regression metrics for
answerable cases:

- **Faithfulness**: precision of answer content tokens supported by retrieved context.
- **Answer relevance**: recall of question content tokens in the answer.
- **Answer completeness**: recall of expected-topic content tokens in the answer.
- **Answer correctness**: F1 overlap between answer and expected-topic content tokens.

It also reports **abstention accuracy** across all cases. For answerable cases,
producing an answer is correct; for unanswerable cases, the answer must equal the
pipeline's standard no-evidence response. Results are aggregated overall and by
risk level and difficulty.

These lexical metrics are transparent and reproducible, making them suitable
for CI regression tracking. They do not understand synonyms, contradictions, or
policy meaning, so releases—especially changes affecting HIGH or CRITICAL risk
questions—still require human review or a separately validated semantic judge.