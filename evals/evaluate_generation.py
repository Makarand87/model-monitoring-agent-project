"""Evaluate answers produced by the policy RAG pipeline.

The evaluator is deterministic and dependency-free. Its lexical metrics are
regression signals, not a substitute for human or calibrated semantic review.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from model_monitoring.rag.answering import answer_policy_question
from model_monitoring.rag.retrieval import PolicyRetriever

DEFAULT_TOP_K = 3
ABSTENTION = "No relevant policy evidence was retrieved."
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what",
    "when", "which", "with",
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load generation cases and validate the required reference fields."""
    # cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    cases: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        case, position = decoder.raw_decode(text, position)
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {len(cases) + 1} must be a JSON object")
        cases.append(case)
 
    required = {"question", "expected_document", "expected_passage/topic", "risk_level", "difficulty"}
    for number, case in enumerate(cases, start=1):
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Case {number} is missing fields: {sorted(missing)}")
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases




def _tokens(text: str) -> set[str]:
    """Return normalized content tokens while retaining numbers and negation."""
    return {
        token for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
        if token not in _STOP_WORDS
    }


def _overlap_scores(candidate: str, reference: str) -> tuple[float, float, float]:
    candidate_tokens, reference_tokens = _tokens(candidate), _tokens(reference)
    overlap = len(candidate_tokens & reference_tokens)
    precision = overlap / len(candidate_tokens) if candidate_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_answer(case: dict[str, Any], answer: str, contexts: Iterable[str]) -> dict[str, Any]:
    """Score a generated answer against its reference topic and RAG context."""
    expected_document = case["expected_document"]
    abstained = answer.strip().lower() == ABSTENTION.lower()
    result: dict[str, Any] = {
        "question": case["question"], "expected_document": expected_document,
        "risk_level": case["risk_level"], "difficulty": case["difficulty"],
        "answer": answer, "abstained": abstained,
        "abstention_correct": abstained if expected_document is None else not abstained,
    }
    if expected_document is None:
        result.update({"scored": False, "reason": "unanswerable case; only abstention is scored"})
        return result

    reference = case.get("expected_passage/topic")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Answerable cases require a non-empty expected_passage/topic")
    context = " ".join(contexts)
    faithfulness, _, _ = _overlap_scores(answer, context)
    _, completeness, correctness = _overlap_scores(answer, reference)
    question_terms, answer_terms = _tokens(case["question"]), _tokens(answer)
    relevance = len(question_terms & answer_terms) / len(question_terms) if question_terms else 0.0
    result.update({
        "scored": True, "reference": reference, "faithfulness": faithfulness,
        "answer_relevance": relevance, "answer_completeness": completeness,
        "answer_correctness": correctness,
    })
    return result


def aggregate(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Average generation metrics and report abstention accuracy separately."""
    items = list(results)
    scored = [item for item in items if item["scored"]]
    metrics = ("faithfulness", "answer_relevance", "answer_completeness", "answer_correctness")
    summary: dict[str, Any] = {
        "case_count": len(items), "answerable_case_count": len(scored),
        "unanswerable_case_count": len(items) - len(scored),
        "abstention_accuracy": sum(item["abstention_correct"] for item in items) / len(items) if items else None,
    }
    summary.update({metric: sum(item[metric] for item in scored) / len(scored) if scored else None for metric in metrics})
    return summary


def _grouped(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[result[field]].append(result)
    return {name: aggregate(group) for name, group in sorted(groups.items())}


def run_evaluation(dataset: Path, policies_dir: Path, db_path: Path, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    cases = load_cases(dataset)
    retriever = PolicyRetriever(policies_dir=policies_dir, db_path=db_path)
    indexed_chunks = retriever.build_index()
    results = []
    for case in cases:
        response = answer_policy_question(case["question"], retriever, top_k=top_k)
        results.append(evaluate_answer(case, str(response["answer"]), [item["text"] for item in response["passages"]]))
    return {
        "configuration": {"dataset": str(dataset), "policies_dir": str(policies_dir),
                          "top_k": top_k, "indexed_chunks": indexed_chunks,
                          "metric_method": "deterministic content-token overlap"},
        "cases": results,
        "metrics": {"overall": aggregate(results),
                    "by_risk_level": _grouped(results, "risk_level"),
                    "by_difficulty": _grouped(results, "difficulty")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/rag_retrieval_dataset.jsonl"))
    parser.add_argument("--policies-dir", type=Path, default=Path("policies"))
    parser.add_argument("--db-path", type=Path, default=Path(".rag/generation_evaluation_vectors.sqlite3"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args()
    
    report = run_evaluation(args.dataset, args.policies_dir, args.db_path, args.top_k)
    rendered = json.dumps(report, indent=2)
    # print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()