"""Evaluate policy retrieval against the repository's JSONL golden set.

Run from the repository root with::

    python evals/evaluate_retrieval.py

The report is JSON so it can be inspected by a person or consumed by CI.
Unanswerable cases are retained in the per-case output, but are excluded from
ranking metrics because ``PolicyRetriever`` has no abstention mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from model_monitoring.rag.retrieval import (
    PolicyRetriever,
    chunk_documents,
    load_markdown_documents,
)

REQUIRED_CASES = 30
DEFAULT_TOP_K = 3


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the JSONL evaluation cases."""
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != REQUIRED_CASES:
        raise ValueError(f"Expected {REQUIRED_CASES} cases, found {len(cases)} in {path}")
    required = {"question", "expected_document", "risk_level", "difficulty"}
    for number, case in enumerate(cases, start=1):
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Case {number} is missing fields: {sorted(missing)}")
    return cases


def _dcg(relevance: list[int]) -> float:
    return sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, start=1))


def evaluate_case(
    case: dict[str, Any],
    retrieved: list[Any],
    relevant_chunk_count: int,
    top_k: int,
) -> dict[str, Any]:
    """Score one case using binary source-document relevance."""
    expected = case["expected_document"]
    sources = [passage.source for passage in retrieved]
    result: dict[str, Any] = {
        "question": case["question"],
        "expected_document": expected,
        "risk_level": case["risk_level"],
        "difficulty": case["difficulty"],
        "retrieved": [
            {
                "rank": rank,
                "source": passage.source,
                "chunk_index": passage.chunk_index,
                "score": passage.score,
            }
            for rank, passage in enumerate(retrieved, start=1)
        ],
    }
    if expected is None:
        result.update({"scored": False, "reason": "unanswerable case; retriever cannot abstain"})
        return result

    relevance = [int(source == expected) for source in sources]
    first_relevant_rank = next((rank for rank, value in enumerate(relevance, start=1) if value), None)
    ideal_relevant = min(relevant_chunk_count, top_k)
    ideal = [1] * ideal_relevant + [0] * (top_k - ideal_relevant)
    result.update(
        {
            "scored": True,
            "hit_at_1": relevance[0] if relevance else 0,
            "hit_at_3": int(any(relevance[:3])),
            "reciprocal_rank": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
            "context_precision": sum(relevance) / top_k,
            "context_recall": sum(relevance) / relevant_chunk_count,
            "ndcg": _dcg(relevance) / _dcg(ideal) if ideal_relevant else 0.0,
        }
    )
    return result


def aggregate(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Average all metrics across answerable cases in a collection."""
    scored = [result for result in results if result["scored"]]
    metric_names = (
        "hit_at_1",
        "hit_at_3",
        "reciprocal_rank",
        "context_precision",
        "context_recall",
        "ndcg",
    )
    summary: dict[str, Any] = {"case_count": len(scored)}
    summary.update(
        {name: sum(result[name] for result in scored) / len(scored) if scored else None for name in metric_names}
    )
    summary["mrr"] = summary.pop("reciprocal_rank")
    return summary


def grouped_metrics(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[result[field]].append(result)
    return {name: aggregate(group) for name, group in sorted(groups.items())}


def run_evaluation(
    dataset: Path,
    policies_dir: Path,
    db_path: Path,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if top_k < 3:
        raise ValueError("top_k must be at least 3 to calculate Hit@3")

    cases = load_cases(dataset)
    retriever = PolicyRetriever(policies_dir=policies_dir, db_path=db_path)
    indexed_chunks = retriever.build_index()
    chunks = chunk_documents(
        load_markdown_documents(policies_dir),
        chunk_size=retriever.chunk_size,
        chunk_overlap=retriever.chunk_overlap,
    )
    relevant_counts: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        relevant_counts[chunk.source] += 1

    results = [
        evaluate_case(
            case,
            retriever.retrieve(case["question"], top_k=top_k),
            relevant_counts.get(case["expected_document"], 0),
            top_k,
        )
        for case in cases
    ]
    return {
        "configuration": {
            "dataset": str(dataset),
            "policies_dir": str(policies_dir),
            "top_k": top_k,
            "indexed_chunks": indexed_chunks,
            "total_cases": len(cases),
            "unanswerable_cases_excluded_from_metrics": sum(not item["scored"] for item in results),
            "relevance_definition": "a retrieved chunk is relevant when its source equals expected_document",
        },
        "cases": results,
        "metrics": {
            "overall": aggregate(results),
            "by_risk_level": grouped_metrics(results, "risk_level"),
            "by_difficulty": grouped_metrics(results, "difficulty"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/rag_retrieval_dataset.jsonl"))
    parser.add_argument("--policies-dir", type=Path, default=Path("policies"))
    parser.add_argument("--db-path", type=Path, default=Path(".rag/evaluation_vectors.sqlite3"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args()

    report = run_evaluation(args.dataset, args.policies_dir, args.db_path, args.top_k)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
