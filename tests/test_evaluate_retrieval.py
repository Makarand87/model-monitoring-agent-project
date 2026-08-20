from pathlib import Path

import pytest
from evals.evaluate_retrieval import aggregate, evaluate_case, load_cases
from model_monitoring.rag.retrieval import RetrievedPassage


def _passage(source: str) -> RetrievedPassage:
    return RetrievedPassage("text", source, source, 0, 0.5, {})


def test_repository_dataset_contains_exactly_30_cases() -> None:
    assert len(load_cases(Path("evals/rag_retrieval_dataset.jsonl"))) == 30


def test_case_metrics_use_binary_document_relevance() -> None:
    case = {
        "question": "question",
        "expected_document": "gold.md",
        "risk_level": "HIGH",
        "difficulty": "EASY",
    }
    result = evaluate_case(
        case,
        [_passage("other.md"), _passage("gold.md"), _passage("gold.md")],
        relevant_chunk_count=4,
        top_k=3,
    )

    assert result["hit_at_1"] == 0
    assert result["hit_at_3"] == 1
    assert result["reciprocal_rank"] == 0.5
    assert result["context_precision"] == pytest.approx(2 / 3)
    assert result["context_recall"] == 0.5
    assert 0 < result["ndcg"] < 1


def test_unanswerable_case_is_excluded_from_aggregate() -> None:
    case = {
        "question": "question",
        "expected_document": None,
        "risk_level": "HIGH",
        "difficulty": "HARD",
    }
    result = evaluate_case(case, [_passage("other.md")], 0, 3)

    assert result["scored"] is False
    assert aggregate([result])["case_count"] == 0
    assert aggregate([result])["mrr"] is None
