import pytest

from evals.evaluate_generation import ABSTENTION, aggregate, evaluate_answer


def _case(expected_document: str | None = "policy.md") -> dict[str, str | None]:
    return {
        "question": "What action is required for a RED PSI breach?",
        "expected_document": expected_document,
        "expected_passage/topic": "RED PSI breach requires escalation to MRM and investigation",
        "risk_level": "HIGH", "difficulty": "EASY",
    }


def test_grounded_complete_answer_scores_all_generation_dimensions() -> None:
    answer = "A RED PSI breach requires escalation to MRM and investigation."
    result = evaluate_answer(_case(), answer, [answer])
    assert result["faithfulness"] == 1.0
    assert result["answer_completeness"] == 1.0
    assert result["answer_correctness"] == 1.0
    assert result["answer_relevance"] > 0.5
    assert result["abstention_correct"] is True


def test_unsupported_words_reduce_faithfulness() -> None:
    result = evaluate_answer(_case(), "A RED PSI breach requires escalation to MRM and immediate model retirement.",
                             ["A RED PSI breach requires escalation to MRM."])
    assert result["faithfulness"] < 1.0
    assert result["answer_completeness"] < 1.0


def test_unanswerable_case_only_scores_abstention() -> None:
    result = evaluate_answer(_case(None), ABSTENTION, [])
    assert result["scored"] is False
    assert result["abstention_correct"] is True
    assert aggregate([result])["abstention_accuracy"] == 1.0
    assert aggregate([result])["faithfulness"] is None


def test_answerable_case_requires_reference_topic() -> None:
    case = _case()
    case["expected_passage/topic"] = ""
    with pytest.raises(ValueError, match="expected_passage/topic"):
        evaluate_answer(case, "answer", ["context"])