import json
import sys
from types import SimpleNamespace

import pytest

from evals.evaluate_generation import judge_answer


def test_ragas_evaluation_dependencies_are_importable() -> None:
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    assert all(
        dependency is not None
        for dependency in (
            EvaluationDataset,
            evaluate,
            LangchainEmbeddingsWrapper,
            LangchainLLMWrapper,
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
    )


def test_judge_answer_validates_and_returns_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class Responses:
        @staticmethod
        def create(**kwargs):
            assert kwargs["text"]["format"]["type"] == "json_object"
            assert "json" in kwargs["input"]
            return type("Response", (), {"output_text": json.dumps({"score": 5, "reason": "Grounded"})})()

    class Client:
        responses = Responses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: Client()))

    assert judge_answer("q", "reference", "answer", "judge") == {
        "score": 5,
        "reason": "Grounded",
    }


def test_judge_answer_rejects_out_of_range_score(monkeypatch: pytest.MonkeyPatch) -> None:
    response = type("Response", (), {"output_text": '{"score": 6}'})()
    client = type("Client", (), {"responses": type("Responses", (), {"create": lambda self, **kwargs: response})()})()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))

    with pytest.raises(ValueError, match="invalid score"):
        judge_answer("q", "reference", "answer", "judge")
