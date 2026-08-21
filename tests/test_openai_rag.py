from pathlib import Path
from types import SimpleNamespace

from model_monitoring.openai_rag.retrieval import OpenAIEmbedder, OpenAIPolicyRetriever
from model_monitoring.rag.retrieval import PolicyRetriever


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def create(self, *, model: str, input: str) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input})
        lowered = input.lower()
        embedding = [
            float("psi" in lowered or "population" in lowered),
            float("escalation" in lowered or "escalate" in lowered),
            float("fraud" in lowered),
        ]
        return SimpleNamespace(data=[SimpleNamespace(embedding=embedding)])


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


def _write_policy(directory: Path, name: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text, encoding="utf-8")


def test_openai_embedder_uses_configured_model() -> None:
    client = _FakeOpenAIClient()
    embedder = OpenAIEmbedder(model="test-embedding-model", client=client)

    vector = embedder.embed("PSI requires escalation")

    assert vector == [1.0, 1.0, 0.0]
    assert client.embeddings.calls == [
        {"model": "test-embedding-model", "input": "PSI requires escalation"}
    ]


def test_openai_retriever_keeps_policy_retriever_retrieve_unchanged() -> None:
    assert OpenAIPolicyRetriever.retrieve is PolicyRetriever.retrieve


def test_openai_retriever_builds_index_and_returns_policy_metadata(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    db_path = tmp_path / "openai_vectors.sqlite3"
    _write_policy(
        policies,
        "monitoring_policy.md",
        """---
document_type: monitoring_policy
---

# Monitoring Policy

A PSI breach above 0.25 is RED and requires escalation to Model Risk Management.
""",
    )
    _write_policy(
        policies,
        "fraud_policy.md",
        """# Fraud Policy

Fraud models are reviewed for fraud capture and false positives.
""",
    )

    client = _FakeOpenAIClient()
    retriever = OpenAIPolicyRetriever(
        policies_dir=policies,
        db_path=db_path,
        client=client,
    )

    indexed = retriever.build_index()
    passages = retriever.retrieve(
        "What escalation is required for a PSI population shift?",
        top_k=2,
    )

    assert indexed == 2
    assert passages[0].source == "monitoring_policy.md"
    assert "PSI" in passages[0].text
    assert passages[0].metadata["document_type"] == "monitoring_policy"
    assert passages[0].score >= passages[1].score
    assert len(client.embeddings.calls) == 3
