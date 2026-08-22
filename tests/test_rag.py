from pathlib import Path
from textwrap import dedent

import pytest

from model_monitoring.rag.answering import answer_policy_question, build_grounded_answer
from model_monitoring.rag.retrieval import PolicyRetriever, load_markdown_documents


def _write_policy(directory: Path, name: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        dedent(text).strip() + "\n",
        encoding="utf-8",
    )


def test_load_markdown_documents_preserves_source_metadata(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    _write_policy(
        policies,
        "monitoring_policy.md",
        """---
document_type: monitoring_policy
product: application_scorecard
---
        

# Monitoring Policy

PSI at or above 0.25 is RED and requires escalation.
""",
    )

    documents = load_markdown_documents(policies)
    print('\n-'*40)
    print((policies / "monitoring_policy.md").read_text(encoding="utf-8"))
    print('-'*40)
    
    assert len(documents) == 1
    assert documents[0].source.endswith("monitoring_policy.md")
    assert documents[0].metadata["document_type"] == "monitoring_policy"
    assert documents[0].metadata["product"] == "application_scorecard"
    assert documents[0].metadata.get("source", documents[0].source).endswith(
        "monitoring_policy.md"
    )


def test_retrieval_returns_relevant_psi_passage_and_source(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    db_path = tmp_path / "vectors.sqlite3"

    _write_policy(
        policies,
        "monitoring_policy.md",
        """# Monitoring Policy
        
        
        When PSI is 0.25 or higher, status is RED. A RED PSI breach requires escalation to the model owner and Model Risk Management, followed by investigation of the population shift.
""",
    )
    _write_policy(
        policies,
        "fraud_model_monitoring.md",
        """# Fraud Monitoring

Fraud models are monitored for precision, recall, false positives, alert volume and fraud capture.
""",
    )

    retriever = PolicyRetriever(policies_dir=policies, db_path=db_path)
    indexed = retriever.build_index()
    passages = retriever.retrieve(
        "What action is required when PSI exceeds 0.25?",
        top_k=2,
    )

    assert indexed >= 2
    assert passages
    assert passages[0].source.endswith("monitoring_policy.md")
    assert "PSI" in passages[0].text
    assert "escalation" in passages[0].text.lower()
    assert passages[0].score >= passages[1].score


def test_answer_layer_uses_retrieved_evidence_without_retrieving(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    db_path = tmp_path / "vectors.sqlite3"
    _write_policy(
        policies,
        "escalation_procedure.md",
        """# Escalation Procedure

When PSI is 0.25 or higher, the PSI status is RED. The required action is escalation to the model owner and MRM, followed by investigation of the segments driving the shift.
""",
    )

    retriever = PolicyRetriever(policies_dir=policies, db_path=db_path)
    retriever.build_index()
    passages = retriever.retrieve("What action is required when PSI exceeds 0.25?")

    answer = build_grounded_answer(
        "What action is required when PSI exceeds 0.25?",
        passages,
    )

    assert "RED" in answer
    assert "escalation" in answer.lower()


def test_end_to_end_response_contains_answer_passage_and_metadata(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    db_path = tmp_path / "vectors.sqlite3"
    _write_policy(
        policies,
        "escalation_procedure.md",
        """---
document_type: escalation_procedure
---

# Escalation Procedure

When PSI is 0.25 or higher, the PSI status is RED. The required action is escalation to the model owner and MRM, followed by investigation of the segments driving the shift.
""",
    )

    retriever = PolicyRetriever(policies_dir=policies, db_path=db_path)
    retriever.build_index()
    response = answer_policy_question(
        "What action is required when PSI exceeds 0.25?",
        retriever,
        top_k=1,
    )

    assert "escalation" in response["answer"].lower()
    assert response["passages"]
    assert response["passages"][0]["source"].endswith("escalation_procedure.md")
    assert response["passages"][0]["metadata"].get("document_type") == "escalation_procedure"


def test_repository_policy_corpus_answers_requested_psi_question(tmp_path: Path) -> None:
    retriever = PolicyRetriever(
        policies_dir=Path(__file__).resolve().parents[1] / "policies",
        db_path=tmp_path / "repository_policy_vectors.sqlite3",
    )
    retriever.build_index()

    response = answer_policy_question(
        "What action is required when PSI exceeds 0.25?",
        retriever,
        top_k=3,
    )

    sources = {passage["source"].rsplit("/", 1)[-1] for passage in response["passages"]}
    assert sources & {"monitoring_policy.md", "escalation_procedure.md"}
    assert "escalation" in response["answer"].lower()
    assert "red" in response["answer"].lower()


def test_empty_query_is_rejected(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    db_path = tmp_path / "vectors.sqlite3"
    _write_policy(policies, "policy.md", "# Policy\n\nSome policy text.")

    retriever = PolicyRetriever(policies_dir=policies, db_path=db_path)
    retriever.build_index()

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("   ")


def test_read_only_retriever_can_search_but_cannot_rebuild(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    db_path = tmp_path / "vectors.sqlite3"
    _write_policy(
        policies,
        "policy.md",
        "# Monitoring Policy\n\nA RED PSI breach requires escalation.",
    )
    PolicyRetriever(policies_dir=policies, db_path=db_path).build_index()

    retriever = PolicyRetriever(
        policies_dir=policies,
        db_path=db_path,
        read_only=True,
    )

    assert retriever.retrieve("RED PSI escalation", top_k=1)
    with pytest.raises(PermissionError, match="read-only"):
        retriever.build_index()
