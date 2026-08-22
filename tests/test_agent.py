"""Tests for the read-only monitoring agent boundary."""

import logging

import pytest

from model_monitoring.agent import (
    MonitoringAgent,
    MonitoringRecommendation,
    PolicyEvidence,
    RecommendedAction,
)
from model_monitoring.models import Breach, MetricName
from model_monitoring.rag.retrieval import RetrievedPassage


class SpyPolicySearch:
    def __init__(self, passages: list[RetrievedPassage]) -> None:
        self.passages = passages
        self.queries: list[tuple[str, int]] = []
        self.build_index_called = False

    def build_index(self) -> None:
        self.build_index_called = True
        raise AssertionError("The agent must not build or mutate the policy index")

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]:
        self.queries.append((query, top_k))
        return self.passages[:top_k]


@pytest.fixture
def policy_passage() -> RetrievedPassage:
    return RetrievedPassage(
        text=(
            "A RED monitoring breach requires escalation to the model owner and "
            "Model Risk Management, followed by investigation."
        ),
        source="monitoring_policy.md",
        path="policies/monitoring_policy.md",
        chunk_index=2,
        score=0.91,
        metadata={"document_type": "monitoring_policy"},
    )


def test_agent_runs_tools_in_required_order_and_logs_each_call(
    policy_passage: RetrievedPassage,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy_search = SpyPolicySearch([policy_passage])
    caplog.set_level(logging.INFO, logger="model_monitoring.agent")

    result = MonitoringAgent(policy_search).run("M001", "2026-07")

    assert [entry.tool_name for entry in result.tool_call_log] == [
        "get_model_metrics",
        "get_historical_metrics",
        "detect_breaches",
        "search_policy",
    ]
    assert all(entry.status == "success" for entry in result.tool_call_log)
    assert all(entry.output is not None for entry in result.tool_call_log)
    tool_records = [
        record for record in caplog.records if "monitoring_tool_call" in record.message
    ]
    assert len(tool_records) == 4
    assert policy_search.build_index_called is False


def test_agent_uses_latest_history_and_deterministic_breaches(
    policy_passage: RetrievedPassage,
) -> None:
    result = MonitoringAgent(SpyPolicySearch([policy_passage])).run(
        "M001",
        "2026-07",
    )

    assert result.previous_metrics is not None
    assert result.previous_metrics.period == "2026-06"
    assert {breach.metric for breach in result.breaches} == {
        MetricName.PSI,
        MetricName.AUC_CHANGE,
    }
    assert "psi" in result.tool_call_log[3].arguments["query"]
    assert "auc_change" in result.tool_call_log[3].arguments["query"]


def test_recommendation_cites_retrieved_policy_evidence(
    policy_passage: RetrievedPassage,
) -> None:
    result = MonitoringAgent(SpyPolicySearch([policy_passage])).run(
        "M001",
        "2026-07",
    )

    assert result.recommendation.policy_evidence[0].citation == (
        "monitoring_policy.md#chunk-2"
    )
    assert result.recommendation.actions
    assert all(
        "monitoring_policy.md#chunk-2" in action.citations
        for action in result.recommendation.actions
    )
    assert "escalation" in result.recommendation.actions[0].action.lower()


def test_recommendation_builder_receives_only_breaches_and_policy_evidence(
    policy_passage: RetrievedPassage,
) -> None:
    captured: dict[str, object] = {}

    def recommendation_builder(
        breaches: list[Breach],
        evidence: list[PolicyEvidence],
    ) -> MonitoringRecommendation:
        captured["breaches"] = breaches
        captured["evidence"] = evidence
        return MonitoringRecommendation(
            summary="Review deterministic results.",
            actions=[
                RecommendedAction(
                    action="Escalate for human review.",
                    citations=[item.citation for item in evidence],
                )
            ],
            policy_evidence=evidence,
        )

    result = MonitoringAgent(
        SpyPolicySearch([policy_passage]),
        recommendation_builder=recommendation_builder,
    ).run("M001", "2026-07")

    assert captured == {
        "breaches": result.breaches,
        "evidence": result.recommendation.policy_evidence,
    }
    assert not any(
        isinstance(item, dict) and "auc" in item
        for item in captured.values()
    )


def test_missing_policy_evidence_prevents_action_recommendation() -> None:
    result = MonitoringAgent(SpyPolicySearch([])).run("M001", "2026-07")

    assert result.recommendation.policy_evidence == []
    assert result.recommendation.actions[0].citations == []
    assert "Do not initiate an action" in result.recommendation.actions[0].action


def test_agent_rejects_uncited_custom_recommendation(
    policy_passage: RetrievedPassage,
) -> None:
    def uncited_builder(
        breaches: list[Breach],
        evidence: list[PolicyEvidence],
    ) -> MonitoringRecommendation:
        return MonitoringRecommendation(
            summary="Escalate.",
            actions=[RecommendedAction(action="Escalate.")],
            policy_evidence=evidence,
        )

    agent = MonitoringAgent(
        SpyPolicySearch([policy_passage]),
        recommendation_builder=uncited_builder,
    )

    with pytest.raises(ValueError, match="must cite policy evidence"):
        agent.run("M001", "2026-07")


def test_failed_tool_call_is_logged(policy_passage: RetrievedPassage) -> None:
    agent = MonitoringAgent(SpyPolicySearch([policy_passage]))

    with pytest.raises(LookupError):
        agent.run("M001", "2025-01")

    assert len(agent.tool_call_log) == 1
    assert agent.tool_call_log[0].tool_name == "get_model_metrics"
    assert agent.tool_call_log[0].status == "error"
    assert "LookupError" in (agent.tool_call_log[0].error or "")
