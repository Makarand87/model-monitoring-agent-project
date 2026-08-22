"""Read-only orchestration for deterministic model monitoring and policy advice."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from model_monitoring.application import (
    DEFAULT_MONITORING_PATH,
    detect_breaches,
    get_historical_metrics,
    get_model_metrics,
)
from model_monitoring.models import Breach, MonitoringMetrics, MonitoringStatus
from model_monitoring.rag.retrieval import RetrievedPassage


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class PolicySearch(Protocol):
    """Read-only part of the policy retriever used by the agent."""

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]: ...


class ToolCallLogEntry(BaseModel):
    """Serializable audit record for one deterministic tool invocation."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["success", "error"]
    duration_ms: float = Field(ge=0.0)
    output: Any | None = None
    error: str | None = None


class PolicyEvidence(BaseModel):
    """Policy passage retained with enough metadata to cite and inspect it."""

    model_config = ConfigDict(extra="forbid")

    citation: str
    text: str
    source: str
    path: str
    chunk_index: int = Field(ge=0)
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class RecommendedAction(BaseModel):
    """Text-only advice with explicit supporting policy citations."""

    model_config = ConfigDict(extra="forbid")

    action: str
    citations: list[str] = Field(default_factory=list)


class MonitoringRecommendation(BaseModel):
    """Interpretation of deterministic results; it performs no calculations."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    actions: list[RecommendedAction]
    policy_evidence: list[PolicyEvidence]


class MonitoringAgentResult(BaseModel):
    """Complete read-only agent response and tool-call audit trail."""

    model_config = ConfigDict(extra="forbid")

    current_metrics: MonitoringMetrics
    historical_metrics: list[MonitoringMetrics]
    previous_metrics: MonitoringMetrics | None
    breaches: list[Breach]
    recommendation: MonitoringRecommendation
    tool_call_log: list[ToolCallLogEntry]


RecommendationBuilder = Callable[
    [list[Breach], list[PolicyEvidence]], MonitoringRecommendation
]


class MonitoringAgent:
    """Coordinate read-only tools and produce policy-grounded recommendations.

    The recommendation builder receives only deterministic breach results and
    retrieved policy evidence. It never receives raw monitoring history, which
    prevents an LLM-backed implementation from recalculating PSI or AUC changes.
    """

    def __init__(
        self,
        policy_search: PolicySearch,
        *,
        monitoring_path: str | Path = DEFAULT_MONITORING_PATH,
        top_k: int = 3,
        recommendation_builder: RecommendationBuilder | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.policy_search = policy_search
        self.monitoring_path = Path(monitoring_path)
        self.top_k = top_k
        self.recommendation_builder = (
            recommendation_builder or build_monitoring_recommendation
        )
        self.logger = logger or LOGGER
        self._tool_call_log: list[ToolCallLogEntry] = []

    @property
    def tool_call_log(self) -> tuple[ToolCallLogEntry, ...]:
        """Return an immutable snapshot of the current run's audit log."""

        return tuple(self._tool_call_log)

    def run(self, model_id: str, period: str) -> MonitoringAgentResult:
        """Execute the fixed read-only monitoring flow for one model-period."""

        self._tool_call_log = []
        current = self._get_model_metrics(model_id, period)
        history = self._get_historical_metrics(model_id, current.period)
        previous = history[-1] if history else None
        breaches = self._detect_breaches(current, previous)
        passages = self._search_policy(_build_policy_query(breaches))
        evidence = [_policy_evidence(passage) for passage in passages]
        if evidence:
            recommendation = self.recommendation_builder(breaches, evidence)
            _validate_recommendation_citations(recommendation, evidence)
        else:
            recommendation = build_monitoring_recommendation(breaches, evidence)

        return MonitoringAgentResult(
            current_metrics=current,
            historical_metrics=history,
            previous_metrics=previous,
            breaches=breaches,
            recommendation=recommendation,
            tool_call_log=list(self._tool_call_log),
        )

    def _get_model_metrics(self, model_id: str, period: str) -> MonitoringMetrics:
        return self._call_tool(
            "get_model_metrics",
            {"model_id": model_id, "period": period},
            lambda: get_model_metrics(
                model_id,
                period,
                monitoring_path=self.monitoring_path,
            ),
        )

    def _get_historical_metrics(
        self,
        model_id: str,
        before_period: str,
    ) -> list[MonitoringMetrics]:
        return self._call_tool(
            "get_historical_metrics",
            {"model_id": model_id, "before_period": before_period},
            lambda: get_historical_metrics(
                model_id,
                before_period=before_period,
                monitoring_path=self.monitoring_path,
            ),
        )

    def _detect_breaches(
        self,
        current: MonitoringMetrics,
        previous: MonitoringMetrics | None,
    ) -> list[Breach]:
        return self._call_tool(
            "detect_breaches",
            {"current": current, "historical": previous},
            lambda: detect_breaches(current, previous),
        )

    def _search_policy(self, query: str) -> list[RetrievedPassage]:
        return self._call_tool(
            "search_policy",
            {"query": query, "top_k": self.top_k},
            lambda: self.policy_search.retrieve(query, top_k=self.top_k),
        )

    def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        operation: Callable[[], T],
    ) -> T:
        started = perf_counter()
        serialized_arguments = _serialize(arguments)
        try:
            output = operation()
        except Exception as exc:
            entry = ToolCallLogEntry(
                timestamp=datetime.now(UTC).isoformat(),
                tool_name=tool_name,
                arguments=serialized_arguments,
                status="error",
                duration_ms=round((perf_counter() - started) * 1000, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
            self._record_tool_call(entry, level=logging.ERROR)
            raise

        entry = ToolCallLogEntry(
            timestamp=datetime.now(UTC).isoformat(),
            tool_name=tool_name,
            arguments=serialized_arguments,
            status="success",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            output=_serialize(output),
        )
        self._record_tool_call(entry, level=logging.INFO)
        return output

    def _record_tool_call(self, entry: ToolCallLogEntry, *, level: int) -> None:
        self._tool_call_log.append(entry)
        self.logger.log(
            level,
            "monitoring_tool_call %s",
            json.dumps(entry.model_dump(mode="json"), sort_keys=True),
        )


def build_monitoring_recommendation(
    breaches: list[Breach],
    evidence: list[PolicyEvidence],
) -> MonitoringRecommendation:
    """Interpret supplied tool results without recomputing monitoring metrics."""

    citations = [item.citation for item in evidence]
    if not breaches:
        return MonitoringRecommendation(
            summary="No deterministic PSI or AUC breach was reported.",
            actions=[
                RecommendedAction(
                    action="Continue monitoring on the policy-defined schedule.",
                    citations=citations,
                )
            ],
            policy_evidence=evidence,
        )

    breach_summary = ", ".join(
        f"{item.metric.value}={item.value:g} ({item.status.value})"
        for item in breaches
    )
    severity = (
        "RED"
        if any(item.status is MonitoringStatus.RED for item in breaches)
        else "AMBER"
    )
    actions = [
        RecommendedAction(
            action=(
                f"Follow the cited {severity} escalation and notification procedure "
                "for the reported breaches."
            ),
            citations=citations,
        ),
        RecommendedAction(
            action=(
                "Investigate the drivers of the reported breach values and document "
                "the assessment; treat the deterministic tool output as authoritative."
            ),
            citations=citations,
        ),
    ]
    if not evidence:
        actions = [
            RecommendedAction(
                action=(
                    "Do not initiate an action until applicable policy evidence is "
                    "available; route the result for human review."
                ),
                citations=[],
            )
        ]

    return MonitoringRecommendation(
        summary=f"Deterministic monitoring reported: {breach_summary}.",
        actions=actions,
        policy_evidence=evidence,
    )


def _build_policy_query(breaches: list[Breach]) -> str:
    if not breaches:
        return (
            "What monitoring cadence and evidence requirements apply when no PSI "
            "or AUC breach is reported?"
        )

    results = "; ".join(
        (
            f"metric {item.metric.value}, status {item.status.value}, "
            f"reported value {item.value:g}, breached threshold {item.threshold:g}"
        )
        for item in breaches
    )
    return (
        "What policy escalation, notification, investigation, and documentation "
        f"requirements apply to these deterministic monitoring results: {results}?"
    )


def _policy_evidence(passage: RetrievedPassage) -> PolicyEvidence:
    return PolicyEvidence(
        citation=f"{passage.source}#chunk-{passage.chunk_index}",
        text=passage.text,
        source=passage.source,
        path=passage.path,
        chunk_index=passage.chunk_index,
        score=passage.score,
        metadata=passage.metadata,
    )


def _validate_recommendation_citations(
    recommendation: MonitoringRecommendation,
    evidence: list[PolicyEvidence],
) -> None:
    available = {item.citation for item in evidence}
    returned = {item.citation for item in recommendation.policy_evidence}
    if returned != available:
        raise ValueError(
            "Recommendation policy evidence must match the retrieved passages"
        )

    for action in recommendation.actions:
        if not action.citations:
            raise ValueError("Every recommended action must cite policy evidence")
        unknown = set(action.citations) - available
        if unknown:
            raise ValueError(
                f"Recommended action contains unknown policy citations: {sorted(unknown)}"
            )


def _serialize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value
