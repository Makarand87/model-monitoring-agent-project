"""Tests for deterministic monitoring functions and orchestration."""

import math

import pytest

from model_monitoring.application import (
    calculate_metric_change,
    classify_auc_change,
    classify_psi,
    detect_breaches,
    get_historical_metrics,
    get_model_metrics,
    run_monitoring,
)
from model_monitoring.models import MetricName, MonitoringStatus


def test_get_model_metrics_returns_requested_record() -> None:
    metrics = get_model_metrics("M001", "2026-07")
    assert metrics.auc == 0.69
    assert metrics.psi == 0.27


def test_get_historical_metrics_returns_only_earlier_periods() -> None:
    history = get_historical_metrics("M001", before_period="2026-07")
    assert [row.period for row in history] == ["2026-05", "2026-06"]


@pytest.mark.parametrize(
    ("model_id", "period"),
    [("001", "2026-07"), ("M001", "2026-13")],
)
def test_get_model_metrics_rejects_invalid_keys(model_id: str, period: str) -> None:
    with pytest.raises(ValueError):
        get_model_metrics(model_id, period)


def test_get_model_metrics_raises_for_missing_record() -> None:
    with pytest.raises(LookupError):
        get_model_metrics("M001", "2025-01")


def test_calculate_metric_change_returns_current_minus_history() -> None:
    assert calculate_metric_change(0.73, 0.75) == pytest.approx(-0.02)
    assert calculate_metric_change(0.12, 0.10, relative=True) == pytest.approx(0.20)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_calculate_metric_change_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        calculate_metric_change(value, 0.50)


def test_relative_change_rejects_zero_historical_value() -> None:
    with pytest.raises(ValueError):
        calculate_metric_change(0.10, 0.0, relative=True)


@pytest.mark.parametrize(
    ("psi", "expected"),
    [
        (0.08, MonitoringStatus.GREEN),
        (0.10, MonitoringStatus.GREEN),
        (0.100001, MonitoringStatus.AMBER),
        (0.25, MonitoringStatus.AMBER),
        (0.250001, MonitoringStatus.RED),
    ],
)
def test_classify_psi_normal_and_boundaries(
    psi: float,
    expected: MonitoringStatus,
) -> None:
    assert classify_psi(psi).status is expected


def test_classify_psi_red_result_matches_requested_shape() -> None:
    result = classify_psi(0.27)
    assert result.model_dump(mode="json") == {
        "metric": "psi",
        "value": 0.27,
        "status": "RED",
        "threshold": 0.25,
        "breach": True,
    }


@pytest.mark.parametrize("invalid", [-0.01, math.nan, math.inf])
def test_classify_psi_rejects_invalid_values(invalid: float) -> None:
    with pytest.raises(ValueError):
        classify_psi(invalid)


@pytest.mark.parametrize(
    ("auc_change", "expected"),
    [
        (0.02, MonitoringStatus.GREEN),
        (0.03, MonitoringStatus.GREEN),
        (0.030001, MonitoringStatus.AMBER),
        (0.05, MonitoringStatus.AMBER),
        (0.050001, MonitoringStatus.RED),
    ],
)
def test_classify_auc_change_normal_and_boundaries(
    auc_change: float,
    expected: MonitoringStatus,
) -> None:
    assert classify_auc_change(auc_change).status is expected


def test_classify_auc_change_rejects_negative_deterioration() -> None:
    with pytest.raises(ValueError):
        classify_auc_change(-0.01)


def test_detect_breaches_returns_psi_and_auc_breaches() -> None:
    current = get_model_metrics("M001", "2026-07")
    previous = get_model_metrics("M001", "2026-06")
    breaches = detect_breaches(current, previous)
    assert {breach.metric for breach in breaches} == {
        MetricName.PSI,
        MetricName.AUC_CHANGE,
    }


def test_run_monitoring_returns_complete_typed_result() -> None:
    result = run_monitoring("M001", "2026-07")
    assert result.model_metadata.model_id == "M001"
    assert result.previous_metrics is not None
    assert result.previous_metrics.period == "2026-06"
    assert result.metric_changes["auc"] == pytest.approx(-0.04)
    assert result.overall_status is MonitoringStatus.RED
    assert len(result.breaches) == 2
