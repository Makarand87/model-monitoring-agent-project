"""Ordinary, deterministic model-performance monitoring functions."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence

from model_monitoring.loaders import load_inventory, load_monitoring
from model_monitoring.models import (
    Breach,
    MetricName,
    ModelMetadata,
    MonitoringMetrics,
    MonitoringResult,
    MonitoringStatus,
    Threshold,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY_PATH = PROJECT_ROOT / "data" / "model_inventory.md"
DEFAULT_MONITORING_PATH = PROJECT_ROOT / "data" / "monitoring_table.md"

PSI_THRESHOLD = Threshold(metric=MetricName.PSI, green_max=0.10, amber_max=0.25)
AUC_CHANGE_THRESHOLD = Threshold(
    metric=MetricName.AUC_CHANGE,
    green_max=0.03,
    amber_max=0.05,
)


def _validate_model_id(model_id: str) -> str:
    value = model_id.strip()
    if re.fullmatch(r"M\d{3}", value) is None:
        raise ValueError("model_id must use the format M followed by three digits")
    return value


def _validate_period(period: str) -> str:
    value = period.strip()
    if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value) is None:
        raise ValueError("period must use YYYY-MM format")
    return value


def _validate_number(value: float, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if non_negative and number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def get_model_metrics(
    model_id: str,
    period: str,
    *,
    records: Sequence[MonitoringMetrics] | None = None,
    monitoring_path: Path = DEFAULT_MONITORING_PATH,
) -> MonitoringMetrics:
    """Return the unique monitoring record for a model and month."""

    validated_id = _validate_model_id(model_id)
    validated_period = _validate_period(period)
    available = list(records) if records is not None else load_monitoring(monitoring_path)
    matches = [
        row
        for row in available
        if row.model_id == validated_id and row.period == validated_period
    ]
    if not matches:
        raise LookupError(f"No metrics found for {validated_id} in {validated_period}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate metrics found for {validated_id} in {validated_period}")
    return matches[0]


def get_historical_metrics(
    model_id: str,
    *,
    before_period: str | None = None,
    records: Sequence[MonitoringMetrics] | None = None,
    monitoring_path: Path = DEFAULT_MONITORING_PATH,
) -> list[MonitoringMetrics]:
    """Return a model's observations in chronological order, optionally before a month."""

    validated_id = _validate_model_id(model_id)
    validated_period = _validate_period(before_period) if before_period is not None else None
    available = list(records) if records is not None else load_monitoring(monitoring_path)
    history = [
        row
        for row in available
        if row.model_id == validated_id
        and (validated_period is None or row.period < validated_period)
    ]
    return sorted(history, key=lambda row: row.period)


def calculate_metric_change(
    current_value: float,
    historical_value: float,
    *,
    relative: bool = False,
) -> float:
    """Calculate current minus historical value, optionally relative to history."""

    current = _validate_number(current_value, "current_value")
    historical = _validate_number(historical_value, "historical_value")
    difference = current - historical
    if not relative:
        return round(difference, 12)
    if historical == 0:
        raise ValueError("historical_value cannot be zero for relative change")
    return round(difference / historical, 12)


def _classify(value: float, threshold: Threshold) -> Breach:
    number = _validate_number(value, threshold.metric.value, non_negative=True)
    if number <= threshold.green_max:
        status = MonitoringStatus.GREEN
        breached_threshold = threshold.green_max
    elif number <= threshold.amber_max:
        status = MonitoringStatus.AMBER
        breached_threshold = threshold.green_max
    else:
        status = MonitoringStatus.RED
        breached_threshold = threshold.amber_max

    return Breach(
        metric=threshold.metric,
        value=number,
        status=status,
        threshold=breached_threshold,
        breach=status is not MonitoringStatus.GREEN,
    )


def classify_psi(psi: float) -> Breach:
    """Classify PSI using the approved 0.10 and 0.25 boundaries."""

    return _classify(psi, PSI_THRESHOLD)


def classify_auc_change(auc_change: float) -> Breach:
    """Classify a non-negative AUC deterioration using 0.03 and 0.05 boundaries."""

    return _classify(auc_change, AUC_CHANGE_THRESHOLD)


def _classifications(
    current: MonitoringMetrics,
    historical: MonitoringMetrics | None,
) -> list[Breach]:
    auc_deterioration = 0.0
    if historical is not None:
        raw_auc_change = calculate_metric_change(current.auc, historical.auc)
        auc_deterioration = max(0.0, -raw_auc_change)
    return [classify_psi(current.psi), classify_auc_change(auc_deterioration)]


def detect_breaches(
    current: MonitoringMetrics,
    historical: MonitoringMetrics | None = None,
) -> list[Breach]:
    """Return amber and red PSI/AUC classifications for the current observation."""

    return [result for result in _classifications(current, historical) if result.breach]


def run_monitoring(
    model_id: str,
    period: str,
    *,
    inventory_path: Path = DEFAULT_INVENTORY_PATH,
    monitoring_path: Path = DEFAULT_MONITORING_PATH,
) -> MonitoringResult:
    """Run deterministic monitoring for one model-period and return a typed result."""

    validated_id = _validate_model_id(model_id)
    metadata_matches = [
        model for model in load_inventory(inventory_path) if model.model_id == validated_id
    ]
    if not metadata_matches:
        raise LookupError(f"Model {validated_id} is not present in the inventory")

    current = get_model_metrics(validated_id, period, monitoring_path=monitoring_path)
    history = get_historical_metrics(
        validated_id,
        before_period=current.period,
        monitoring_path=monitoring_path,
    )
    previous = history[-1] if history else None

    changes: dict[str, float] = {}
    if previous is not None:
        changes = {
            "auc": calculate_metric_change(current.auc, previous.auc),
            "psi": calculate_metric_change(current.psi, previous.psi),
            "bad_rate": calculate_metric_change(current.bad_rate, previous.bad_rate),
            "approval_rate": calculate_metric_change(
                current.approval_rate,
                previous.approval_rate,
            ),
        }

    classifications = _classifications(current, previous)
    breaches = [result for result in classifications if result.breach]
    severity = {
        MonitoringStatus.GREEN: 0,
        MonitoringStatus.AMBER: 1,
        MonitoringStatus.RED: 2,
    }
    overall_status = max(classifications, key=lambda item: severity[item.status]).status

    return MonitoringResult(
        model_metadata=ModelMetadata.model_validate(metadata_matches[0]),
        current_metrics=current,
        previous_metrics=previous,
        metric_changes=changes,
        classifications=classifications,
        breaches=breaches,
        overall_status=overall_status,
    )


def main() -> None:
    """Command-line entry point for a single model-period monitoring run."""

    parser = argparse.ArgumentParser(description="Run model performance monitoring")
    parser.add_argument("model_id", help="Model identifier, for example M001")
    parser.add_argument("period", help="Monitoring period in YYYY-MM format")
    args = parser.parse_args()
    result = run_monitoring(args.model_id, args.period)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
