"""Unit tests for Pydantic record validation."""

import pytest
from pydantic import ValidationError

from model_monitoring.models import ModelInventoryRecord, MonitoringRecord


def test_inventory_record_accepts_valid_data() -> None:
    record = ModelInventoryRecord(
        model_id="M001",
        model_name="Application Scorecard",
        risk_tier="High",
        owner="Credit Risk",
    )
    assert record.model_id == "M001"


@pytest.mark.parametrize(
    ("field", "value"),
    [("auc", 1.01), ("psi", -0.01), ("bad_rate", -0.1), ("approval_rate", 1.1)],
)
def test_monitoring_record_rejects_invalid_metric(field: str, value: float) -> None:
    payload = {
        "model_id": "M001",
        "period": "2026-07",
        "auc": 0.75,
        "psi": 0.08,
        "bad_rate": 0.064,
        "approval_rate": 0.68,
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        MonitoringRecord(**payload)
