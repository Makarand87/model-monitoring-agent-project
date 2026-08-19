"""Unit tests for Pydantic records using the supplied Markdown datasets."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from model_monitoring.loaders import load_inventory, load_markdown_table, load_monitoring
from model_monitoring.models import ModelInventoryRecord, MonitoringRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = PROJECT_ROOT / "data" / "model_inventory.md"
MONITORING_PATH = PROJECT_ROOT / "data" / "monitoring_table.md"


@pytest.fixture
def inventory_records() -> list[ModelInventoryRecord]:
    return load_inventory(INVENTORY_PATH)


@pytest.fixture
def monitoring_records() -> list[MonitoringRecord]:
    return load_monitoring(MONITORING_PATH)


def test_load_markdown_table_returns_inventory_models() -> None:
    records = load_markdown_table(INVENTORY_PATH, ModelInventoryRecord)

    assert records == load_inventory(INVENTORY_PATH)
    assert all(isinstance(record, ModelInventoryRecord) for record in records)


def test_loaded_inventory_records_are_valid(
    inventory_records: list[ModelInventoryRecord],
) -> None:
    assert len(inventory_records) == 20
    assert {record.model_id for record in inventory_records} == {
        f"M{number:03d}" for number in range(1, 21)
    }
    assert all(record.model_name and record.owner for record in inventory_records)


def test_loaded_monitoring_records_are_valid(
    monitoring_records: list[MonitoringRecord],
) -> None:
    assert len(monitoring_records) == 60
    assert all(0.0 <= record.auc <= 1.0 for record in monitoring_records)
    assert all(record.psi >= 0.0 for record in monitoring_records)
    assert all(0.0 <= record.bad_rate <= 1.0 for record in monitoring_records)
    assert all(0.0 <= record.approval_rate <= 1.0 for record in monitoring_records)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("auc", 1.01), ("psi", -0.01), ("bad_rate", -0.1), ("approval_rate", 1.1)],
)
def test_monitoring_model_rejects_invalid_metric_from_loaded_record(
    field: str,
    invalid_value: float,
    monitoring_records: list[MonitoringRecord],
) -> None:
    payload = monitoring_records[0].model_dump()
    payload[field] = invalid_value

    with pytest.raises(ValidationError):
        MonitoringRecord.model_validate(payload)


# #######################################################


@pytest.fixture
def valid_inventory_record() -> ModelInventoryRecord:
    return ModelInventoryRecord(
        model_id="M001",
        model_name="Application Scorecard",
        risk_tier="High",
        owner="Credit Risk",
    )


def test_inventory_record_accepts_valid_data(
    valid_inventory_record: ModelInventoryRecord,
) -> None:
    record = valid_inventory_record
    assert record.model_id == "M001"


@pytest.fixture
def monitoring_payload() -> dict[str, object]:
    return {
        "model_id": "M001",
        "period": "2026-07",
        "auc": 1.75,
        "psi": 0.08,
        "bad_rate": 0.064,
        "approval_rate": 0.68,
    }


@pytest.mark.parametrize(("field", "value"),
    [("auc", 1.01), ("psi", -0.01), ("bad_rate", -0.1), ("approval_rate", 1.1)],)
def test_monitoring_record_rejects_invalid_metric(
    field: str,
    value: float,
    monitoring_payload: dict[str, object],
) -> None:
    payload = monitoring_payload.copy()
    payload[field] = value
    with pytest.raises(ValidationError):
        MonitoringRecord(**payload)
