"""Integration tests for the supplied Markdown datasets."""

from pathlib import Path

from model_monitoring.loaders import load_inventory, load_monitoring

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_inventory_contains_expected_20_models() -> None:
    inventory = load_inventory(PROJECT_ROOT / "data" / "model_inventory.md")
    assert len(inventory) == 20
    assert len({record.model_id for record in inventory}) == 20


def test_monitoring_has_three_periods_for_every_inventory_model() -> None:
    inventory = load_inventory(PROJECT_ROOT / "data" / "model_inventory.md")
    monitoring = load_monitoring(PROJECT_ROOT / "data" / "monitoring_table.md")

    inventory_ids = {record.model_id for record in inventory}
    monitoring_ids = {record.model_id for record in monitoring}
    assert monitoring_ids == inventory_ids
    assert len(monitoring) == 60

    for model_id in inventory_ids:
        periods = {record.period for record in monitoring if record.model_id == model_id}
        assert periods == {"2026-05", "2026-06", "2026-07"}


def test_m001_example_values_are_preserved() -> None:
    monitoring = load_monitoring(PROJECT_ROOT / "data" / "monitoring_table.md")
    july = next(row for row in monitoring if row.model_id == "M001" and row.period == "2026-07")
    assert (july.auc, july.psi, july.bad_rate, july.approval_rate) == (0.69, 0.27, 0.078, 0.62)
