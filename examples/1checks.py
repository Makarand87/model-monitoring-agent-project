"""Integration tests for the supplied Markdown datasets."""

from pathlib import Path
from model_monitoring.loaders import load_inventory, load_monitoring

PROJECT_ROOT = Path(__file__).resolve().parents[1]


inventory = load_inventory(PROJECT_ROOT / "data" / "model_inventory.md")
monitoring = load_monitoring(PROJECT_ROOT / "data" / "monitoring_table.md")

inventory_ids = {record.model_id for record in inventory}
monitoring_ids = {record.model_id for record in monitoring}
