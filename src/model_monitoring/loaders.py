"""Load Markdown tables into validated Pydantic records."""

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from model_monitoring.models import ModelInventoryRecord, MonitoringRecord

RecordT = TypeVar("RecordT", bound=BaseModel)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def load_markdown_table(path: Path, record_type: type[RecordT]) -> list[RecordT]:
    """Parse the first Markdown table in *path* and validate every data row."""

    table_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("|")
    ]
    if len(table_lines) < 2:
        raise ValueError(f"No Markdown table found in {path}")

    headers = _cells(table_lines[0])
    records: list[RecordT] = []
    for line in table_lines[2:]:  # second line is the Markdown separator
        values = _cells(line)
        if len(values) != len(headers):
            raise ValueError(f"Malformed table row in {path}: {line}")
        records.append(record_type.model_validate(dict(zip(headers, values, strict=True))))
    return records


def load_inventory(path: Path) -> list[ModelInventoryRecord]:
    """Load and validate the model inventory table."""

    return load_markdown_table(path, ModelInventoryRecord)


def load_monitoring(path: Path) -> list[MonitoringRecord]:
    """Load and validate the monthly monitoring table."""

    return load_markdown_table(path, MonitoringRecord)
