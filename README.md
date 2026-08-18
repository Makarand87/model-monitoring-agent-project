# model-monitoring-agent-project

Initial Python foundation for a model-performance monitoring application. This phase deliberately contains no agents, RAG, MCP, API, database, or user interface.

## What is included

- Pydantic schemas for model inventory and monthly monitoring records.
- A small loader that converts Markdown table rows into validated objects.
- An inventory of 20 models: 10 application scorecards, 5 behaviour scorecards, 3 collection models, and 2 fraud models.
- Three months of synthetic monitoring data for all 20 models (60 observations).
- pytest tests for schemas, data completeness, model coverage, and the supplied M001 values.
- A provisional, documentation-only threshold policy for later implementation.

## Setup and test

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest
```

## Repository structure

```text
model-monitoring-agent/
├── src/model_monitoring/
│   ├── __init__.py
│   ├── loaders.py
│   └── models.py
├── data/
│   ├── model_inventory.md
│   └── monitoring_table.md
├── policies/
│   └── monitoring_thresholds.md
├── tests/
│   ├── test_data_files.py
│   └── test_models.py
├── evals/README.md
├── notebooks/README.md
├── .env
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## File-by-file explanation

| File | Why it exists |
|---|---|
| `src/model_monitoring/__init__.py` | Defines the importable package and its small public interface. |
| `src/model_monitoring/models.py` | Holds the Pydantic schemas, allowed risk tiers, identifier format, and metric bounds. |
| `src/model_monitoring/loaders.py` | Reads Markdown tables and validates each row before the application uses it. |
| `data/model_inventory.md` | Provides the initial governed list of 20 models and their ownership/materiality details. |
| `data/monitoring_table.md` | Provides 60 synthetic monthly observations covering every inventory model. |
| `policies/monitoring_thresholds.md` | Records provisional Green/Amber/Red thresholds separately from code for governance review. |
| `tests/test_models.py` | Checks that valid records pass and impossible metric values fail validation. |
| `tests/test_data_files.py` | Checks inventory uniqueness, full monitoring coverage, periods, row counts, and M001 sample values. |
| `evals/README.md` | Preserves a clearly scoped location for future evaluation work without implementing it now. |
| `notebooks/README.md` | Preserves a location for exploratory analysis and keeps production logic out of notebooks. |
| `.env` | Supplies harmless local runtime defaults; future secrets should remain local and never be committed. |
| `.gitignore` | Prevents Python build/test artefacts, virtual environments, coverage output, and `.env` from entering Git. |
| `requirements.txt` | Declares the only initial runtime/development dependencies: Pydantic and pytest. |
| `pytest.ini` | Tells pytest to import from `src/`, discover tests in `tests/`, and use concise output. |
| `README.md` | Explains scope, setup, structure, and the purpose of every file. |

## Intentional limitations

The metrics are synthetic examples, not production evidence. Thresholds are provisional documentation. Alert logic, baseline selection, time-series storage, APIs, dashboards, agents, RAG, and MCP should be added only in later, separately tested phases.


# Evaluations

Reserved for future evaluation datasets and evaluation code. No agent, RAG, or MCP evaluations are implemented in this initial phase.
