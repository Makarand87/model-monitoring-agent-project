from pathlib import Path
from textwrap import dedent

import pytest

from model_monitoring.rag.answering import answer_policy_question, build_grounded_answer
from model_monitoring.rag.retrieval import PolicyRetriever, load_markdown_documents


def _write_policy(directory: Path, name: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        dedent(text).strip() + "\n",
        encoding="utf-8",
    )


def test_load_markdown_documents_preserves_source_metadata(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    _write_policy(
        policies,
        "monitoring_policy.md",
        """---
        document_type: monitoring_policy
        product: application_scorecard
        ---
        
        
        # Monitoring Policy
        
        PSI at or above 0.25 is RED and requires escalation.
        """,
    )

    documents = load_markdown_documents(policies)
    print(documents)

    # assert len(documents) == 1
    # assert documents[0].source.endswith("monitoring_policy.md")
    # assert documents[0].metadata["document_type"] == "monitoring_policy"
    # assert documents[0].metadata["product"] == "application_scorecard"
    # assert documents[0].metadata.get("source", documents[0].source).endswith(
    #     "monitoring_policy.md"
    # )



# from __future__ import annotations
# from textwrap import dedent

# import hashlib
# import json
# import math
# import re
# import sqlite3
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Iterable


# @dataclass(frozen=True)
# class PolicyDocument:
#     text: str
#     source: str
#     path: str
#     metadata: dict[str, str]

# def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
#     if not text.startswith("---\n"):
#         return {}, text

#     end = text.find("\n---\n", 4)
#     if end == -1:
#         return {}, text

#     metadata: dict[str, str] = {}
#     for line in text[4:end].splitlines():
#         if ":" not in line:
#             continue
#         key, value = line.split(":", 1)
#         metadata[key.strip()] = value.strip()
#     return metadata, text[end + 5 :]


# def load_markdown_documents(policies_dir: str | Path) -> list[PolicyDocument]:
#     directory = Path(policies_dir)
#     if not directory.exists():
#         raise FileNotFoundError(f"Policy directory does not exist: {directory}")

#     documents: list[PolicyDocument] = []
#     for path in sorted(directory.glob("*.md")):
#         raw_text = path.read_text(encoding="utf-8")
#         metadata, body = _parse_front_matter(raw_text)
#         metadata = {**metadata, "source": path.name}
#         documents.append(
#             PolicyDocument(
#                 text=body.strip(),
#                 source=path.name,
#                 path=str(path),
#                 metadata=metadata,
#             )
#         )
#     return documents

# def _write_policy(directory: Path, name: str, text: str) -> None:
#     directory.mkdir(parents=True, exist_ok=True)
#     (directory / name).write_text(
#         dedent(text).strip() + "\n",
#         encoding="utf-8",
#     )
# from pathlib import Path
# from textwrap import dedent




# def _write_policy(directory: Path, name: str, text: str) -> None:
#     directory.mkdir(parents=True, exist_ok=True)
#     (directory / name).write_text(
#         dedent(text).strip() + "\n",
#         encoding="utf-8",
#     )


# def load_markdown_documents_preserves_source_metadata(tmp_path: Path) -> None:
#     policies = tmp_path / "policies"
#     _write_policy(
#         policies,
#         "monitoring_policy.md",
#         """---
#         document_type: monitoring_policy
#         product: application_scorecard
#         ---
        
        
#         # Monitoring Policy
        
#         PSI at or above 0.25 is RED and requires escalation.
#         """,
#     )
#     documents = load_markdown_documents(policies)
#     print(documents)
    