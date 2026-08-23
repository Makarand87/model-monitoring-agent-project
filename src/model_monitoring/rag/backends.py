"""Backend construction shared by command-line evaluations."""

from __future__ import annotations

from pathlib import Path

from .retrieval import HashingEmbedder, OpenAIEmbedder, PolicyRetriever

BACKENDS = ("hash", "openai")


def build_retriever(backend: str, policies_dir: Path, db_path: Path, embedding_model: str) -> PolicyRetriever:
    """Build a retriever whose embedding implementation matches ``backend``."""
    if backend == "hash":
        embedder = HashingEmbedder()
    elif backend == "openai":
        embedder = OpenAIEmbedder(model=embedding_model)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return PolicyRetriever(policies_dir=policies_dir, db_path=db_path, embedder=embedder)
