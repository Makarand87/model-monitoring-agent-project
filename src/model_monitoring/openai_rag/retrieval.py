from __future__ import annotations

from pathlib import Path
from typing import Any

from model_monitoring.rag.retrieval import PolicyRetriever


class OpenAIEmbedder:
    """OpenAI-backed embedding adapter compatible with ``PolicyRetriever``.

    The adapter intentionally exposes the same small ``embed(text)`` contract
    used by the existing local ``HashingEmbedder``. This keeps retrieval logic,
    chunking, storage, and ``PolicyRetriever.retrieve(...)`` unchanged.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")

        if client is None:
            from openai import OpenAI

            client = OpenAI()

        self.model = model
        self.client = client

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return list(response.data[0].embedding)


class OpenAIPolicyRetriever(PolicyRetriever):
    """Policy retriever that swaps only the embedding implementation.

    ``retrieve(query, top_k=3)`` is inherited directly from ``PolicyRetriever``
    so callers can switch retrievers without changing their retrieval code.
    """

    def __init__(
        self,
        policies_dir: str | Path = "policies",
        db_path: str | Path = ".rag/openai_policy_vectors.sqlite3",
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        embedding_model: str = "text-embedding-3-small",
        client: Any | None = None,
    ) -> None:
        super().__init__(
            policies_dir=policies_dir,
            db_path=db_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedder=OpenAIEmbedder(model=embedding_model, client=client),
        )
