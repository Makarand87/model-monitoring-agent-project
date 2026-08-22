from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class PolicyDocument:
    text: str
    source: str
    path: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    text: str
    source: str
    path: str
    chunk_index: int
    metadata: dict[str, str]


@dataclass(frozen=True)
class RetrievedPassage:
    text: str
    source: str
    path: str
    chunk_index: int
    score: float
    metadata: dict[str, str]


class HashingEmbedder:
    """Deterministic local text embedder with no API or model download.

    It hashes normalized word tokens into a fixed-size vector and L2 normalizes
    the result. This is deliberately small and transparent for the learning
    project. A semantic embedding model can replace this class later without
    changing the retriever interface.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class Embedder(Protocol):
    """Minimal interface accepted by the local vector store."""

    def embed(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """Semantic embeddings supplied by the OpenAI embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI()

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


class LocalVectorStore:
    """A tiny SQLite-backed local vector store for policy chunks."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_vectors (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )

    def replace(self, chunks: Iterable[PolicyChunk], embedder: Embedder) -> None:
        rows = []
        for chunk in chunks:
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.source,
                    chunk.path,
                    chunk.chunk_index,
                    json.dumps(chunk.metadata, sort_keys=True),
                    json.dumps(embedder.embed(chunk.text)),
                )
            )

        with self._connect() as connection:
            connection.execute("DELETE FROM policy_vectors")
            connection.executemany(
                """
                INSERT INTO policy_vectors (
                    chunk_id, text, source, path, chunk_index,
                    metadata_json, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[RetrievedPassage]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT text, source, path, chunk_index,
                       metadata_json, embedding_json
                FROM policy_vectors
                """
            ).fetchall()

        ranked: list[RetrievedPassage] = []
        for text, source, path, chunk_index, metadata_json, embedding_json in rows:
            embedding = json.loads(embedding_json)
            score = _dot(query_embedding, embedding)
            ranked.append(
                RetrievedPassage(
                    text=text,
                    source=source,
                    path=path,
                    chunk_index=chunk_index,
                    score=score,
                    metadata=json.loads(metadata_json),
                )
            )

        ranked.sort(key=lambda passage: passage.score, reverse=True)
        return ranked[:top_k]


class PolicyRetriever:
    """Indexes markdown policy files and retrieves top-k policy passages."""

    def __init__(
        self,
        policies_dir: str | Path = "policies",
        db_path: str | Path = ".rag/policy_vectors.sqlite3",
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        embedder: Embedder | None = None,
    ) -> None:
        self.policies_dir = Path(policies_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder or HashingEmbedder()
        self.store = LocalVectorStore(db_path)

    def build_index(self) -> int:
        documents = load_markdown_documents(self.policies_dir)
        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.store.replace(chunks, self.embedder)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]:
        if not query.strip():
            raise ValueError("query must not be empty")
        return self.store.search(self.embedder.embed(query), top_k=top_k)


def load_markdown_documents(policies_dir: str | Path) -> list[PolicyDocument]:
    directory = Path(policies_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Policy directory does not exist: {directory}")

    documents: list[PolicyDocument] = []
    for path in sorted(directory.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = _parse_front_matter(raw_text)
        metadata = {**metadata, "source": path.name}
        documents.append(
            PolicyDocument(
                text=body.strip(),
                source=path.name,
                path=str(path),
                metadata=metadata,
            )
        )
    return documents


def chunk_documents(
    documents: Iterable[PolicyDocument],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[PolicyChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks: list[PolicyChunk] = []
    for document in documents:
        for index, text in enumerate(
            _chunk_text(document.text, chunk_size, chunk_overlap)
        ):
            chunk_id = hashlib.sha256(
                f"{document.path}:{index}:{text}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=document.source,
                    path=document.path,
                    chunk_index=index,
                    metadata=document.metadata,
                )
            )
    return chunks


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            prefix = current[-overlap:] if overlap else ""
            current = f"{prefix}\n\n{paragraph}".strip()
        else:
            for start in range(0, len(paragraph), chunk_size - overlap):
                piece = paragraph[start : start + chunk_size]
                if piece:
                    chunks.append(piece)
            current = ""

    if current:
        chunks.append(current)
    return chunks


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, text[end + 5 :]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))
