from __future__ import annotations

import re
from dataclasses import asdict
from typing import Iterable

from .retrieval import PolicyRetriever, RetrievedPassage


def build_grounded_answer(
    query: str,
    passages: Iterable[RetrievedPassage],
    max_sentences: int = 2,
) -> str:
    """Build a small extractive answer from already-retrieved passages.

    This function deliberately does not retrieve documents. Keeping it separate
    lets retrieval quality be evaluated independently from answer quality.
    """
    passage_list = list(passages)
    if not passage_list:
        return "No relevant policy evidence was retrieved."

    query_terms = set(_tokenize(query))
    candidates: list[tuple[float, str]] = []

    for rank, passage in enumerate(passage_list):
        sentences = re.split(r"(?<=[.!?])\s+", passage.text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = set(_tokenize(sentence))
            overlap = len(query_terms & terms)
            policy_signal = sum(
                1
                for term in ("psi", "red", "escalation", "escalate", "mrm", "investigation")
                if term in terms
            )
            score = overlap + (0.35 * policy_signal) - (0.05 * rank)
            candidates.append((score, sentence))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    for _, sentence in candidates:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    return " ".join(selected) if selected else passage_list[0].text


def answer_policy_question(
    query: str,
    retriever: PolicyRetriever,
    top_k: int = 3,
) -> dict[str, object]:
    """Convenience orchestration for the demo: retrieve first, answer second."""
    passages = retriever.retrieve(query, top_k=top_k)
    return {
        "question": query,
        "answer": build_grounded_answer(query, passages),
        "passages": [asdict(passage) for passage in passages],
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
