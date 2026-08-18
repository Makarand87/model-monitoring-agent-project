"""Small, local RAG components for policy retrieval."""

from .answering import answer_policy_question, build_grounded_answer
from .retrieval import PolicyRetriever, RetrievedPassage

__all__ = [
    "PolicyRetriever",
    "RetrievedPassage",
    "answer_policy_question",
    "build_grounded_answer",
]
