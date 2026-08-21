from __future__ import annotations

import json

from model_monitoring.rag.answering import answer_policy_question

from .retrieval import OpenAIPolicyRetriever


def main() -> None:
    retriever = OpenAIPolicyRetriever()
    indexed = retriever.build_index()
    question = "What action is required when PSI exceeds 0.25?"
    result = answer_policy_question(question, retriever, top_k=3)

    print(f"Indexed {indexed} policy chunks with OpenAI embeddings.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
