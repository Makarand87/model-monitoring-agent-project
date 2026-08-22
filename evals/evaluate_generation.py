"""Evaluate answers on the same 30-case golden set with RAGAS and an LLM judge.

Hash mode creates deterministic extractive answers; OpenAI mode uses the chosen
chat model. Both modes require ``OPENAI_API_KEY`` for model-based evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from evals.evaluate_retrieval import DEFAULT_TOP_K, load_cases
from model_monitoring.rag.answering import build_grounded_answer
from model_monitoring.rag.backends import BACKENDS, build_retriever


def generate_openai_answer(question: str, contexts: list[str], model: str) -> str:
    """Generate an answer constrained to the retrieved policy context."""
    from openai import OpenAI

    response = OpenAI().responses.create(
        model=model,
        instructions=(
            "Answer only from the supplied policy context. If it does not contain "
            "the answer, say that the policy corpus does not specify it. Be concise."
        ),
        input=f"Question: {question}\n\nPolicy context:\n" + "\n\n---\n\n".join(contexts),
    )
    return response.output_text


def judge_answer(question: str, reference: str, answer: str, model: str) -> dict[str, Any]:
    """Return a structured 1-5 correctness score from an independent LLM call."""
    from openai import OpenAI

    response = OpenAI().responses.create(
        model=model,
        instructions=(
            "Act as a strict answer-correctness judge. Compare the candidate with the "
            "reference. Return JSON with integer score (1 wholly incorrect to 5 fully "
            "correct) and a brief reason. For an UNANSWERABLE reference, full credit "
            "requires the candidate to abstain rather than invent a value."
        ),
        input=f"Question: {question}\nReference: {reference}\nCandidate: {answer}",
        text={"format": {"type": "json_object"}},
    )
    result = json.loads(response.output_text)
    score = result.get("score")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"Judge returned invalid score: {score!r}")
    return {"score": score, "reason": str(result.get("reason", ""))}


def run_ragas(rows: list[dict[str, Any]], judge_model: str, embedding_model: str) -> list[dict[str, Any]]:
    """Run RAGAS faithfulness, relevancy, and context metrics for all rows."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

    dataset = EvaluationDataset.from_list(
        [{"user_input": row["question"], "response": row["answer"],
          "retrieved_contexts": row["contexts"], "reference": row["reference"]} for row in rows]
    )
    llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
    result = evaluate(dataset=dataset, metrics=[
        Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm), LLMContextRecall(llm=llm),
    ])
    return result.to_pandas().to_dict(orient="records")


def run_evaluation(dataset: Path, policies_dir: Path, db_path: Path, backend: str = "hash",
                   top_k: int = DEFAULT_TOP_K, embedding_model: str = "text-embedding-3-small",
                   generation_model: str = "gpt-5-mini", judge_model: str = "gpt-5-mini") -> dict[str, Any]:
    """Generate and evaluate answers for every golden case."""
    cases = load_cases(dataset)
    retriever = build_retriever(backend, policies_dir, db_path, embedding_model)
    indexed_chunks = retriever.build_index()
    rows: list[dict[str, Any]] = []
    for case in cases:
        passages = retriever.retrieve(case["question"], top_k=top_k)
        contexts = [passage.text for passage in passages]
        answer = (build_grounded_answer(case["question"], passages) if backend == "hash"
                  else generate_openai_answer(case["question"], contexts, generation_model))
        rows.append({"question": case["question"], "reference": case["expected_passage/topic"],
                     "answer": answer, "contexts": contexts,
                     "sources": [passage.source for passage in passages],
                     "risk_level": case["risk_level"], "difficulty": case["difficulty"]})

    for row, scores in zip(rows, run_ragas(rows, judge_model, embedding_model), strict=True):
        row["ragas"] = {key: value for key, value in scores.items() if key not in
                        {"user_input", "response", "retrieved_contexts", "reference"}}
        row["llm_judge"] = judge_answer(row["question"], row["reference"], row["answer"], judge_model)

    names = sorted(rows[0]["ragas"]) if rows else []
    return {"configuration": {"dataset": str(dataset), "backend": backend, "top_k": top_k,
            "total_cases": len(rows), "indexed_chunks": indexed_chunks,
            "embedding_model": embedding_model, "generation_model": generation_model if backend == "openai" else None,
            "judge_model": judge_model}, "cases": rows,
            "metrics": {"ragas": {name: mean(float(row["ragas"][name]) for row in rows) for name in names},
                        "llm_judge_mean": mean(row["llm_judge"]["score"] for row in rows)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/rag_retrieval_dataset.jsonl"))
    parser.add_argument("--policies-dir", type=Path, default=Path("policies"))
    parser.add_argument("--db-path", type=Path, default=Path(".rag/generation_vectors.sqlite3"))
    parser.add_argument("--backend", choices=BACKENDS, default="hash")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--generation-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_evaluation(**{key: value for key, value in vars(args).items() if key != "output"})
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
