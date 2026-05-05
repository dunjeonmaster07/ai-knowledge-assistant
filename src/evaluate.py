"""
RAGAS evaluation pipeline for the RAG system.

Measures:
  Faithfulness       — Is the answer grounded in the retrieved context?
  LLMContextRecall   — Did the retriever find the chunks needed to answer?
  FactualCorrectness — Does the generated answer match the ground truth?
  SemanticSimilarity — How close is the generated answer to the ground truth in meaning?

Usage:
    python -m src.evaluate
"""

import json
import time
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    FactualCorrectness,
    SemanticSimilarity,
)

from src.config import EVAL_DIR, LLM_MODEL, EMBEDDING_MODEL
from src.chain import ask_question_for_eval


def load_eval_questions(path: Path | None = None) -> list[dict]:
    """Load the evaluation dataset from JSON."""
    path = path or EVAL_DIR / "eval_questions.json"
    with open(path) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} evaluation questions from {path.name}\n")
    return questions


def collect_rag_responses(questions: list[dict]) -> list[dict]:
    """Run each question through the RAG pipeline and collect structured results."""
    dataset = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        ground_truth = q["ground_truth"]

        print(f"  [{i}/{len(questions)}] {question[:60]}...")

        result = ask_question_for_eval(question)

        dataset.append({
            "user_input": result["user_input"],
            "response": result["response"],
            "retrieved_contexts": result["retrieved_contexts"],
            "reference": ground_truth,
        })

    print(f"\nCollected responses for {len(dataset)} questions.\n")
    return dataset


def run_evaluation(
    questions: list[dict] | None = None,
    save_results: bool = True,
) -> dict:
    """Full evaluation pipeline: load questions → run RAG → score with RAGAS."""
    print("=" * 60)
    print("RAGAS EVALUATION PIPELINE")
    print("=" * 60 + "\n")

    if questions is None:
        questions = load_eval_questions()

    print("Step 1/3: Running questions through RAG pipeline...\n")
    dataset = collect_rag_responses(questions)

    print("Step 2/3: Building RAGAS evaluation dataset...\n")
    eval_dataset = EvaluationDataset.from_list(dataset)

    evaluator_llm = LangchainLLMWrapper(ChatGroq(model=LLM_MODEL))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    print("Step 3/3: Running RAGAS metrics (this may take a few minutes)...\n")
    metrics = [
        Faithfulness(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings),
    ]

    start = time.time()
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    elapsed = time.time() - start

    scores = print_scorecard(results, elapsed)

    if save_results:
        save_evaluation_results(results, scores)

    return results


def _extract_scores(results) -> dict[str, float | None]:
    """Pull aggregate scores out of the EvaluationResult."""
    df = results.to_pandas()
    scores = {}

    for canonical in ["faithfulness", "context_recall", "factual_correctness", "semantic_similarity"]:
        matching_cols = [c for c in df.columns if c.startswith(canonical)]
        if matching_cols:
            values = df[matching_cols[0]].dropna()
            scores[canonical] = float(values.mean()) if len(values) > 0 else None
        else:
            scores[canonical] = None

    return scores


def print_scorecard(results, elapsed: float) -> dict:
    """Print a clean scorecard of RAGAS metrics."""
    scores = _extract_scores(results)

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION SCORECARD")
    print("=" * 60)

    labels = {
        "faithfulness": "Faithfulness",
        "context_recall": "LLM Context Recall",
        "factual_correctness": "Factual Correctness",
        "semantic_similarity": "Semantic Similarity",
    }

    for key, label in labels.items():
        value = scores.get(key)
        if value is not None:
            print(f"  {label + ':':<24s} {value:.4f}")
        else:
            print(f"  {label + ':':<24s} N/A")

    print(f"\n  Evaluation time: {elapsed:.1f}s")
    print("=" * 60)

    return scores


def save_evaluation_results(results, scores: dict) -> None:
    """Save detailed per-question results to a JSON file."""
    output_path = EVAL_DIR / "eval_results.json"

    df = results.to_pandas()
    records = df.to_dict(orient="records")

    output = {
        "summary": scores,
        "per_question": records,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
