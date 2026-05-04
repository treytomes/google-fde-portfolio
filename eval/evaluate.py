"""
RAGAS evaluation of the RAG pipeline over eval/test_set.json.

Usage:
    python3 eval/evaluate.py

Requires GEMINI_API_KEY in .env. Expect 3-5 minutes on free tier due to
rate limits — RAGAS makes multiple LLM calls per question for scoring.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    ContextRecall,
)

from embedder import load_embeddings
from retrieval import retrieve
from generation import generate

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
TEST_SET_PATH = Path(__file__).parent / "test_set.json"
INTER_QUESTION_DELAY = 12  # seconds between questions — free-tier rate limit courtesy


def build_samples(client, embeddings, chunks, test_set: list[dict]) -> list[SingleTurnSample]:
    samples = []
    for i, item in enumerate(test_set, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{len(test_set)}] {question[:70]}...")
        context_chunks = retrieve(question, client, embeddings, chunks, k=5)
        answer = generate(question, context_chunks, client)

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer["text"],
            retrieved_contexts=[c["text"] for c in context_chunks],
            reference=ground_truth,
        ))

        if i < len(test_set):
            time.sleep(INTER_QUESTION_DELAY)

    return samples


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key, "GEMINI_API_KEY not set — check .env"

    print("Loading corpus embeddings...")
    embeddings, chunks = load_embeddings(CORPUS_DIR)
    print(f"  {embeddings.shape[0]} chunks, {embeddings.shape[1]} dims")

    print("Loading test set...")
    test_set = json.loads(TEST_SET_PATH.read_text())
    print(f"  {len(test_set)} questions")

    client = genai.Client(api_key=api_key)

    print("\nGenerating answers for each question...")
    samples = build_samples(client, embeddings, chunks, test_set)

    print("\nRunning RAGAS evaluation (this takes a few minutes)...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key)
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

    dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=emb),
            LLMContextPrecisionWithoutReference(llm=llm),
            ContextRecall(llm=llm),
        ],
    )

    df = results.to_pandas()

    print("\n" + "═" * 72)
    print("  RAGAS EVALUATION RESULTS")
    print("═" * 72)

    metric_cols = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference", "context_recall"]
    display_names = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "llm_context_precision_without_reference": "Context Precision",
        "context_recall": "Context Recall",
    }

    print("\n  Per-question scores:\n")
    for i, row in df.iterrows():
        q = test_set[i]["question"]
        print(f"  Q{i+1}: {q[:65]}...")
        for col in metric_cols:
            val = row.get(col)
            bar = "█" * int((val or 0) * 10) if val is not None else ""
            score_str = f"{val:.3f}" if val is not None else "  N/A"
            print(f"       {display_names[col]:<22} {score_str}  {bar}")
        print()

    print("  Aggregate means:\n")
    means = df[metric_cols].mean()
    for col in metric_cols:
        val = means[col]
        bar = "█" * int(val * 10)
        gate = "✓" if val >= 0.7 else "✗"
        print(f"  {gate}  {display_names[col]:<22} {val:.3f}  {bar}")

    print("\n" + "═" * 72)
    print("  Thresholds: ✓ = ≥ 0.70   ✗ = < 0.70")
    print("═" * 72)

    results_path = Path(__file__).parent / "results.json"
    output = {
        "means": {display_names[c]: round(float(means[c]), 4) for c in metric_cols},
        "per_question": [
            {
                "question": test_set[i]["question"],
                "topic": test_set[i]["topic"],
                **{display_names[c]: round(float(row.get(c, 0) or 0), 4) for c in metric_cols},
            }
            for i, row in df.iterrows()
        ],
    }
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
