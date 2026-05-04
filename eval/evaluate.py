"""
RAGAS evaluation of the RAG pipeline over eval/test_set.json.

Usage:
    python3 eval/evaluate.py

Requires GEMINI_API_KEY in .env.

Judge LLM fallback chain:
  1. gemini-2.5-flash-lite   (preferred — fast, sufficient quota)
  2. Ollama gemma3:1b        (fast local fallback, ~44s/metric/question)
  3. Ollama gemma4:e2b       (slow local fallback, ~235s/metric/question)

Pull local models with: ollama pull gemma3:1b && ollama pull gemma4:e2b
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate, EvaluationDataset, RunConfig, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from embedder import load_embeddings
from retrieval import retrieve
from generation import generate, OLLAMA_MODEL, OLLAMA_BASE_URL

CORPUS_DIR    = Path(__file__).parent.parent / "corpus"
TEST_SET_PATH = Path(__file__).parent / "test_set.json"

OLLAMA_FAST_MODEL = "gemma3:1b"   # ~44s/metric/question — preferred local model
OLLAMA_SLOW_MODEL = OLLAMA_MODEL  # ~235s/metric/question — fallback if 1b not pulled

INTER_QUESTION_DELAY = 12  # seconds between questions — free-tier rate limit courtesy

GEMINI_RUN_CONFIG = RunConfig(timeout=120, max_workers=4,  max_retries=5)
OLLAMA_RUN_CONFIG = RunConfig(timeout=600, max_workers=1,  max_retries=3)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
DISPLAY_NAMES = {
    "faithfulness":        "Faithfulness",
    "answer_relevancy":    "Answer Relevancy",
    "context_precision":   "Context Precision",
    "context_recall":      "Context Recall",
}


def _gemini_quota_ok(api_key: str) -> bool:
    try:
        genai.Client(api_key=api_key).models.generate_content(
            model="gemini-2.5-flash-lite", contents="ok"
        )
        return True
    except Exception:
        return False


def _ollama_model_available(model: str) -> bool:
    try:
        import ollama
        names = [m.model for m in ollama.list().models]
        return any(model in n for n in names)
    except Exception:
        return False


def _make_judge_llm_and_config(api_key: str):
    """Return (ragas_llm, ragas_emb, run_config, label) for the best available judge."""
    emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    )
    if _gemini_quota_ok(api_key):
        llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key)
        )
        return llm, emb, GEMINI_RUN_CONFIG, "gemini-2.5-flash-lite"

    from langchain_ollama import ChatOllama
    for model in [OLLAMA_FAST_MODEL, OLLAMA_SLOW_MODEL]:
        if _ollama_model_available(model):
            llm = LangchainLLMWrapper(
                ChatOllama(model=model, base_url=OLLAMA_BASE_URL, num_ctx=8192)
            )
            label = f"Ollama {model} (Gemini quota exhausted)"
            return llm, emb, OLLAMA_RUN_CONFIG, label

    raise RuntimeError(
        "No judge LLM available. Either restore Gemini quota or run: "
        f"ollama pull {OLLAMA_FAST_MODEL}"
    )


def build_samples(client, embeddings, chunks, test_set: list[dict]) -> list[SingleTurnSample]:
    samples = []
    for i, item in enumerate(test_set, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{len(test_set)}] {question[:70]}...")
        context_chunks = retrieve(question, client, embeddings, chunks, k=5)
        answer         = generate(question, context_chunks, client)

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

    print("\nSelecting RAGAS judge LLM...")
    llm, emb, run_config, label = _make_judge_llm_and_config(api_key)
    print(f"  Judge LLM  : {label}")
    print(f"  Embeddings : gemini-embedding-001")

    for metric in METRICS:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = emb

    print("\nRunning RAGAS evaluation...")
    if "Ollama" in label:
        print("  (Using local model — this will take a while)")

    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=METRICS,
        run_config=run_config,
    )

    df = results.to_pandas()
    means = df[METRIC_COLS].mean()

    print("\n" + "═" * 72)
    print("  RAGAS EVALUATION RESULTS")
    print("═" * 72)

    print("\n  Per-question scores:\n")
    for i, row in df.iterrows():
        q = test_set[i]["question"]
        print(f"  Q{i+1}: {q[:65]}...")
        for col in METRIC_COLS:
            val = row.get(col)
            bar  = "█" * int((val or 0) * 10) if val is not None else ""
            s    = f"{val:.3f}" if val is not None else "  N/A"
            print(f"       {DISPLAY_NAMES[col]:<22} {s}  {bar}")
        print()

    print("  Aggregate means:\n")
    for col in METRIC_COLS:
        val  = means[col]
        bar  = "█" * int(val * 10)
        gate = "✓" if val >= 0.7 else "✗"
        print(f"  {gate}  {DISPLAY_NAMES[col]:<22} {val:.3f}  {bar}")

    print("\n" + "═" * 72)
    print("  Thresholds: ✓ = ≥ 0.70   ✗ = < 0.70")
    print("  Judge LLM  :", label)
    print("═" * 72)

    results_path = Path(__file__).parent / "results.json"
    output = {
        "judge_llm": label,
        "means": {DISPLAY_NAMES[c]: round(float(means[c]), 4) for c in METRIC_COLS},
        "per_question": [
            {
                "question": test_set[i]["question"],
                "topic": test_set[i]["topic"],
                **{DISPLAY_NAMES[c]: round(float(row.get(c, 0) or 0), 4) for c in METRIC_COLS},
            }
            for i, row in df.iterrows()
        ],
    }
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
