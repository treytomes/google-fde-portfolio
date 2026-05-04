# Grounded Q&A Assistant on Google Cloud Vertex AI
### Technical Summary — May 2026

---

## What It Does

This project is a question-answering assistant that answers questions about Google
Cloud by reading the official documentation — not by guessing. Ask it how a feature
works, what a service costs, or how to configure authentication, and it returns an
answer with direct citations to the source pages it used.

The key constraint: the system is only allowed to answer from what it retrieved. If
the documentation doesn't cover the question, it says so. This makes it useful in
enterprise contexts where hallucinated or outdated answers cause real problems.

---

## The Business Problem

Enterprise teams adopting a new cloud platform face a knowledge gap. Documentation
is extensive but scattered. Engineers waste time searching; sales and solutions teams
give inconsistent answers. An assistant grounded in the vendor's own documentation
closes that gap — and every answer it gives can be audited back to a source.

---

## How It Works

The architecture follows the Retrieval-Augmented Generation (RAG) pattern:

1. **Corpus** — Official Google Cloud documentation pages are fetched and cleaned
2. **Embedding** — Each document chunk is converted to a vector using
   `gemini-embedding-001` (3,072 dimensions), capturing semantic meaning
3. **Retrieval** — At query time, the question is embedded with the same model and
   the five most similar documentation chunks are retrieved by cosine similarity
4. **Generation** — The retrieved chunks are passed to `gemini-2.5-flash` as context,
   which synthesizes a grounded answer with source citations

A typical query costs under $0.001 at Vertex AI rates (generation cost verified;
embedding cost estimated — verify against GCP billing for production use).

---

## Google Cloud Components

| Component | Role |
|---|---|
| `gemini-embedding-001` | Converts text to semantic vectors for retrieval |
| `gemini-2.5-flash` | Synthesizes answers from retrieved context |
| Vertex AI (ADC auth) | Enterprise API access with project-level billing and quota |
| `google-genai` SDK | Unified SDK for both free-tier development and Vertex AI production |

---

## Reliability — RAGAS Evaluation

The pipeline was evaluated against 12 held-out questions across 8 topics using
RAGAS, an open-source framework that uses an independent language model to score
answer quality. Results (Vertex AI judge, `gemini-2.5-flash`):

| Metric | Score | What it means |
|---|---|---|
| **Faithfulness** | **0.96** | 96% of answer claims are directly supported by retrieved text |
| **Context Recall** | **0.93** | Retrieval finds the relevant documentation in 93% of cases |
| **Answer Relevancy** | **0.86** | Answers address what was actually asked |
| **Context Precision** | **0.73** | Retrieved chunks are on-topic (lower on pricing queries — see below) |

The one weak area — Context Precision on pricing questions — reflects a corpus
quality issue: the pricing page is large and noisy, so retrieval pulls in unrelated
pricing rows alongside the relevant ones. RAGAS identified this precisely, which
illustrates its value: it doesn't just report a score, it tells you where to improve.

---

## The AWS Parallel

This is the same pattern deployed on AWS Bedrock at City Electric Supply: S3 corpus,
Bedrock Knowledge Bases for embeddings and vector search, Claude for generation,
agentic tool integrations, and org-wide rollout. This project reproduces that
architecture on Google Cloud to validate the platform differences firsthand before
advising customers on it. The pattern is identical; the SDK surfaces differ.
