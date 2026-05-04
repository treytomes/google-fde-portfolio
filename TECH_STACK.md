# Tech Stack — Google Cloud FDE IV Portfolio

## Overview

This project is a document Q&A assistant built on Google Cloud, demonstrating
the same RAG + agentic architecture previously built on AWS Bedrock at City
Electric Supply. The stack maps directly to GCP equivalents at every layer.

---

## Language & Runtime

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | Consistent with GCP SDK ecosystem |
| Package manager | pip / venv | Standard; no build system needed at this scale |
| Presentation format | Jupyter Notebook | Rich output (inline results, markdown narrative, RAGAS visualizations) without web server complexity |

---

## Google Cloud Platform

| Service | Purpose | AWS Bedrock Equivalent |
|---|---|---|
| **Vertex AI** | Platform for all ML/AI services | Amazon Bedrock (platform) |
| **Gemini 1.5 Pro** (via Vertex AI) | Text generation / LLM inference | Bedrock Claude / Titan |
| **Vertex AI Embeddings** (`text-embedding-004`) | Embed queries and document chunks | Bedrock Titan Embeddings |
| **Vertex AI Vector Search** *(stretch)* | Managed ANN vector index | Bedrock Knowledge Bases (managed) |
| **Vertex AI Agent Builder** *(reference)* | Managed agent/search platform | Bedrock Agents |
| **GCP IAM / ADC** | Authentication | AWS IAM / credential chain |

### SDK

```
google-cloud-aiplatform   # Vertex AI Python SDK (primary)
vertexai                  # High-level Vertex AI SDK (same package, different namespace)
```

**Not used:** `google-generativeai` — that targets the direct Gemini API, not Vertex AI.

### Auth

Application Default Credentials (ADC):
```bash
gcloud auth application-default login
```
SDK picks up credentials automatically; no credential file management needed for local dev.
Every SDK call requires `project` and `location` at init:
```python
import vertexai
vertexai.init(project=PROJECT_ID, location="us-central1")
```

---

## RAG Pipeline

| Component | MVP | Stretch |
|---|---|---|
| Chunking | Custom Python (fixed-size + overlap) | LangChain text splitters |
| Embedding store | numpy in-memory + JSON persistence | Vertex AI Vector Search |
| Similarity search | Cosine similarity (numpy) | Vertex AI Vector Search ANN |
| Retrieval | Top-k cosine search | Managed vector index query |

### Core libraries (MVP)

```
numpy         # vector math / cosine similarity
```

---

## Orchestration

| Option | Status | Notes |
|---|---|---|
| **LangGraph** | Primary | Graph-based agent; Gemini as LLM node; explicit state machine |
| **Google ADK** | Alternative | Google's Agent Development Kit; more opinionated |

LangGraph is preferred — it's framework-agnostic and maps cleanly to the LangChain
graph model, making the AWS-to-GCP comparison story cleaner.

### Libraries

```
langgraph
langchain-google-vertexai   # LangChain wrapper for Vertex AI / Gemini
```

---

## Evaluation

| Tool | Purpose |
|---|---|
| **RAGAS** | RAG pipeline evaluation framework |

### Metrics targeted

| Metric | What it measures |
|---|---|
| Faithfulness | Are answers grounded in retrieved context? (no hallucination) |
| Answer Relevance | Does the answer address the question? |
| Context Precision | Is retrieved context ranked well? (relevant chunks high) |
| Context Recall | Does retrieved context cover the ground-truth answer? |

### Libraries

```
ragas
datasets   # HuggingFace datasets (RAGAS input format)
```

---

## Development & Tooling

| Tool | Purpose |
|---|---|
| `gcloud` CLI | GCP project setup, ADC auth, API enablement |
| `python-dotenv` | `.env` for `PROJECT_ID` and config (not secrets) |
| VS Code + Claude Code | IDE and AI pair programming |
| GitHub | Version control; Issues for SDD task tracking |

---

## Dependency Summary

```
# GCP / Vertex AI
google-cloud-aiplatform

# Orchestration
langgraph
langchain-google-vertexai

# RAG (MVP)
numpy

# Evaluation
ragas
datasets

# Dev utility
python-dotenv
jupyter
```

> A `requirements.txt` will be generated once the initial environment is validated
> (Day 1 quickstart complete).

---

## Corpus

| Property | Detail |
|---|---|
| **Source** | GCP / Vertex AI official documentation (`cloud.google.com`) |
| **Format** | HTML / Markdown (scraped or downloaded) |
| **Why** | Meta-relevant — the assistant answers questions about the platform it's built on. No licensing concerns. Strong FDE demo: "Ask it anything about Vertex AI." |
| **Stretch** | Add AWS-to-GCP service mapping docs as a second source to enable cross-cloud comparison queries |

---

## What's Intentionally Out of Scope

- **Databases / persistent storage** — embeddings stored in-memory or flat JSON for MVP
- **Web server / frontend** — Jupyter Notebook is the presentation layer
- **Docker / containerization** — not needed for a portfolio demo
- **CI/CD** — manual runs; GitHub for version control only
- **Vertex AI Vector Search** at launch — MVP uses numpy; Vector Search is a named stretch goal
