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

## Two-Path Backend Strategy

The notebook supports two backends, selected by a single variable at the top:

```python
BACKEND = "gemini_api"   # Free development path (default)
# BACKEND = "vertex_ai"  # Enterprise path — requires GCP billing account
```

This design strengthens the FDE pitch: it demonstrates explicit awareness of the
difference between the Gemini API (direct, consumer/developer) and Vertex AI
(enterprise, managed, IAM-controlled), which is a common customer decision point.

### Path 1 — Gemini API (Free Development Path)

Used for all development, testing, and the primary demo run.

| Component | Detail |
|---|---|
| SDK | `google-generativeai` |
| Auth | `GEMINI_API_KEY` in `.env` — free tier, no billing account required |
| Model (generation) | `gemini-2.5-flash` — confirmed working on free tier |
| Model (embeddings) | `gemini-embedding-001` — 3072 dimensions, confirmed working on free tier |
| AWS equivalent | Direct Bedrock API with API key (vs. IAM-gated Bedrock) |

Free tier limits are well within portfolio demo usage. No billing surprises.

### Path 2 — Vertex AI (Verification Path)

Used for a final verification run before the project is called done. Confirms the
same code works against the enterprise-grade platform.

| Component | Detail |
|---|---|
| SDK | `google-cloud-aiplatform` / `vertexai` |
| Auth | Application Default Credentials (ADC) via `gcloud auth application-default login` |
| Model (generation) | `gemini-2.5-flash` via Vertex AI |
| Model (embeddings) | `gemini-embedding-001` via Vertex AI |
| Cost | ~$1 for a full demo run — covered by GCP free trial credits |
| AWS equivalent | Bedrock with IAM role / service account auth |

Every Vertex AI SDK call requires project and location at init:
```python
import vertexai
vertexai.init(project=PROJECT_ID, location="us-central1")
```

### SDK Comparison

| Concern | Gemini API (`google-generativeai`) | Vertex AI (`google-cloud-aiplatform`) |
|---|---|---|
| Auth | API key | ADC / service account |
| Billing | Free tier available | Billing account required |
| Enterprise features | Limited | Full (VPC-SC, CMEK, audit logs) |
| Regional control | No | Yes |
| SLA | No | Yes |
| When to use | Prototyping, demos, personal projects | Production, regulated industries |

---

## RAG Pipeline

| Component | MVP | Stretch |
|---|---|---|
| Chunking | Custom Python (fixed-size + overlap) | LangChain text splitters |
| Embedding store | numpy in-memory + JSON persistence | Vertex AI Vector Search |
| Similarity search | Cosine similarity (numpy) | Vertex AI Vector Search ANN |
| Retrieval | Top-k cosine search | Managed vector index query |

The RAG pipeline is backend-agnostic — only the embedding call differs between paths.

### Core libraries

```
numpy         # vector math / cosine similarity
```

---

## Orchestration (Stretch)

| Option | Status | Notes |
|---|---|---|
| **LangGraph** | Stretch (Day 4) | Graph-based agent; Gemini as LLM node; explicit state machine |
| **Google ADK** | Reference only | Google's Agent Development Kit; more opinionated |

### Libraries (if stretch is implemented)

```
langgraph
langchain-google-genai      # LangChain wrapper for Gemini API path
langchain-google-vertexai   # LangChain wrapper for Vertex AI path
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
| `gcloud` CLI | GCP project setup, ADC auth for Vertex AI verification |
| `python-dotenv` | `.env` for `GEMINI_API_KEY`, `PROJECT_ID`, and config |
| VS Code + Claude Code | IDE and AI pair programming |
| GitHub | Version control; Issues for SDD task tracking |

---

## Dependency Summary

```
# Gemini API (free development path)
google-generativeai

# Vertex AI (verification path)
google-cloud-aiplatform

# RAG (MVP)
numpy

# Evaluation
ragas
datasets

# Orchestration (stretch)
langgraph
langchain-google-genai
langchain-google-vertexai

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
