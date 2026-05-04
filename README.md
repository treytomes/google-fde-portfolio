# Grounded Q&A Assistant — Vertex AI / Gemini

A RAG pipeline and agentic Q&A assistant built on Google Cloud Vertex AI, evaluated
with RAGAS. Demonstrates the architecture an enterprise team would deploy to let
employees query vendor documentation reliably — answers grounded in source text, every
claim auditable back to the page it came from.

Built as a hands-on GCP parallel to an AWS Bedrock production deployment at City Electric
Supply: same RAG + agentic pattern, different cloud surface.

---

## Notebooks

| Notebook | What it shows |
|---|---|
| [notebooks/rag_pipeline.ipynb](notebooks/rag_pipeline.ipynb) | Full RAG pipeline — corpus → embeddings → retrieval → generation → RAGAS evaluation |
| [notebooks/agent_demo.ipynb](notebooks/agent_demo.ipynb) | LangGraph tool-calling agent — routes queries to RAG corpus or direct Gemini answer |
| [notebooks/day1_quickstart.ipynb](notebooks/day1_quickstart.ipynb) | Day 1 SDK verification — Gemini and embedding calls on both free-tier and Vertex AI |

---

## Architecture

```
User query
    │
    ▼
[Embed query]  ←  gemini-embedding-001 (3072 dims)
    │
    ▼
[Cosine search]  ←  104 chunks from 20 GCP docs pages (numpy vector store)
    │
    ▼
[Generate answer]  ←  gemini-2.5-flash with retrieved context
    │
    ▼
Answer + source citations
```

The agent layer (agent_demo.ipynb) adds a LangGraph ReAct loop on top: Gemini decides
whether to invoke the RAG pipeline (`search_docs`) or answer from training knowledge
directly (`answer_direct`), based on whether the question requires GCP documentation.

---

## RAGAS Evaluation Results

Evaluated against 12 held-out questions across 8 topics. Judge: `gemini-2.5-flash` on
Vertex AI.

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | **0.96** | ≥ 0.70 ✓ |
| Context Recall | **0.93** | ≥ 0.70 ✓ |
| Answer Relevancy | **0.86** | ≥ 0.70 ✓ |
| Context Precision | **0.73** | ≥ 0.70 ✓ |

Context Precision is lowest on pricing queries — the pricing page is large and noisy,
so retrieval pulls in unrelated rows alongside the relevant ones. RAGAS surfaces this
precisely, which is the point.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| SDK | `google-genai` | One SDK for both free-tier API and Vertex AI — no `google-cloud-aiplatform` needed |
| Embedding | `gemini-embedding-001` | 3072 dims; confirmed working on both API paths |
| Generation | `gemini-2.5-flash` | Best free-tier + Vertex AI availability; fast and cheap |
| Vector store | numpy cosine similarity | No infra dependency; sufficient for a 104-chunk corpus |
| Orchestration | LangGraph (`langchain.agents.create_agent`) | Explicit, auditable tool routing |
| Evaluation | RAGAS 0.4.3 | Four standard RAG metrics; Vertex AI judge, no daily quota |
| Auth | ADC (`gcloud auth application-default login`) | Same auth flow as production GCP workloads |

---

## Setup

**Prerequisites:** Python 3.12+, a GCP project with Vertex AI API enabled, and either
a `GEMINI_API_KEY` (free tier) or ADC credentials configured.

```bash
git clone https://github.com/treytomes/google-fde-portfolio.git
cd google-fde-portfolio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY and/or GCP_PROJECT_ID
```

**For Vertex AI (ADC auth):**
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Then open either notebook in Jupyter and set the `BACKEND` toggle at the top of the
setup cell:
- `"gemini_api"` — free Gemini API, no billing needed, 20 req/day limit
- `"vertex_ai"` — Vertex AI, uses GCP billing credits, no daily quota

---

## Project Structure

```
google-fde-portfolio/
├── notebooks/
│   ├── rag_pipeline.ipynb      # Main demo: RAG pipeline + RAGAS evaluation
│   ├── agent_demo.ipynb        # Agent demo: LangGraph tool routing
│   └── day1_quickstart.ipynb   # SDK + model verification
├── src/
│   ├── agent.py                # LangGraph agent (build_agent, run_query)
│   ├── build_corpus.py         # Fetch and clean GCP docs pages
│   ├── chunker.py              # Split documents into ~800-char chunks
│   ├── embedder.py             # Batch embedding with rate-limit handling
│   ├── retrieval.py            # Cosine similarity search
│   ├── generation.py           # Gemini generation with context; returns usage metadata
│   └── cost.py                 # Per-query cost tracking at Vertex AI rates
├── corpus/                     # Fetched GCP docs (20 pages, 104 chunks)
├── eval/
│   ├── test_set.json           # 12 Q&A pairs with ground truth
│   └── evaluate.py             # RAGAS evaluation script
└── writeup/
    └── customer_summary.md     # One-page business-facing summary
```

---

## Cost

A typical RAG query on Vertex AI costs under **$0.001** (generation only; embedding
cost is near-zero at this corpus scale). The RAGAS evaluation of 12 questions runs in
~3 minutes and costs roughly $0.01–0.02 total in judge calls.

Pricing used: `gemini-2.5-flash` standard — $0.30 input / $2.50 output per 1M tokens
(verified from Vertex AI pricing page, May 2026). Embedding pricing is estimated —
verify against your GCP billing console for production use.
