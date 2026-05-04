# Google FDE Portfolio Project — Claude Context

## What This Is

A portfolio project demonstrating readiness for the Google Cloud Forward Deployed
Engineer (FDE IV) role in Addison, TX. The project builds a RAG pipeline and agentic
Q&A assistant on Vertex AI / Gemini, evaluated with RAGAS.

See PROJECT_PLAN.md for the full implementation plan and day-by-day tasks.

## Who Trey Is

Senior software engineer, 20+ years experience. Strong in Python, C#, AWS, .NET.
Built an enterprise AI assistant on AWS Bedrock at City Electric Supply — RAG pipelines,
agentic tool integrations, event-driven architecture, org-wide rollout. That project
is the direct parallel to what this portfolio project is demonstrating on GCP.

This is a focused learning sprint. Trey is competent with the architecture and AI
concepts; the gap is GCP-specific tooling (Vertex AI SDK, GCP auth, Gemini API).
Treat him as an experienced engineer getting oriented to a new cloud, not a beginner.

## Target Stack

- **Language:** Python
- **SDK (dev):** `google-genai` — the current Gemini API SDK (`google-generativeai` is deprecated as of 2025, do not use)
- **SDK (verification):** `google-cloud-aiplatform` / `vertexai` — Vertex AI SDK, Day 4 only
- **Model (generation):** `gemini-2.5-flash` — confirmed working on free tier (`gemini-2.0-flash` has limit: 0 quota on free-tier projects)
- **Model (embeddings):** `gemini-embedding-001` — 3072 dimensions, confirmed working (`text-embedding-004` not available on free-tier API)
- **Auth (dev):** `GEMINI_API_KEY` in `.env` — free tier, no billing account needed
- **Auth (verification):** Application Default Credentials (ADC) via `gcloud auth application-default login`
- **RAG:** Manual vector store (numpy/cosine) for MVP; Vertex AI Vector Search for stretch
- **Orchestration:** LangGraph (with Gemini as LLM node) — stretch goal
- **Evaluation:** RAGAS

The notebook has a single `BACKEND` toggle at the top. All development uses `"gemini_api"`.
The final verification switches to `"vertex_ai"` to confirm enterprise compatibility.

## Project Structure

```
google-fde-portfolio/
├── CLAUDE.md               # this file
├── PROJECT_PLAN.md         # full plan with day-by-day tasks
├── TECH_STACK.md           # detailed technology choices and rationale
├── notebooks/              # Jupyter notebooks — primary deliverable
│   ├── day1_quickstart.ipynb   # Day 1: SDK + model verification (complete)
│   └── rag_pipeline.ipynb      # Days 2-3: full RAG demo (in progress)
├── src/                    # shared Python helper modules
│   ├── chunker.py          # document chunking
│   ├── embedder.py         # embedding calls (both backends)
│   ├── retrieval.py        # cosine similarity search
│   ├── generation.py       # Gemini generation with context
│   └── agent.py            # LangGraph agentic wrapper (stretch)
├── corpus/                 # fetched and processed GCP docs
├── eval/
│   ├── test_set.json       # Q&A pairs with ground truth
│   └── evaluate.py         # RAGAS evaluation script
└── writeup/
    └── customer_summary.md # one-page business-facing summary
```

## Key GCP Concepts (orientation notes)

- **ADC auth:** Run `gcloud auth application-default login` once locally. SDK picks it
  up automatically. No need to manage credentials files explicitly for local dev.
- **Vertex AI vs Bedrock:** Vertex AI Studio ≈ Bedrock Playground; Vertex AI Search ≈
  Bedrock Knowledge Bases; Vertex AI Agent Builder ≈ Bedrock Agents; Model Garden ≈
  Bedrock model catalog.
- **Gemini SDK (dev path):** `from google import genai; client = genai.Client(api_key=...)` —
  current `google-genai` SDK. Do NOT use `google-generativeai` — it is deprecated.
- **Gemini SDK (verification path):** `from vertexai.generative_models import GenerativeModel` —
  Vertex AI SDK, used for the final verification run only.
- **Model gotchas:** `gemini-2.0-flash` has `limit: 0` quota on free-tier projects — use
  `gemini-2.5-flash`. `text-embedding-004` returns 404 on the free-tier API — use
  `gemini-embedding-001` (3072 dims).
- **Project/location:** Every Vertex AI call needs `project` and `location` params.
  Initialize once with `vertexai.init(project=PROJECT_ID, location="us-central1")`.

## Implementation Priorities

1. Get auth working first — everything else depends on it
2. Get a single Gemini call working before building the pipeline
3. Get a single embedding working before building the retrieval system
4. Build incrementally — working at each step before moving to the next
5. RAGAS evaluation is not optional — it's the differentiator for the FDE pitch

## Resume Gate

Once Day 4 is complete and the notebook runs end-to-end with `BACKEND = "vertex_ai"`:
- Add GCP back to `~/Documents/job-search-2026/resume-2026-fde4.html` competencies
- Specific addition: `GCP (Vertex AI · Gemini · Vertex AI Search)`

Do not add it before the Vertex AI verification run succeeds. The free Gemini API
development path (Days 1–3) does not satisfy this gate.

## The Pitch

This project enables a specific interview answer: "I've built this architecture on
AWS Bedrock in production. I wanted to understand Vertex AI before talking to your
customers about it, so I built the equivalent here. Here's what's the same, here's
what's different, here's what surprised me." That answer is more credible than
claiming GCP expertise — it shows the learning pattern an FDE needs.
