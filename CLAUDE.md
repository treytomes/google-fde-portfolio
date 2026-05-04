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
- **Model (dev):** Gemini API via `google-generativeai` — free tier, `GEMINI_API_KEY` in `.env`
- **Model (verification):** Gemini via Vertex AI Python SDK (`google-cloud-aiplatform`) — final verification run only
- **Embeddings:** `text-embedding-004` — same model, available on both paths
- **RAG:** Manual vector store (numpy/cosine) for MVP; Vertex AI Vector Search for stretch
- **Orchestration:** LangGraph (with Gemini as LLM node) — stretch goal
- **Evaluation:** RAGAS
- **Auth (dev):** Gemini API key — no billing account needed
- **Auth (verification):** Application Default Credentials (ADC) via `gcloud auth application-default login`

The notebook has a single `BACKEND` toggle at the top. All development uses `"gemini_api"`.
The final verification switches to `"vertex_ai"` to confirm enterprise compatibility.

## Project Structure

```
google-fde-portfolio/
├── CLAUDE.md               # this file
├── PROJECT_PLAN.md         # full plan with day-by-day tasks
├── src/
│   ├── embeddings.py       # chunk + embed corpus
│   ├── retrieval.py        # vector search
│   ├── generation.py       # Gemini call with retrieved context
│   ├── pipeline.py         # end-to-end Q&A
│   └── agent.py            # LangGraph agentic wrapper (stretch)
├── corpus/                 # source documents
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
- **Gemini SDK (dev path):** `import google.generativeai as genai` — direct Gemini API,
  free tier, used for all development.
- **Gemini SDK (verification path):** `from vertexai.generative_models import GenerativeModel` —
  Vertex AI SDK, used for the final verification run only.
- **Project/location:** Every Vertex AI call needs `project` and `location` params.
  Initialize once with `vertexai.init(project=PROJECT_ID, location="us-central1")`.

## Implementation Priorities

1. Get auth working first — everything else depends on it
2. Get a single Gemini call working before building the pipeline
3. Get a single embedding working before building the retrieval system
4. Build incrementally — working at each step before moving to the next
5. RAGAS evaluation is not optional — it's the differentiator for the FDE pitch

## Resume Gate

Once the Vertex AI quickstart is confirmed working (Day 1 complete):
- Add GCP back to `~/Documents/job-search-2026/resume-2026-fde4.html` competencies
- Specific addition: `GCP (Vertex AI · Gemini · Vertex AI Search)`

Do not add it before the quickstart is done.

## The Pitch

This project enables a specific interview answer: "I've built this architecture on
AWS Bedrock in production. I wanted to understand Vertex AI before talking to your
customers about it, so I built the equivalent here. Here's what's the same, here's
what's different, here's what surprised me." That answer is more credible than
claiming GCP expertise — it shows the learning pattern an FDE needs.
