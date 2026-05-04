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
- **SDK (all paths):** `google-genai` — handles both free-tier API and Vertex AI via
  `genai.Client(api_key=...)` vs `genai.Client(vertexai=True, project=..., location=...)`.
  `google-generativeai` is deprecated — do not use. `google-cloud-aiplatform` / `vertexai`
  SDK is NOT needed — `google-genai` replaces it entirely.
- **Model (generation):** `gemini-2.5-flash` — confirmed on both free tier and Vertex AI
- **Model (embeddings):** `gemini-embedding-001` — 3072 dimensions, confirmed on both paths
- **Auth (dev):** `GEMINI_API_KEY` in `.env` — free tier, no billing account needed
- **Auth (Vertex AI):** Application Default Credentials (ADC) via
  `gcloud auth application-default login` + `gcloud auth application-default set-quota-project GCP_PROJECT_ID`
- **GCP project:** `gen-lang-client-0948275824`, location `us-south1`
- **Billing:** Account `013827-052849-7D0788` linked; $300 credit available until 2026-07-20
- **RAG:** Manual vector store (numpy/cosine); Vertex AI Vector Search for stretch
- **Orchestration:** LangGraph (with Gemini as LLM node) — stretch goal (Issue #18)
- **Evaluation:** RAGAS

The notebook has a 3-way `BACKEND` toggle: `"ollama"` (local, quota-free) /
`"gemini_api"` (free tier, 20 req/day) / `"vertex_ai"` (enterprise, uses billing credits).

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
│   ├── generation.py       # Gemini generation with context; returns usage_metadata
│   ├── cost.py             # CostTracker — per-query and session cost at Vertex AI rates
│   └── agent.py            # LangGraph agentic wrapper (Day 5 stretch — not yet created)
├── corpus/                 # fetched and processed GCP docs
├── eval/
│   ├── test_set.json       # Q&A pairs with ground truth
│   └── evaluate.py         # RAGAS evaluation script
└── writeup/
    └── customer_summary.md # one-page business-facing summary
```

## Key GCP Concepts (orientation notes)

- **ADC auth:** Run `gcloud auth application-default login` then
  `gcloud auth application-default set-quota-project PROJECT_ID`. Both steps required —
  the second links billing so Vertex AI calls don't get PERMISSION_DENIED.
- **One SDK for everything:** `google-genai` handles both paths. Free tier:
  `genai.Client(api_key=KEY)`. Vertex AI: `genai.Client(vertexai=True, project=ID, location=LOC)`.
  Do NOT use `google-generativeai` (deprecated) or `google-cloud-aiplatform` (not needed).
- **Vertex AI API must be enabled:** Visit the GCP console and enable
  `aiplatform.googleapis.com` on the project before making Vertex AI calls.
- **Vertex AI vs Bedrock:** Vertex AI Studio ≈ Bedrock Playground; Vertex AI Search ≈
  Bedrock Knowledge Bases; Vertex AI Agent Builder ≈ Bedrock Agents; Model Garden ≈
  Bedrock model catalog.
- **Model gotchas:** `gemini-2.0-flash` has `limit: 0` quota on free-tier projects — use
  `gemini-2.5-flash`. `text-embedding-004` returns 404 on the free-tier API — use
  `gemini-embedding-001` (3072 dims).
- **Pricing (verified May 2026):** `gemini-2.5-flash` standard: $0.30 input / $2.50 output
  per 1M tokens. `gemini-2.5-flash-lite`: $0.10 / $0.40. Embedding pricing not listed on
  the pricing page in a parseable section — verify from GCP billing console.

## Implementation Priorities

1. ~~Get auth working first~~ — done (Day 1)
2. ~~Get a single Gemini call working before building the pipeline~~ — done (Day 1)
3. ~~Get a single embedding working before building the retrieval system~~ — done (Day 1)
4. ~~Build RAG pipeline incrementally~~ — done (Day 2): chunker → embedder → retrieval → generation → ask()
5. ~~RAGAS evaluation~~ — done (Day 3): all 4 metrics ≥ 0.70; Faithfulness 0.98, Context Recall 0.94
6. ~~Vertex AI verification~~ — done (Day 4): both models confirmed on Vertex AI via google-genai
7. LangGraph agentic wrapper — Day 5 stretch (Issue #18)

## Resume Gate

**CLEARED (2026-05-04).** Day 4 Vertex AI verification succeeded.
- Add `GCP (Vertex AI · Gemini · Vertex AI Search)` to `~/Documents/job-search-2026/resume-2026-fde4.html`

## Corpus Notes (learned during Day 2)

- **Rebrand banner:** Most GCP docs pages start with "Vertex AI is transitioning to become
  part of Gemini Enterprise Agent Platform..." — this is site-wide nav noise, not content.
  Strip it in `extract_text()` in `src/build_corpus.py`.
- **Per-page cap:** `MAX_CHARS = 50_000` prevents single pages dominating the chunk index.
  `gemini_pricing` and `vertex_ai_access_control` both hit this cap. Pricing in particular
  surfaces as retrieval noise — consider a lower per-page cap or dropping it.
- **Embedding rate limits:** Free-tier embedding quota is tight. Batch size of 10 with 2s
  inter-batch delay avoids most 429s; exponential backoff (10s base, 5 retries) handles the
  rest. After any corpus rebuild, delete `corpus/embeddings.npy` and `corpus/chunks.json`
  before re-embedding.

## RAGAS Notes

- **Required packages:** `ragas`, `datasets`, `langchain-google-genai`, `langchain-ollama`
- **Judge LLM fallback chain:** Vertex AI `gemini-2.5-flash` (preferred, no daily quota) →
  free-tier `gemini-2.5-flash-lite` → Ollama `gemma3:1b` → Ollama `gemma4:e2b`
- **Vertex AI judge:** `ChatGoogleGenerativeAI(model="gemini-2.5-flash", vertexai=True, project=..., location=...)`
  with `LangchainLLMWrapper`. Same for embeddings via `GoogleGenerativeAIEmbeddings`.
- **RAGAS API version (0.4.3):** Use old `ragas.metrics` singletons (`faithfulness`,
  `answer_relevancy`, `context_precision`, `context_recall`) with `LangchainLLMWrapper`.
  The `ragas.metrics.collections` namespace is incompatible with `evaluate()`.
- **RunConfig:** `timeout=180, max_workers=4` for Gemini/Vertex AI. `timeout=600, max_workers=1`
  for Ollama. NaN results occur when a job times out — handled gracefully in display/JSON.
- **Runtime:** ~3 minutes on Vertex AI for 12 questions. Pipeline cost tracked per-question
  via `CostTracker`; RAGAS judge costs billed separately to the same GCP project.
- **Key metrics:** `faithfulness` (answer grounded in context), `answer_relevancy`
  (answer addresses the question), `context_precision` (retrieved chunks are relevant),
  `context_recall` (relevant chunks were retrieved). Threshold: ≥ 0.70 to pass.

## Cost Tracking Notes

- `src/cost.py` — `CostTracker` active on all three backends; labeled appropriately
- **Verified pricing (May 2026, from corpus/gemini_pricing.txt):**
  - `gemini-2.5-flash` standard: $0.30 input / $2.50 output per 1M tokens
  - `gemini-2.5-flash-lite` standard: $0.10 input / $0.40 output per 1M tokens
- **Embedding pricing:** NOT verified from primary source. Corpus hits 10K char cap before
  reaching embedding section of pricing page. $0.025/1M tokens is a proxy — do not cite
  without checking GCP billing console.
- Typical query: ~$0.001 (generation only; embedding cost is near-zero at this scale)

## The Pitch

This project enables a specific interview answer: "I've built this architecture on
AWS Bedrock in production. I wanted to understand Vertex AI before talking to your
customers about it, so I built the equivalent here. Here's what's the same, here's
what's different, here's what surprised me." That answer is more credible than
claiming GCP expertise — it shows the learning pattern an FDE needs.
