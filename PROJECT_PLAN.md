# Google Cloud FDE IV — Portfolio Project Plan

## Purpose

This project demonstrates readiness for the Google Cloud Forward Deployed Engineer (FDE IV)
role in Addison, TX. It needs to show two things simultaneously:

1. Ability to build on GCP / Vertex AI
2. Ability to frame the work as a business solution, not just a tech demo

The target interviewer is a VP-level stakeholder, not a fellow engineer.

---

## Background

Trey Tomes has built the equivalent of this project on AWS Bedrock — an internal
enterprise AI assistant at City Electric Supply using RAG pipelines, agentic tool
integrations, and event-driven architecture. The AWS/Vertex AI parallel is a core
part of the FDE pitch: same architecture, different cloud. The portfolio project makes
that claim concrete and provable.

The PHP hiring precedent: Trey joined City Electric's WMS team with no PHP experience.
He was handed a project on Friday and delivered by Monday. This project is the same
move — concentrated study, working implementation, demonstrate the pattern.

---

## Target Stack

| Layer | Development path | Verification path |
|---|---|---|
| Model inference | Gemini API (`google-generativeai`, free tier) | Gemini via Vertex AI (`google-cloud-aiplatform`) |
| Embeddings | Gemini API `text-embedding-004` (free tier) | Vertex AI `text-embedding-004` |
| Auth | `GEMINI_API_KEY` in `.env` | Application Default Credentials (ADC) |
| RAG / retrieval | numpy cosine similarity (MVP); Vertex AI Vector Search (stretch) | ← same |
| Orchestration | LangGraph (Gemini as LLM node) — stretch goal | ← same |
| Evaluation | RAGAS (faithfulness, answer relevance, context precision/recall) | ← same |
| Language | Python | ← same |
| Presentation | Jupyter Notebook | ← same |

The notebook backend is selected by a single variable (`BACKEND = "gemini_api"` or `"vertex_ai"`).
All development and the primary demo run use the free Gemini API path.

---

## Project Description

**A RAG-enhanced Gemini chatbot that answers questions about Vertex AI**, built and
demonstrated in a Jupyter Notebook.

The corpus is the GCP / Vertex AI official documentation — making the demo
self-referential and directly relevant to the FDE role: the assistant knows what
it was built on, and you can ask it anything about the platform.

**Deliverables:**

1. **Jupyter Notebook** — the primary deliverable. Walks through every pipeline stage
   with narrative markdown, inline outputs, and an interactive Q&A demo at the end.
2. **Working RAG pipeline** — GCP docs chunked and embedded via Vertex AI, cosine
   retrieval, generation via Gemini with retrieved context
3. **RAGAS evaluation** — test set of 10–20 Q&A pairs, evaluation run with scores
   displayed inline in the notebook, brief analysis
4. **One-page writeup** — framed as a customer presentation (see `writeup/` directory)
5. **Agentic wrapper (stretch)** — LangGraph agent that decides whether to retrieve
   from the corpus or answer from model knowledge

---

## Notebook Structure (Demo Experience)

The notebook is the demo artifact. A reviewer should be able to run it top-to-bottom
and experience a working RAG chatbot. Each section has a markdown header, brief
explanation, and visible output.

| Section | Content |
|---|---|
| 1. Setup & Auth | `vertexai.init(...)`, confirm credentials, print model info |
| 2. Corpus Ingestion | Load GCP docs, chunk, show sample chunks |
| 3. Embedding | Embed chunks via Vertex AI, show vector shape and sample |
| 4. Retrieval | Query → embed → cosine search → display top-k chunks |
| 5. Generation | Pass retrieved context to Gemini, display answer |
| 6. Interactive Q&A | Input cell: type any question about Vertex AI, see full RAG trace |
| 7. RAGAS Evaluation | Run evaluation, display scores as a table or bar chart |
| 8. Analysis | Markdown discussion: what worked, what didn't, enterprise considerations |

Section 6 is the demo centerpiece — a cell where you enter a question and see:
- The query
- The retrieved chunks (with source URLs)
- The generated answer

---

## Implementation Plan

### Day 1 — Environment Setup & Gemini API Quickstart
- [ ] Create GCP project (or use existing)
- [ ] Obtain Gemini API key (free tier) — for development path
- [ ] Set up Python venv, install `google-generativeai`, configure `.env`
- [ ] Run Gemini API quickstart: generate text with `gemini-2.0-flash`
- [ ] Run embedding quickstart: embed a string with `text-embedding-004`, confirm output shape
- [ ] Confirm free-tier auth and SDK are working end-to-end

### Day 2 — Corpus & RAG Pipeline
- [ ] Fetch and prepare GCP / Vertex AI documentation corpus (HTML → clean text)
- [ ] Chunk documents (fixed-size with overlap)
- [ ] Embed chunks using Vertex AI text-embedding model
- [ ] Build retrieval: embed query, cosine similarity search, return top-k chunks
- [ ] Build generation: format retrieved chunks as context, call Gemini, return answer
- [ ] Wire into a working Q&A loop in the notebook

### Day 3 — Evaluation, Polish & Write-up
- [ ] Build test set (10–20 Q&A pairs with ground truth answers about Vertex AI)
- [ ] Run RAGAS evaluation: faithfulness, answer relevance, context precision, context recall
- [ ] Display RAGAS scores inline in the notebook (table or chart)
- [ ] Analyze results — where does the pipeline fall short?
- [ ] Polish notebook narrative: section headers, markdown explanations, clean outputs
- [ ] Write one-page customer-facing summary (`writeup/customer_summary.md`)

### Day 4 — Vertex AI Verification
- [ ] Enable Vertex AI API on GCP project
- [ ] Configure Application Default Credentials (`gcloud auth application-default login`)
- [ ] Switch notebook to `BACKEND = "vertex_ai"`, run top-to-bottom
- [ ] Confirm Gemini generation and embeddings work via Vertex AI SDK
- [ ] Note any behavioral differences vs. the Gemini API path
- [ ] Update resume: add `GCP (Vertex AI · Gemini · Vertex AI Search)` to competencies

### Day 5 (Stretch) — Agentic Wrapper
- [ ] Add LangGraph agent node: route query to RAG retrieval vs. direct Gemini answer
- [ ] Integrate into notebook as an optional Section 9
- [ ] Compare agentic vs. naive RAG on a few test queries

---

## Key Concepts to Internalize

Before or during Day 1, ensure these are solid:

- **Vertex AI vs Bedrock mental model** — Vertex AI Studio ≈ Bedrock Playground;
  Vertex AI Search ≈ Bedrock Knowledge Bases; Vertex AI Agent Builder ≈ Bedrock Agents
- **GCP auth model** — ADC, service accounts, how they differ from AWS IAM roles
- **Gemini API** — multimodal, context window, tool use / function calling syntax
- **RAGAS metrics** — what faithfulness vs. answer relevance actually measure
- **LangGraph basics** — nodes, edges, state, cycles (builds on LangChain graph model)

---

## GCP Add-back to Resume

Once Day 4 is complete and the Vertex AI verification run succeeds:
- Add `GCP (Vertex AI · Gemini · Vertex AI Search)` to the `resume-2026-fde4.html`
  competencies section under Infrastructure or AI/ML
- Do not add GCP back until the Vertex AI backend is confirmed working end-to-end

---

## Files in This Project

- `PROJECT_PLAN.md` — this file
- `CLAUDE.md` — Claude Code context for this project
- `TECH_STACK.md` — detailed technology choices and rationale
- `notebooks/` — Jupyter notebooks (primary deliverable; `src/` for any shared helper modules)
- `corpus/` — fetched and processed GCP documentation
- `eval/` — RAGAS test set and evaluation results
- `writeup/` — customer-facing one-page summary

---

## Related Files (job search context)

- `~/Documents/job-search-2026/resume-2026-fde4.html` — FDE IV resume (GCP removed pending this work)
- `~/Documents/job-search-2026/study-curriculum-prompt-engineer.md` — full curriculum including FDE IV section
- `~/Documents/job-search-2026/job-tracker-ai.md` — job tracker (Google Cloud FDE entry)
