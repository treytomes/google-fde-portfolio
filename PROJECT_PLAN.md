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

| Layer | Technology |
|---|---|
| Model inference | Gemini via Vertex AI (Python SDK) |
| Embeddings | Vertex AI text-embedding model |
| RAG / retrieval | Vertex AI Search OR manual vector store |
| Orchestration | LangGraph (Gemini as LLM node) or Google ADK |
| Evaluation | RAGAS (faithfulness, answer relevance, context precision/recall) |
| Auth | GCP service account / Application Default Credentials (ADC) |
| Language | Python |

---

## Project Description

**A document Q&A assistant over a domain-specific corpus**, evaluated with RAGAS.

The domain doesn't matter — what matters is the architecture and the evaluation story.
Good candidate corpora:
- AWS vs GCP service mapping documentation (directly relevant to FDE context)
- Technical documentation from an open-source project
- Any publicly available domain-specific document set

**Deliverables:**

1. **Working RAG pipeline** — documents chunked and embedded via Vertex AI, retrieval
   over user queries, generation via Gemini
2. **Agentic wrapper (stretch)** — a LangGraph agent that decides whether to retrieve
   from documents or answer from model knowledge
3. **RAGAS evaluation** — test set of 10–20 Q&A pairs, evaluation run with scores,
   brief analysis of results
4. **One-page writeup** — framed as a customer presentation:
   - Business problem
   - Architecture diagram or description
   - Evaluation results
   - What I'd do differently at enterprise scale
   - AWS Bedrock comparison (what's the same, what's different, what surprised me)

---

## Implementation Plan

### Day 1 — GCP Setup & Vertex AI Quickstart
- [ ] Create GCP project (or use existing)
- [ ] Enable Vertex AI API
- [ ] Configure Application Default Credentials (`gcloud auth application-default login`)
- [ ] Run Vertex AI quickstart: generate text with Gemini via Python SDK
- [ ] Run embedding quickstart: embed a string, confirm output shape
- [ ] Confirm auth and SDK are working end-to-end

### Day 2 — RAG Pipeline
- [ ] Select and prepare corpus (chunk documents)
- [ ] Embed chunks using Vertex AI text-embedding model
- [ ] Store embeddings (in-memory numpy or simple JSON for MVP; Vertex AI Vector Search for stretch)
- [ ] Build retrieval function: embed query, cosine similarity search, return top-k chunks
- [ ] Build generation function: format retrieved chunks as context, call Gemini, return answer
- [ ] Wire together into a working Q&A loop

### Day 3 — Evaluation & Write-up
- [ ] Install RAGAS, build test set (10–20 Q&A pairs with ground truth answers)
- [ ] Run RAGAS evaluation: faithfulness, answer relevance, context precision, context recall
- [ ] Analyze results — where does the pipeline fall short?
- [ ] Write one-page customer-facing summary
- [ ] Optional: add LangGraph agentic wrapper (retrieve vs. direct answer decision)

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

Once Day 1 is complete and Vertex AI SDK is working:
- Add `GCP (Vertex AI · Gemini · Vertex AI Search)` to the `resume-2026-fde4.html`
  competencies section under Infrastructure or AI/ML
- Do not add GCP back until the quickstart is confirmed working

---

## Files in This Project

- `PROJECT_PLAN.md` — this file
- `CLAUDE.md` — Claude Code context for this project
- `src/` — Python implementation
- `corpus/` — document corpus for RAG
- `eval/` — RAGAS evaluation scripts and results
- `writeup/` — customer-facing one-page summary

---

## Related Files (job search context)

- `~/Documents/job-search-2026/resume-2026-fde4.html` — FDE IV resume (GCP removed pending this work)
- `~/Documents/job-search-2026/study-curriculum-prompt-engineer.md` — full curriculum including FDE IV section
- `~/Documents/job-search-2026/job-tracker-ai.md` — job tracker (Google Cloud FDE entry)
