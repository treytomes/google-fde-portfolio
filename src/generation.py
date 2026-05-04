"""
RAG-augmented generation using Gemini with Ollama fallback.

Usage:
    from generation import generate

    answer = generate(query, context_chunks, client)
    print(answer["text"])
    print(answer["sources"])

Fallback chain (free-tier quota is 20 req/day shared across models):
  1. gemini-2.5-flash      (preferred)
  2. gemini-2.5-flash-lite (Gemini fallback on daily quota exhaustion)
  3. ollama/gemma4:e2b     (local fallback when all Gemini quota is exhausted)
"""

import time

GENERATION_MODEL          = "gemini-2.5-flash"
GENERATION_MODEL_FALLBACK = "gemini-2.5-flash-lite"
OLLAMA_MODEL              = "gemma4:e2b"
OLLAMA_BASE_URL           = "http://localhost:11434"

_SYSTEM_PROMPT = """\
You are an assistant that answers questions about Google Cloud and Vertex AI.
Answer ONLY using the provided context. If the context doesn't contain enough
information to answer the question, say so explicitly rather than speculating.\
"""

_MAX_RETRIES = 4
_BASE_DELAY  = 5


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] (Source: {chunk['source_url']})\n{chunk['text']}")
    return "\n\n".join(parts)


def _generate_ollama(prompt: str) -> str:
    import ollama
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192},
    )
    return response.message.content


def generate(query: str, context_chunks: list[dict], client,
             model: str = GENERATION_MODEL) -> dict:
    context = _format_context(context_chunks)
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    sources = list(dict.fromkeys(c["source_url"] for c in context_chunks))

    # Ollama path — used when client is None (all Gemini quota exhausted)
    if client is None:
        print(f"Using local Ollama ({OLLAMA_MODEL})...")
        return {"text": _generate_ollama(prompt), "sources": sources}

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return {"text": response.text, "sources": sources}
        except Exception as e:
            err = str(e)
            # Cascade: flash → flash-lite → ollama
            if "429" in err and "GenerateRequestsPerDayPerProjectPerModel" in err:
                if model == GENERATION_MODEL:
                    print(f"Daily quota exhausted for {model} — falling back to {GENERATION_MODEL_FALLBACK}")
                    return generate(query, context_chunks, client, model=GENERATION_MODEL_FALLBACK)
                print(f"Daily quota exhausted for {model} — falling back to local Ollama ({OLLAMA_MODEL})")
                return generate(query, context_chunks, client=None, model=model)
            if "503" not in err and "UNAVAILABLE" not in err:
                raise
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = _BASE_DELAY * (2 ** attempt)
            print(f"503 UNAVAILABLE — retrying in {delay}s (attempt {attempt + 1}/{_MAX_RETRIES})...")
            time.sleep(delay)
