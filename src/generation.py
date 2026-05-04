"""
RAG-augmented generation using Gemini.

Usage:
    from generation import generate

    answer = generate(query, context_chunks, client)
    print(answer["text"])
    print(answer["sources"])
"""

import time

# gemini-2.5-flash is preferred; fall back to flash-lite if daily quota (20 req/day
# on free tier) is exhausted. Pass model= to generate() to override.
GENERATION_MODEL = "gemini-2.5-flash"
GENERATION_MODEL_FALLBACK = "gemini-2.5-flash-lite"

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


def generate(query: str, context_chunks: list[dict], client,
             model: str = GENERATION_MODEL) -> dict:
    context = _format_context(context_chunks)
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
            )
            sources = list(dict.fromkeys(c["source_url"] for c in context_chunks))
            return {"text": response.text, "sources": sources}
        except Exception as e:
            err = str(e)
            # Auto-fallback on daily quota exhaustion
            if "429" in err and "GenerateRequestsPerDayPerProjectPerModel" in err:
                if model != GENERATION_MODEL_FALLBACK:
                    print(f"Daily quota exhausted for {model} — falling back to {GENERATION_MODEL_FALLBACK}")
                    return generate(query, context_chunks, client, model=GENERATION_MODEL_FALLBACK)
                raise
            if "503" not in err and "UNAVAILABLE" not in err:
                raise
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = _BASE_DELAY * (2 ** attempt)
            print(f"503 UNAVAILABLE — retrying in {delay}s (attempt {attempt + 1}/{_MAX_RETRIES})...")
            time.sleep(delay)
