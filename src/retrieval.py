"""
Cosine similarity retrieval over pre-embedded corpus chunks.

Usage:
    from retrieval import retrieve

    results = retrieve(query, client, embeddings, chunks, k=5)
    for r in results:
        print(r["score"], r["doc_title"], r["text"][:120])
"""

import numpy as np
from google.genai import types

EMBEDDING_MODEL = "gemini-embedding-001"


def _embed_query(client, query: str) -> np.ndarray:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)


def retrieve(query: str, client, embeddings: np.ndarray,
             chunks: list[dict], k: int = 5) -> list[dict]:
    query_vec = _embed_query(client, query)
    # Embeddings are unit vectors (confirmed), so dot product == cosine similarity
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:k]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in top_indices
    ]
