"""
Embed corpus chunks using the Gemini embedding API and persist to disk.

Usage:
    from embedder import embed_chunks, load_embeddings

    # Embed and save (slow — calls the API)
    embedder = Embedder(client)
    embedder.embed_and_save(chunks, corpus_dir)

    # Load cached results (fast)
    embeddings, chunks = load_embeddings(corpus_dir)
"""

import json
import time
from pathlib import Path

import numpy as np
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 10  # free-tier embedding quota is tight; small batches avoid 429s
BATCH_DELAY = 2  # seconds between batches


def _embed_batch_with_retry(client, batch: list[str], task_type: str,
                             retries: int = 5, base_delay: float = 10) -> list:
    for attempt in range(retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in result.embeddings]
        except Exception as e:
            if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                raise
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  429 rate limit — retrying in {delay}s (attempt {attempt + 1}/{retries})...")
            time.sleep(delay)
    return []  # unreachable


class Embedder:
    def __init__(self, client):
        self.client = client

    def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        all_vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  batch {batch_num}/{total_batches} ({len(batch)} texts)...", end=" ", flush=True)
            vectors = _embed_batch_with_retry(self.client, batch, task_type)
            all_vectors.extend(vectors)
            print("ok")
            if i + BATCH_SIZE < len(texts):
                time.sleep(BATCH_DELAY)
        return np.array(all_vectors, dtype=np.float32)

    def embed_and_save(self, chunks: list[dict], corpus_dir: Path) -> np.ndarray:
        texts = [c["text"] for c in chunks]
        print(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")
        embeddings = self.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

        np.save(corpus_dir / "embeddings.npy", embeddings)
        (corpus_dir / "chunks.json").write_text(
            json.dumps(chunks, indent=2), encoding="utf-8"
        )
        print(f"Saved embeddings.npy  shape={embeddings.shape}")
        print(f"Saved chunks.json     {len(chunks)} entries")
        return embeddings


def load_embeddings(corpus_dir: Path) -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(corpus_dir / "embeddings.npy")
    chunks = json.loads((corpus_dir / "chunks.json").read_text(encoding="utf-8"))
    return embeddings, chunks
