"""
Split corpus documents into overlapping word-based chunks.

Usage:
    from chunker import chunk_corpus
    chunks = chunk_corpus(corpus_dir, manifest)
"""

from pathlib import Path


_ACRONYMS = {"Ai": "AI", "Gcp": "GCP", "Api": "API", "Rag": "RAG", "Adc": "ADC"}

def _slug_to_title(slug: str) -> str:
    words = slug.replace("_", " ").title().split()
    return " ".join(_ACRONYMS.get(w, w) for w in words)


def chunk_document(text: str, source_url: str, doc_title: str,
                   chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < 30:  # discard near-empty tail fragments
            break
        chunks.append({
            "text": " ".join(chunk_words),
            "source_url": source_url,
            "chunk_index": len(chunks),
            "doc_title": doc_title,
        })
    return chunks


def chunk_corpus(corpus_dir: Path, manifest: dict,
                 chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    all_chunks = []
    for page in manifest["pages"]:
        filepath = corpus_dir / page["filename"]
        if not filepath.exists():
            continue
        text = filepath.read_text(encoding="utf-8")
        doc_title = _slug_to_title(page["slug"])
        chunks = chunk_document(
            text,
            source_url=page["url"],
            doc_title=doc_title,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks
