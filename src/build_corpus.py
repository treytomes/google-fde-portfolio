"""
Fetch and clean GCP/Vertex AI documentation pages into corpus/.

Each page is saved as a .txt file alongside a manifest entry in
corpus/manifest.json that records the source URL and title.

Usage:
    python3 src/build_corpus.py
"""

import json
import re
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

# Representative Vertex AI / Gemini docs pages.
# Covers the topics the RAG assistant should know: platform overview, models,
# embeddings, RAG, grounding, auth, and enterprise considerations.
PAGES = [
    # Platform overview
    ("vertex_ai_overview",          "https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform"),
    ("gemini_enterprise_platform",  "https://cloud.google.com/products/gemini-enterprise-agent-platform"),

    # Gemini models
    ("gemini_models",               "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models"),
    ("gemini_api_overview",         "https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal"),
    ("gemini_multimodal",           "https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-chat-prompts-gemini"),

    # Embeddings
    ("embeddings_overview",         "https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings"),
    ("embeddings_api_reference",    "https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api"),

    # RAG Engine (rebranded from Vertex AI RAG)
    ("rag_overview",                "https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview"),
    ("rag_quickstart",              "https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-quickstart"),

    # Grounding & context
    ("grounding_overview",          "https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/ground-with-google-search"),
    ("context_cache",               "https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview"),

    # Agent Builder / search
    ("agent_builder_overview",      "https://cloud.google.com/generative-ai-app-builder/docs/parse-chunk-documents"),
    ("vertex_ai_search",            "https://cloud.google.com/generative-ai-app-builder/docs/create-datastore-ingest"),
    ("agent_builder_intro",         "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/introduction"),

    # Vector Search
    ("vector_search_overview",      "https://cloud.google.com/vertex-ai/docs/vector-search/overview"),

    # Auth & security
    ("gcp_auth_adc",                "https://cloud.google.com/docs/authentication/application-default-credentials"),
    ("vertex_ai_access_control",    "https://cloud.google.com/vertex-ai/docs/general/access-control"),

    # Responsible AI
    ("responsible_ai",              "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/responsible-ai"),

    # Pricing
    ("gemini_pricing",              "https://cloud.google.com/vertex-ai/generative-ai/pricing"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-rag-bot/1.0)"}
DELAY_SECONDS = 1.5    # polite crawl delay
MAX_CHARS     = 50_000  # default per-page cap

# Per-slug overrides — lower cap for pages that are large but low-signal for Q&A.
# gemini_pricing is 50K chars of pricing tables; 2-3 chunks is enough to answer
# "how much does X cost?" without letting it dominate retrieval for every other topic.
MAX_CHARS_OVERRIDE = {
    "gemini_pricing": 10_000,
}

# Site-wide boilerplate patterns to strip after text extraction.
# These appear on nearly every GCP docs page and add noise without signal.
_BOILERPLATE_PATTERNS = [
    r"Vertex AI is transitioning to become part of Gemini Enterprise Agent Platform\."
    r"\s*See the most up-to-date information in the Agent Platform documentation\s*\.?",
    r"Note:\s*Vertex AI Search is being renamed to Agent Search\."
    r"\s*We are in the process of updating content to reflect the new branding\.",
    r"Stay organized with collections\s+Save and categorize content based on your preferences\.",
    r"Send feedback",
]
_BOILERPLATE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _BOILERPLATE_PATTERNS),
    re.IGNORECASE,
)


def fetch(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
    resp.raise_for_status()
    return resp.text


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements before extraction
    for tag in soup.select("nav, header, footer, script, style, .devsite-nav, "
                           ".devsite-header, .devsite-footer, .devsite-feedback, "
                           "[aria-hidden=true]"):
        tag.decompose()

    content = soup.select_one("article") or soup.select_one("main") or soup.body
    if not content:
        return ""

    text = content.get_text(separator=" ", strip=True)

    # Collapse whitespace and normalize unicode
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Strip site-wide boilerplate that appears on nearly every GCP docs page
    text = _BOILERPLATE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slug_to_filename(slug: str) -> str:
    return f"{slug}.txt"


def main():
    CORPUS_DIR.mkdir(exist_ok=True)
    manifest = []
    skipped = []

    for slug, url in PAGES:
        filename = slug_to_filename(slug)
        filepath = CORPUS_DIR / filename

        print(f"Fetching: {slug} ...")
        try:
            html = fetch(url)
            text = extract_text(html)

            if len(text) < 200:
                print(f"  WARNING: very short content ({len(text)} chars) — skipping")
                skipped.append({"slug": slug, "url": url, "reason": "too short"})
                continue

            cap = MAX_CHARS_OVERRIDE.get(slug, MAX_CHARS)
            if len(text) > cap:
                text = text[:cap]
                print(f"  TRUNCATED to {cap:,} chars")
            filepath.write_text(text, encoding="utf-8")
            manifest.append({"slug": slug, "url": url, "filename": filename, "chars": len(text)})
            print(f"  OK  {len(text):,} chars → {filename}")

        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append({"slug": slug, "url": url, "reason": str(e)})

        time.sleep(DELAY_SECONDS)

    # Write manifest
    manifest_path = CORPUS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps({"pages": manifest, "skipped": skipped}, indent=2), encoding="utf-8")

    print(f"\nDone. {len(manifest)} pages saved, {len(skipped)} skipped.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
