"""
Cost estimation for RAG pipeline queries at Vertex AI rates.

Active on all backends (ollama, gemini_api, vertex_ai). On non-Vertex paths
the output is labeled "estimated at Vertex AI rates" so development costs
are visible and citable in the FDE pitch.

Pricing source: https://cloud.google.com/vertex-ai/generative-ai/docs/pricing
(as of May 2026 — update constants here if pricing changes)

Verified from corpus/gemini_pricing.txt (scraped from the pricing page):
  gemini-2.5-flash standard:     $0.30 input / $2.50 output per 1M tokens
  gemini-2.5-flash-lite standard: $0.10 input / $0.40 output per 1M tokens

NOTE: gemini-embedding-001 pricing is not publicly listed on the pricing page
in a parseable form. The $0.025/1M tokens figure below is sourced from the
Google AI Studio pricing page (https://ai.google.dev/pricing) which lists
text-embedding-004 at $0.025/1M tokens — used here as a reasonable proxy.
Verify against your actual GCP billing for production use.
"""

# ── Pricing constants ──────────────────────────────────────────────────────────
# gemini-2.5-flash  (≤200K token prompts, standard on-demand)
# Source: corpus/gemini_pricing.txt (Vertex AI pricing page, May 2026)
_FLASH_INPUT_PER_M   = 0.30   # USD per 1M input tokens
_FLASH_OUTPUT_PER_M  = 2.50   # USD per 1M output tokens

# gemini-2.5-flash-lite (standard on-demand)
# Source: corpus/gemini_pricing.txt
_FLASH_LITE_INPUT_PER_M  = 0.10
_FLASH_LITE_OUTPUT_PER_M = 0.40

# gemini-embedding-001 — UNVERIFIED ESTIMATE
# The Vertex AI pricing page does not list embedding model prices in a retrievable
# section (the page exceeds our corpus cap before reaching embeddings).
# $0.025/1M tokens is taken from AI Studio pricing for text-embedding-004 as a proxy.
# DO NOT cite this number in production without checking your actual GCP billing.
_EMBEDDING_PER_M_TOKENS = 0.025  # USD per 1M tokens — PROXY, not verified for Vertex AI
_CHARS_PER_TOKEN = 4              # rough approximation for cost estimation
_EMBEDDING_PER_M_CHARS = _EMBEDDING_PER_M_TOKENS / _CHARS_PER_TOKEN

_MODEL_PRICING = {
    "gemini-2.5-flash":      (_FLASH_INPUT_PER_M,      _FLASH_OUTPUT_PER_M),
    "gemini-2.5-flash-lite": (_FLASH_LITE_INPUT_PER_M, _FLASH_LITE_OUTPUT_PER_M),
}
_DEFAULT_PRICING = (_FLASH_INPUT_PER_M, _FLASH_OUTPUT_PER_M)


def _gen_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    input_rate, output_rate = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _emb_cost(char_count: int) -> float:
    return char_count * _EMBEDDING_PER_M_CHARS / 1_000_000


class CostTracker:
    def __init__(self, backend: str = "gemini_api"):
        self._backend   = backend
        self._session   = 0.0
        self._exchange  = 0.0
        self._gen_input = 0
        self._gen_output = 0
        self._gen_model  = ""
        self._emb_chars  = 0

    def reset_exchange(self) -> None:
        self._exchange   = 0.0
        self._gen_input  = 0
        self._gen_output = 0
        self._gen_model  = ""
        self._emb_chars  = 0

    def record_generation(self, input_tokens: int, output_tokens: int, model: str) -> None:
        cost = _gen_cost(input_tokens, output_tokens, model)
        self._gen_input  = input_tokens
        self._gen_output = output_tokens
        self._gen_model  = model
        self._exchange  += cost
        self._session   += cost

    def record_embedding(self, char_count: int) -> None:
        cost = _emb_cost(char_count)
        self._emb_chars  = char_count
        self._exchange  += cost
        self._session   += cost

    @property
    def exchange_cost(self) -> float:
        return self._exchange

    @property
    def session_total(self) -> float:
        return self._session

    def exchange_summary(self) -> str:
        label = "Vertex AI" if self._backend == "vertex_ai" else "estimated at Vertex AI rates"
        lines = [f"  COST ESTIMATE  ({label})\n"]

        if self._emb_chars:
            emb = _emb_cost(self._emb_chars)
            lines.append(f"    Embedding (query)   : {self._emb_chars:>6,} chars   → ${emb:.6f}  (est.)")

        if self._gen_input or self._gen_output:
            in_cost  = self._gen_input  * _MODEL_PRICING.get(self._gen_model, _DEFAULT_PRICING)[0] / 1_000_000
            out_cost = self._gen_output * _MODEL_PRICING.get(self._gen_model, _DEFAULT_PRICING)[1] / 1_000_000
            lines.append(f"    Generation (input)  : {self._gen_input:>6,} tokens  → ${in_cost:.6f}")
            lines.append(f"    Generation (output) : {self._gen_output:>6,} tokens  → ${out_cost:.6f}")

        lines.append(f"    {'─' * 47}")
        lines.append(f"    This query          :               ~ ${self._exchange:.6f}")
        lines.append(f"    Session total       :               ~ ${self._session:.6f}")

        return "\n".join(lines)
