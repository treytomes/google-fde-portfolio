"""
Cost estimation for RAG pipeline queries at Vertex AI rates.

Active on all backends (ollama, gemini_api, vertex_ai). On non-Vertex paths
the output is labeled "estimated at Vertex AI rates" so development costs
are visible and citable in the FDE pitch.

Pricing source: https://cloud.google.com/vertex-ai/generative-ai/docs/pricing
(as of May 2026 — update constants here if pricing changes)
"""

# ── Pricing constants ──────────────────────────────────────────────────────────
# gemini-2.5-flash  (≤200K token prompts)
_FLASH_INPUT_PER_M   = 0.15   # USD per 1M input tokens
_FLASH_OUTPUT_PER_M  = 0.60   # USD per 1M output tokens

# gemini-2.5-flash-lite
_FLASH_LITE_INPUT_PER_M  = 0.10
_FLASH_LITE_OUTPUT_PER_M = 0.40

# gemini-embedding-001
_EMBEDDING_PER_M_CHARS = 0.25  # USD per 1M characters ($0.00000025/char)

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
            lines.append(f"    Embedding (query)   : {self._emb_chars:>6,} chars   → ${emb:.6f}")

        if self._gen_input or self._gen_output:
            in_cost  = self._gen_input  * _MODEL_PRICING.get(self._gen_model, _DEFAULT_PRICING)[0] / 1_000_000
            out_cost = self._gen_output * _MODEL_PRICING.get(self._gen_model, _DEFAULT_PRICING)[1] / 1_000_000
            lines.append(f"    Generation (input)  : {self._gen_input:>6,} tokens  → ${in_cost:.6f}")
            lines.append(f"    Generation (output) : {self._gen_output:>6,} tokens  → ${out_cost:.6f}")

        lines.append(f"    {'─' * 47}")
        lines.append(f"    This query          :               ~ ${self._exchange:.6f}")
        lines.append(f"    Session total       :               ~ ${self._session:.6f}")

        return "\n".join(lines)
