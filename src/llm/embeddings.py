from __future__ import annotations
"""
Provider-aware embedding factory for Rocco.

Returns a LangChain Embeddings object matched to the active LLM provider.
Both the curator pipeline and the assistant module import from here.

Resolution priority (highest → lowest):
  1. Explicit keyword arguments to get_embeddings()
  2. EMBEDDING_MODEL / EMBEDDING_URL / EMBEDDING_API_KEY env vars
  3. Per-provider defaults derived from LLM_PROVIDER
  4. Hard-coded fallback (local BAAI/bge-large-en-v1.5)

Provider → embedding backend mapping:
  openai      → OpenAIEmbeddings  text-embedding-3-small
  sambanova   → OpenAIEmbeddings  E5-Mistral-7B-Instruct  (TACC endpoint)
  gemini      → OpenAIEmbeddings  gemini-embedding-2
  ollama      → HuggingFaceEmbeddings  BAAI/bge-large-en-v1.5  (local)
  huggingface → HuggingFaceEmbeddings  BAAI/bge-large-en-v1.5  (local)
  anthropic   → HuggingFaceEmbeddings  BAAI/bge-large-en-v1.5  (no Anthropic embedding API)
  deepseek    → HuggingFaceEmbeddings  BAAI/bge-large-en-v1.5  (no DeepSeek embedding API)
  other       → HuggingFaceEmbeddings  BAAI/bge-large-en-v1.5  (safe default)

Setting any EMBEDDING_* env var overrides the provider default and can switch a
"local" provider onto an API backend (e.g. EMBEDDING_URL forces API path for ollama).
"""

import os
from langchain_core.embeddings import Embeddings

_LOCAL_PROVIDERS = {"ollama", "huggingface", "anthropic", "deepseek"}

_PROVIDER_API_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "model": "text-embedding-3-small",
        "url": "https://api.openai.com/v1",
    },
    "sambanova": {
        "model": "E5-Mistral-7B-Instruct",
        "url": "https://ai.tejas.tacc.utexas.edu/v1",
    },
    "gemini": {
        "model": "gemini-embedding-2",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

_LOCAL_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> Embeddings:
    """Return a LangChain Embeddings object for the active provider.

    All parameters are optional; omit them to auto-detect from env vars.
    """
    effective_provider = (
        provider
        or os.getenv("LLM_PROVIDER", "").lower()
        or "openai"
    )

    explicit_model = model or os.getenv("EMBEDDING_MODEL")
    explicit_url = api_url or os.getenv("EMBEDDING_URL")
    explicit_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")

    # Use local HuggingFace when the provider has no embedding API,
    # unless the user explicitly sets EMBEDDING_URL or EMBEDDING_MODEL
    # (which signals they want an API path regardless of provider).
    use_local = (
        effective_provider in _LOCAL_PROVIDERS
        and not explicit_url
        and not explicit_model
    ) or (
        effective_provider not in _LOCAL_PROVIDERS
        and effective_provider not in _PROVIDER_API_DEFAULTS
        and not explicit_url
        and not explicit_model
    )

    if use_local:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=explicit_model or _LOCAL_DEFAULT_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # API path
    defaults = _PROVIDER_API_DEFAULTS.get(effective_provider, {})
    final_model = explicit_model or defaults.get("model", "text-embedding-3-small")
    final_url = explicit_url or defaults.get("url") or os.getenv("LLM_BASE_URL") or None
    final_key = explicit_key or "unused"

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=final_model,
        api_key=final_key,
        base_url=final_url,
        # Disable tiktoken pre-tokenization: non-OpenAI endpoints (e.g. LiteLLM/TACC)
        # expect raw strings, not token-id arrays.
        check_embedding_ctx_length=False,
    )
