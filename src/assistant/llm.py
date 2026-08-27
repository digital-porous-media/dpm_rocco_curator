"""
LLM and embeddings for the General Assistant, backed by RoccoClient.

Configuration:
    RoccoClient (src/llm/client.py) is the canonical provider-agnostic LLM client
    that implements both traditional LLM patterns and LangChain's BaseChatModel interface.
    All configuration (provider, API key, model, etc.) is inherited from .env variables.

    Embedding model is auto-selected from LLM_PROVIDER via src.llm.embeddings.get_embeddings().
    Override with EMBEDDING_URL / EMBEDDING_MODEL / EMBEDDING_API_KEY if needed.
"""

from dotenv import load_dotenv
from src.llm.client import RoccoClient
from src.llm.embeddings import get_embeddings

load_dotenv()

_chat_model = None
_embeddings = None


def get_chat_model() -> RoccoClient:
    global _chat_model
    if _chat_model is None:
        _chat_model = RoccoClient(temperature=0.3)
    return _chat_model


def get_embeddings_model():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


def strip_code_fences(text: str | None) -> str:
    """Strip a leading markdown code fence (and its ``json`` language tag) from an LLM
    response, so the body can be handed to ``json.loads``.

    Every prompt in this module's callers asks for "JSON only, no markdown fences" and
    this model intermittently emits them anyway. Only a response that *starts* with a
    fence is unwrapped — a fence appearing mid-response is left alone, since that is
    prose the caller's own parse-failure fallback should handle rather than something to
    guess at.

    Note ``RoccoClient.send_prompt`` already applies an equivalent strip to its own
    return value, so calling this on a ``send_prompt`` result is a no-op kept for
    defence in depth; results from ``llm.invoke(...).content`` genuinely need it.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json"):]
    return cleaned.strip()


class _LazyEmbeddings:
    """Proxy that initializes the embeddings model on first use."""

    def __getattr__(self, name):
        return getattr(get_embeddings_model(), name)


embeddings = _LazyEmbeddings()
