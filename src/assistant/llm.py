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


class _LazyEmbeddings:
    """Proxy that initializes the embeddings model on first use."""

    def __getattr__(self, name):
        return getattr(get_embeddings_model(), name)


embeddings = _LazyEmbeddings()
