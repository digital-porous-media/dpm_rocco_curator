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

# Unified LLM client: RoccoClient (implements BaseChatModel for LangChain/LangGraph)
# Temperature 0.3 for structured outputs (intent classifier); can be overridden per-call
chat_model = RoccoClient(temperature=0.3)

# Embeddings model (auto-selected from LLM_PROVIDER)
embeddings = get_embeddings()
