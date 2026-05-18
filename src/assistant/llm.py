"""
LangChain-compatible LLM and embeddings for the General Assistant.

Uses ChatOpenAI + OpenAIEmbeddings configured from .env — both work with any
OpenAI-compatible endpoint (SambaNova, OpenAI, Ollama, etc.).

If langchain_sambanova is installed, SambaStudioEmbeddings can replace
OpenAIEmbeddings for the embedding model. For now, OpenAIEmbeddings with a
custom base URL covers all supported providers.

Required .env variables:
    LLM_API_KEY     - API key (same as the rest of the project)
    LLM_BASE_URL    - Provider endpoint (e.g. https://ai.tejas.tacc.utexas.edu/v1)
    LLM_MODEL       - Chat model name
    EMBEDDING_URL   - Embeddings endpoint (may differ from LLM_BASE_URL)
    EMBEDDING_MODEL - Embedding model name
    EMBEDDING_API_KEY - API key for embeddings (may differ from LLM_API_KEY)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

chat_model = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL") or None,
    temperature=0.7,
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY")),
    base_url=os.getenv("EMBEDDING_URL", os.getenv("LLM_BASE_URL")) or None,
)
