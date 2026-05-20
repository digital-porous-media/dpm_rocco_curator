"""
LangChain-compatible LLM and embeddings for the General Assistant.

Uses ChatOpenAI + OpenAIEmbeddings configured from .env — both work with any
OpenAI-compatible endpoint (SambaNova, OpenAI, Ollama, etc.).

If langchain_sambanova is installed, SambaStudioEmbeddings can replace
OpenAIEmbeddings for the embedding model. For now, OpenAIEmbeddings with a
custom base URL covers all supported providers.

Required .env variables:
    LLM_API_KEY   - API key (same as the rest of the project)
    LLM_PROVIDER  - Provider alias (openai, sambanova, gemini, ollama, …)
    LLM_MODEL     - Chat model name

Embedding model is auto-selected from LLM_PROVIDER via src.llm.embeddings.get_embeddings().
Override with EMBEDDING_URL / EMBEDDING_MODEL / EMBEDDING_API_KEY if needed.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.llm.embeddings import get_embeddings

load_dotenv()

chat_model = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL") or None,
    temperature=0.7,
)

embeddings = get_embeddings()
