from typing import List, Optional, Dict, Any
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class DocumentEmbedder:
    """Embeddings wrapper for the curator RAG pipeline.

    Accepts a pre-built LangChain ``Embeddings`` object via the ``embeddings``
    parameter (injected by ``src.llm.embeddings.get_embeddings``). When omitted,
    falls back to local ``HuggingFaceEmbeddings`` for backward compatibility.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Args:
            model_name: HuggingFace model ID. Ignored when ``embeddings`` is provided.
            model_kwargs: Passed to ``HuggingFaceEmbeddings``. Ignored when ``embeddings`` is provided.
            encode_kwargs: Passed to the encode call. Ignored when ``embeddings`` is provided.
            embeddings: Pre-built LangChain Embeddings object. When set, the HuggingFace
                parameters above are ignored.
        """
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.model_name = model_name
            self.model_kwargs = model_kwargs or {'device': 'cpu'}
            self.encode_kwargs = encode_kwargs or {'normalize_embeddings': True}
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs,
                encode_kwargs=self.encode_kwargs,
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings and return their dense vectors."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string and return its dense vector."""
        return self.embeddings.embed_query(text)

    def get_embeddings(self) -> Embeddings:
        """Return the underlying LangChain ``Embeddings`` object (e.g., for use with FAISS)."""
        return self.embeddings
    