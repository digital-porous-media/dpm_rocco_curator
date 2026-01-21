from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class BaseEmbedder(ABC):
    """Base class for all embedders"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass
    
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Get the underlying LangChain Embeddings object"""
        pass


class DocumentEmbedder(BaseEmbedder):
    """HuggingFace embeddings implementation"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {'device': 'cpu'}
        self.encode_kwargs = encode_kwargs or {'normalize_embeddings': True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
    
    def get_embeddings(self) -> Embeddings:
        return self.embeddings
    