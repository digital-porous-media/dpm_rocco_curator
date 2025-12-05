from typing import List, Optional, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from src.ingestor.embedder import DocumentEmbedder


class VectorStoreManager:
    """Manages vector store operations (create, save, load, query)"""
    
    def __init__(self, embedder: DocumentEmbedder):
        """
        Initialize vector store manager.
        
        Args:
            embedder: DocumentEmbedder instance to use for vectorization
        """
        self.embedder = embedder
        self.vector_store: Optional[VectorStore] = None
    
    def create_from_documents(self, documents: List[Document]) -> VectorStore:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Created vector store
        """
        self.vector_store = FAISS.from_documents(
            documents,
            self.embedder.get_embeddings()
        )
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_from_documents first.")
        
        self.vector_store.add_documents(documents)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store.save_local(str(path))
    
    def load(self, path: Union[str, Path]) -> VectorStore:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path containing the saved vector store
            
        Returns:
            Loaded vector store
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at: {path}")
        
        self.vector_store = FAISS.load_local(
            str(path),
            self.embedder.get_embeddings(),
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded from: {path}")
        return self.vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_vector_store(self) -> VectorStore:
        """Get the underlying vector store object"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store