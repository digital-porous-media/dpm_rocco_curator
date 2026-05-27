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
        import logging
        logger = logging.getLogger(__name__)

        texts = [doc.page_content for doc in documents]
        logger.info(f"Embedding {len(documents)} documents...")

        # Embed in batches to avoid timeouts and handle failures gracefully
        batch_size = 32
        all_embeddings = []
        valid_docs = []
        valid_texts = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]

            try:
                batch_embeddings = self.embedder.embed_documents(batch_texts)
                # Verify we got embeddings for all texts in batch
                if len(batch_embeddings) != len(batch_texts):
                    logger.warning(
                        f"Batch {i//batch_size}: Expected {len(batch_texts)} embeddings, "
                        f"got {len(batch_embeddings)}. Skipping batch."
                    )
                    continue

                all_embeddings.extend(batch_embeddings)
                valid_texts.extend(batch_texts)
                valid_docs.extend(batch_docs)
            except Exception as e:
                logger.error(f"Failed to embed batch {i//batch_size}: {str(e)}")
                continue

        if not valid_docs:
            raise ValueError("No documents were successfully embedded")

        if len(valid_docs) < len(documents):
            logger.warning(
                f"Successfully embedded {len(valid_docs)}/{len(documents)} documents "
                f"({len(documents) - len(valid_docs)} failed or skipped)"
            )

        # Create FAISS store from pre-computed embeddings using from_embeddings
        text_embedding_pairs = list(zip(valid_texts, all_embeddings))
        self.vector_store = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embedder.embeddings,
            metadatas=[doc.metadata for doc in valid_docs],
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