# src/rocco/retriever/retriever.py

class DocumentRetriever:
    """Retrieves top-K relevant chunks from embeddings / vector DB"""

    def __init__(self, vector_db):
        self.vector_db = vector_db

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return top-k most relevant chunks"""
        # TODO: similarity search + reranking
        return ["relevant chunk 1", "chunk 2"]
