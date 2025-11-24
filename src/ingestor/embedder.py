# src/rocco/ingestor/embedder.py

class Embedder:
    """Generates embeddings for chunks"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        # TODO: call embeddings API
        return [[0.0]*768 for _ in texts]
