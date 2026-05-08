from typing import List, Optional, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseIngestor


class DocumentIngestor(BaseIngestor):
    """Ingests documents (PDF, DOCX) for RAG pipeline"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        """
        Args:
            chunk_size: Maximum character length per chunk. Must be >= 100. Defaults to 500.
            chunk_overlap: Character overlap between consecutive chunks. Must be < ``chunk_size``. Defaults to 100.
            separators: Ordered list of separators tried by ``RecursiveCharacterTextSplitter``.
                Defaults to ``["\\n\\n", "\\n", ".", " ", ""]``.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        if chunk_size < 100:
            raise ValueError(f"chunk_size ({chunk_size}) should be at least 100")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def _load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on file extension"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = path.suffix.lower()

        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        return loader.load()

    def ingest(self, file_paths: Union[str, List[str]]) -> List[Document]:
        """
        Load and chunk document(s).

        Args:
            file_paths: Single file path (str) or list of file paths (List[str])

        Returns:
            List of chunked Document objects with enriched metadata
        """
        # Convert single string to list for uniform processing
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        all_chunks = []

        for file_path in file_paths:
            try:
                documents = self._load_document(file_path)
                print("Loaded docs:", len(documents))
                chunks = self.splitter.split_documents(documents)
                print("Chunks:", len(chunks))

                # Enrich chunk metadata with doc_title and chunk_index
                doc_title = Path(file_path).stem  # filename without extension
                for chunk_idx, chunk in enumerate(chunks):
                    chunk.metadata["doc_title"] = doc_title
                    chunk.metadata["chunk_index"] = chunk_idx

                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Failed to ingest {file_path}: {e}")

        return all_chunks
