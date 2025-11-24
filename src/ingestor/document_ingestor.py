class DocumentIngestor:
    """Extracts and processes text from documents"""

    def extract_text(self, file_path: str) -> str:
        # TODO: PDF / text extraction
        return "Full text of document"

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
