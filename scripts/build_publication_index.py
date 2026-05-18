"""
Build the publication FAISS index from curated PDFs.

Chunks all PDFs in data/curated_papers/ using the existing DocumentIngestor
and writes the index to data/publication_vector_store/.

Usage:
    python scripts/build_publication_index.py

Environment variables required: LLM_API_KEY (for embeddings)
"""

# TODO (Intern A, Week 3): implement
#   1. Glob data/curated_papers/*.pdf
#   2. Use src/ingestor/document_ingestor.py to chunk + embed
#   3. Tag each chunk with dataset_id if filename matches a known dataset
#   4. Persist FAISS index to data/publication_vector_store/
