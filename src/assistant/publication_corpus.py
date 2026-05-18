"""
FAISS index over chunked full-text of curated paper PDFs.

Papers live in data/curated_papers/ (gitignored).
Index is built by scripts/build_publication_index.py and stored in
data/publication_vector_store/ (gitignored).

Each chunk is tagged with dataset_id where known, enabling cross-references
between a search result and its associated publication.
"""

# TODO (Intern A, Week 3): implement PublicationCorpus class
#   - __init__: load FAISS index from data/publication_vector_store/
#   - search(query, top_k=5): return chunks with metadata (doc_title, page, dataset_id, snippet)
#   - Source label on all results: "[paper match]"
