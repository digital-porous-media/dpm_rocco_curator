"""
Semantic Scholar API wrapper for live literature search.

Used as fallback when publication FAISS confidence < 0.75.
Requires SEMANTIC_SCHOLAR_API_KEY in environment (one shared server key).
"""

# TODO (Intern B, Week 3): implement LiteratureSearch class
#   - search(query, limit=5): call Semantic Scholar API, return title/abstract/url
#   - Source label on all results: "[semantic scholar]"
#   - Routing logic lives in tools.py; this class is pure API wrapper
