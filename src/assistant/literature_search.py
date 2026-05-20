"""
Semantic Scholar API wrapper for live literature search.

Used as fallback when publication FAISS confidence < 0.75.
If SEMANTIC_SCHOLAR_API_KEY is set in the environment, requests are authenticated;
otherwise unauthenticated requests are used (shared rate limit, fine for dev).
"""

import os
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,abstract,year,externalIds,citationCount,openAccessPdf,url"


@dataclass
class LiteratureResult:
    title: str
    authors: list[str]
    abstract: str | None
    year: int | None
    doi: str | None
    citation_count: int | None
    pdf_url: str | None
    url: str
    source: str = "[semantic scholar]"


class LiteratureSearch:
    def __init__(self):
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self._headers = {"x-api-key": api_key} if api_key else {}

    def search(self, query: str, limit: int = 5) -> list[LiteratureResult]:
        """Search Semantic Scholar for papers matching query."""
        try:
            resp = requests.get(
                f"{_BASE_URL}/paper/search",
                params={"query": query, "limit": limit, "fields": _FIELDS},
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Semantic Scholar search failed: %s", e)
            return []

        return [self._parse(p) for p in resp.json().get("data", [])]

    def recommendations(self, paper_id: str, limit: int = 5) -> list[LiteratureResult]:
        """Get recommended papers similar to a given Semantic Scholar paper ID."""
        try:
            resp = requests.get(
                f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}",
                params={"limit": limit, "fields": _FIELDS},
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Semantic Scholar recommendations failed: %s", e)
            return []

        return [self._parse(p) for p in resp.json().get("recommendedPapers", [])]

    @staticmethod
    def _parse(paper: dict) -> LiteratureResult:
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        oa = paper.get("openAccessPdf") or {}
        return LiteratureResult(
            title=paper.get("title", ""),
            authors=authors,
            abstract=paper.get("abstract"),
            year=paper.get("year"),
            doi=(paper.get("externalIds") or {}).get("DOI"),
            citation_count=paper.get("citationCount"),
            pdf_url=oa.get("url"),
            url=paper.get("url", ""),
        )
