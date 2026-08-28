from __future__ import annotations
"""
Semantic Scholar API wrapper for live literature search.

Called directly by the search_literature tool, and internally as a fallback by
get_educational_context/get_workflow_guidance when no portal tutorial matches.
If SEMANTIC_SCHOLAR_API_KEY is set in the environment, requests are authenticated;
otherwise unauthenticated requests are used (shared rate limit, fine for dev).

Rate limiting: 1 request/second enforced per instance via a sleep-based throttle.
This matches the Semantic Scholar authenticated rate limit (1 RPS per API key).
Note: if the server moves to multi-process deployment (e.g. Gunicorn workers),
replace this with a Redis-backed distributed rate limiter (e.g. the `limits` library).
"""

import os
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,abstract,year,externalIds,citationCount,openAccessPdf,url"


@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    citation_count: Optional[int]
    pdf_url: Optional[str]
    url: str
    source: str = "[semantic scholar]"


class LiteratureSearch:
    def __init__(self):
        """Initialize LiteratureSearch with optional Semantic Scholar API key and rate limiter."""
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self._headers = {"x-api-key": api_key} if api_key else {}
        self._min_interval = 1.0
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def _throttle(self):
        """Enforce a minimum interval between API calls to respect the 1 RPS rate limit."""
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()

    def _get_with_retry(self, url: str, params: dict, max_retries: int = 3) -> requests.Response | None:
        """GET with exponential backoff on 429."""
        for attempt in range(max_retries):
            self._throttle()
            try:
                resp = requests.get(url, params=params, headers=self._headers, timeout=10)
                if resp.status_code == 429:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    logger.warning("Semantic Scholar rate limited (429); retrying in %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                logger.warning("Semantic Scholar request failed: %s", e)
                return None
        logger.warning("Semantic Scholar: gave up after %d retries (429)", max_retries)
        return None

    def search_external_literature(self, query: str, max_results: int = 5) -> list[Paper]:
        """Search Semantic Scholar for papers matching query."""
        resp = self._get_with_retry(
            f"{_BASE_URL}/paper/search",
            params={"query": query, "limit": max_results, "fields": _FIELDS},
        )
        if resp is None:
            return []
        return [self._parse(p) for p in resp.json().get("data", [])]

    def recommendations(self, paper_id: str, limit: int = 5) -> list[Paper]:
        """Get recommended papers similar to a given Semantic Scholar paper ID."""
        resp = self._get_with_retry(
            f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}",
            params={"limit": limit, "fields": _FIELDS},
        )
        if resp is None:
            return []

        return [self._parse(p) for p in resp.json().get("recommendedPapers", [])]


    @staticmethod
    def _parse(paper: dict) -> Paper:
        """Parse a raw Semantic Scholar API paper dict into a Paper dataclass."""
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        oa = paper.get("openAccessPdf") or {}
        return Paper(
            title=paper.get("title", ""),
            authors=authors,
            abstract=paper.get("abstract"),
            year=paper.get("year"),
            doi=(paper.get("externalIds") or {}).get("DOI"),
            citation_count=paper.get("citationCount"),
            pdf_url=oa.get("url"),
            url=paper.get("url", ""),
        )
