"""
Unit tests for src/assistant/literature_search.py.

All tests are fully mocked — no API key or network access required.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.assistant.literature_search import LiteratureSearch, Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_paper_dict(**overrides) -> dict:
    base = {
        "title": "Pore-scale imaging of multiphase flow",
        "authors": [{"name": "Blunt MJ"}, {"name": "Bijeljic B"}],
        "abstract": "We image multiphase flow at the pore scale using micro-CT.",
        "year": 2013,
        "externalIds": {"DOI": "10.1039/c3cs60120j"},
        "citationCount": 512,
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        "url": "https://www.semanticscholar.org/paper/abc123",
    }
    base.update(overrides)
    return base


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.status_code = status_code
    if status_code >= 400:
        mock.raise_for_status.side_effect = __import__("requests").HTTPError(
            response=mock
        )
    else:
        mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# search_external_literature
# ---------------------------------------------------------------------------

class TestSearchExternalLiterature:
    def test_search_returns_results(self):
        """Valid API response returns a list of Paper with correct fields."""
        data = {"data": [_fake_paper_dict()]}
        with patch("src.assistant.literature_search.requests.get") as mock_get, \
             patch("src.assistant.literature_search.time.sleep"):
            mock_get.return_value = _mock_response(data)
            results = LiteratureSearch().search_external_literature("pore-scale imaging")

        assert len(results) == 1
        p = results[0]
        assert isinstance(p, Paper)
        assert p.title == "Pore-scale imaging of multiphase flow"
        assert p.authors == ["Blunt MJ", "Bijeljic B"]
        assert p.year == 2013
        assert p.doi == "10.1039/c3cs60120j"
        assert p.citation_count == 512
        assert p.pdf_url == "https://example.com/paper.pdf"
        assert p.source == "[semantic scholar]"

    def test_search_empty_data(self):
        """Empty data array returns empty list."""
        with patch("src.assistant.literature_search.requests.get") as mock_get, \
             patch("src.assistant.literature_search.time.sleep"):
            mock_get.return_value = _mock_response({"data": []})
            results = LiteratureSearch().search_external_literature("obscure query")

        assert results == []

    def test_search_http_error(self):
        """HTTP error (e.g. 429) returns empty list without raising."""
        with patch("src.assistant.literature_search.requests.get") as mock_get, \
             patch("src.assistant.literature_search.time.sleep"):
            mock_get.return_value = _mock_response({}, status_code=429)
            results = LiteratureSearch().search_external_literature("anything")

        assert results == []

    def test_search_network_error(self):
        """Network-level RequestException returns empty list without raising."""
        import requests as req_lib
        with patch("src.assistant.literature_search.requests.get") as mock_get, \
             patch("src.assistant.literature_search.time.sleep"):
            mock_get.side_effect = req_lib.RequestException("timeout")
            results = LiteratureSearch().search_external_literature("anything")

        assert results == []

    def test_auth_header_when_key_set(self, monkeypatch):
        """x-api-key header is sent when SEMANTIC_SCHOLAR_API_KEY is set."""
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key-abc")
        with patch("src.assistant.literature_search.requests.get") as mock_get, \
             patch("src.assistant.literature_search.time.sleep"):
            mock_get.return_value = _mock_response({"data": []})
            LiteratureSearch().search_external_literature("query")

        _, kwargs = mock_get.call_args
        assert kwargs.get("headers", {}).get("x-api-key") == "test-key-abc"

    def test_no_auth_header_without_key(self, monkeypatch):
        """x-api-key header is absent when SEMANTIC_SCHOLAR_API_KEY is not set."""
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        with patch("src.assistant.literature_search.requests.get") as mock_get, \
             patch("src.assistant.literature_search.time.sleep"):
            mock_get.return_value = _mock_response({"data": []})
            LiteratureSearch().search_external_literature("query")

        _, kwargs = mock_get.call_args
        assert "x-api-key" not in kwargs.get("headers", {})


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_recommendations_returns_results(self):
        """Valid recommendations response returns list of Paper."""
        data = {"recommendedPapers": [_fake_paper_dict(title="Similar Paper")]}
        with patch("src.assistant.literature_search.requests.get") as mock_get, \
             patch("src.assistant.literature_search.time.sleep"):
            mock_get.return_value = _mock_response(data)
            results = LiteratureSearch().recommendations("abc123", limit=3)

        assert len(results) == 1
        assert results[0].title == "Similar Paper"

    def test_recommendations_http_error(self):
        """HTTP error on recommendations returns empty list without raising."""
        with patch("src.assistant.literature_search.requests.get") as mock_get, \
             patch("src.assistant.literature_search.time.sleep"):
            mock_get.return_value = _mock_response({}, status_code=500)
            results = LiteratureSearch().recommendations("abc123")

        assert results == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestThrottle:
    def test_throttle_sleeps_between_rapid_calls(self):
        """Two calls made faster than min_interval triggers a sleep."""
        ls = LiteratureSearch()
        ls._last_call = time.monotonic()  # pretend a call just happened

        with patch("src.assistant.literature_search.time.sleep") as mock_sleep, \
             patch("src.assistant.literature_search.requests.get") as mock_get:
            mock_get.return_value = _mock_response({"data": []})
            ls.search_external_literature("query")

        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        assert sleep_duration > 0
