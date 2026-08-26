"""
Pytest fixtures for src/assistant/ tests.

Provides:
  neo4j_driver     — mock Neo4j driver; records queries without a live database
  mock_graph_store — GraphStore with Neo4j replaced by the mock driver
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Neo4j mock
# ---------------------------------------------------------------------------

class _FakeRecord:
    """Mimics a neo4j.Record accessed by key."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def data(self):
        return self._data


class _FakeResult:
    def __init__(self, records: list[dict]):
        self._records = [_FakeRecord(r) for r in records]

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class _FakeSession:
    """Session that returns empty results for any query."""

    def __init__(self):
        self.queries: list[str] = []

    def run(self, query: str, **kwargs) -> _FakeResult:
        self.queries.append(query)
        return _FakeResult([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _FakeDriver:
    def __init__(self):
        self._session = _FakeSession()

    def session(self, **kwargs) -> _FakeSession:
        return self._session

    def verify_connectivity(self):
        pass

    def close(self):
        pass

    @property
    def queries(self) -> list[str]:
        return self._session.queries


@pytest.fixture
def neo4j_driver() -> _FakeDriver:
    """A mock Neo4j driver that accepts queries without a live database."""
    return _FakeDriver()


# ---------------------------------------------------------------------------
# GraphStore with mocked Neo4j
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_graph_store():
    """
    GraphStore with Neo4j disabled via USE_NEO4J=false.

    graph_store.py was written with deferred langchain_neo4j imports inside
    __init__, so module-level patching of individual driver classes is
    unreliable.  USE_NEO4J=false is the code's own supported fallback: search()
    and cypher_qa() return empty results without requiring a live database.

    Usage:
        def test_something(mock_graph_store):
            results = mock_graph_store.search("sandstone", top_k=3)
            assert isinstance(results, list)
            assert results == []  # disabled path always returns empty
    """
    with patch.dict("os.environ", {"USE_NEO4J": "false"}):
        from src.assistant.graph_store import GraphStore
        yield GraphStore()


# ---------------------------------------------------------------------------
# Auto-mark live (real-network) tests
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(items):
    """
    Any test that requests the `chat_model` fixture makes a real LLM call
    (and, transitively, real Neo4j/Semantic Scholar calls via tools.py) — see
    test_search_integration.py's `chat_model` fixture. Auto-apply the `live`
    marker so these are excluded by default (`addopts = -m "not live"` in
    pytest.ini) without requiring every test author to remember to tag them.
    """
    for item in items:
        if "chat_model" in item.fixturenames:
            item.add_marker(pytest.mark.live)
