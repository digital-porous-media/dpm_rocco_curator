"""
Pytest fixtures for src/assistant/ tests.

Provides:
  neo4j_driver   — mock Neo4j driver; records queries without a live database
  small_faiss    — tiny in-memory FAISS index (3 docs) for PublicationCorpus tests
  mock_graph_store — GraphStore with Neo4j replaced by the mock driver
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
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
# Small in-memory FAISS index
# ---------------------------------------------------------------------------

THREE_DOCS = [
    {
        "doc_title": "Blunt et al. 2013",
        "page": 1,
        "dataset_id": "DRP-11",
        "snippet": "Pore-scale imaging of multiphase flow in Bentheimer sandstone.",
    },
    {
        "doc_title": "Gostick et al. 2019",
        "page": 2,
        "dataset_id": "DRP-1",
        "snippet": "PoreSpy: a Python toolkit for quantitative analysis of porous media images.",
    },
    {
        "doc_title": "Wildenschild & Sheppard 2013",
        "page": 3,
        "dataset_id": None,
        "snippet": "X-ray imaging and analysis techniques for quantifying pore-scale structure.",
    },
]

EMBEDDING_DIM = 8  # tiny dimension — fast, no model required


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic fake embeddings: hash-based unit vectors."""
    vecs = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**31))
        v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        vecs.append(v)
    return np.array(vecs)


@pytest.fixture
def small_faiss():
    """
    A tiny FAISS flat index pre-loaded with THREE_DOCS.

    Returns a dict with keys:
        index      — faiss.IndexFlatIP (inner-product / cosine on unit vecs)
        metadata   — list of doc metadata dicts (same order as index vectors)
        embed_fn   — callable: list[str] -> np.ndarray of unit vectors
        dim        — embedding dimension
    """
    faiss = pytest.importorskip("faiss")

    texts = [d["snippet"] for d in THREE_DOCS]
    vecs = _fake_embed(texts)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vecs)

    return {
        "index": index,
        "metadata": list(THREE_DOCS),
        "embed_fn": _fake_embed,
        "dim": EMBEDDING_DIM,
    }


# ---------------------------------------------------------------------------
# GraphStore with mocked Neo4j
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_graph_store(neo4j_driver):
    """
    GraphStore instance with the Neo4j driver replaced by neo4j_driver.

    USE_NEO4J is forced to "true" so the constructor runs the Neo4j code
    path, but all driver calls hit the fake session.

    Usage:
        def test_something(mock_graph_store):
            results = mock_graph_store.search("sandstone", top_k=3)
            assert isinstance(results, list)
    """
    with patch.dict("os.environ", {"USE_NEO4J": "true",
                                    "NEO4J_URI": "bolt://localhost:7687",
                                    "NEO4J_USER": "neo4j",
                                    "NEO4J_PASSWORD": "test"}):
        with patch("src.assistant.graph_store.GraphDatabase") as mock_gdb, \
             patch("src.assistant.graph_store.Neo4jVector") as mock_vec, \
             patch("src.assistant.graph_store.Neo4jGraph") as mock_graph:

            mock_gdb.driver.return_value = neo4j_driver

            # Neo4jVector.from_existing_index returns a mock vectorstore
            mock_vectorstore = MagicMock()
            mock_vectorstore.similarity_search_with_score.return_value = []
            mock_vec.from_existing_index.return_value = mock_vectorstore

            # Neo4jGraph just needs to not error
            mock_graph.return_value = MagicMock()

            from src.assistant.graph_store import GraphStore
            store = GraphStore()
            store._vectorstore = mock_vectorstore
            store._component_vectorstore = mock_vectorstore
            yield store
