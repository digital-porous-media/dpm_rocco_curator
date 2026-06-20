"""
Unit tests for Neo4jGraphStore. All Neo4j driver calls are mocked —
no live database required.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from graph_store import Neo4jGraphStore, SearchResult



# Helpers

def make_store(use_neo4j: bool = True) -> Neo4jGraphStore:
    """
    Builds a Neo4jGraphStore with a mocked driver.
    Patches GraphDatabase.driver so no real connection is attempted.
    """
    env = {"USE_NEO4J": "true" if use_neo4j else "false"}
    with patch.dict("os.environ", env):
        with patch("neo4j_graph_store.GraphDatabase.driver") as mock_cls:
            mock_cls.return_value = MagicMock()
            store = Neo4jGraphStore(url="bolt://localhost:7687", username="neo4j", password="test")
    return store


def mock_session_returning(records: list[dict]):
    """
    Patches store.driver.session() so that session.run() returns the
    given list of dicts as mock Record objects.
    """
    def side_effect(store: Neo4jGraphStore):
        mock_result = [MagicMock(**{"__iter__": lambda s: iter(r.items()), **r}) for r in records]
        # Make dict(record) work by defining keys/values on the mock
        real_records = records

        session_mock = MagicMock()
        run_result   = MagicMock()
        run_result.__iter__ = lambda s: iter(
            [_make_record(r) for r in real_records]
        )
        session_mock.__enter__ = lambda s: session_mock
        session_mock.__exit__  = MagicMock(return_value=False)
        session_mock.run.return_value = run_result
        store.driver.session.return_value = session_mock
        return store

    return side_effect


def _make_record(d: dict):
    """Returns an object that dict() can consume (mimics neo4j.Record)."""
    m = MagicMock()
    m.keys.return_value = list(d.keys())
    m.__getitem__ = lambda s, k: d[k]
    m.items       = lambda: d.items()
    # dict(record) iterates over keys and calls record[key]
    m.__iter__    = lambda s: iter(d.keys())
    return m



# USE_NEO4J flag

class TestUseNeo4jFlag:
    def test_driver_stays_none_when_disabled(self):
        """Driver must not be initialized when USE_NEO4J=false."""
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            store = Neo4jGraphStore()
        assert store.driver is None

    def test_semantic_search_returns_empty_when_disabled(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            store = Neo4jGraphStore()
        assert store.semantic_search([0.1, 0.2, 0.3]) == []

    def test_filter_by_metadata_returns_empty_when_disabled(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            store = Neo4jGraphStore()
        assert store.filter_by_metadata({"rockType": "Sandstone"}) == []

    def test_search_datasets_returns_empty_when_disabled(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            store = Neo4jGraphStore()
        assert store.search_datasets([0.1, 0.2], filters={"rockType": "Sandstone"}) == []



# Connection lifecycle

class TestConnectionLifecycle:
    def test_connect_raises_on_failure(self):
        with patch.dict("os.environ", {"USE_NEO4J": "true"}):
            with patch("neo4j_graph_store.GraphDatabase.driver") as mock_cls:
                mock_cls.return_value.verify_connectivity.side_effect = Exception("refused")
                with pytest.raises(RuntimeError, match="Could not connect"):
                    Neo4jGraphStore()

    def test_execute_cypher_raises_when_driver_inactive(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            store = Neo4jGraphStore()
        with pytest.raises(RuntimeError, match="driver is not active"):
            store.execute_cypher("MATCH (n) RETURN n")

    def test_context_manager_closes_driver(self):
        store = make_store()
        mock_driver = store.driver
        with store:
            pass
        mock_driver.close.assert_called_once()



# semantic_search

class TestSemanticSearch:
    def test_returns_search_results(self):
        store = make_store()
        fake_rows = [
            {"dataset_id": "ds-001", "score": 0.95, "props": {"rockType": "Sandstone"}},
            {"dataset_id": "ds-002", "score": 0.80, "props": {"rockType": "Limestone"}},
        ]

        with patch.object(store, "execute_cypher", return_value=fake_rows):
            results = store.semantic_search([0.1, 0.2, 0.3], k=2)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].dataset_id == "ds-001"
        assert results[0].score      == 0.95
        assert results[0].properties == {"rockType": "Sandstone"}

    def test_passes_correct_params_to_cypher(self):
        store     = make_store()
        embedding = [0.1, 0.2, 0.3]

        with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
            store.semantic_search(embedding, k=3, index_name="my-index")

        _, called_params = mock_exec.call_args[0]
        assert called_params["embedding"]  == embedding
        assert called_params["k"]          == 3
        assert called_params["index_name"] == "my-index"

    def test_uses_vector_index_procedure(self):
        store = make_store()

        with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
            store.semantic_search([0.0])

        called_query = mock_exec.call_args[0][0]
        assert "db.index.vector.queryNodes" in called_query

    def test_returns_empty_list_on_no_results(self):
        store = make_store()
        with patch.object(store, "execute_cypher", return_value=[]):
            assert store.semantic_search([0.1]) == []



# filter_by_metadata

class TestFilterByMetadata:
    def test_returns_dataset_ids(self):
        store = make_store()
        with patch.object(store, "execute_cypher",
                          return_value=[{"dataset_id": "ds-001"}, {"dataset_id": "ds-002"}]):
            result = store.filter_by_metadata({"rockType": "Sandstone"})

        assert result == ["ds-001", "ds-002"]

    def test_builds_correct_where_clause(self):
        store   = make_store()
        filters = {"rockType": "Sandstone", "segmented": "true"}

        with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
            store.filter_by_metadata(filters)

        called_query, called_params = mock_exec.call_args[0]
        assert "n.rockType = $param_rockType"   in called_query
        assert "n.segmented = $param_segmented" in called_query
        assert called_params["param_rockType"]  == "Sandstone"
        assert called_params["param_segmented"] == "true"

    def test_fallback_scan_when_empty_filters(self):
        store = make_store()
        with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
            store.filter_by_metadata({})

        called_query = mock_exec.call_args[0][0]
        assert "LIMIT 20" in called_query

    def test_rejects_invalid_key(self):
        store = make_store()
        with pytest.raises(ValueError, match="invalid characters"):
            store.filter_by_metadata({"bad-key!": "value"})

    def test_custom_label(self):
        store = make_store()
        with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
            store.filter_by_metadata({"porosity": "0.3"}, label="CoreSample")

        called_query = mock_exec.call_args[0][0]
        assert "MATCH (n:CoreSample)" in called_query

    def test_supports_arbitrary_croissant_fields(self):
        """filter_by_metadata must handle any key without code changes."""
        store   = make_store()
        filters = {"license": "CC-BY-4.0", "distribution": "HuggingFace", "recordSet": "tabular"}

        with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
            store.filter_by_metadata(filters)

        _, called_params = mock_exec.call_args[0]
        assert called_params["param_license"]      == "CC-BY-4.0"
        assert called_params["param_distribution"] == "HuggingFace"
        assert called_params["param_recordSet"]    == "tabular"



# search_datasets

class TestSearchDatasets:
    def test_delegates_to_semantic_search_when_no_filters(self):
        """Without filters, should call semantic_search directly (no extra WHERE)."""
        store     = make_store()
        embedding = [0.1, 0.2]

        with patch.object(store, "semantic_search", return_value=[]) as mock_sem:
            store.search_datasets(embedding, filters=None, k=3)

        mock_sem.assert_called_once_with(embedding, k=3, index_name="dataset-embeddings")

    def test_combines_vector_and_filter_in_one_query(self):
        """With filters, a single Cypher query must contain both the vector call and WHERE."""
        store   = make_store()
        filters = {"rockType": "Sandstone"}

        with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
            store.search_datasets([0.1, 0.2], filters=filters, k=5)

        called_query, called_params = mock_exec.call_args[0]
        assert "db.index.vector.queryNodes" in called_query
        assert "WHERE"                      in called_query
        assert "n.rockType = $param_rockType" in called_query
        assert called_params["param_rockType"] == "Sandstone"

    def test_returns_search_results(self):
        store     = make_store()
        fake_rows = [
            {"dataset_id": "ds-001", "score": 0.92, "props": {"rockType": "Sandstone"}},
        ]

        with patch.object(store, "execute_cypher", return_value=fake_rows):
            results = store.search_datasets([0.1], filters={"rockType": "Sandstone"})

        assert len(results)           == 1
        assert results[0].dataset_id  == "ds-001"
        assert results[0].score       == 0.92

    def test_rejects_invalid_filter_key(self):
        store = make_store()
        with pytest.raises(ValueError, match="invalid characters"):
            store.search_datasets([0.1], filters={"bad key!": "value"})

    def test_custom_index_name(self):
        store = make_store()
        with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
            store.search_datasets([0.1], filters={"rockType": "x"}, index_name="custom-idx")

        _, called_params = mock_exec.call_args[0]
        assert called_params["index_name"] == "custom-idx"



# get_schema_blueprint

class TestGetSchemaBlueprint:
    def test_returns_correct_structure(self):
        store = make_store()

        def fake_execute(query, params=None):
            if "db.labels"            in query: return [{"label": "DigitalDataset"}]
            if "db.relationshipTypes" in query: return [{"relType": "HAS_CORE"}]
            return []

        with patch.object(store, "execute_cypher", side_effect=fake_execute):
            schema = store.get_schema_blueprint()

        assert schema["node_labels"]        == ["DigitalDataset"]
        assert schema["relationship_types"] == ["HAS_CORE"]
