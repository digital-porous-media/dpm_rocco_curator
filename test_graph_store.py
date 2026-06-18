"""
Unit tests for Neo4jGraphStore. The Neo4j driver is always mocked so
these run without a live database.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from graph_store import Neo4jGraphStore



# Helpers

def make_store(use_neo4j: bool = True) -> Neo4jGraphStore:
    """
    Returns a Neo4jGraphStore with a fully mocked driver.
    Skips the real connect() call regardless of USE_NEO4J.
    """
    env = {"USE_NEO4J": str(use_neo4j).lower()}
    with patch.dict("os.environ", env):
        with patch("neo4j_graph_store.GraphDatabase.driver") as mock_driver_cls:
            mock_driver_cls.return_value = MagicMock()
            store = Neo4jGraphStore(
                url="bolt://localhost:7687",
                username="neo4j",
                password="test",
            )
    return store



# Connection / lifecycle

def test_driver_not_initialized_when_flag_false():
    """Driver should stay None when USE_NEO4J=false."""
    with patch.dict("os.environ", {"USE_NEO4J": "false"}):
        store = Neo4jGraphStore()
    assert store.driver is None


def test_execute_cypher_raises_when_driver_inactive():
    """
    execute_cypher must raise RuntimeError (not silently return [])
    so callers can distinguish 'no results' from 'DB was off'.
    """
    with patch.dict("os.environ", {"USE_NEO4J": "false"}):
        store = Neo4jGraphStore()

    with pytest.raises(RuntimeError, match="driver is not active"):
        store.execute_cypher("MATCH (n) RETURN n")


def test_context_manager_closes_driver():
    """__exit__ should call driver.close()."""
    store = make_store()
    mock_driver = store.driver

    with store:
        pass

    mock_driver.close.assert_called_once()


def test_connect_raises_on_bad_credentials():
    """RuntimeError should bubble up when verify_connectivity fails."""
    with patch.dict("os.environ", {"USE_NEO4J": "true"}):
        with patch("neo4j_graph_store.GraphDatabase.driver") as mock_driver_cls:
            mock_driver_cls.return_value.verify_connectivity.side_effect = Exception(
                "Auth failure"
            )
            with pytest.raises(RuntimeError, match="Could not connect"):
                Neo4jGraphStore(url="bolt://bad-host:7687")



# filter_by_metadata — Cypher generation

def test_filter_by_metadata_builds_correct_query():
    """
    Verify that filter_by_metadata passes the right Cypher string and
    parameterized values to execute_cypher.
    """
    store = make_store()
    filters = {"rockType": "Sandstone", "segmented": "true"}

    with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
        store.filter_by_metadata(filters)

    called_query, called_params = mock_exec.call_args[0]

    assert "MATCH (n:DigitalDataset)" in called_query
    assert "n.rockType = $param_rockType" in called_query
    assert "n.segmented = $param_segmented" in called_query
    assert called_params["param_rockType"] == "Sandstone"
    assert called_params["param_segmented"] == "true"


def test_filter_by_metadata_fallback_when_empty():
    """Empty filter dict should fall back to an unfiltered LIMIT 20 query."""
    store = make_store()

    with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
        store.filter_by_metadata({})

    called_query = mock_exec.call_args[0][0]
    assert "LIMIT 20" in called_query


def test_filter_by_metadata_rejects_invalid_keys():
    """Malicious or malformed property keys must raise ValueError."""
    store = make_store()

    with pytest.raises(ValueError, match="invalid characters"):
        store.filter_by_metadata({"bad-key!": "value"})


def test_filter_by_metadata_custom_label():
    """The node label should be configurable."""
    store = make_store()

    with patch.object(store, "execute_cypher", return_value=[]) as mock_exec:
        store.filter_by_metadata({"porosity": "0.3"}, label="CoreSample")

    called_query = mock_exec.call_args[0][0]
    assert "MATCH (n:CoreSample)" in called_query



# get_schema_blueprint

def test_get_schema_blueprint_structure():
    """Schema helper should return the expected top-level keys."""
    store = make_store()

    def fake_execute(query, params=None):
        if "db.labels" in query:
            return [{"label": "DigitalDataset"}, {"label": "Well"}]
        if "db.relationshipTypes" in query:
            return [{"relType": "HAS_CORE"}, {"relType": "BELONGS_TO"}]
        return []

    with patch.object(store, "execute_cypher", side_effect=fake_execute):
        schema = store.get_schema_blueprint()

    assert schema["node_labels"] == ["DigitalDataset", "Well"]
    assert schema["relationship_types"] == ["HAS_CORE", "BELONGS_TO"]