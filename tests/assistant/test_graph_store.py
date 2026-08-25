"""
Unit tests for GraphStore raw-driver methods. All Neo4j driver calls are mocked —
no live database required.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call

from src.assistant.graph_store import (
    GraphStore,
    SearchResult,
    DatasetProfileMatch,
    DatasetProfileAmbiguous,
)


# Helpers

def make_store(use_neo4j: bool = True) -> GraphStore:
    """
    Builds a GraphStore with a mocked driver.
    Mocks the LLM, langchain, and Neo4j dependencies so no real connection is attempted.
    """
    # Temporarily set USE_NEO4J for this creation (necessary because _neo4j_enabled() checks env)
    original_use_neo4j = os.environ.get("USE_NEO4J", "true")
    if use_neo4j:
        os.environ["USE_NEO4J"] = "true"
    else:
        os.environ["USE_NEO4J"] = "false"

    try:
        # Mock the LLM and langchain imports that happen inside __init__
        with patch("src.assistant.llm.get_chat_model") as mock_chat:
            with patch("src.assistant.llm.get_embeddings_model") as mock_emb:
                # Mock connect() to prevent real Neo4j connection attempts
                with patch.object(GraphStore, "connect") as mock_connect:
                    mock_chat.return_value = MagicMock()
                    mock_emb.return_value = MagicMock()
                    with patch("langchain_neo4j.Neo4jGraph"):
                        with patch("langchain_neo4j.Neo4jVector"):
                            with patch("langchain_neo4j.GraphCypherQAChain"):
                                store = GraphStore()

        # Mock the driver after creation
        if use_neo4j:
            store._driver = MagicMock()
    finally:
        # Restore original environment
        os.environ["USE_NEO4J"] = original_use_neo4j

    return store


def mock_session_returning(records: list[dict]):
    """
    Patches store.driver.session() so that session.run() returns the
    given list of dicts as mock Record objects.
    """
    def side_effect(store: GraphStore):
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
        store._driver.session.return_value = session_mock
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
            with patch("src.assistant.llm.get_chat_model"):
                with patch("src.assistant.llm.get_embeddings_model"):
                    with patch("langchain_neo4j.Neo4jGraph"):
                        with patch("langchain_neo4j.Neo4jVector"):
                            with patch("langchain_neo4j.GraphCypherQAChain"):
                                store = GraphStore()
                                assert store._driver is None

    def test_semantic_search_returns_empty_when_disabled(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.llm.get_chat_model"):
                with patch("src.assistant.llm.get_embeddings_model"):
                    with patch("langchain_neo4j.Neo4jGraph"):
                        with patch("langchain_neo4j.Neo4jVector"):
                            with patch("langchain_neo4j.GraphCypherQAChain"):
                                store = GraphStore()
                                assert store.semantic_search([0.1, 0.2, 0.3]) == []

    def test_filter_by_metadata_returns_empty_when_disabled(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.llm.get_chat_model"):
                with patch("src.assistant.llm.get_embeddings_model"):
                    with patch("langchain_neo4j.Neo4jGraph"):
                        with patch("langchain_neo4j.Neo4jVector"):
                            with patch("langchain_neo4j.GraphCypherQAChain"):
                                store = GraphStore()
                                assert store.filter_by_metadata({"rockType": "Sandstone"}) == []

    def test_search_datasets_returns_empty_when_disabled(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.llm.get_chat_model"):
                with patch("src.assistant.llm.get_embeddings_model"):
                    with patch("langchain_neo4j.Neo4jGraph"):
                        with patch("langchain_neo4j.Neo4jVector"):
                            with patch("langchain_neo4j.GraphCypherQAChain"):
                                store = GraphStore()
                                assert store.search_datasets([0.1, 0.2], filters={"rockType": "Sandstone"}) == []



# Connection lifecycle

class TestConnectionLifecycle:
    def test_execute_cypher_raises_when_driver_inactive(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            store = GraphStore()
        with pytest.raises(RuntimeError, match="driver is not active"):
            store.execute_cypher("MATCH (n) RETURN n")

    def test_context_manager_closes_driver(self):
        store = make_store()
        mock_driver = store._driver
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

        mock_sem.assert_called_once_with(embedding, k=3, index_name="datasetEmbedding")

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


# component_search

class TestComponentSearch:
    def _make_doc(self, metadata: dict, text: str = "some component text"):
        doc = MagicMock()
        doc.page_content = text
        doc.metadata = metadata
        return doc

    def test_excludes_component_not_matching_filter(self):
        store = make_store()
        docs = [
            self._make_doc({"doi": "1", "porousMediaType": "carbonate", "segmented": "yes"}),
            self._make_doc({"doi": "2", "porousMediaType": "sandstone", "segmented": "yes"}),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        store._component_index = MagicMock()
        store._component_index.as_retriever.return_value = mock_retriever

        results = store.component_search("query", filters={"porousMediaType": "sandstone"})

        assert len(results) == 1
        assert results[0]["metadata"]["doi"] == "2"

    def test_no_filters_returns_all(self):
        store = make_store()
        docs = [
            self._make_doc({"doi": "1", "porousMediaType": "carbonate"}),
            self._make_doc({"doi": "2", "porousMediaType": "sandstone"}),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        store._component_index = MagicMock()
        store._component_index.as_retriever.return_value = mock_retriever

        results = store.component_search("query")

        assert len(results) == 2

    def test_missing_metadata_key_does_not_exclude(self):
        store = make_store()
        docs = [self._make_doc({"doi": "1"})]  # no porousMediaType key at all
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        store._component_index = MagicMock()
        store._component_index.as_retriever.return_value = mock_retriever

        results = store.component_search("query", filters={"porousMediaType": "sandstone"})

        assert len(results) == 1


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


# get_dataset_profile

_FULL_PROFILE_ROW = {
    "d": {"datasetNumber": 42, "title": "Bentheimer Sandstone", "doi": "10.1234/drp42"},
    "samples": [{"identifier": "s1", "title": "Core 1", "porousMediaType": "sandstone"}],
    "digital_datasets": [{"identifier": "dd1", "title": "Scan 1", "fileTypes": ["tiff"]}],
    "analysis_datasets": [{"identifier": "ad1", "title": "PNM 1"}],
    "related_publications": [],
    "related_software": [],
    "related_datasets": [],
    "sample_to_digital_edges": [{"sample": "s1", "digitalDataset": "dd1"}, {"sample": None, "digitalDataset": None}],
    "digital_to_analysis_edges": [{"digitalDataset": "dd1", "analysisDataset": "ad1"}],
}


def _tier(query: str) -> str:
    """Identifies which of get_dataset_profile's Cypher queries `query` is, by a
    substring unique to each — used by these tests' fake execute_cypher side effects."""
    if "OPTIONAL MATCH" in query:
        return "full_profile"
    if "{datasetNumber: $ref}" in query:
        return "dataset_number"
    if "$bare" in query:
        return "doi"
    if "CONTAINS" in query:
        return "title"
    raise AssertionError(f"Unrecognized query shape: {query}")


class TestGetDatasetProfile:
    def test_matches_by_dataset_number_exact(self):
        store = make_store()

        def fake_execute(query, params=None):
            tier = _tier(query)
            if tier == "dataset_number":
                return [{"datasetNumber": 42, "title": "Bentheimer Sandstone", "doi": "10.1234/drp42"}]
            if tier == "full_profile":
                return [_FULL_PROFILE_ROW]
            raise AssertionError(f"Unexpected tier reached: {tier}")

        with patch.object(store, "execute_cypher", side_effect=fake_execute):
            result = store.get_dataset_profile("42")

        assert isinstance(result, DatasetProfileMatch)
        assert result.dataset["title"] == "Bentheimer Sandstone"

    def test_matches_by_doi_exact_when_not_numeric(self):
        store = make_store()

        def fake_execute(query, params=None):
            tier = _tier(query)
            if tier == "doi":
                # Case-folding happens inside the Cypher query (toLower), not in Python —
                # the passed param preserves the reference's original casing.
                assert params["bare"] == "10.1234/DRP42"
                return [{"datasetNumber": 42, "title": "Bentheimer Sandstone", "doi": "10.1234/DRP42"}]
            if tier == "full_profile":
                return [_FULL_PROFILE_ROW]
            return []  # dataset_number tier: not a numeric reference

        with patch.object(store, "execute_cypher", side_effect=fake_execute):
            result = store.get_dataset_profile("10.1234/DRP42")

        assert isinstance(result, DatasetProfileMatch)

    def test_matches_by_title_contains_when_not_number_or_doi(self):
        store = make_store()

        def fake_execute(query, params=None):
            tier = _tier(query)
            if tier == "title":
                assert params["ref"] == "Bentheimer"
                return [{"datasetNumber": 42, "title": "Bentheimer Sandstone", "doi": "10.1234/drp42"}]
            if tier == "full_profile":
                return [_FULL_PROFILE_ROW]
            return []  # dataset_number/doi tiers: reference is neither

        with patch.object(store, "execute_cypher", side_effect=fake_execute):
            result = store.get_dataset_profile("Bentheimer")

        assert isinstance(result, DatasetProfileMatch)

    def test_zero_matches_returns_none(self):
        store = make_store()
        with patch.object(store, "execute_cypher", return_value=[]):
            assert store.get_dataset_profile("Nonexistent Dataset") is None

    def test_multiple_matches_returns_ambiguous_without_second_round_trip(self):
        store = make_store()
        candidates = [
            {"datasetNumber": 1, "title": "Bentheimer A", "doi": "10.1/a"},
            {"datasetNumber": 2, "title": "Bentheimer B", "doi": "10.1/b"},
        ]

        def fake_execute(query, params=None):
            return candidates if _tier(query) == "title" else []

        with patch.object(store, "execute_cypher", side_effect=fake_execute) as mock_exec:
            result = store.get_dataset_profile("Bentheimer")

        assert isinstance(result, DatasetProfileAmbiguous)
        assert result.candidates == candidates
        # No full-profile query should have run.
        assert not any(_tier(c.args[0]) == "full_profile" for c in mock_exec.call_args_list)

    def test_single_match_assembles_subnodes_and_input_for_edges(self):
        store = make_store()

        def fake_execute(query, params=None):
            tier = _tier(query)
            if tier == "dataset_number":
                return [{"datasetNumber": 42, "title": "Bentheimer Sandstone", "doi": "10.1234/drp42"}]
            return [_FULL_PROFILE_ROW]

        with patch.object(store, "execute_cypher", side_effect=fake_execute):
            result = store.get_dataset_profile("42")

        assert result.samples == [{"identifier": "s1", "title": "Core 1", "porousMediaType": "sandstone"}]
        assert result.digital_datasets == [{"identifier": "dd1", "title": "Scan 1", "fileTypes": ["tiff"]}]
        assert result.analysis_datasets == [{"identifier": "ad1", "title": "PNM 1"}]
        # The null-identifier edge pair must be dropped.
        assert result.sample_to_digital_edges == [{"sample": "s1", "digitalDataset": "dd1"}]
        assert result.digital_to_analysis_edges == [{"digitalDataset": "dd1", "analysisDataset": "ad1"}]

    def test_related_publication_and_optional_software_dataset_included(self):
        store = make_store()
        row = dict(_FULL_PROFILE_ROW)
        row["related_publications"] = [{"identifier": "rp1", "title": "A paper", "doi": "10.9/x"}]
        # RelatedSoftware/RelatedDataset's PART_OF relationship to Dataset is not confirmed
        # against the live schema (see get_dataset_profile's docstring) — this test only
        # checks that whatever the query returns is threaded through unmodified.
        row["related_software"] = [{"identifier": "rs1", "title": "Some Tool"}]
        row["related_datasets"] = [{"identifier": "rds1", "title": "Companion Dataset"}]

        def fake_execute(query, params=None):
            tier = _tier(query)
            if tier == "dataset_number":
                return [{"datasetNumber": 42, "title": "Bentheimer Sandstone", "doi": "10.1234/drp42"}]
            return [row]

        with patch.object(store, "execute_cypher", side_effect=fake_execute):
            result = store.get_dataset_profile("42")

        assert result.related_publications == row["related_publications"]
        assert result.related_software == row["related_software"]
        assert result.related_datasets == row["related_datasets"]

    def test_disabled_returns_none(self):
        store = make_store(use_neo4j=False)
        with patch.object(store, "execute_cypher") as mock_exec:
            assert store.get_dataset_profile("42") is None
        mock_exec.assert_not_called()

    def test_full_profile_query_nulls_embedding_vectors(self):
        """The full-profile query must never let a real 4096-float datasetEmbedding/
        componentEmbedding vector cross the wire — this is the wire-level defense layer
        for the reported context-window-exceeded bug (a real Bentheimer Sandstone-sized
        dataset, once embedded by build_dataset_vector_index.py, was enough to blow the
        model's context window on a single call)."""
        store = make_store()
        captured_queries = []

        def fake_execute(query, params=None):
            captured_queries.append(query)
            tier = _tier(query)
            if tier == "dataset_number":
                return [{"datasetNumber": 42, "title": "Bentheimer Sandstone", "doi": "10.1234/drp42"}]
            return [_FULL_PROFILE_ROW]

        with patch.object(store, "execute_cypher", side_effect=fake_execute):
            store.get_dataset_profile("42")

        full_profile_query = next(q for q in captured_queries if _tier(q) == "full_profile")
        assert "datasetEmbedding: null" in full_profile_query
        assert full_profile_query.count("componentEmbedding: null") == 3  # s, dd, ad
