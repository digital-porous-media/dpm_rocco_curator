import os
import re
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Guards against Cypher injection through dynamic property key names.
# Keys must start with a letter/underscore and contain only alphanumerics/underscores.
_SAFE_KEY_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


@dataclass
class SearchResult:
    """
    A single result returned by semantic_search or search_datasets.

    Attributes:
        dataset_id: The unique identifier of the matched node.
        score:      Cosine similarity score from the vector index (0–1).
        properties: All other node properties (rockType, porosity, etc.).
    """
    dataset_id: str
    score: float
    properties: dict[str, Any]


# Type alias used by filter_by_metadata, which only needs IDs.
DatasetId = str


def _neo4j_enabled() -> bool:
    """Returns True unless USE_NEO4J is explicitly set to 'false'."""
    return os.getenv("USE_NEO4J", "true").lower() == "true"


def _validate_keys(properties: dict) -> None:
    """
    Raises ValueError if any property key contains characters that could
    break out of a Cypher identifier (e.g. spaces, hyphens, injection payloads).
    """
    for key in properties:
        if not _SAFE_KEY_RE.match(key):
            raise ValueError(
                f"Property key '{key}' contains invalid characters. "
                "Only alphanumeric and underscore are allowed."
            )


def _build_where_clause(properties: dict) -> tuple[str, dict]:
    """
    Converts a filter dict into a parameterized WHERE clause.

    Example:
        {"rockType": "Sandstone", "porosity": 0.3}
        → ("n.rockType = $param_rockType AND n.porosity = $param_porosity",
           {"param_rockType": "Sandstone", "param_porosity": 0.3})

    Values are always parameterized; keys are pre-validated by _validate_keys.
    """
    clauses = []
    params  = {}

    for key, value in properties.items():
        param_name         = f"param_{key}"
        clauses.append(f"n.{key} = ${param_name}")
        params[param_name] = value

    return " AND ".join(clauses), params


class Neo4jGraphStore:
    """
    Thread-safe Neo4j client with vector search, metadata filtering, and
    combined hybrid search.

    All public methods respect the USE_NEO4J environment flag — when disabled
    they return empty results immediately without touching the database.
    """

    def __init__(self, url: str = None, username: str = None, password: str = None):
        self._uri      = url      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        self._user     = username or os.getenv("NEO4J_USER",     "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self.driver    = None

        if _neo4j_enabled():
            self.connect()


    # Connection lifecycle

    def connect(self):
        """
        Opens the driver and verifies connectivity immediately.
        Raises RuntimeError early so callers are not surprised by failures
        on the first query.
        """
        try:
            self.driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            self.driver.verify_connectivity()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to Neo4j at {self._uri}: {e}"
            ) from e

    def close(self):
        """Shuts down the connection pool and releases sockets."""
        if self.driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


    # Core execution

    def execute_cypher(self, query: str, parameters: dict = None) -> list[dict]:
        """
        Runs a raw Cypher string and returns all records as plain dicts.

        Raises RuntimeError rather than returning [] when the driver is
        inactive — callers must be able to distinguish 'no results' from
        'database was unreachable'.
        """
        if not self.driver:
            raise RuntimeError(
                "Neo4j driver is not active. Call connect() before executing queries."
            )

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            raise RuntimeError(f"Cypher execution failed: {e}") from e


    # Semantic search  (vector index)

    def semantic_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        index_name: str = "dataset-embeddings",
    ) -> list[SearchResult]:
        """
        Finds the k most similar DigitalDataset nodes to the given query
        embedding using Neo4j's built-in vector index.

        Args:
            query_embedding: A pre-computed embedding vector for the query.
                             Call your embedding model (e.g. SambaNova) before
                             passing the result here.
            k:               Number of nearest neighbours to return.
            index_name:      Name of the vector index as configured in Neo4j.
                             Defaults to "dataset-embeddings".

        Returns:
            List of SearchResult, sorted by descending similarity score.
            Returns [] immediately if USE_NEO4J=false.
        """
        if not _neo4j_enabled():
            return []

        # db.index.vector.queryNodes streams approximate nearest-neighbour
        # results directly from the index — no full graph scan needed.
        query = """
            CALL db.index.vector.queryNodes($index_name, $k, $embedding)
            YIELD node AS n, score
            RETURN n.id        AS dataset_id,
                   score,
                   properties(n) AS props
        """
        params = {
            "index_name": index_name,
            "k":          k,
            "embedding":  query_embedding,
        }

        rows = self.execute_cypher(query, params)

        return [
            SearchResult(
                dataset_id=row["dataset_id"],
                score=row["score"],
                properties=row["props"],
            )
            for row in rows
        ]


    # Metadata filter  (structured Cypher)

    def filter_by_metadata(
        self,
        filters: dict,
        label: str = "DigitalDataset",
    ) -> list[DatasetId]:
        """
        Returns dataset IDs whose properties match every key/value in `filters`.

        The filter dict is intentionally open-ended so new Croissant metadata
        fields (e.g. "license", "distribution", "recordSet") can be added
        without changing this function's signature.

        Args:
            filters: Arbitrary property key/value pairs to match on
                     (e.g. {"rockType": "Sandstone", "segmented": "true"}).
            label:   Node label to match against. Defaults to "DigitalDataset".

        Returns:
            List of dataset ID strings.
            Returns [] immediately if USE_NEO4J=false.
            Falls back to a LIMIT 20 scan when filters is empty.
        """
        if not _neo4j_enabled():
            return []

        if not filters:
            rows = self.execute_cypher(
                f"MATCH (n:{label}) RETURN n.id AS dataset_id LIMIT 20"
            )
            return [row["dataset_id"] for row in rows]

        _validate_keys(filters)
        where_clause, params = _build_where_clause(filters)
        query = f"MATCH (n:{label}) WHERE {where_clause} RETURN n.id AS dataset_id"

        rows = self.execute_cypher(query, params)
        return [row["dataset_id"] for row in rows]


    # Combined hybrid search

    def search_datasets(
        self,
        query_embedding: list[float],
        filters: dict = None,
        k: int = 5,
        index_name: str = "dataset-embeddings",
    ) -> list[SearchResult]:
        """
        Hybrid search: vector similarity + optional metadata filters in one
        Cypher query, avoiding a round-trip when both are needed.

        When filters is provided, the WHERE clause is appended directly to
        the vector index call so Neo4j can prune candidates before ranking.
        When filters is None or empty, it degrades to pure semantic search.

        Args:
            query_embedding: Pre-computed embedding vector for the query.
            filters:         Optional metadata filters (same format as
                             filter_by_metadata). Supports any Croissant field.
            k:               Number of results to return.
            index_name:      Name of the Neo4j vector index.

        Returns:
            List of SearchResult sorted by descending similarity score.
            Returns [] immediately if USE_NEO4J=false.
        """
        if not _neo4j_enabled():
            return []

        # No filters — delegate directly to semantic_search (no extra overhead).
        if not filters:
            return self.semantic_search(query_embedding, k=k, index_name=index_name)

        _validate_keys(filters)
        where_clause, params = _build_where_clause(filters)

        # Combine vector ANN call with metadata filtering in one round-trip.
        # The WHERE clause runs after candidate retrieval but before returning,
        # which is the closest Neo4j 5.x gets to pre-filtering on vector search.
        query = f"""
            CALL db.index.vector.queryNodes($index_name, $k, $embedding)
            YIELD node AS n, score
            WHERE {where_clause}
            RETURN n.id        AS dataset_id,
                   score,
                   properties(n) AS props
        """
        params.update({
            "index_name": index_name,
            "k":          k,
            "embedding":  query_embedding,
        })

        rows = self.execute_cypher(query, params)

        return [
            SearchResult(
                dataset_id=row["dataset_id"],
                score=row["score"],
                properties=row["props"],
            )
            for row in rows
        ]


    # Schema introspection

    def get_schema_blueprint(self) -> dict:
        """
        Returns node labels and relationship types for LLM prompt context.
        Uses metadata procedures (no graph scan) so it's fast on large DBs.
        """
        labels    = [r["label"]   for r in self.execute_cypher("CALL db.labels()")]
        rel_types = [r["relType"] for r in self.execute_cypher("CALL db.relationshipTypes()")]

        return {
            "node_labels":        labels,
            "relationship_types": rel_types,
        }
