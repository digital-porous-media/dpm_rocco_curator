"""
Thin wrapper around the Neo4j Python driver for connecting to the TACC graph VM,
executing LLM-generated Cypher queries, and retrieving schema context.
"""
 
import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
 
load_dotenv()
 
# Only alphanumeric and underscore — guards against key injection in dynamic queries
_SAFE_KEY_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
 
 
class Neo4jGraphStore:
    """
    Manages a thread-safe connection pool to a Neo4j instance and exposes
    helpers for raw Cypher execution, metadata-filtered lookups, and schema
    introspection (used to give the LLM query context).
    """
 
    def __init__(self, url: str = None, username: str = None, password: str = None):
        self._uri      = url      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        self._user     = username or os.getenv("NEO4J_USER",     "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self.driver    = None
 
        # Respect the USE_NEO4J flag so tests can disable the live connection.
        if os.getenv("USE_NEO4J", "true").lower() == "true":
            self.connect()
 

    # Connection lifecycle
 
    def connect(self):
        """
        Opens the driver and immediately verifies the network handshake.
        Raises RuntimeError on any connectivity failure so callers know
        early rather than getting silent empty results later.
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
        """Cleanly shuts down the connection pool."""
        if self.driver:
            self.driver.close()
 
    # Context-manager support so callers can use `with Neo4jGraphStore() as g:`
    def __enter__(self):
        return self
 
    def __exit__(self, *_):
        self.close()
 

    # Core query execution
 
    def execute_cypher(self, query: str, parameters: dict = None) -> list[dict]:
        """
        Runs a raw Cypher string (typically LLM-generated) and returns all
        records as a list of plain dicts.
 
        Raises RuntimeError if the driver is not active — callers should
        not silently receive an empty list when the DB is unreachable.
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
 

    # Filtered lookups                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
 
    def filter_by_metadata(
        self,
        properties: dict,
        label: str = "DigitalDataset",
    ) -> list[dict]:
        """
        Builds and runs a parameterized MATCH query for the given property
        filters (e.g. {"rockType": "sandstone", "porosity": 0.3}).
 
        Property *values* are always parameterized.
        Property *keys* are validated against a strict allowlist pattern
        to prevent Cypher injection through key names.
 
        Falls back to an unfiltered LIMIT 20 scan when `properties` is empty.
        """
        if not properties:
            return self.execute_cypher(f"MATCH (n:{label}) RETURN n LIMIT 20")
 
        clauses: list[str] = []
        params:  dict      = {}
 
        for key, value in properties.items():
            if not _SAFE_KEY_RE.match(key):
                raise ValueError(
                    f"Property key '{key}' contains invalid characters. "
                    "Only alphanumeric and underscore are allowed."
                )
            param_name         = f"param_{key}"
            clauses.append(f"n.{key} = ${param_name}")
            params[param_name] = value
 
        where_clause = " AND ".join(clauses)
        query = f"MATCH (n:{label}) WHERE {where_clause} RETURN n"
 
        return self.execute_cypher(query, params)
 

    # Schema introspection
  
    def get_schema_blueprint(self) -> dict:
        """
        Returns the graph's node labels and relationship types as a dict
        that can be injected into the LLM prompt for query context framing.
 
        Uses metadata procedures instead of full graph scans so this stays
        fast even on large databases.
        """
        # db.labels() / db.relationshipTypes() are metadata-only — no graph scan.
        labels        = [r["label"]    for r in self.execute_cypher("CALL db.labels()")]
        rel_types     = [r["relType"]  for r in self.execute_cypher("CALL db.relationshipTypes()")]
 
        return {
            "node_labels":        labels,
            "relationship_types": rel_types,
        }