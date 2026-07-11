from __future__ import annotations
"""
Neo4j vector index + structured Cypher search over dataset nodes.

Graph schema
------------
Node labels:
    Dataset          - root node; properties: title, description, doi, datasetNumber,
                       llmKeywords, datasetEmbedding
    Sample           - properties: title, identifier, location, porousMediaType, porosity,
                       grainSizeAvg/Min/Max, grainSizeUnits, collectionMethod, source,
                       onshoreOffshore, depth, waterDepth, procedure, equipment,
                       algorithmDescription, geographicOrigin, datasetNumber
    DigitalDataset   - properties: title, identifier, voxelDimensions, imagingCenter,
                       imagingEquipmentAndModel, imageFormat, imageDimensions,
                       imageByteOrder, dimensionality, numberOfFiles, fileTypes,
                       segmented, datasetNumber
    AnalysisDataset  - properties: title, identifier, segmented, type,
                       referencedDigitalDataset, referencedSample, numberOfFiles,
                       fileTypes, datasetNumber
    RelatedPublication - title, authors, abstract, link, publicationDate, datasetNumber
    RelatedSoftware  - title, description, link, datasetNumber
    RelatedDataset   - title, description, link, datasetNumber

Relationships:
    PART_OF   (Sample|DigitalDataset|AnalysisDataset → Dataset)
    INPUT_FOR (Dataset → Dataset)

Vector indexes:
    datasetEmbedding  — node: Dataset, property: datasetEmbedding
                        Aggregates title + description + sub-node metadata into one vector.
                        Used by search() and GraphCypherQAChain.
    componentEmbedding — node: DatasetComponent (secondary label on Sample/DigitalDataset/AnalysisDataset)
                         property: componentEmbedding
                         Each sub-node embedded individually with parent Dataset context injected.
                         Used by component_search() for fine-grained retrieval.
    Both built by: scripts/build_dataset_vector_index.py

Alternative approach (not implemented):
    Chunking strategy stores Description + Chunk nodes instead of embedding on Dataset.
    See CurationTools/JsonToNeo4jwChunking.ipynb for reference.

Environment variables required:
    NEO4J_URI      - bolt://localhost:7687 (local) or neo4j+s://... (cloud)
    NEO4J_USER     - typically "neo4j"
    NEO4J_PASSWORD - your password
    USE_NEO4J      - set to "false" to disable graph and fall back to publication FAISS only

APOC note:
    APOC is not required. The Cypher generation prompt explicitly forbids apoc.* calls,
    keeping the code portable across local Neo4j, TACC VM, and AuraDB.
"""

import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

# langchain_neo4j imports are deferred to __init__ so that USE_NEO4J=false
# works without triggering the neo4j driver (which has heavy optional deps).

# Hardcoded schema fed to GraphCypherQAChain (refresh_schema=False means langchain
# won't introspect via apoc.meta.data, so we supply it manually).
MANUAL_SCHEMA = """
Node labels and properties:
  Dataset        — identifier, datasetNumber (int), title, description, doi, authors, license, publicationDate
  Sample         — identifier, datasetNumber (int), title, porousMediaType, porosity (float, 27% populated),
                   source, location, geographicOrigin, grainSizeAvg (float), grainSizeMin (float), grainSizeMax (float)
                   porousMediaType values: beads, carbonate, coal, fibrous_media, granite, other, sandstone, soil
                   source values: artificial, natural
  DigitalDataset — identifier, datasetNumber (int), title, description, voxelDimensions, segmented, numberOfFiles (int), fileTypes (list)
                   segmented values: yes, no
  AnalysisDataset — identifier, datasetNumber (int), title, description, type, segmented, numberOfFiles (int), fileTypes (list)
                   type values: geometric_analysis, other, simulation
  RelatedPublication — title, authors, abstract, link, publicationDate, datasetNumber (int)

Relationships (all use PART_OF or INPUT_FOR — no other relationship types exist):
  (Sample)-[:PART_OF]->(Dataset)
  (DigitalDataset)-[:PART_OF]->(Dataset)
  (AnalysisDataset)-[:PART_OF]->(Dataset)
  (RelatedPublication)-[:PART_OF]->(Dataset)
  (Sample)-[:INPUT_FOR]->(DigitalDataset)
  (DigitalDataset)-[:INPUT_FOR]->(AnalysisDataset)

Important:
  - porosity is on Sample nodes, NOT on Dataset nodes. Always join via PART_OF.
  - porousMediaType (e.g. "sandstone") is on Sample nodes, NOT on Dataset nodes.
  - Use OPTIONAL MATCH for sparse properties (porosity, grainSize*, geographicOrigin).
  - Use case-insensitive matching for string values: toLower(s.porousMediaType) = 'sandstone'
"""

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer
questions about porous media datasets from the Digital Porous Media Portal.

Use ONLY the node labels, relationship types, and properties listed below.
Do not invent labels, relationship types, or properties.
Do not return entire nodes or embedding properties.
Do not use any APOC procedures or functions. Use only standard Cypher.

Fine Tuning:
- Sometimes relevant keywords may be contained in the description instead of just the title.
- People may use "projects" and "datasets" interchangeably.
- When a condition applies to multiple sub-node types (e.g. DigitalDataset and AnalysisDataset),
  use OPTIONAL MATCH for each type separately, then combine with OR in the WHERE clause on named variables.
  Never combine two different labels in a single MATCH pattern like (n:`LabelA OR alias`:LabelB) — that is invalid syntax.
  Example for "segmented datasets":
    MATCH (d:Dataset)
    OPTIONAL MATCH (d)<-[:PART_OF]-(dd:DigitalDataset)
    OPTIONAL MATCH (d)<-[:PART_OF]-(ad:AnalysisDataset)
    WHERE dd.segmented = 'yes' OR ad.segmented = 'yes'
    RETURN DISTINCT d.identifier, d.title
- If you do use UNION, all branches must return the same column names.

Schema:
""" + MANUAL_SCHEMA + """

Question:
{question}

Cypher Query:
"""

_cypher_prompt = PromptTemplate(input_variables=["question"], template=CYPHER_GENERATION_TEMPLATE)


class GraphStore:
    """
    Wraps Neo4j vector similarity search and structured Cypher QA.

    Falls back gracefully when USE_NEO4J=false (returns empty results).
    """

    def __init__(self):
        self._enabled = os.getenv("USE_NEO4J", "true").lower() != "false"
        if not self._enabled:
            self._graph = None
            self._vector_index = None
            self._component_index = None
            self._cypher_chain = None
            self._driver = None
            return

        from langchain_neo4j import Neo4jGraph, Neo4jVector
        from langchain_neo4j import GraphCypherQAChain
        from src.assistant.llm import get_chat_model, get_embeddings_model

        chat_model = get_chat_model()
        embeddings = get_embeddings_model()

        self._graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
            # Skip apoc.meta.data() schema introspection — APOC is not installed.
            # Schema is provided manually to GraphCypherQAChain via the prompt template.
            refresh_schema=False,
        )

        self._vector_index = Neo4jVector.from_existing_index(
            embeddings,
            graph=self._graph,
            index_name="datasetEmbedding",
            node_label="Dataset",
            text_node_property="description",
            embedding_node_property="datasetEmbedding",
            retrieval_query="""
RETURN
    node.description AS text,
    score,
    {
        title: node.title,
        sampleTitles: [(sample)-[:PART_OF]->(node) | sample.title],
        datasetNumber: node.datasetNumber,
        doi: node.doi
    } AS metadata
""",
        )

        self._component_index = Neo4jVector.from_existing_index(
            embeddings,
            graph=self._graph,
            index_name="componentEmbedding",
            node_label="DatasetComponent",
            text_node_property="title",
            embedding_node_property="componentEmbedding",
            retrieval_query="""
MATCH (n)-[:PART_OF]->(d:Dataset)
RETURN
    n.title + coalesce(': ' + n.description, '') AS text,
    score,
    {
        componentType:  labels(n)[0],
        componentTitle: n.title,
        datasetTitle:   d.title,
        datasetNumber:  d.datasetNumber,
        doi:            d.doi
    } AS metadata
""",
        )

        self._cypher_chain = GraphCypherQAChain.from_llm(
            chat_model,
            graph=self._graph,
            verbose=True,
            cypher_prompt=_cypher_prompt,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
            top_k=10,
        )

        # Initialize raw Neo4j driver for low-level primitives (semantic_search, filter_by_metadata, etc.)
        self._uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = os.getenv("NEO4J_USER", "neo4j")
        self._password = os.getenv("NEO4J_PASSWORD", "")
        self._driver = None
        self.connect()

    def search(self, query: str, filters: dict = None, top_k: int = 5) -> list[dict]:
        """
        Vector similarity search over dataset descriptions.

        Args:
            query: Natural language search query.
            filters: Optional dict of property constraints (e.g. {"porousMediaType": "sandstone"}).
                     Keys must be valid Dataset/Sample node properties.
            top_k: Number of results to return.

        Returns:
            List of result dicts with keys: text, score, metadata, source_label.
        """
        if not self._enabled:
            return []

        retriever = self._vector_index.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)

        results = []
        for doc in docs:
            result = {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "source_label": "[graph match]",
            }
            # Apply post-retrieval filter if provided (Cypher-level filtering is a Week 2 enhancement)
            if filters:
                skip = False
                for key, value in filters.items():
                    meta_val = doc.metadata.get(key)
                    if meta_val is not None and str(meta_val).lower() != str(value).lower():
                        skip = True
                        break
                if skip:
                    continue
            results.append(result)

        return results

    def hybrid_search(self, query: str, filters: dict = None, top_k: int = 5) -> list[dict]:
        """
        Hybrid BM25 + vector search with Reciprocal Rank Fusion.

        Runs vector similarity search and Neo4j fulltext (BM25) search in parallel,
        then merges results using RRF. This handles vocabulary mismatch: datasets whose
        descriptions use different words than the query (e.g. "LBM transport simulation"
        vs "velocity field FNO") are caught by BM25 even when vector similarity is low.

        Requires the `datasetDescriptionFulltext` fulltext index (created by
        build_dataset_vector_index.py or manually via CREATE FULLTEXT INDEX).

        Returns list of dicts with keys: text, metadata, source_label ("[hybrid match]").
        """
        if not self._enabled:
            return []

        candidate_k = top_k * 2  # fetch extra candidates so RRF has room to rerank

        # --- Vector search ---
        retriever = self._vector_index.as_retriever(search_kwargs={"k": candidate_k})
        vec_docs = retriever.invoke(query)

        # Build rank lookup by DOI: {doi: rank (0-based)}
        vec_rank: dict[str, int] = {}
        vec_meta: dict[str, dict] = {}  # doi -> metadata dict for result assembly
        for rank, doc in enumerate(vec_docs):
            doi = doc.metadata.get("doi", "")
            if doi and doi not in vec_rank:
                vec_rank[doi] = rank
                vec_meta[doi] = {"meta": doc.metadata, "text": doc.page_content}

        # --- BM25 fulltext search ---
        bm25_rank: dict[str, int] = {}
        bm25_meta: dict[str, dict] = {}
        try:
            with self._driver.session() as session:
                rows = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('datasetDescriptionFulltext', $search_query,
                        {limit: $limit})
                    YIELD node, score
                    RETURN node.doi AS doi, node.title AS title,
                           node.description AS description,
                           node.datasetNumber AS datasetNumber,
                           [(s)-[:PART_OF]->(node) | s.title] AS sampleTitles
                    """,
                    search_query=query,
                    limit=candidate_k,
                ).data()
            for rank, row in enumerate(rows):
                doi = row.get("doi", "")
                if doi and doi not in bm25_rank:
                    bm25_rank[doi] = rank
                    bm25_meta[doi] = {
                        "meta": {
                            "title": row["title"],
                            "doi": doi,
                            "datasetNumber": row.get("datasetNumber"),
                            "sampleTitles": row.get("sampleTitles") or [],
                        },
                        "text": row.get("description") or row.get("title") or "",
                    }
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "hybrid_search: BM25 query failed (%s); falling back to vector only", e
            )

        # --- RRF merge ---
        all_dois = set(vec_rank) | set(bm25_rank)
        penalty = candidate_k + 1  # rank assigned to a DOI absent from one list
        k_rrf = 60  # standard RRF constant

        scored: list[tuple[float, str]] = []
        for doi in all_dois:
            r_vec = vec_rank.get(doi, penalty)
            r_bm25 = bm25_rank.get(doi, penalty)
            rrf = 1.0 / (r_vec + k_rrf) + 1.0 / (r_bm25 + k_rrf)
            scored.append((rrf, doi))
        scored.sort(reverse=True)

        # --- Apply filters and assemble results ---
        results = []
        for _, doi in scored:
            if len(results) >= top_k:
                break
            entry = vec_meta.get(doi) or bm25_meta.get(doi)
            if entry is None:
                continue
            meta = entry["meta"]
            if filters:
                skip = False
                for key, value in filters.items():
                    meta_val = meta.get(key)
                    if meta_val is not None and str(meta_val).lower() != str(value).lower():
                        skip = True
                        break
                if skip:
                    continue
            results.append({
                "text": entry["text"],
                "metadata": meta,
                "source_label": "[hybrid match]",
            })

        return results

    def component_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Vector similarity search over individual Sample, DigitalDataset, and AnalysisDataset
        sub-nodes. Each result links back to its parent Dataset.

        Returns list of dicts with keys: text, score, metadata, source_label.
        metadata keys: componentType, componentTitle, datasetTitle, datasetNumber, doi.
        """
        if not self._enabled:
            return []

        retriever = self._component_index.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "source_label": "[component match]",
            }
            for doc in docs
        ]

    def get_dataset(self, dataset_id: str) -> dict | None:
        """Fetch full Dataset node properties by datasetNumber."""
        if not self._enabled or not self._graph:
            return None
        records = self._graph.query(
            "MATCH (d:Dataset {datasetNumber: $id}) RETURN d",
            params={"id": dataset_id},
        )
        if records:
            return records[0]["d"]
        return None

    def cypher_qa(self, question: str) -> str:
        """
        Answer a structured question about datasets using LLM-generated Cypher.
        Source label: [cypher match]
        """
        if not self._enabled or not self._cypher_chain:
            return "Graph search is disabled (USE_NEO4J=false)."
        result = self._cypher_chain.invoke({"query": question})
        return result.get("result", "No answer found.")

    # ---------------------------------------------------------------------------
    # Raw Neo4j driver methods — low-level primitives for Week 5+ work
    # ---------------------------------------------------------------------------

    def connect(self):
        """
        Opens the driver and verifies connectivity immediately.
        Raises RuntimeError early so callers are not surprised by failures
        on the first query.
        """
        if not self._enabled:
            return
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            self._driver.verify_connectivity()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to Neo4j at {self._uri}: {e}"
            ) from e

    def close(self):
        """Shuts down the connection pool and releases sockets."""
        if self._driver:
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def execute_cypher(self, query: str, parameters: dict = None) -> list[dict]:
        """
        Runs a raw Cypher string and returns all records as plain dicts.

        Raises RuntimeError rather than returning [] when the driver is
        inactive — callers must be able to distinguish 'no results' from
        'database was unreachable'.
        """
        if not self._driver:
            raise RuntimeError(
                "Neo4j driver is not active. Call connect() before executing queries."
            )

        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            raise RuntimeError(f"Cypher execution failed: {e}") from e

    def semantic_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        index_name: str = "datasetEmbedding",
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
                             Defaults to "datasetEmbedding".

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

    def filter_by_metadata(
        self,
        filters: dict,
        label: str = "Dataset",
    ) -> list[DatasetId]:
        """
        Returns dataset IDs whose properties match every key/value in `filters`.

        The filter dict is intentionally open-ended so new Croissant metadata
        fields (e.g. "license", "distribution", "recordSet") can be added
        without changing this function's signature.

        Args:
            filters: Arbitrary property key/value pairs to match on
                     (e.g. {"rockType": "Sandstone", "segmented": "true"}).
            label:   Node label to match against. Defaults to "Dataset".

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

    def search_datasets(
        self,
        query_embedding: list[float],
        filters: dict = None,
        k: int = 5,
        index_name: str = "datasetEmbedding",
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
