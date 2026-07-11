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
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# langchain_neo4j imports are deferred to __init__ so that USE_NEO4J=false
# works without triggering the neo4j driver (which has heavy optional deps).

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer
questions about porous media datasets from the Digital Porous Media Portal.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Do not return entire nodes or embedding properties.
Do not use any APOC procedures or functions. Use only standard Cypher.

Fine Tuning:
- Sometimes relevant keywords may be contained in the description instead of just the title.
- People may use "projects" and "datasets" interchangeably.
- Avoid UNION/UNION ALL unless necessary. Prefer combining conditions with OR in a single MATCH.
- If you do use UNION, all branches must return the same column names.

Schema:
{schema}

Question:
{question}

Cypher Query:
"""

_cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)


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
            return

        from langchain_neo4j import Neo4jGraph, Neo4jVector
        from langchain_neo4j import GraphCypherQAChain
        from src.assistant.llm import chat_model, embeddings

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
