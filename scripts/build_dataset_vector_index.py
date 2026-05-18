"""
Build the Neo4j vector index over Dataset nodes.

Approach: Keywords strategy (from CurationTools/JsonToNeo4jwKeywords.ipynb)
- For each Dataset node, use the LLM to extract keywords from the description
- Embed the description using SambaStudioEmbeddings
- Store the embedding as Dataset.descriptionEmbedding
- Create a vector index named "datasetDescription" on that property

This enables semantic search via Neo4jVector.from_existing_index() in graph_store.py.

Alternative: Chunking strategy (see CurationTools/JsonToNeo4jwChunking.ipynb)
  Stores Description + Chunk nodes with per-chunk embeddings. More granular but
  more complex. Not implemented here.

Usage:
    python scripts/build_dataset_vector_index.py
    python scripts/build_dataset_vector_index.py --metadata-dir data/metadata/

Prerequisites:
    - Neo4j running with NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in .env
    - Metadata JSON files in data/metadata/ (run scripts/scrape_metadata.py first)
    - pip install -e ".[graph]"
"""

import argparse
import glob
import json
import os
from dotenv import load_dotenv

load_dotenv()

# TODO (Intern A, Week 2): implement after GraphStore is working
#
# Suggested implementation steps:
#
# 1. Connect to Neo4j:
#       from langchain_neo4j import Neo4jGraph
#       graph = Neo4jGraph(url=..., username=..., password=...)
#
# 2. Load embeddings (reuse the singleton from src/assistant/llm.py):
#       from src.assistant.llm import embeddings
#
# 3. For each JSON file in metadata_dir:
#       a. Parse the JSON and extract the description field
#       b. Use LLM to extract keywords (see CurationTools/JsonToNeo4jwKeywords.ipynb for prompt)
#       c. Embed the description: vector = embeddings.embed_query(description)
#       d. Write back to Neo4j:
#            graph.query(
#                "MATCH (d:Dataset {datasetNumber: $num}) "
#                "SET d.descriptionEmbedding = $vec, d.llmKeywords = $keywords",
#                params={"num": dataset_number, "vec": vector, "keywords": keywords}
#            )
#
# 4. Create the vector index (run once):
#       graph.query("""
#           CREATE VECTOR INDEX datasetDescription IF NOT EXISTS
#           FOR (d:Dataset) ON (d.descriptionEmbedding)
#           OPTIONS {indexConfig: {`vector.dimensions`: <dim>, `vector.similarity_function`: 'cosine'}}
#       """)
#
# Note: The embedding dimension depends on the EMBEDDING_MODEL. Check the model docs.
# SambaNova E5-Mistral-7B-Instruct produces 4096-dim vectors.


def build_index(metadata_dir: str) -> None:
    files = glob.glob(os.path.join(metadata_dir, "*.json"))
    if not files:
        print(f"No JSON files found in {metadata_dir}. Run scripts/scrape_metadata.py first.")
        return
    print(f"Found {len(files)} metadata files in {metadata_dir}.")
    raise NotImplementedError("build_index not yet implemented — see TODOs above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Neo4j vector index over dataset descriptions.")
    parser.add_argument("--metadata-dir", default="data/metadata/", help="Path to metadata JSON files")
    args = parser.parse_args()
    build_index(args.metadata_dir)
