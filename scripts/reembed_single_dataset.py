"""
Re-embed a single dataset by DOI and update its Neo4j embedding.

Usage:
    python scripts/reembed_single_dataset.py --doi 10.17612/93pd-y471

Useful for patching a specific dataset whose embedding wasn't updated by a full index rebuild.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

load_dotenv()


def reembed(doi: str) -> None:
    from neo4j import GraphDatabase
    from src.assistant.llm import embeddings
    from scripts.build_dataset_vector_index import _build_embedding_text, STRIP_CODE_FROM_DESCRIPTIONS

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )

    with driver.session() as s:
        rows = s.run(
            """
            MATCH (d:Dataset {doi: $doi})
            OPTIONAL MATCH (sam:Sample)-[:PART_OF]->(d)
            OPTIONAL MATCH (dd:DigitalDataset)-[:PART_OF]->(d)
            WITH d,
                 collect(DISTINCT {
                     porousMediaType: sam.porousMediaType, porosity: sam.porosity,
                     grainSizeAvg: sam.grainSizeAvg, source: sam.source
                 }) AS samples,
                 collect(DISTINCT {
                     voxelDimensions: dd.voxelDimensions,
                     imagingEquipmentAndModel: dd.imagingEquipmentAndModel,
                     dimensionality: dd.dimensionality, segmented: dd.segmented
                 }) AS digital_datasets
            RETURN d.identifier AS identifier, d.title AS title,
                   d.description AS description, samples, digital_datasets
            """,
            doi=doi,
        ).data()

    if not rows:
        print(f"No dataset found with DOI: {doi}")
        driver.close()
        return

    row = rows[0]
    samples = [s for s in (row.get("samples") or []) if any(v for v in s.values() if v is not None)]
    digital = [d for d in (row.get("digital_datasets") or []) if any(v for v in d.values() if v is not None)]
    text = _build_embedding_text(
        {"title": row.get("title"), "description": row.get("description")},
        samples,
        digital,
    )

    print(f"Dataset: {row.get('title')}")
    print(f"STRIP_CODE_FROM_DESCRIPTIONS: {STRIP_CODE_FROM_DESCRIPTIONS}")
    print(f"Embedding text length: {len(text)} chars")
    print(f"Embedding text preview:\n{text[:300]}\n...")

    vec = embeddings.embed_documents([text])[0]
    print(f"Embedding dim: {len(vec)}")

    with driver.session() as s:
        s.run(
            "MATCH (d:Dataset {identifier: $id}) SET d.datasetEmbedding = $vec",
            id=row["identifier"],
            vec=vec,
        )
    print(f"Updated embedding for {doi} (identifier: {row['identifier']})")
    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", required=True, help="DOI of the dataset to re-embed")
    args = parser.parse_args()
    reembed(args.doi)
