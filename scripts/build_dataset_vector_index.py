"""
Build the Neo4j vector index over Dataset nodes.

For each Dataset node, assembles a structured text representation from the
dataset's description plus key sub-node properties (rock type, modality,
voxel size, segmentation status, porosity), embeds it, writes the embedding
back to the Dataset node as `datasetEmbedding`, then creates (or verifies)
the vector index.

After indexing, a round-trip smoke test runs GraphStore.search() to confirm
the index is queryable.

Usage:
    python scripts/build_dataset_vector_index.py
    python scripts/build_dataset_vector_index.py --metadata-dir data/metadata/
    python scripts/build_dataset_vector_index.py --skip-verify
    python scripts/build_dataset_vector_index.py --batch-size 20

Prerequisites:
    - Neo4j running with NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in .env
    - Graph already loaded (run scripts/load_graph.py first)
    - LLM_API_KEY + embedding provider configured in .env
    - pip install -e ".[graph]"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _build_embedding_text(dataset: dict, samples: list[dict], digital_datasets: list[dict]) -> str:
    """Assemble a structured text blob for embedding from a dataset and its sub-nodes."""
    parts: list[str] = []

    title = dataset.get("title") or ""
    description = dataset.get("description") or ""
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")

    # Aggregate sample properties across all linked samples
    media_types = sorted({s.get("porousMediaType") for s in samples if s.get("porousMediaType")})
    porosities = [s.get("porosity") for s in samples if s.get("porosity") is not None]
    grain_sizes = [s.get("grainSizeAvg") for s in samples if s.get("grainSizeAvg") is not None]
    sources = sorted({s.get("source") for s in samples if s.get("source")})

    if media_types:
        parts.append(f"Rock/media type: {', '.join(str(m) for m in media_types)}")
    if porosities:
        avg_por = sum(porosities) / len(porosities)
        parts.append(f"Porosity: {avg_por:.1f}%")
    if grain_sizes:
        parts.append(f"Grain size (avg): {grain_sizes[0]}")
    if sources:
        parts.append(f"Sample source: {', '.join(str(s) for s in sources)}")

    # Aggregate digital dataset properties
    voxel_dims = [dd.get("voxelDimensions") for dd in digital_datasets if dd.get("voxelDimensions")]
    modalities = sorted({dd.get("imagingEquipmentAndModel") for dd in digital_datasets if dd.get("imagingEquipmentAndModel")})
    dimensionalities = sorted({dd.get("dimensionality") for dd in digital_datasets if dd.get("dimensionality")})
    seg_flags = [dd.get("segmented") for dd in digital_datasets if dd.get("segmented") is not None]

    if voxel_dims:
        parts.append(f"Voxel size: {voxel_dims[0]}")
    if modalities:
        parts.append(f"Imaging modality: {', '.join(str(m) for m in modalities)}")
    if dimensionalities:
        parts.append(f"Dimensionality: {', '.join(str(d) for d in dimensionalities)}")
    if seg_flags:
        is_seg = any(seg_flags)
        parts.append(f"Segmented: {'yes' if is_seg else 'no'}")

    return "\n".join(parts) if parts else title or "Unknown dataset"


def _parent_header(parent_title: str, parent_description: str) -> list[str]:
    """Return the shared parent-context lines prepended to every sub-node text blob."""
    parts = []
    if parent_title:
        parts.append(f"Dataset: {parent_title}")
    if parent_description:
        parts.append(f"Dataset description: {parent_description}")
    return parts


def _build_sample_text(props: dict, parent_title: str, parent_description: str) -> str:
    parts = _parent_header(parent_title, parent_description)
    if props.get("title"):
        parts.append(f"Sample: {props['title']}")
    if props.get("description"):
        parts.append(f"Description: {props['description']}")
    if props.get("porousMediaType"):
        parts.append(f"Rock/media type: {props['porousMediaType']}")
    if props.get("porosity") is not None:
        parts.append(f"Porosity: {props['porosity']}%")
    if props.get("grainSizeAvg") is not None:
        grain = str(props["grainSizeAvg"])
        if props.get("grainSizeUnits"):
            grain += f" {props['grainSizeUnits']}"
        parts.append(f"Grain size (avg): {grain}")
    if props.get("source"):
        parts.append(f"Source: {props['source']}")
    if props.get("geographicOrigin"):
        parts.append(f"Geographic origin: {props['geographicOrigin']}")
    if props.get("collectionMethod"):
        parts.append(f"Collection method: {props['collectionMethod']}")
    return "\n".join(parts) if parts else props.get("title") or "Unknown sample"


def _build_digital_text(props: dict, parent_title: str, parent_description: str) -> str:
    parts = _parent_header(parent_title, parent_description)
    if props.get("title"):
        parts.append(f"Scan: {props['title']}")
    if props.get("description"):
        parts.append(f"Description: {props['description']}")
    if props.get("voxelDimensions"):
        parts.append(f"Voxel size: {props['voxelDimensions']}")
    if props.get("imagingEquipmentAndModel"):
        parts.append(f"Imaging: {props['imagingEquipmentAndModel']}")
    if props.get("dimensionality"):
        parts.append(f"Dimensionality: {props['dimensionality']}")
    if props.get("segmented") is not None:
        parts.append(f"Segmented: {'yes' if props['segmented'] else 'no'}")
    return "\n".join(parts) if parts else props.get("title") or "Unknown scan"


def _build_analysis_text(props: dict, parent_title: str, parent_description: str) -> str:
    parts = _parent_header(parent_title, parent_description)
    if props.get("title"):
        parts.append(f"Analysis: {props['title']}")
    if props.get("description"):
        parts.append(f"Description: {props['description']}")
    if props.get("type"):
        parts.append(f"Type: {props['type']}")
    if props.get("segmented") is not None:
        parts.append(f"Segmented: {'yes' if props['segmented'] else 'no'}")
    return "\n".join(parts) if parts else props.get("title") or "Unknown analysis"


_SUBNODE_TEXT_BUILDERS = {
    "Sample": _build_sample_text,
    "DigitalDataset": _build_digital_text,
    "AnalysisDataset": _build_analysis_text,
}


# ---------------------------------------------------------------------------
# IndexBuilder
# ---------------------------------------------------------------------------

class IndexBuilder:
    """Embeds Dataset nodes and builds the Neo4j vector index."""

    _EMBEDDING_DIM_CACHE: int | None = None

    def __init__(self, batch_size: int = 16):
        from neo4j import GraphDatabase
        from src.assistant.llm import embeddings

        self._driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
        )
        self._driver.verify_connectivity()
        self._embeddings = embeddings
        self._batch_size = batch_size

    def close(self):
        self._driver.close()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, skip_verify: bool = False) -> None:
        datasets = self._fetch_datasets()
        if not datasets:
            print("No Dataset nodes found. Run scripts/load_graph.py first.")
            return

        print(f"Found {len(datasets)} Dataset nodes. Building dataset embeddings...")
        t0 = time.time()
        for batch_start in range(0, len(datasets), self._batch_size):
            batch = datasets[batch_start : batch_start + self._batch_size]
            self._embed_batch(batch, batch_start, len(datasets))
        elapsed = time.time() - t0
        print(f"\nEmbedded {len(datasets)} datasets in {elapsed:.1f}s.")

        dim = self._EMBEDDING_DIM_CACHE
        if dim:
            self._create_vector_index(dim)
        else:
            print("Warning: could not determine embedding dimension; skipping dataset index creation.")

        subnodes = self._fetch_subnodes()
        if subnodes:
            print(f"\nFound {len(subnodes)} sub-nodes (Sample/DigitalDataset/AnalysisDataset). Building component embeddings...")
            t1 = time.time()
            for batch_start in range(0, len(subnodes), self._batch_size):
                batch = subnodes[batch_start : batch_start + self._batch_size]
                self._embed_subnodes_batch(batch, batch_start, len(subnodes))
            elapsed2 = time.time() - t1
            print(f"\nEmbedded {len(subnodes)} components in {elapsed2:.1f}s.")
            if dim:
                self._create_component_index(dim)
        else:
            print("No sub-nodes found; skipping component index.")

        if not skip_verify:
            self._verify()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_datasets(self) -> list[dict]:
        with self._driver.session() as session:
            records = session.run("""
                MATCH (d:Dataset)
                OPTIONAL MATCH (s:Sample)-[:PART_OF]->(d)
                OPTIONAL MATCH (dd:DigitalDataset)-[:PART_OF]->(d)
                WITH d,
                     collect(DISTINCT {
                         porousMediaType: s.porousMediaType,
                         porosity: s.porosity,
                         grainSizeAvg: s.grainSizeAvg,
                         source: s.source
                     }) AS samples,
                     collect(DISTINCT {
                         voxelDimensions: dd.voxelDimensions,
                         imagingEquipmentAndModel: dd.imagingEquipmentAndModel,
                         dimensionality: dd.dimensionality,
                         segmented: dd.segmented
                     }) AS digital_datasets
                RETURN
                    d.identifier     AS identifier,
                    d.datasetNumber  AS datasetNumber,
                    d.title          AS title,
                    d.description    AS description,
                    samples          AS samples,
                    digital_datasets AS digital_datasets
            """)
            return [dict(r) for r in records]

    def _embed_batch(self, batch: list[dict], offset: int, total: int) -> None:
        texts = []
        for row in batch:
            samples = [s for s in (row.get("samples") or []) if any(v for v in s.values() if v is not None)]
            digital = [d for d in (row.get("digital_datasets") or []) if any(v for v in d.values() if v is not None)]
            text = _build_embedding_text(
                {"title": row.get("title"), "description": row.get("description")},
                samples,
                digital,
            )
            texts.append(text)

        vectors = self._embeddings.embed_documents(texts)

        if self._EMBEDDING_DIM_CACHE is None and vectors:
            IndexBuilder._EMBEDDING_DIM_CACHE = len(vectors[0])

        with self._driver.session() as session:
            for row, vector in zip(batch, vectors):
                session.run(
                    "MATCH (d:Dataset {identifier: $id}) SET d.datasetEmbedding = $vec",
                    id=row["identifier"],
                    vec=vector,
                )

        done = offset + len(batch)
        print(f"  [{done}/{total}] embedded", end="\r", flush=True)

    def _fetch_subnodes(self) -> list[dict]:
        with self._driver.session() as session:
            records = session.run("""
                MATCH (n)-[:PART_OF]->(d:Dataset)
                WHERE n:Sample OR n:DigitalDataset OR n:AnalysisDataset
                RETURN
                    labels(n)[0]      AS label,
                    n.identifier      AS identifier,
                    properties(n)     AS props,
                    d.title           AS parentTitle,
                    d.description     AS parentDescription
            """)
            return [dict(r) for r in records]

    def _embed_subnodes_batch(self, batch: list[dict], offset: int, total: int) -> None:
        texts = []
        for row in batch:
            builder = _SUBNODE_TEXT_BUILDERS.get(row["label"])
            if builder:
                text = builder(dict(row["props"]), row.get("parentTitle") or "", row.get("parentDescription") or "")
            else:
                text = row.get("parentTitle") or "Unknown component"
            texts.append(text)

        vectors = self._embeddings.embed_documents(texts)

        with self._driver.session() as session:
            for row, vector in zip(batch, vectors):
                session.run(
                    "MATCH (n {identifier: $id}) SET n:DatasetComponent, n.componentEmbedding = $vec",
                    id=row["identifier"],
                    vec=vector,
                )

        done = offset + len(batch)
        print(f"  [{done}/{total}] embedded", end="\r", flush=True)

    def _create_component_index(self, dim: int) -> None:
        with self._driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX `componentEmbedding` IF NOT EXISTS
                FOR (n:DatasetComponent) ON (n.componentEmbedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
        print(f"Vector index 'componentEmbedding' ready (dim={dim}).")

    def _create_vector_index(self, dim: int) -> None:
        with self._driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX `datasetEmbedding` IF NOT EXISTS
                FOR (d:Dataset) ON (d.datasetEmbedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
        print(f"Vector index 'datasetEmbedding' ready (dim={dim}).")

    def _verify(self) -> None:
        print("\nRound-trip verification...")
        try:
            from src.assistant.graph_store import GraphStore
            store = GraphStore()

            results = store.search("sandstone porosity microCT", top_k=3)
            if results:
                print(f"  Dataset search OK — got {len(results)} result(s). Top: {results[0]['metadata'].get('title', '?')!r}")
            else:
                print("  Warning: dataset search returned 0 results (index may still be building).")

            comp_results = store.component_search("carbonate grain size", top_k=3)
            if comp_results:
                top = comp_results[0]["metadata"]
                print(f"  Component search OK — got {len(comp_results)} result(s). Top: {top.get('componentTitle', '?')!r} in {top.get('datasetTitle', '?')!r}")
            else:
                print("  Warning: component search returned 0 results (index may still be building).")
        except Exception as exc:
            print(f"  Verification failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Embed Dataset nodes and build Neo4j vector index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Embedding batch size (default: 16)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the round-trip GraphStore.search() smoke test")
    args = parser.parse_args()

    builder = IndexBuilder(batch_size=args.batch_size)
    try:
        builder.run(skip_verify=args.skip_verify)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
