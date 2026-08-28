"""
Build the Neo4j vector index over Dataset nodes.

For each Dataset node, assembles a structured text representation from the
dataset's description plus key sub-node properties (rock type, modality,
voxel size, segmentation status, porosity), embeds it, writes the embedding
back to the Dataset node as `datasetEmbedding`, then creates (or verifies)
the vector index.

A third step builds per-dataset **fact sheets** (`Dataset.factSheet` /
`Dataset.factSheetText` / `Dataset.factSheetEmbedding`) — an edge-preserving
prose summary of each dataset's sub-node graph, used by the assistant's
`reason_about_dataset_content` tool. See _build_fact_sheet below for why this
does not reuse _build_embedding_text.

After indexing, a round-trip smoke test runs GraphStore.search() to confirm
the index is queryable.

Usage:
    python scripts/build_dataset_vector_index.py
    python scripts/build_dataset_vector_index.py --metadata-dir data/metadata/
    python scripts/build_dataset_vector_index.py --skip-verify
    python scripts/build_dataset_vector_index.py --batch-size 20
    python scripts/build_dataset_vector_index.py --only fact-sheets
    python scripts/build_dataset_vector_index.py --only fact-sheets --retry-missing

Prerequisites:
    - Neo4j running with NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in .env
    - Graph already loaded (run scripts/load_graph.py first)
    - LLM_API_KEY + embedding provider configured in .env
    - pip install -e ".[graph]"
"""

from __future__ import annotations

import argparse
import json
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

# Set to False to embed raw descriptions (may include code blocks and boilerplate).
# Set to True to strip code-like content before embedding, improving semantic
# signal for descriptions that contain Python/Matlab code or file format instructions.
# Changing this flag requires rebuilding the index: python scripts/build_dataset_vector_index.py
STRIP_CODE_FROM_DESCRIPTIONS = True


import re as _re


def _clean_description_for_embedding(text: str) -> str:
    """Strip code blocks, URLs, and boilerplate from a description before embedding.

    The stored description in Neo4j is not modified — this only affects the text
    fed to the embedding model.
    """
    lines = text.splitlines()
    cleaned = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        # Toggle fenced code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        # Skip lines that look like code or boilerplate
        if _re.match(r"^(from|import)\s+\w", stripped):
            continue
        if _re.match(r"^\w[\w.]*\s*=\s*\w", stripped):  # assignment
            continue
        if _re.match(r"^[\w.]+\(", stripped):  # bare function call
            continue
        if _re.match(r"^[#%]", stripped):  # Python/shell/Matlab comment
            continue
        if _re.match(r"^https?://", stripped):  # URL-only line
            continue
        if _re.match(r"^-{3,}$", stripped):  # horizontal rule
            continue
        # Skip repeated section headers that add no semantic content
        if _re.match(r"^(load data in|filenames and keys|for comments|usage:|prerequisites:|please see the readme|see the repo)", stripped, _re.IGNORECASE):
            continue
        # Skip lines that are only a URL (possibly trailing context)
        if _re.search(r"https?://\S+$", stripped) and len(stripped.split()) <= 10:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def _build_embedding_text(dataset: dict, samples: list[dict], digital_datasets: list[dict]) -> str:
    """Assemble a structured text blob for embedding from a dataset and its sub-nodes."""
    parts: list[str] = []

    title = dataset.get("title") or ""
    description = dataset.get("description") or ""
    if STRIP_CODE_FROM_DESCRIPTIONS:
        description = _clean_description_for_embedding(description)
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
# Fact sheets (Dataset.factSheet / .factSheetText / .factSheetEmbedding)
# ---------------------------------------------------------------------------
#
# A fact sheet is the raw material the assistant's reason_about_dataset_content
# tool reasons over: everything about one dataset that carries inferential signal
# for a relationship/comparison/content question ("paired tomographic and segmented
# images", "the same sample imaged at different resolutions"), and nothing else.
#
# This deliberately does NOT reuse _build_embedding_text above. That function
# flattens sub-node properties into single aggregated lines (one merged
# "Rock/media type: ..." across every Sample, voxel_dims[0] picking just the first
# value found) because that is the right shape for embedding similarity — but it is
# the wrong shape here, since it discards exactly what a fact sheet needs: WHICH
# specific DigitalDatasets belong to WHICH specific Sample. "Does this sample have
# scans at two different resolutions?" is unanswerable from the aggregated form and
# trivially answerable from the edge-preserving form below.
#
# Fact sheets cache RAW MATERIAL ONLY, never a verdict: whether a dataset satisfies
# the relationship a given question describes is inherently query-dependent and is
# always judged live, per query, by the reasoning tool's own LLM pass.

# Per-node-type property allowlist. Deliberately narrow — license/numberOfFiles/
# fileTypes carry no inferential value for relational questions and only add bulk to
# every shortlisted fact sheet that reaches the model's context at query time.
_FACT_SHEET_FIELDS = {
    "samples": ("porousMediaType",),
    "digitalDatasets": ("voxelDimensions", "segmented"),
    "analysisDatasets": ("type", "segmented"),
}

# Bounds on one fact sheet's size. A shortlist of ~40 fact sheets goes into a single
# LLM call, so an unbounded description or sub-node list on one dataset can crowd out
# the other 39 (or blow the context window outright — see HANDOFF.md's embedding-vector
# incident for the same failure mode from a different angle). Truncation is never
# silent: every cap appends an explicit note with the real omitted count.
FACT_SHEET_MAX_NODES_PER_TYPE = 25
FACT_SHEET_DATASET_DESC_CHARS = 1500
FACT_SHEET_NODE_DESC_CHARS = 500
FACT_SHEET_ABSTRACT_CHARS = 800

# Fact sheets are far larger than the title+description blobs the other two embedding
# passes send, and the embedding endpoint enforces a limit on TOTAL characters per
# request, not on the number of items in it. Measured against the live TACC/SambaNova
# E5-Mistral-7B-Instruct endpoint: a request totalling ~14k characters succeeds, ~40k
# fails with a 500 whose body carries a per-item {"embedding": null, "error":
# "unexpected_error"} for most items — so a fixed batch size of 16 (fine for the other
# passes) fails on the very first fact-sheet batch. Batch by character budget instead,
# with headroom. Any single sheet larger than the budget still goes out on its own; a
# 21k-character sheet (the corpus maximum) embeds fine alone.
FACT_SHEET_EMBED_CHAR_BUDGET = 12_000

# Per-item cap on the text sent to the embedding model. E5-Mistral-7B-Instruct has a
# 4096-token limit and `check_embedding_ctx_length=False` is set on OpenAIEmbeddings (the
# TACC/LiteLLM endpoint expects raw strings), so LangChain does NOT truncate for us — an
# over-long sheet just 500s. Token density varies a lot between sheets (one dense with
# numeric voxel-dimension strings and identifiers tokenizes far denser than one that is
# mostly prose), so the cap is deliberately conservative: measured live, the 17 sheets that
# failed at full length ranged from 10.5k to 20.9k characters and ALL 17 embed successfully
# at 8k.
#
# This truncates ONLY the text handed to the embedding model. The stored `factSheetText` is
# always complete — BM25 (`datasetFactSheetFulltext`) indexes the whole thing, and the
# reasoning pass reads the whole thing. The embedding serves ranking alone, and a sheet's
# leading section (title, description, samples, scans) carries the ranking signal.
FACT_SHEET_EMBED_MAX_CHARS = 8_000


def _truncate(text: str | None, limit: int) -> str:
    """Trim `text` to `limit` chars, marking the cut explicitly so neither the LLM nor
    a downstream reader mistakes a truncated description for a complete one."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …[description truncated]"


def _fact_sheet_node(props: dict, fields: tuple[str, ...]) -> dict:
    """Project one sub-node down to the fact sheet's allowlisted fields plus its
    identifier/title/description. Empty values are dropped entirely — a fact sheet must
    never carry a blank field, both to save context and because an empty field read as
    an asserted value is exactly the kind of thing that produces an overclaimed answer."""
    node = {}
    for key in ("identifier", "title"):
        if props.get(key):
            node[key] = props[key]
    description = _truncate(props.get("description"), FACT_SHEET_NODE_DESC_CHARS)
    if description:
        node["description"] = description
    for key in fields:
        value = props.get(key)
        if value not in (None, "", []):
            node[key] = value
    return node


def _fact_sheet_nodes(nodes: list[dict], fields: tuple[str, ...]) -> tuple[list[dict], int]:
    """Project and cap a list of sub-nodes. Returns (kept, omitted_count)."""
    projected = [_fact_sheet_node(n, fields) for n in nodes]
    projected = [n for n in projected if n]
    kept = projected[:FACT_SHEET_MAX_NODES_PER_TYPE]
    return kept, len(projected) - len(kept)


def _build_fact_sheet(row: dict) -> dict:
    """Assemble one dataset's fact sheet from an edge-preserving profile row (see
    IndexBuilder._fetch_profile_row, which mirrors GraphStore.get_dataset_profile()'s
    query shape rather than adding a second aggregation query).

    Returns a plain JSON-serializable dict. Descriptions are code-stripped with the
    same _clean_description_for_embedding pass used for embeddings — DRP descriptions
    frequently embed Python/Matlab snippets and file-loading boilerplate, which is
    noise for both BM25 ranking and LLM reasoning.
    """
    dataset = dict(row.get("dataset") or {})
    description = dataset.get("description") or ""
    if STRIP_CODE_FROM_DESCRIPTIONS:
        description = _clean_description_for_embedding(description)

    samples, samples_omitted = _fact_sheet_nodes(
        row.get("samples") or [], _FACT_SHEET_FIELDS["samples"]
    )
    digital, digital_omitted = _fact_sheet_nodes(
        row.get("digitalDatasets") or [], _FACT_SHEET_FIELDS["digitalDatasets"]
    )
    analysis, analysis_omitted = _fact_sheet_nodes(
        row.get("analysisDatasets") or [], _FACT_SHEET_FIELDS["analysisDatasets"]
    )

    publications = []
    for pub in (row.get("relatedPublications") or [])[:FACT_SHEET_MAX_NODES_PER_TYPE]:
        entry = {}
        if pub.get("title"):
            entry["title"] = pub["title"]
        abstract = _truncate(pub.get("abstract"), FACT_SHEET_ABSTRACT_CHARS)
        if abstract:
            entry["abstract"] = abstract
        if entry:
            publications.append(entry)

    # Structural edges, resolved from identifiers to titles where possible: "which
    # DigitalDatasets belong to which Sample" is the whole point of the fact sheet, and
    # a bare identifier ("dd-1147") is not something an LLM can reason about or cite.
    label_by_id = {}
    for node in samples + digital + analysis:
        if node.get("identifier"):
            label_by_id[node["identifier"]] = node.get("title") or node["identifier"]

    analysis_by_digital: dict = {}
    for edge in row.get("digitalToAnalysisEdges") or []:
        analysis_by_digital.setdefault(edge["digitalDataset"], []).append(edge["analysisDataset"])

    pipelines = []
    linked_digital_ids = set()
    for edge in row.get("sampleToDigitalEdges") or []:
        sample_id, digital_id = edge["sample"], edge["digitalDataset"]
        linked_digital_ids.add(digital_id)
        for analysis_id in analysis_by_digital.get(digital_id) or [None]:
            chain = {
                "sample": label_by_id.get(sample_id, sample_id),
                "digitalDataset": label_by_id.get(digital_id, digital_id),
            }
            if analysis_id:
                chain["analysisDataset"] = label_by_id.get(analysis_id, analysis_id)
            pipelines.append(chain)

    # An analysis computed straight off a sample, with no intermediate scan.
    for edge in row.get("sampleToAnalysisEdges") or []:
        pipelines.append({
            "sample": label_by_id.get(edge["sample"], edge["sample"]),
            "analysisDataset": label_by_id.get(edge["analysisDataset"], edge["analysisDataset"]),
        })

    unlinked_digital = [
        n.get("title") or n.get("identifier")
        for n in digital
        if n.get("identifier") and n["identifier"] not in linked_digital_ids
    ]

    # Cap the chain list too, not just the node lists: a large multi-scan collection can
    # have hundreds of pipeline chains, and an uncapped list took one live dataset's fact
    # sheet to 129k characters — on its own more than half the model's whole context.
    unlinked_digital = [u for u in unlinked_digital if u]
    pipelines_omitted = max(0, len(pipelines) - FACT_SHEET_MAX_NODES_PER_TYPE)
    unlinked_omitted = max(0, len(unlinked_digital) - FACT_SHEET_MAX_NODES_PER_TYPE)

    sheet = {
        "datasetNumber": dataset.get("datasetNumber"),
        "title": dataset.get("title"),
        "doi": dataset.get("doi"),
        "description": _truncate(description, FACT_SHEET_DATASET_DESC_CHARS),
        "samples": samples,
        "digitalDatasets": digital,
        "analysisDatasets": analysis,
        "relatedPublications": publications,
        "pipelines": pipelines[:FACT_SHEET_MAX_NODES_PER_TYPE],
        "unlinkedDigitalDatasets": unlinked_digital[:FACT_SHEET_MAX_NODES_PER_TYPE],
    }
    omitted = {
        k: v
        for k, v in (
            ("samples", samples_omitted),
            ("digitalDatasets", digital_omitted),
            ("analysisDatasets", analysis_omitted),
            ("pipelines", pipelines_omitted),
            ("unlinkedDigitalDatasets", unlinked_omitted),
        )
        if v
    }
    if omitted:
        sheet["omittedCounts"] = omitted
    return sheet


def _batch_by_char_budget(texts: list[str], budget: int = FACT_SHEET_EMBED_CHAR_BUDGET) -> list[list[int]]:
    """Group indices into batches whose combined text stays under `budget` characters.

    Returns a list of index lists (indices, not texts, so the caller can map results back
    to their rows). A single text longer than the budget gets a batch to itself rather
    than being dropped or truncated — the embedding endpoint accepts a large single item,
    it just won't accept many of them at once.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_chars = 0
    for i, text in enumerate(texts):
        size = len(text)
        if current and current_chars + size > budget:
            batches.append(current)
            current, current_chars = [], 0
        current.append(i)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _render_fact_sheet_text(sheet: dict) -> str:
    """Render a fact sheet as prose. This exact text is what gets embedded
    (factSheetEmbedding), BM25-indexed (datasetFactSheetFulltext), AND handed to the
    reasoning LLM — one rendering, so what the ranker matched on is literally what the
    model then reasons over, with no second divergent representation to keep in sync.
    """
    lines = []
    header = f"Dataset {sheet.get('datasetNumber')}: {sheet.get('title') or 'Untitled'}"
    if sheet.get("doi"):
        header += f" (DOI: {sheet['doi']})"
    lines.append(header)
    if sheet.get("description"):
        lines.append(f"Description: {sheet['description']}")

    def _node_lines(nodes: list[dict], heading: str, omitted: int) -> None:
        if not nodes:
            return
        lines.append(f"\n{heading} ({len(nodes) + omitted}):")
        for node in nodes:
            label = node.get("title") or node.get("identifier") or "(untitled)"
            attrs = [
                f"{k}: {v}"
                for k, v in node.items()
                if k not in ("identifier", "title", "description")
            ]
            lines.append(f"- {label}" + (f" — {'; '.join(attrs)}" if attrs else ""))
            if node.get("description"):
                lines.append(f"  Description: {node['description']}")
        if omitted:
            lines.append(f"- ... and {omitted} more not listed here.")

    omitted_counts = sheet.get("omittedCounts") or {}
    _node_lines(sheet.get("samples") or [], "Samples", omitted_counts.get("samples", 0))
    _node_lines(sheet.get("digitalDatasets") or [], "Digital datasets (images/scans)",
                omitted_counts.get("digitalDatasets", 0))
    _node_lines(sheet.get("analysisDatasets") or [], "Analysis datasets",
                omitted_counts.get("analysisDatasets", 0))

    publications = sheet.get("relatedPublications") or []
    if publications:
        lines.append(f"\nRelated publications ({len(publications)}):")
        for pub in publications:
            lines.append(f"- {pub.get('title') or '(untitled)'}")
            if pub.get("abstract"):
                lines.append(f"  Abstract: {pub['abstract']}")

    pipelines = sheet.get("pipelines") or []
    if pipelines:
        lines.append("\nStructure (Sample -> Digital dataset -> Analysis dataset):")
        for chain in pipelines:
            parts = [chain.get("sample"), chain.get("digitalDataset"), chain.get("analysisDataset")]
            line = "- " + " -> ".join(p for p in parts if p)
            if chain.get("analysisDataset") and not chain.get("digitalDataset"):
                line += "  (analysis computed directly from the sample, no intermediate scan)"
            lines.append(line)
        if omitted_counts.get("pipelines"):
            lines.append(f"- ... and {omitted_counts['pipelines']} more chains not listed here.")

    unlinked = sheet.get("unlinkedDigitalDatasets") or []
    if unlinked:
        lines.append("\nDigital datasets with no recorded sample link:")
        lines.extend(f"- {name}" for name in unlinked)
        if omitted_counts.get("unlinkedDigitalDatasets"):
            lines.append(
                f"- ... and {omitted_counts['unlinkedDigitalDatasets']} more not listed here."
            )

    return "\n".join(lines)


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

    def run(self, skip_verify: bool = False, only: str | None = None,
            retry_missing: bool = False) -> None:
        datasets = self._fetch_datasets()
        if not datasets:
            print("No Dataset nodes found. Run scripts/load_graph.py first.")
            return

        if retry_missing:
            # Transient endpoint failures (504s, intermittent 500s) on a ~30-minute build
            # shouldn't require rebuilding everything to recover a handful of datasets.
            with self._driver.session() as session:
                missing = {
                    r["identifier"] for r in session.run(
                        "MATCH (d:Dataset) WHERE d.factSheetEmbedding IS NULL "
                        "RETURN d.identifier AS identifier"
                    )
                }
            datasets = [d for d in datasets if d["identifier"] in missing]
            print(f"--retry-missing: {len(datasets)} dataset(s) still lack a fact-sheet embedding.")
            if not datasets:
                print("Nothing to retry.")
                return

        dim = None
        if only in (None, "embeddings"):
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

        if only in (None, "fact-sheets"):
            self._build_fact_sheets(datasets, dim)

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
                    d.doi            AS doi,
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

        if len(vectors) != len(batch):
            # API returned a partial batch — fall back to one-at-a-time so nothing is skipped.
            print(f"\n  Warning: API returned {len(vectors)}/{len(batch)} vectors at offset {offset}. "
                  f"Retrying individually...")
            vectors = [self._embeddings.embed_documents([t])[0] for t in texts]

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

        if len(vectors) != len(batch):
            print(f"\n  Warning: API returned {len(vectors)}/{len(batch)} vectors at offset {offset}. "
                  f"Retrying individually...")
            vectors = [self._embeddings.embed_documents([t])[0] for t in texts]

        with self._driver.session() as session:
            for row, vector in zip(batch, vectors):
                session.run(
                    "MATCH (n {identifier: $id}) SET n:DatasetComponent, n.componentEmbedding = $vec",
                    id=row["identifier"],
                    vec=vector,
                )

        done = offset + len(batch)
        print(f"  [{done}/{total}] embedded", end="\r", flush=True)

    # ------------------------------------------------------------------
    # Fact sheets
    # ------------------------------------------------------------------

    # One bulk query per sub-node type / edge type, grouped by datasetNumber in Python.
    # Deliberately NOT the single multi-OPTIONAL-MATCH-then-collect query
    # GraphStore.get_dataset_profile() uses for one dataset at a time: four OPTIONAL
    # MATCHes cross-multiply before the collect (samples x digital x analysis x
    # publications), so a dataset with a few dozen of each produces six figures of
    # intermediate rows. That is tolerable once, interactively; run 184 times in a row it
    # dominates the whole build. The fact sheet needs the same edge-preserving SHAPE, not
    # the same query text — these bulk reads produce exactly that shape, and each one is
    # a flat relationship scan.
    _PROFILE_BULK_QUERIES = {
        "samples": """
            MATCH (s:Sample)-[:PART_OF]->(d:Dataset)
            RETURN d.datasetNumber AS datasetNumber,
                   s{.*, componentEmbedding: null} AS node
        """,
        "digitalDatasets": """
            MATCH (dd:DigitalDataset)-[:PART_OF]->(d:Dataset)
            RETURN d.datasetNumber AS datasetNumber,
                   dd{.*, componentEmbedding: null} AS node
        """,
        "analysisDatasets": """
            MATCH (ad:AnalysisDataset)-[:PART_OF]->(d:Dataset)
            RETURN d.datasetNumber AS datasetNumber,
                   ad{.*, componentEmbedding: null} AS node
        """,
        "relatedPublications": """
            MATCH (rp:RelatedPublication)-[:PART_OF]->(d:Dataset)
            RETURN d.datasetNumber AS datasetNumber, rp{.title, .abstract} AS node
        """,
    }

    # INPUT_FOR points CHILD -> PARENT ("was derived from"), the same direction as
    # PART_OF — see graph_store.py's schema docstring. Written the other way round these
    # match zero rows, which would leave every fact sheet with an empty structure section
    # and defeat the whole reason the fact sheet is edge-preserving.
    _PROFILE_EDGE_QUERIES = {
        "sampleToDigitalEdges": """
            MATCH (dd:DigitalDataset)-[:INPUT_FOR]->(s:Sample)
            MATCH (dd)-[:PART_OF]->(d:Dataset)
            RETURN d.datasetNumber AS datasetNumber,
                   {sample: s.identifier, digitalDataset: dd.identifier} AS edge
        """,
        "digitalToAnalysisEdges": """
            MATCH (ad:AnalysisDataset)-[:INPUT_FOR]->(dd:DigitalDataset)
            MATCH (ad)-[:PART_OF]->(d:Dataset)
            RETURN d.datasetNumber AS datasetNumber,
                   {digitalDataset: dd.identifier, analysisDataset: ad.identifier} AS edge
        """,
        # An analysis computed straight off a sample with no intermediate scan (55 such
        # edges live). Rare, but real structure — dropping it would leave those analyses
        # looking unattached.
        "sampleToAnalysisEdges": """
            MATCH (ad:AnalysisDataset)-[:INPUT_FOR]->(s:Sample)
            MATCH (ad)-[:PART_OF]->(d:Dataset)
            RETURN d.datasetNumber AS datasetNumber,
                   {sample: s.identifier, analysisDataset: ad.identifier} AS edge
        """,
    }

    def _fetch_profile_rows(self, datasets: list[dict]) -> dict:
        """Fetch every dataset's edge-preserving sub-node graph, keyed by datasetNumber.

        Embeddings are nulled out via map projection so 4096-float vectors never cross
        the wire (same guard as GraphStore.get_dataset_profile() — see HANDOFF.md for what
        happens when they do).
        """
        rows = {
            d["datasetNumber"]: {
                "dataset": {
                    "datasetNumber": d.get("datasetNumber"),
                    "title": d.get("title"),
                    "doi": d.get("doi"),
                    "description": d.get("description"),
                },
                "samples": [], "digitalDatasets": [], "analysisDatasets": [],
                "relatedPublications": [], "sampleToDigitalEdges": [],
                "digitalToAnalysisEdges": [], "sampleToAnalysisEdges": [],
            }
            for d in datasets if d.get("datasetNumber") is not None
        }

        with self._driver.session() as session:
            for key, query in self._PROFILE_BULK_QUERIES.items():
                for record in session.run(query):
                    row = rows.get(record["datasetNumber"])
                    node = dict(record["node"] or {})
                    # A map projection nulls a key rather than dropping it, so a node whose
                    # only properties were embeddings collapses to an all-null map — skip
                    # those rather than emit an empty fact-sheet entry.
                    if row is not None and any(v is not None for v in node.values()):
                        row[key].append(node)
            for key, query in self._PROFILE_EDGE_QUERIES.items():
                for record in session.run(query):
                    row = rows.get(record["datasetNumber"])
                    edge = dict(record["edge"] or {})
                    if row is not None and all(edge.values()):
                        row[key].append(edge)
        return rows

    def _build_fact_sheets(self, datasets: list[dict], dim: int | None) -> None:
        """Build, embed, and store one fact sheet per dataset, then create the fact-sheet
        vector + fulltext indexes used by GraphStore.rank_fact_sheets().

        Cheaper than the embedding steps above per dataset (a Cypher fetch + JSON
        serialization, no API call) apart from the one embedding pass at the end, and,
        like the embeddings, written with SET so re-running is safe.
        """
        print(f"\nBuilding fact sheets for {len(datasets)} datasets...")
        t0 = time.time()
        profile_rows = self._fetch_profile_rows(datasets)
        rendered: list[tuple[dict, dict, str]] = []  # (dataset row, sheet, text)
        for row in datasets:
            dataset_number = row.get("datasetNumber")
            if dataset_number is None:
                print(f"  Warning: dataset {row.get('identifier')!r} has no datasetNumber; skipped.")
                continue
            profile_row = profile_rows.get(dataset_number)
            if not profile_row:
                continue
            sheet = _build_fact_sheet(profile_row)
            rendered.append((row, sheet, _render_fact_sheet_text(sheet)))
        print(f"Assembled {len(rendered)} fact sheets in {time.time() - t0:.1f}s.")

        if not rendered:
            print("No fact sheets built; skipping fact-sheet indexes.")
            return

        # Embed a per-item-capped variant; the FULL text is what gets stored (see
        # FACT_SHEET_EMBED_MAX_CHARS).
        all_texts = [text[:FACT_SHEET_EMBED_MAX_CHARS] for _, _, text in rendered]
        oversized = sum(1 for _, _, text in rendered if len(text) > FACT_SHEET_EMBED_MAX_CHARS)
        if oversized:
            print(f"  ({oversized} sheet(s) capped at {FACT_SHEET_EMBED_MAX_CHARS} chars for "
                  f"embedding only; stored text and BM25 indexing use the full sheet.)")
        batches = _batch_by_char_budget(all_texts)
        print(f"Embedding fact sheets ({len(batches)} batches, "
              f"<= {FACT_SHEET_EMBED_CHAR_BUDGET} chars each)...")
        t1 = time.time()
        done = 0
        failed: list = []
        for indices in batches:
            texts = [all_texts[i] for i in indices]
            try:
                vectors = self._embeddings.embed_documents(texts)
                if len(vectors) != len(texts):
                    raise ValueError(f"API returned {len(vectors)}/{len(texts)} vectors")
            except Exception as exc:
                # Retry one at a time: the endpoint limits total characters per request,
                # so a batch failure usually still leaves every individual item embeddable.
                # A single item that fails on its own is skipped LOUDLY, not silently — the
                # rest of the build is still worth completing.
                print(f"\n  Batch of {len(texts)} failed ({str(exc)[:80]}); retrying individually...")
                vectors = []
                for i, text in zip(indices, texts):
                    try:
                        vectors.append(self._embeddings.embed_documents([text])[0])
                    except Exception as inner:
                        number = rendered[i][0].get("datasetNumber")
                        print(f"    Dataset {number} ({len(text)} chars) could not be embedded: "
                              f"{str(inner)[:100]}")
                        failed.append(number)
                        vectors.append(None)

            if dim is None:
                dim = next((len(v) for v in vectors if v), None)

            with self._driver.session() as session:
                for i, vector in zip(indices, vectors):
                    row, sheet, text = rendered[i]
                    if vector is None:
                        # Still store the fact sheet itself — it remains usable by the BM25
                        # fulltext index and by a title-restricted fetch; only this dataset's
                        # vector-ranking contribution is lost.
                        session.run(
                            "MATCH (d:Dataset {identifier: $id}) "
                            "SET d.factSheet = $sheet, d.factSheetText = $text",
                            id=row["identifier"], sheet=json.dumps(sheet, default=str), text=text,
                        )
                        continue
                    session.run(
                        "MATCH (d:Dataset {identifier: $id}) "
                        "SET d.factSheet = $sheet, d.factSheetText = $text, "
                        "    d.factSheetEmbedding = $vec",
                        id=row["identifier"],
                        sheet=json.dumps(sheet, default=str),
                        text=text,
                        vec=vector,
                    )
            done += len(indices)
            print(f"  [{done}/{len(rendered)}] embedded", end="\r", flush=True)

        print(f"\nEmbedded {len(rendered) - len(failed)}/{len(rendered)} fact sheets "
              f"in {time.time() - t1:.1f}s.")
        if failed:
            print(f"  WARNING: {len(failed)} dataset(s) stored without an embedding "
                  f"(BM25-only ranking for these): {failed}")

        if dim:
            self._create_fact_sheet_indexes(dim)
        else:
            print("Warning: could not determine embedding dimension; skipping fact-sheet vector index.")

    def _create_fact_sheet_indexes(self, dim: int) -> None:
        with self._driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX `factSheetEmbedding` IF NOT EXISTS
                FOR (d:Dataset) ON (d.factSheetEmbedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
        print(f"Vector index 'factSheetEmbedding' ready (dim={dim}).")
        with self._driver.session() as session:
            session.run("""
                CREATE FULLTEXT INDEX datasetFactSheetFulltext IF NOT EXISTS
                FOR (d:Dataset) ON EACH [d.factSheetText]
                OPTIONS { indexConfig: { `fulltext.analyzer`: 'english' } }
            """)
        print("Fulltext index 'datasetFactSheetFulltext' ready.")

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
        with self._driver.session() as session:
            session.run("""
                CREATE FULLTEXT INDEX datasetDescriptionFulltext IF NOT EXISTS
                FOR (d:Dataset) ON EACH [d.title, d.description]
                OPTIONS { indexConfig: { `fulltext.analyzer`: 'english' } }
            """)
        print("Fulltext index 'datasetDescriptionFulltext' ready.")

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

            ranked = store.rank_fact_sheets("paired tomographic and segmented images", top_k=5)
            if ranked:
                sheets = store.fetch_fact_sheets(ranked[:1])
                top_title = sheets[0].get("title", "?") if sheets else "?"
                print(f"  Fact-sheet ranking OK — got {len(ranked)} result(s). Top: {top_title!r}")
            else:
                print("  Warning: fact-sheet ranking returned 0 results (index may still be building).")
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
    parser.add_argument("--retry-missing", action="store_true",
                        help="Only process datasets that still lack a factSheetEmbedding — for "
                             "recovering from transient endpoint failures without a full rebuild. "
                             "Use with --only fact-sheets.")
    parser.add_argument("--only", choices=["embeddings", "fact-sheets"], default=None,
                        help="Build only one stage (default: both). 'fact-sheets' skips the "
                             "dataset/component embedding passes and rebuilds only "
                             "Dataset.factSheet + the fact-sheet vector/fulltext indexes.")
    args = parser.parse_args()

    builder = IndexBuilder(batch_size=args.batch_size)
    try:
        builder.run(skip_verify=args.skip_verify, only=args.only,
                    retry_missing=args.retry_missing)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
