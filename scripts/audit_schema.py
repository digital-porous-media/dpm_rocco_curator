"""
Audit the DRP Neo4j graph schema from local metadata JSON files.

Scans data/metadata/*.json to compute:
  - Node counts per label
  - % non-null coverage for every property
  - Distinct values for low-cardinality enum fields
  - Relationship type counts

Optionally queries a live Neo4j instance (--neo4j) to verify coverage
matches what was actually loaded into the graph.

Use --verify to cross-check that all 176 datasets are present in Neo4j
with correct property values, sub-node counts, and relationship counts.

Usage:
    # Offline audit from JSON files (no Neo4j required)
    python scripts/audit_schema.py --folder data/metadata/

    # Write the intern reference doc
    python scripts/audit_schema.py --folder data/metadata/ --output docs/neo4j_schema.md

    # Verify graph completeness + property correctness against live Neo4j
    python scripts/audit_schema.py --folder data/metadata/ --verify --output docs/neo4j_schema.md

    # Full audit + verify
    python scripts/audit_schema.py --folder data/metadata/ --neo4j --verify --output docs/neo4j_schema.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

DATASET_FIELDS = ["title", "description", "doi", "authors", "license", "publicationDate"]
SAMPLE_FIELDS = [
    "name", "porousMediaType", "porosity", "source", "geographicalLocation",
    "geographicOrigin", "grainSizeAvg", "grainSizeMin", "grainSizeMax", "grainSizeUnits",
    "collectionMethod", "onshoreOffshore", "depth", "waterDepth", "procedure",
    "equipment", "algorithmDescription",
]
DIGITAL_DATASET_FIELDS = [
    "voxelX", "voxelY", "voxelZ", "voxelUnits", "isSegmented",
    "imagingCenter", "imagingEquipmentAndModel", "imageFormat",
    "imageDimensions", "imageByteOrder", "dimensionality", "description",
]
ANALYSIS_DATASET_FIELDS = ["datasetType", "isSegmented", "description"]
RELATED_PUB_FIELDS = ["publicationTitle", "publicationAuthor", "publicationDescription",
                      "publicationLink", "publicationDateOfPublication"]
RELATED_SOFTWARE_FIELDS = ["softwareTitle", "softwareDescription", "softwareLink"]
RELATED_DATASET_FIELDS = ["datasetTitle", "datasetDescription", "datasetLink"]

# Low-cardinality fields to collect distinct values for
ENUM_FIELDS = {
    "Dataset": ["license"],
    "Sample": ["porousMediaType", "source", "onshoreOffshore"],
    "DigitalDataset": ["isSegmented", "voxelUnits"],
    "AnalysisDataset": ["datasetType", "isSegmented"],
}


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

class NodeStats:
    def __init__(self, label: str, fields: list[str]):
        self.label = label
        self.fields = fields
        self.total = 0
        self.present: dict[str, int] = defaultdict(int)
        self.enums: dict[str, set] = defaultdict(set)

    def record(self, value: dict[str, Any], enum_fields: list[str] = None):
        self.total += 1
        for f in self.fields:
            v = value.get(f)
            if v is not None and v != "" and v != []:
                self.present[f] += 1
        if enum_fields:
            for f in enum_fields:
                v = value.get(f)
                if v is not None and v != "":
                    self.enums[f].add(str(v))

    def coverage(self, field: str) -> float:
        if self.total == 0:
            return 0.0
        return self.present[field] / self.total * 100


# ---------------------------------------------------------------------------
# JSON audit
# ---------------------------------------------------------------------------

def audit_json(folder: Path) -> dict[str, NodeStats]:
    stats = {
        "Dataset": NodeStats("Dataset", DATASET_FIELDS),
        "Sample": NodeStats("Sample", SAMPLE_FIELDS),
        "DigitalDataset": NodeStats("DigitalDataset", DIGITAL_DATASET_FIELDS),
        "AnalysisDataset": NodeStats("AnalysisDataset", ANALYSIS_DATASET_FIELDS),
        "RelatedPublication": NodeStats("RelatedPublication", RELATED_PUB_FIELDS),
        "RelatedSoftware": NodeStats("RelatedSoftware", RELATED_SOFTWARE_FIELDS),
        "RelatedDataset": NodeStats("RelatedDataset", RELATED_DATASET_FIELDS),
    }
    rel_counts: dict[str, int] = defaultdict(int)

    files = sorted(folder.glob("*.json"))
    if not files:
        print(f"No JSON files found in {folder}")
        return stats

    for path in files:
        with open(path) as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  WARN: skipping {path.name} — {e}")
                continue

        nodes = meta.get("nodes", [])
        links = meta.get("links", [])

        # Find root node
        root = next((n for n in nodes if n.get("id") == "NODE_ROOT"), None)
        if root is None:
            continue

        root_val = root["value"]
        stats["Dataset"].record(root_val, ENUM_FIELDS["Dataset"])

        for pub in root_val.get("relatedPublications", []):
            stats["RelatedPublication"].record(pub)

        for sw in root_val.get("relatedSoftware", []):
            stats["RelatedSoftware"].record(sw)

        for rd in root_val.get("relatedDatasets", []):
            stats["RelatedDataset"].record(rd)

        for node in nodes:
            dtype = node.get("value", {}).get("dataType")
            if dtype == "sample":
                stats["Sample"].record(node["value"], ENUM_FIELDS["Sample"])
            elif dtype == "digital_dataset":
                stats["DigitalDataset"].record(node["value"], ENUM_FIELDS["DigitalDataset"])
            elif dtype == "analysis_data":
                stats["AnalysisDataset"].record(node["value"], ENUM_FIELDS["AnalysisDataset"])

        # Relationship types. Mirrors scripts/load_graph.py's _establish_connection,
        # which decides PART_OF vs INPUT_FOR purely on `source == root_id` — not on
        # node-type pattern matching — so classify links the same way here or the
        # sub-type buckets below undercount and "other" links get mis-bucketed.
        sample_pattern = re.compile(r"drp\.project\.sample")
        digital_pattern = re.compile(r"drp\.project\.digital_dataset")
        analysis_pattern = re.compile(r"drp\.project\.analysis_dataset")

        for link in links:
            src, tgt = link.get("source", ""), link.get("target", "")
            if "NODE_ROOT" in src:
                rel_counts["PART_OF: * → Dataset"] += 1
            elif sample_pattern.search(src) and digital_pattern.search(tgt):
                rel_counts["INPUT_FOR: Sample → DigitalDataset"] += 1
            elif digital_pattern.search(src) and analysis_pattern.search(tgt):
                rel_counts["INPUT_FOR: DigitalDataset → AnalysisDataset"] += 1
            else:
                rel_counts["INPUT_FOR (other)"] += 1

    stats["_rel_counts"] = rel_counts  # type: ignore
    return stats


# ---------------------------------------------------------------------------
# Neo4j audit (optional)
# ---------------------------------------------------------------------------

def audit_neo4j() -> dict[str, dict[str, float]]:
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    neo4j_coverage: dict[str, dict[str, float]] = {}

    # (neo4j label, display key used by print_coverage_table, actual Neo4j property name)
    # display key must match the raw JSON field name in *_FIELDS so the coverage
    # table can look it up; actual Neo4j property name is what load_graph.py wrote.
    checks = [
        ("Dataset", "title", "title"), ("Dataset", "description", "description"),
        ("Dataset", "doi", "doi"), ("Dataset", "authors", "authors"),
        ("Dataset", "license", "license"), ("Dataset", "publicationDate", "publicationDate"),
        ("Dataset", "descriptionEmbedding", "descriptionEmbedding"),

        ("Sample", "name", "title"), ("Sample", "porousMediaType", "porousMediaType"),
        ("Sample", "porosity", "porosity"), ("Sample", "source", "source"),
        ("Sample", "geographicalLocation", "location"),
        ("Sample", "geographicOrigin", "geographicOrigin"), ("Sample", "grainSizeAvg", "grainSizeAvg"),
        ("Sample", "grainSizeMin", "grainSizeMin"), ("Sample", "grainSizeMax", "grainSizeMax"),
        ("Sample", "grainSizeUnits", "grainSizeUnits"), ("Sample", "collectionMethod", "collectionMethod"),
        ("Sample", "onshoreOffshore", "onshoreOffshore"), ("Sample", "depth", "depth"),
        ("Sample", "waterDepth", "waterDepth"), ("Sample", "procedure", "procedure"),
        ("Sample", "equipment", "equipment"), ("Sample", "algorithmDescription", "algorithmDescription"),

        # voxelX/Y/Z/voxelUnits are combined into one dd.voxelDimensions string by
        # load_graph.py's _voxel_dim() — all four raw fields share the same coverage
        # number since there's no way to check them individually once merged.
        ("DigitalDataset", "voxelX", "voxelDimensions"), ("DigitalDataset", "voxelY", "voxelDimensions"),
        ("DigitalDataset", "voxelZ", "voxelDimensions"), ("DigitalDataset", "voxelUnits", "voxelDimensions"),
        # load_graph.py stores JSON "isSegmented" under the Neo4j property "segmented"
        ("DigitalDataset", "isSegmented", "segmented"),
        ("DigitalDataset", "imagingCenter", "imagingCenter"),
        ("DigitalDataset", "imagingEquipmentAndModel", "imagingEquipmentAndModel"),
        ("DigitalDataset", "imageFormat", "imageFormat"), ("DigitalDataset", "imageDimensions", "imageDimensions"),
        ("DigitalDataset", "imageByteOrder", "imageByteOrder"), ("DigitalDataset", "dimensionality", "dimensionality"),
        ("DigitalDataset", "description", "description"),

        ("AnalysisDataset", "datasetType", "type"),
        ("AnalysisDataset", "isSegmented", "segmented"),
        ("AnalysisDataset", "description", "description"),

        ("RelatedPublication", "publicationTitle", "title"),
        ("RelatedPublication", "publicationAuthor", "authors"),
        ("RelatedPublication", "publicationDescription", "abstract"),
        ("RelatedPublication", "publicationLink", "link"),
        ("RelatedPublication", "publicationDateOfPublication", "publicationDate"),

        ("RelatedSoftware", "softwareTitle", "title"),
        ("RelatedSoftware", "softwareDescription", "description"),
        ("RelatedSoftware", "softwareLink", "link"),

        ("RelatedDataset", "datasetTitle", "title"),
        ("RelatedDataset", "datasetDescription", "description"),
        ("RelatedDataset", "datasetLink", "link"),
    ]

    with driver.session() as session:
        for label, display_key, neo4j_prop in checks:
            result = session.run(
                f"MATCH (n:{label}) RETURN "
                f"count(n.{neo4j_prop}) * 100.0 / count(n) AS cov"
            )
            row = result.single()
            cov = round(row["cov"], 1) if row else 0.0
            neo4j_coverage.setdefault(label, {})[display_key] = cov

    driver.close()
    return neo4j_coverage


# ---------------------------------------------------------------------------
# Neo4j verification (--verify)
# ---------------------------------------------------------------------------

# PART_OF/INPUT_FOR-eligible node types, per scripts/load_graph.py:
#   - Sample, DigitalDataset, AnalysisDataset, RelatedPublication, RelatedSoftware,
#     RelatedDataset each get exactly one unconditional PART_OF → Dataset relationship
#     (lines 378, 415, 444, 314, 331, 346), so expected PART_OF = sum of their totals.
#   - INPUT_FOR relationships are created per JSON "links" entry (line 465), so the
#     expected count is derived from the links-based rel_counts computed by audit_json().
_PART_OF_NODE_TYPES = [
    "Sample", "DigitalDataset", "AnalysisDataset",
    "RelatedPublication", "RelatedSoftware", "RelatedDataset",
]


def verify_neo4j(folder: Path, stats: dict[str, NodeStats]) -> dict:
    """
    Cross-check graph completeness and property correctness against live Neo4j.

    `stats` is the dict returned by audit_json(folder) — used to derive expected
    sub-node and relationship counts from the *current* metadata instead of stale
    hardcoded values.

    Returns a dict with keys: missing, unexpected, mismatches, count_diffs, rel_diffs.
    """
    from neo4j import GraphDatabase

    expected_counts = {
        "Sample": stats["Sample"].total,
        "DigitalDataset": stats["DigitalDataset"].total,
        "AnalysisDataset": stats["AnalysisDataset"].total,
        "RelatedPublication": stats["RelatedPublication"].total,
    }
    rel_counts = stats.get("_rel_counts", {})
    expected_rel_part_of = sum(stats[label].total for label in _PART_OF_NODE_TYPES)
    expected_rel_input_for = (
        rel_counts.get("INPUT_FOR: Sample → DigitalDataset", 0)
        + rel_counts.get("INPUT_FOR: DigitalDataset → AnalysisDataset", 0)
        + rel_counts.get("INPUT_FOR (other)", 0)
    )

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    results = {
        "missing": [],       # dataset numbers in JSON but not in Neo4j
        "unexpected": [],    # dataset numbers in Neo4j but not in JSON folder
        "mismatches": [],    # (dataset_number, field, json_val, neo4j_val)
        "count_diffs": {},   # label → (expected, actual), only entries that differ
        "rel_diffs": {},     # rel_type → (expected, actual), only entries that differ
        "sub_node_counts": {},  # label → (expected, actual), all labels
        "rel_counts_verify": {},  # rel_type → (expected, actual), both rel types
        "total_neo4j": 0,
        "total_json": 0,
    }

    # --- Step 1: completeness diff ---
    json_nums = {
        int(re.findall(r"\d+", p.stem)[0])
        for p in folder.glob("*.json")
        if re.search(r"\d+", p.stem)
    }
    results["total_json"] = len(json_nums)

    with driver.session() as session:
        rows = session.run(
            "MATCH (d:Dataset) RETURN d.datasetNumber AS num, "
            "d.identifier AS id, d.title AS title, d.doi AS doi, "
            "d.description AS description, d.authors AS authors, "
            "d.publicationDate AS pubDate"
        ).data()

    neo4j_by_num = {r["num"]: r for r in rows}
    results["total_neo4j"] = len(neo4j_by_num)

    results["missing"] = sorted(json_nums - set(neo4j_by_num.keys()))
    results["unexpected"] = sorted(set(neo4j_by_num.keys()) - json_nums)

    # --- Step 2: property spot-check ---
    spot_fields = {
        "title": "title",
        "doi": "doi",
        "description": "description",
        "authors": "authors",
        "publicationDate": "publicationDate",
    }
    for path in sorted(folder.glob("*.json")):
        nums = re.findall(r"\d+", path.stem)
        if not nums:
            continue
        num = int(nums[0])
        if num not in neo4j_by_num:
            continue  # already flagged as missing
        with open(path) as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                continue
        root = next((n for n in meta["nodes"] if n.get("id") == "NODE_ROOT"), None)
        if not root:
            continue
        v = root["value"]
        import sys as _sys, importlib as _il
        _sys.path.insert(0, str(Path(__file__).parent))
        _lg = _il.import_module("load_graph")
        root_key = _lg._find_root(meta["nodes"])
        json_authors = _lg._author_names(meta["nodes"], root_key)
        json_vals = {
            "title": v.get("title"),
            "doi": v.get("doi"),
            "description": v.get("description"),
            "authors": json_authors,
            "publicationDate": v.get("publicationDate"),
        }
        neo_key_map = {"publicationDate": "pubDate"}
        neo_row = neo4j_by_num[num]
        for field, neo_key in spot_fields.items():
            j_val = json_vals.get(field)
            n_val = neo_row.get(neo_key_map.get(neo_key, neo_key))
            if j_val != n_val:
                results["mismatches"].append((num, field, j_val, n_val))

    # --- Step 3: sub-node count check ---
    with driver.session() as session:
        for label, expected in expected_counts.items():
            actual = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            results["sub_node_counts"][label] = (expected, actual)
            if actual != expected:
                results["count_diffs"][label] = (expected, actual)

    # --- Step 4: relationship count check ---
    with driver.session() as session:
        part_of = session.run("MATCH ()-[r:PART_OF]->() RETURN count(r) AS c").single()["c"]
        input_for = session.run("MATCH ()-[r:INPUT_FOR]->() RETURN count(r) AS c").single()["c"]
        results["rel_counts_verify"]["PART_OF"] = (expected_rel_part_of, part_of)
        results["rel_counts_verify"]["INPUT_FOR"] = (expected_rel_input_for, input_for)
        if part_of != expected_rel_part_of:
            results["rel_diffs"]["PART_OF"] = (expected_rel_part_of, part_of)
        if input_for != expected_rel_input_for:
            results["rel_diffs"]["INPUT_FOR"] = (expected_rel_input_for, input_for)

    driver.close()
    return results


def print_verify_results(v: dict):
    print(f"\n{'='*60}")
    print("  Graph Verification")
    print(f"{'='*60}")

    total_json = v["total_json"]
    total_neo = v["total_neo4j"]
    ok = "✓" if total_neo == total_json and not v["missing"] else "✗"
    print(f"  {ok} Datasets: {total_neo}/{total_json} loaded")

    if v["missing"]:
        print(f"  ✗ Missing from graph (DRP-N): {v['missing']}")
    if v["unexpected"]:
        print(f"  ✗ In graph but not in JSON folder: {v['unexpected']}")

    if v["mismatches"]:
        print(f"\n  Property mismatches ({len(v['mismatches'])}):")
        for num, field, j_val, n_val in v["mismatches"]:
            j_str = str(j_val)[:60] if j_val else "None"
            n_str = str(n_val)[:60] if n_val else "None"
            print(f"    DRP-{num} [{field}]  JSON: {j_str!r}  Neo4j: {n_str!r}")
    else:
        print("  ✓ No property mismatches (title, doi, description, authors, publicationDate)")

    print(f"\n  Sub-node counts:")
    for label, (expected, actual) in v["sub_node_counts"].items():
        if label in v["count_diffs"]:
            print(f"    ✗ {label:<20} expected {expected}, got {actual}")
        else:
            print(f"    ✓ {label:<20} {actual}")

    print(f"\n  Relationship counts:")
    for rel, (expected, actual) in v["rel_counts_verify"].items():
        if rel in v["rel_diffs"]:
            print(f"    ✗ {rel:<12} expected {expected}, got {actual}")
        else:
            print(f"    ✓ {rel:<12} {actual}")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_coverage_table(stats: dict[str, NodeStats], neo4j: dict | None = None):
    order = ["Dataset", "Sample", "DigitalDataset", "AnalysisDataset",
             "RelatedPublication", "RelatedSoftware", "RelatedDataset"]
    for label in order:
        s = stats[label]
        print(f"\n{'='*60}")
        print(f"  {label}  (n={s.total})")
        print(f"{'='*60}")
        print(f"  {'Property':<40} {'JSON%':>6}  {'Neo4j%':>7}")
        print(f"  {'-'*40} {'-'*6}  {'-'*7}")
        for f in s.fields:
            pct = s.coverage(f)
            neo_pct = ""
            if neo4j and label in neo4j:
                neo_val = neo4j[label].get(f)
                neo_pct = f"{neo_val:6.1f}%" if neo_val is not None else "      -"
            print(f"  {f:<40} {pct:5.1f}%  {neo_pct:>7}")

        if label in ENUM_FIELDS and s.enums:
            print()
            for f in ENUM_FIELDS[label]:
                vals = sorted(s.enums.get(f, set()))
                if vals:
                    print(f"  {f} values: {', '.join(vals)}")

    rel_counts = stats.get("_rel_counts", {})
    if rel_counts:
        print(f"\n{'='*60}")
        print("  Relationship Counts (from links arrays)")
        print(f"{'='*60}")
        for rel, cnt in sorted(rel_counts.items()):
            print(f"  {rel:<50} {cnt:>6}")


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def _pct_badge(pct: float) -> str:
    if pct >= 90:
        return f"{pct:.0f}%"
    elif pct >= 50:
        return f"{pct:.0f}% ⚠️"
    else:
        return f"{pct:.0f}% ❌"


def _neo4j_prop(raw_field: str, label: str) -> str:
    """Map raw JSON field name to the Neo4j property name used in load_graph.py."""
    mapping = {
        # Sample
        "name": "title",
        "geographicalLocation": "location",
        # RelatedPublication
        "publicationTitle": "title",
        "publicationAuthor": "authors",
        "publicationDescription": "abstract",
        "publicationLink": "link",
        "publicationDateOfPublication": "publicationDate",
        # RelatedSoftware
        "softwareTitle": "title",
        "softwareDescription": "description",
        "softwareLink": "link",
        # RelatedDataset
        "datasetTitle": "title",
        "datasetDescription": "description",
        "datasetLink": "link",
        # DigitalDataset
        "voxelX": "voxelDimensions (combined)",
        "voxelY": "voxelDimensions (combined)",
        "voxelZ": "voxelDimensions (combined)",
        "voxelUnits": "voxelDimensions (combined)",
        "isSegmented": "segmented",
        # AnalysisDataset
        "datasetType": "type",
    }
    return mapping.get(raw_field, raw_field)


def build_markdown(stats: dict[str, NodeStats], neo4j: dict | None = None) -> str:
    lines = []
    lines += [
        "# Neo4j Graph Schema — DRP Portal",
        "",
        "> Auto-generated by `scripts/audit_schema.py` from `data/metadata/`.  ",
        "> ⚠️ = 50–89% coverage (use `IS NOT NULL` guard in Cypher).  ",
        "> ❌ = <50% coverage (sparse; treat as optional).  ",
        "",
    ]

    # ------------------------------------------------------------------ nodes
    lines += ["## Node Labels", ""]

    node_sections = {
        "Dataset": {
            "description": "Root node. One per DRP project.",
            "fields": [
                ("identifier", "string", None, "Unique. Format: `NODE_ROOT+drp.project.published.DRP-N`"),
                ("datasetNumber", "int", None, "Integer N from projectId. Always set."),
                ("title", "string", "title", None),
                ("description", "string", "description", None),
                ("doi", "string", "doi", None),
                ("authors", "string", "authors", "Concatenated `First Last` names"),
                ("license", "string", "license", None),
                ("publicationDate", "string", "publicationDate", None),
                ("descriptionEmbedding", "float[]", None, "Set by `build_dataset_vector_index.py`. Used for semantic search."),
                ("llmKeywords", "string[]", None, "Planned; not yet populated."),
            ],
            "label": "Dataset",
            "raw_fields": DATASET_FIELDS,
        },
        "Sample": {
            "description": "Physical or synthetic porous media sample.",
            "fields": [
                ("identifier", "string", None, "Node id from JSON"),
                ("title", "string", "name", "Stored as `title` in Neo4j"),
                ("location", "string", "geographicalLocation", None),
                ("porousMediaType", "string", "porousMediaType", None),
                ("porosity", "float", "porosity", None),
                ("source", "string", "source", None),
                ("geographicOrigin", "string", "geographicOrigin", None),
                ("grainSizeAvg", "float", "grainSizeAvg", None),
                ("grainSizeMin", "float", "grainSizeMin", None),
                ("grainSizeMax", "float", "grainSizeMax", None),
                ("grainSizeUnits", "string", "grainSizeUnits", None),
                ("collectionMethod", "string", "collectionMethod", None),
                ("onshoreOffshore", "string", "onshoreOffshore", None),
                ("depth", "string", "depth", None),
                ("waterDepth", "string", "waterDepth", None),
                ("procedure", "string", "procedure", None),
                ("equipment", "string", "equipment", None),
                ("algorithmDescription", "string", "algorithmDescription", None),
                ("datasetNumber", "int", None, "Always set."),
            ],
            "label": "Sample",
            "raw_fields": SAMPLE_FIELDS,
        },
        "DigitalDataset": {
            "description": "3D imaging data (CT scan, micro-CT, etc.) derived from a Sample.",
            "fields": [
                ("identifier", "string", None, "Node id from JSON"),
                ("title", "string", None, "From node label"),
                ("voxelDimensions", "string", "voxelX/Y/Z+voxelUnits", "Combined: `X, Y, Z units (in µm): ...`"),
                ("description", "string", "description", None),
                ("segmented", "string", "isSegmented", "`yes` or `no`"),
                ("imagingCenter", "string", "imagingCenter", None),
                ("imagingEquipmentAndModel", "string", "imagingEquipmentAndModel", None),
                ("imageFormat", "string", "imageFormat", None),
                ("imageDimensions", "string", "imageDimensions", None),
                ("imageByteOrder", "string", "imageByteOrder", None),
                ("dimensionality", "string", "dimensionality", None),
                ("referencedSample", "string", "sample", "UUID of linked Sample node"),
                ("numberOfFiles", "int", None, "len(fileObjs). Always set."),
                ("fileTypes", "string[]", None, "Derived from file extensions."),
                ("datasetNumber", "int", None, "Always set."),
            ],
            "label": "DigitalDataset",
            "raw_fields": DIGITAL_DATASET_FIELDS,
        },
        "AnalysisDataset": {
            "description": "Computed result (simulation, measurement) derived from a DigitalDataset.",
            "fields": [
                ("identifier", "string", None, "Node id from JSON"),
                ("title", "string", None, "From node label"),
                ("description", "string", "description", None),
                ("segmented", "string", "isSegmented", "`yes` or `no`"),
                ("type", "string", "datasetType", "`simulation` or `measurement`"),
                ("referencedDigitalDataset", "string", "digitalDataset", "UUID of linked DigitalDataset"),
                ("referencedSample", "string", "sample", "UUID of linked Sample"),
                ("numberOfFiles", "int", None, "Always set."),
                ("fileTypes", "string[]", None, "Derived from file extensions."),
                ("datasetNumber", "int", None, "Always set."),
            ],
            "label": "AnalysisDataset",
            "raw_fields": ANALYSIS_DATASET_FIELDS,
        },
        "RelatedPublication": {
            "description": "Article written about or using the dataset. Linked to its parent Dataset via PART_OF.",
            "fields": [
                ("title", "string", "publicationTitle", None),
                ("authors", "string", "publicationAuthor", None),
                ("abstract", "string", "publicationDescription", None),
                ("link", "string", "publicationLink", "Publication URL (not DOI)"),
                ("publicationDate", "string", "publicationDateOfPublication", None),
                ("datasetNumber", "int", None, "Always set."),
            ],
            "label": "RelatedPublication",
            "raw_fields": RELATED_PUB_FIELDS,
        },
        "RelatedSoftware": {
            "description": "Software used to produce or analyze the dataset.",
            "fields": [
                ("title", "string", "softwareTitle", None),
                ("description", "string", "softwareDescription", None),
                ("link", "string", "softwareLink", None),
                ("datasetNumber", "int", None, "Always set."),
            ],
            "label": "RelatedSoftware",
            "raw_fields": RELATED_SOFTWARE_FIELDS,
        },
        "RelatedDataset": {
            "description": "External dataset referenced by this project.",
            "fields": [
                ("title", "string", "datasetTitle", None),
                ("description", "string", "datasetDescription", None),
                ("link", "string", "datasetLink", None),
                ("datasetNumber", "int", None, "Always set."),
            ],
            "label": "RelatedDataset",
            "raw_fields": RELATED_DATASET_FIELDS,
        },
    }

    for section_label, meta in node_sections.items():
        s = stats[section_label]
        lines += [f"### {section_label}  (n={s.total})", ""]
        lines.append(meta["description"])
        lines += [""]
        lines += [f"| Neo4j property | Type | Coverage | Notes |",
                  f"|---|---|---|---|"]
        for neo_prop, ptype, raw_field, notes in meta["fields"]:
            if raw_field and raw_field in s.fields:
                pct = s.coverage(raw_field)
                cov_str = _pct_badge(pct)
            else:
                cov_str = "100%" if neo_prop in ("identifier", "datasetNumber", "numberOfFiles", "fileTypes", "title") else "—"
            note_str = notes or ""
            lines.append(f"| `{neo_prop}` | {ptype} | {cov_str} | {note_str} |")

        # Enum values
        enum_fields_for_label = ENUM_FIELDS.get(section_label, [])
        for ef in enum_fields_for_label:
            vals = sorted(s.enums.get(ef, set()))
            if vals:
                neo_name = _neo4j_prop(ef, section_label)
                lines.append(f"| | | | **`{neo_name}` values:** {', '.join(f'`{v}`' for v in vals)} |")

        lines.append("")

    # ------------------------------------------------------------------ relationships
    lines += ["## Relationship Types", ""]
    lines += ["| Relationship | Direction | Meaning |",
              "|---|---|---|",
              "| `PART_OF` | `Sample → Dataset` | Sample belongs to this dataset project |",
              "| `PART_OF` | `DigitalDataset → Dataset` | Imaging data belongs to this project |",
              "| `PART_OF` | `AnalysisDataset → Dataset` | Analysis result belongs to this project |",
              "| `PART_OF` | `RelatedPublication → Dataset` | Publication linked to this project |",
              "| `PART_OF` | `RelatedSoftware → Dataset` | Software linked to this project |",
              "| `PART_OF` | `RelatedDataset → Dataset` | External dataset linked to this project |",
              "| `INPUT_FOR` | `Sample → DigitalDataset` | Sample was imaged to produce this digital dataset |",
              "| `INPUT_FOR` | `DigitalDataset → AnalysisDataset` | Image data was input for this analysis |",
              ""]

    # ------------------------------------------------------------------ vector index
    lines += [
        "## Vector Index",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Name | `datasetDescription` |",
        "| Node label | `Dataset` |",
        "| Property | `descriptionEmbedding` |",
        "| Dimensions | 4096 (E5-Mistral-7B-Instruct) |",
        "| Built by | `scripts/build_dataset_vector_index.py` |",
        "",
    ]

    # ------------------------------------------------------------------ cypher reference
    lines += [
        "## Cypher Quick Reference",
        "",
        "### Find datasets by porous media type",
        "```cypher",
        "MATCH (s:Sample)-[:PART_OF]->(d:Dataset)",
        "WHERE s.porousMediaType = 'sandstone'",
        "RETURN d.title, d.datasetNumber, d.doi",
        "ORDER BY d.datasetNumber",
        "```",
        "",
        "### Find segmented digital datasets",
        "```cypher",
        "MATCH (dd:DigitalDataset)-[:PART_OF]->(d:Dataset)",
        "WHERE dd.segmented = 'yes'",
        "RETURN d.title, dd.title, dd.voxelDimensions",
        "```",
        "",
        "### Find all datasets with a related publication",
        "```cypher",
        "MATCH (rp:RelatedPublication)-[:PART_OF]->(d:Dataset)",
        "RETURN d.title, d.datasetNumber, rp.title, rp.link",
        "ORDER BY d.datasetNumber",
        "```",
        "",
        "### Full graph for one dataset",
        "```cypher",
        "MATCH (n)-[:PART_OF]->(d:Dataset {datasetNumber: 1})",
        "RETURN d, n",
        "```",
        "",
        "### Semantic search (requires descriptionEmbedding populated)",
        "```cypher",
        "CALL db.index.vector.queryNodes('datasetDescription', 5, $embedding)",
        "YIELD node, score",
        "RETURN node.title, node.datasetNumber, score",
        "```",
        "",
        "### Filter with IS NOT NULL guard (for sparse fields)",
        "```cypher",
        "MATCH (s:Sample)-[:PART_OF]->(d:Dataset)",
        "WHERE s.porosity IS NOT NULL",
        "RETURN d.title, s.porosity",
        "ORDER BY s.porosity DESC",
        "```",
        "",
    ]

    # ------------------------------------------------------------------ coverage notes
    lines += [
        "## Coverage Notes",
        "",
        "Fields marked ⚠️ (50–89%) or ❌ (<50%) are sparsely populated. "
        "Always add an `IS NOT NULL` guard when filtering on these fields, "
        "otherwise queries silently return fewer results than expected.",
        "",
        "Fields guaranteed to be present on every node: "
        "`identifier`, `title`, `datasetNumber`, `numberOfFiles` (where applicable).",
        "",
        "Dataset `description` is present on nearly all nodes and is the primary "
        "field used for semantic vector search.",
        "",
    ]

    # ------------------------------------------------------------------ graceful degradation
    lines += [
        "## Graceful Degradation Tiers",
        "",
        "The assistant should **always attempt a query** regardless of field coverage. "
        "These tiers govern how the assistant communicates when results come back empty.",
        "",
        "### Tier 1 — High confidence (≥90% coverage)",
        "",
        "If a query on these fields returns no results, it is a genuine miss.",
        'Report confidently: *"No datasets match your criteria."*',
        "",
        "| Node | Fields |",
        "|---|---|",
        "| Dataset | `title`, `description`, `authors`, `publicationDate`, `doi` |",
        "| Sample | `porousMediaType`, `source` |",
        "| DigitalDataset | `segmented` |",
        "| AnalysisDataset | `type`, `segmented` |",
        "",
        "### Tier 2 — Low confidence on empty results (50–89% coverage)",
        "",
        "If a query on these fields returns no results, add a caveat: "
        "*\"This field is not present on all datasets — the absence of results may reflect "
        "missing metadata rather than no match.\"*",
        "",
        "| Node | Fields | Coverage |",
        "|---|---|---|",
        "| Sample | `location` | 89.8% |",
        "| DigitalDataset | `voxelDimensions` | 63–69% |",
        "| RelatedPublication | `abstract` | 68% |",
        "| RelatedPublication | `link` | 97.4% |",
        "",
        "### Tier 3 — Very sparse; always note sparsity (<50% coverage)",
        "",
        "Always mention sparsity alongside results (or their absence): "
        "*\"Note: this metadata is not consistently captured across all datasets.\"* "
        "For 0% fields, be explicit: "
        "*\"This field is not present in the current portal metadata.\"*",
        "",
        "| Node | Fields | Coverage |",
        "|---|---|---|",
        "| Sample | `porosity` | 26.7% |",
        "| Sample | `geographicOrigin` | 16.5% |",
        "| Sample | `grainSizeAvg/Min/Max` | ~15% |",
        "| Sample | `grainSizeUnits` | 0% |",
        "| Sample | `collectionMethod`, `onshoreOffshore`, `depth`, `waterDepth`, `procedure`, `equipment`, `algorithmDescription` | 0% |",
        "| DigitalDataset | `imagingCenter`, `imagingEquipmentAndModel`, `imageFormat`, `imageDimensions`, `imageByteOrder`, `dimensionality` | 0% |",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit DRP Neo4j schema coverage from metadata JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--folder", default="data/metadata/",
                        help="Directory of metadata JSON files (default: data/metadata/)")
    parser.add_argument("--neo4j", action="store_true",
                        help="Also query live Neo4j instance for coverage verification")
    parser.add_argument("--verify", action="store_true",
                        help="Verify graph completeness and property correctness against live Neo4j")
    parser.add_argument("--output", default=None,
                        help="Write Markdown reference doc to this path (e.g. docs/neo4j_schema.md)")
    args = parser.parse_args()

    folder = Path(args.folder)
    print(f"Auditing JSON files in: {folder.resolve()}")
    stats = audit_json(folder)

    neo4j_coverage = None
    if args.neo4j:
        print("Querying live Neo4j instance for coverage...")
        try:
            neo4j_coverage = audit_neo4j()
        except Exception as e:
            print(f"  WARN: Neo4j coverage audit failed — {e}")

    print_coverage_table(stats, neo4j=neo4j_coverage)

    if args.verify:
        print("\nVerifying graph against JSON files...")
        try:
            v = verify_neo4j(folder, stats)
            print_verify_results(v)
        except Exception as e:
            print(f"  WARN: Neo4j verification failed — {e}")

    if args.output:
        md = build_markdown(stats, neo4j=neo4j_coverage)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nReference doc written to: {out}")


if __name__ == "__main__":
    main()
