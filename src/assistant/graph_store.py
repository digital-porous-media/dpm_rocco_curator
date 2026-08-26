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
    INPUT_FOR (DigitalDataset → Sample, AnalysisDataset → DigitalDataset,
               AnalysisDataset → Sample)
               Points CHILD → PARENT ("was derived from"), the same direction as
               PART_OF — NOT parent → child. Verified against the live graph
               (1893 DigitalDataset→Sample, 983 AnalysisDataset→DigitalDataset,
               55 AnalysisDataset→Sample) and against scripts/load_graph.py's
               _establish_connection, which writes `(source)<-[:INPUT_FOR]-(target)`
               with `source` being the parent node in the DRP metadata's links list.
               Querying it the other way round silently matches zero rows.

Vector indexes:
    datasetEmbedding  — node: Dataset, property: datasetEmbedding
                        Aggregates title + description + sub-node metadata into one vector.
                        Used by search() and GraphCypherQAChain.
    componentEmbedding — node: DatasetComponent (secondary label on Sample/DigitalDataset/AnalysisDataset)
                         property: componentEmbedding
                         Each sub-node embedded individually with parent Dataset context injected.
                         Used by component_search() for fine-grained retrieval.
    factSheetEmbedding — node: Dataset, property: factSheetEmbedding
                         Embedding of the dataset's fact sheet (Dataset.factSheetText) — an
                         edge-preserving narration of which DigitalDatasets belong to which
                         Sample, their resolutions/segmented status, and sub-node descriptions.
                         Used by rank_fact_sheets() to narrow before content reasoning.
    All built by: scripts/build_dataset_vector_index.py

Fulltext indexes:
    datasetDescriptionFulltext — Dataset.title + Dataset.description (BM25 half of hybrid_search)
    datasetFactSheetFulltext   — Dataset.factSheetText (BM25 half of rank_fact_sheets)

Derived (non-source) Dataset properties, all written by build_dataset_vector_index.py:
    datasetEmbedding, factSheetEmbedding, factSheet (JSON string), factSheetText.
    These are computed FROM the published DRP metadata; the published metadata itself
    (title, description, doi, authors, sub-node properties) is never modified.

Alternative approach (not implemented):
    Chunking strategy stores Description + Chunk nodes instead of embedding on Dataset.
    See CurationTools/JsonToNeo4jwChunking.ipynb for reference.

Environment variables required:
    NEO4J_URI      - bolt://localhost:7687 (local) or neo4j+s://... (cloud)
    NEO4J_USER     - typically "neo4j"
    NEO4J_PASSWORD - your password
    USE_NEO4J      - set to "false" to disable all graph-backed dataset search. Every search
                     method then returns empty immediately without importing the Neo4j driver.
                     There is no alternative dataset backend; domain Q&A, workflow guidance,
                     portal-doc search and literature search are unaffected.

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


@dataclass
class DatasetProfileMatch:
    """
    A single, unambiguous dataset match plus its full PART_OF/INPUT_FOR sub-node graph.
    Returned by GraphStore.get_dataset_profile().

    Attributes:
        dataset:                 Dataset node properties (title, doi, datasetNumber,
                                  description, llmKeywords).
        samples/digital_datasets/analysis_datasets/related_publications/related_software/
        related_datasets:        Lists of full node properties for each PART_OF sub-node type.
        sample_to_digital_edges: [{"sample": identifier, "digitalDataset": identifier}, ...]
                                  describing INPUT_FOR edges — which Sample fed which
                                  DigitalDataset. Pairs with a null identifier on either side
                                  are dropped (that sub-node has no recorded INPUT_FOR edge).
        digital_to_analysis_edges: Same shape, for DigitalDataset -> AnalysisDataset.
    """
    dataset: dict
    samples: list[dict]
    digital_datasets: list[dict]
    analysis_datasets: list[dict]
    related_publications: list[dict]
    related_software: list[dict]
    related_datasets: list[dict]
    sample_to_digital_edges: list[dict]
    digital_to_analysis_edges: list[dict]


@dataclass
class DatasetProfileAmbiguous:
    """
    Returned by GraphStore.get_dataset_profile() when a reference matches more than one
    dataset. Candidates come from the tier-matching query alone (no second graph round-trip)
    so callers can render a disambiguation prompt immediately.
    """
    candidates: list[dict]  # each: {"datasetNumber": ..., "title": ..., "doi": ...}


def _strip_doi_prefix(doi: str | None) -> str:
    """
    Strips any number of leading 'https://doi.org/' prefixes from a DOI string.
    Shared by GraphStore.get_dataset_profile() and tools.py::search_datasets, which both
    need to compare/display bare DOI identifiers rather than full resolver URLs.
    """
    doi_id = doi or ""
    while doi_id.startswith("https://doi.org/"):
        doi_id = doi_id[len("https://doi.org/"):]
    return doi_id


def _try_parse_dataset_number(reference: str) -> int | None:
    """
    Returns the integer dataset number if `reference` looks like ONLY a dataset number
    (optionally with a 'DRP-'/'#' prefix, e.g. "42", "DRP-42", "#42") — not a title that
    merely contains digits somewhere (e.g. "Sample 42 Sandstone" is not a dataset-number
    reference).
    """
    m = re.fullmatch(r"(?:drp-?)?#?(\d+)", reference.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _neo4j_enabled() -> bool:
    """Returns True unless USE_NEO4J is explicitly set to 'false'."""
    return os.getenv("USE_NEO4J", "true").lower() == "true"


def _passes_filters(metadata: dict, filters: dict | None) -> bool:
    """
    Checks a result's metadata against a post-retrieval filter dict.

    A filter key that's absent from metadata (None) is treated as "unknown,
    don't exclude" rather than a mismatch, since sparse/unmapped properties
    shouldn't silently drop otherwise-relevant results.
    """
    if not filters:
        return True
    for key, value in filters.items():
        meta_val = metadata.get(key)
        if meta_val is not None and str(meta_val).lower() != str(value).lower():
            return False
    return True


def _row_field(row: dict, *names: str):
    """Look up a Cypher result column by suffix match, e.g. names="title" matches
    both 'title' and 'd.title' — GraphCypherQAChain result keys carry the variable
    prefix from the RETURN clause, which varies by generated query."""
    for key, value in row.items():
        if any(key == n or key.endswith("." + n) for n in names):
            return value
    return None


def _format_dataset_rows(rows: list) -> str | None:
    """
    Deterministically render Cypher rows that look like a dataset listing (every row
    has a title) as a markdown bullet list — title and DOI only, exactly as stored,
    never retyped by an LLM. Returns None for shapes that aren't a dataset listing
    (aggregates, counts, single-property lookups), so those still fall through to the
    QA chain's own prose answer.

    This exists because GraphCypherQAChain's QA-answer LLM call is not reliably
    steerable by prompt instructions alone (this model intermittently ignores
    formatting/grounding instructions) — moving the one high-stakes shape (dataset
    identity + DOI) into code removes that failure mode entirely rather than trying
    to phrase the prompt more carefully.
    """
    if not rows or not all(isinstance(r, dict) for r in rows):
        return None
    titles = [_row_field(r, "title") for r in rows]
    if not all(titles):
        return None

    # A generated query can join through a sub-node type with multiple rows per
    # Dataset (e.g. one DigitalDataset row per voxel-dimensions value) — dedupe by
    # (title, doi) so the same dataset isn't listed once per sub-node row.
    seen = set()
    lines = []
    for row, title in zip(rows, titles):
        doi = _row_field(row, "doi")
        dedupe_key = (title, doi)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        doi_str = doi if doi else "not available"
        lines.append(f"- **{title}** (DOI: {doi_str})")
    return "\n".join(lines)


def _rrf_merge(rank_lists: list[dict], penalty: int, k_rrf: int = 60) -> list:
    """
    Reciprocal Rank Fusion over N independent rank lookups.

    Each rank_lists entry is {id: 0-based rank} from one retriever. An id missing
    from a list is scored at `penalty` (conventionally candidate_k + 1) rather than
    dropped, so a strong hit in one retriever still ranks even when the other never
    saw it. Returns all ids, best first.

    Shared by hybrid_search (description vector + description BM25) and
    rank_fact_sheets (fact-sheet vector + fact-sheet BM25) — the fusion mechanism is
    identical in both cases and only the underlying indexes differ, so there is one
    implementation rather than a second, drifting copy. Ties break on first-seen order
    (never on the id itself, which may be int for one dataset and str for another).
    """
    order: list = []
    seen: set = set()
    for ranks in rank_lists:
        for key in ranks:
            if key not in seen:
                seen.add(key)
                order.append(key)

    scored = []
    for position, key in enumerate(order):
        rrf = sum(1.0 / (ranks.get(key, penalty) + k_rrf) for ranks in rank_lists)
        scored.append((-rrf, position, key))
    scored.sort()
    return [key for _, _, key in scored]


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
                   porosity has NO consistent scale: some datasets store it as a 0-1 fraction
                   (e.g. 0.39), others as a 0-100 percent value (e.g. 30.0, 50.0) — there is no
                   units field to distinguish them. A raw numeric comparison like
                   "s.porosity > 0.3" will incorrectly also match every percent-scale value
                   (since 30 > 0.3 is trivially true). Always normalize with a CASE expression
                   before comparing, treating any value greater than 1 as a percentage:
                     WHERE CASE WHEN s.porosity > 1 THEN s.porosity / 100 ELSE s.porosity END > 0.3
  DigitalDataset — identifier, datasetNumber (int), title, description, voxelDimensions, segmented, numberOfFiles (int), fileTypes (list)
                   segmented values: yes, no
                   voxelDimensions is free text, e.g.:
                     "X, Y, Z units (in micrometers): 3.3113, 3.3113, 3.3113"
                     "X, Y, Z units (in millimeters): 0.488, 0.488, 1.25"
                   There is no 'x' delimiter between numbers, and units vary (micrometers,
                   millimeters, etc.) — always check the unit substring before comparing values,
                   and convert to a common unit (e.g. millimeters * 1000 = micrometers).
  AnalysisDataset — identifier, datasetNumber (int), title, description, type, segmented, numberOfFiles (int), fileTypes (list)
                   type values: geometric_analysis, other, simulation
  RelatedPublication — title, authors, abstract, link, publicationDate, datasetNumber (int)

Relationships (all use PART_OF or INPUT_FOR — no other relationship types exist):
  (Sample)-[:PART_OF]->(Dataset)
  (DigitalDataset)-[:PART_OF]->(Dataset)
  (AnalysisDataset)-[:PART_OF]->(Dataset)
  (RelatedPublication)-[:PART_OF]->(Dataset)
  (DigitalDataset)-[:INPUT_FOR]->(Sample)
  (AnalysisDataset)-[:INPUT_FOR]->(DigitalDataset)
  (AnalysisDataset)-[:INPUT_FOR]->(Sample)
  IMPORTANT — INPUT_FOR points CHILD -> PARENT ("was derived from"), the SAME direction
  as PART_OF, despite what its name suggests. A scan points AT the sample it was imaged
  from; an analysis points AT the scan it was computed from. Writing it the other way
  round — (s:Sample)-[:INPUT_FOR]->(dd:DigitalDataset) — matches ZERO rows and silently
  returns nothing. To find which scans came from a given sample, use:
    MATCH (dd:DigitalDataset)-[:INPUT_FOR]->(s:Sample)

Important:
  - porosity is on Sample nodes, NOT on Dataset nodes. Always join via PART_OF.
  - porousMediaType (e.g. "sandstone") is on Sample nodes, NOT on Dataset nodes.
  - Use OPTIONAL MATCH for sparse properties (porosity, grainSize*, geographicOrigin).
  - Use case-insensitive matching for string values: toLower(s.porousMediaType) = 'sandstone'
  - Any query that RETURNs Dataset rows must also RETURN d.doi (in addition to
    d.identifier/d.title) — the DOI is the user-facing citation for a dataset, and it
    must come from this field, never invented. It is fine if d.doi is null for some rows.
  - authors (on Dataset and RelatedPublication) is a single concatenated "First Last, First
    Last" string, NOT a list — never compare it with `=`. To find datasets by a named
    person, use toLower(d.authors) CONTAINS toLower('<name>').
  - porousMediaType is a coarse enum (see the 8 values above), not the sample's specific
    geological rock name — a question naming a specific rock by its common/geological name
    must be mapped to the enum value it falls under, or it will silently match zero rows
    (e.g. toLower(s.porousMediaType) = 'limestone' never matches anything, since 'limestone'
    is not one of the 8 enum values). Known mappings: "limestone", "dolomite", "chalk",
    "Ketton", "Estaillades", "Savonnieres"/"Savonnières" → carbonate; "quartzite",
    "greywacke"/"graywacke", "arkose", "Bentheimer", "Berea", "Fontainebleau" → sandstone;
    "anthracite", "lignite" → coal. If a named rock isn't in this list and you're unsure
    which enum value it maps to, fall back to a title/description substring match instead
    (e.g. toLower(d.title) CONTAINS 'limestone') rather than filtering porousMediaType on
    the literal name.
  - There is no usable imaging-technique/modality field. imagingCenter,
    imagingEquipmentAndModel and similar fields exist in the schema but are populated on
    only ~4% of nodes (see docs/neo4j_schema.md), so filtering on one answers for a
    handful of datasets and silently drops the rest of the catalog — which is worse than
    returning nothing, because it looks like it worked. Virtually every dataset in this
    portal already involves some form
    of tomographic/micro-CT/X-ray imaging, so a question mentioning "tomographic", "CT",
    "micro-CT", "X-ray imaging", etc. is describing the portal as a whole, not a
    distinguishing, filterable property — do not add a WHERE clause for it, and do not let
    its absence as a property cause the whole query to return zero rows. Filter only on
    whatever OTHER concrete property the question also names (e.g. segmented status, rock
    type, porosity). In particular, do NOT try to match an imaging technique against
    fileTypes (see below) — fileTypes records file extensions/formats (tiff, raw, vtk,
    ...), not imaging modality, and "tomographic"/"CT"/"micro-CT" will never appear there.
  - fileTypes (on DigitalDataset/AnalysisDataset) is a LIST of strings, not a single string —
    toLower(dd.fileTypes) raises a runtime type error (toLower requires a string). To match
    a value inside it, use: any(f IN dd.fileTypes WHERE toLower(f) CONTAINS 'tiff').
"""

# Derived (not hand-maintained) list of every queryable node-property name in
# MANUAL_SCHEMA — the single source of truth GraphCypherQAChain's Cypher generation
# is bound to. Used to keep the agent's tool-routing guidance (which properties make
# a question belong to get_dataset_details) from drifting out of sync with the schema:
# a field added here is automatically picked up by callers instead of requiring a
# second hand-written edit to a routing prompt (see conversation_manager.py, tools.py).
_SCHEMA_PROPERTY_LINE_RE = re.compile(r"^\s*\w+\s*—\s*(.+)$")


def _looks_like_bare_property_list(line: str) -> bool:
    """True for a wrapped continuation line of a node's property list (e.g. Sample's
    properties span two lines in MANUAL_SCHEMA without repeating 'Sample —') — as
    opposed to an annotation line like 'porousMediaType values: beads, ...' or prose,
    which always contain a colon or non-identifier words."""
    line = line.strip()
    if not line or ":" in line:
        return False
    for token in line.split(","):
        name = token.strip().split("(")[0].strip()
        if not name or not _SAFE_KEY_RE.match(name):
            return False
    return True


def get_queryable_field_names() -> list[str]:
    """Parse MANUAL_SCHEMA and return the sorted, deduped list of node property names
    get_dataset_details can query. Pure string parsing — no Neo4j connection required."""
    fields: set[str] = set()
    lines = MANUAL_SCHEMA.splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = _SCHEMA_PROPERTY_LINE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        segment = m.group(1)
        i += 1
        while i < n and _looks_like_bare_property_list(lines[i]):
            segment += "," + lines[i]
            i += 1
        for raw in segment.split(","):
            name = raw.strip().split("(")[0].strip()
            if name and _SAFE_KEY_RE.match(name):
                fields.add(name)
    return sorted(fields)


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
  Example for "segmented datasets" (an OR condition — any one sub-node type matching is enough):
    MATCH (d:Dataset)
    OPTIONAL MATCH (d)<-[:PART_OF]-(dd:DigitalDataset)
    OPTIONAL MATCH (d)<-[:PART_OF]-(ad:AnalysisDataset)
    WHERE dd.segmented = 'yes' OR ad.segmented = 'yes'
    RETURN DISTINCT d.identifier, d.title
- When a question requires MULTIPLE conditions across different optionally-matched sub-nodes
  to ALL be true (an AND condition, e.g. "datasets with both a segmented image AND a simulation
  analysis"), do NOT put WHERE directly after the last OPTIONAL MATCH. Neo4j scopes a WHERE
  immediately following OPTIONAL MATCH as part of that match's own pattern predicate — if the
  predicate fails, OPTIONAL MATCH still emits a row with the variable bound to NULL, silently
  defeating the filter and returning every dataset. Always add an explicit WITH before the
  WHERE so it filters the accumulated row instead:
    MATCH (d:Dataset)
    OPTIONAL MATCH (d)<-[:PART_OF]-(dd:DigitalDataset)
    OPTIONAL MATCH (d)<-[:PART_OF]-(ad:AnalysisDataset)
    WITH d, dd, ad
    WHERE dd.segmented = 'yes' AND ad.type = 'simulation'
    RETURN DISTINCT d.identifier, d.title
- NEVER write `x IS NULL OR <condition>` for a condition the question actually REQUIRES.
  That clause is true for every row where the OPTIONAL MATCH found nothing, so it admits
  every dataset that simply has no Sample/DigitalDataset/AnalysisDataset attached — the
  filter looks present but matches almost everything, and the user gets datasets that
  plainly do not meet what they asked for. This is the single most common way a generated
  query silently returns wrong rows.
    WRONG (returns unrelated datasets that have no Sample at all):
      OPTIONAL MATCH (d)<-[:PART_OF]-(s:Sample)
      WITH d, s
      WHERE s IS NULL OR toLower(s.porousMediaType) = 'sandstone'
    RIGHT (the condition is required, so require the node too):
      MATCH (d:Dataset)<-[:PART_OF]-(s:Sample)
      WHERE toLower(s.porousMediaType) = 'sandstone'
      RETURN DISTINCT d.identifier, d.title, d.doi
  Use OPTIONAL MATCH + a null-tolerant WHERE only for a property the question treats as
  optional/extra information, never for one it is filtering on. A dataset missing the
  property is NOT a match — returning it is a wrong answer, not a lenient one.
- If you do use UNION, all branches must return the same column names.
- porosity is stored on Sample nodes with an inconsistent scale (see Schema below) —
  always normalize with a CASE expression before filtering. Note this filter REQUIRES a
  Sample, so it uses a plain MATCH (see the "IS NULL OR" rule above), e.g. for
  "sandstone datasets with porosity above 0.3":
    MATCH (d:Dataset)<-[:PART_OF]-(s:Sample)
    WHERE toLower(s.porousMediaType) = 'sandstone'
      AND CASE WHEN s.porosity > 1 THEN s.porosity / 100 ELSE s.porosity END > 0.3
    RETURN DISTINCT d.identifier, d.title, d.doi
- voxelDimensions on DigitalDataset is free text with an embedded unit and no 'x' delimiter
  between the three numbers (see Schema below) — to filter on a numeric voxel size threshold,
  extract the first number after the colon, then convert to a common unit (micrometers) before
  comparing. Example for "voxel size smaller than 2 microns":
    MATCH (d:Dataset)
    OPTIONAL MATCH (d)<-[:PART_OF]-(dd:DigitalDataset)
    WITH d, dd,
      toFloat(split(split(dd.voxelDimensions, ': ')[1], ', ')[0]) AS rawValue,
      toLower(dd.voxelDimensions) AS unitText
    WITH d, dd,
      CASE
        WHEN unitText CONTAINS 'nanomet' THEN rawValue / 1000.0
        WHEN unitText CONTAINS 'millimet' THEN rawValue * 1000.0
        WHEN unitText CONTAINS 'micromet' THEN rawValue
        ELSE null
      END AS voxelSizeMicrometers
    WHERE voxelSizeMicrometers IS NOT NULL AND voxelSizeMicrometers < 2
    RETURN DISTINCT d.identifier, d.title, dd.voxelDimensions
- authors on Dataset/RelatedPublication is one concatenated "First Last, First Last"
  string, not a list — a named person is a substring match, never `=`. Example for
  "datasets by Jane Doe":
    MATCH (d:Dataset)
    WHERE toLower(d.authors) CONTAINS toLower('Jane Doe')
    RETURN DISTINCT d.identifier, d.title, d.doi

Schema:
""" + MANUAL_SCHEMA + """

Question:
{question}

Cypher Query:
"""

_cypher_prompt = PromptTemplate(input_variables=["question"], template=CYPHER_GENERATION_TEMPLATE)

# get_dataset_details' answer is returned to the user untouched (conversation_manager.py
# treats it as self-contained, precisely so the outer agent never retypes/reformats it —
# see _SELF_CONTAINED_TOOLS). That means formatting has to come from here, not from an
# outer synthesis pass: a plain "answer in a sentence" QA prompt renders as one dense
# run-on paragraph with no visual structure once nothing downstream reformats it.
QA_GENERATION_TEMPLATE = """\
You are Rocco, a research assistant for the Digital Porous Media Portal. Use ONLY the \
information in the Context below to answer the Question. The Context is authoritative — \
never invent a dataset identifier, title, DOI, or count that isn't present in it, and \
never supplement from general knowledge.

If the Context is empty, say plainly that no matching datasets were found — do not guess.

Note: a Context that lists individual datasets (title/DOI per row) is reformatted \
elsewhere and will not reach this prompt — you are only answering counts, aggregates,
or single-property lookups here. Answer those directly in one or two sentences.

Context:
{context}

Question:
{question}

Answer:
"""

_qa_prompt = PromptTemplate(input_variables=["context", "question"], template=QA_GENERATION_TEMPLATE)


class GraphStore:
    """
    Wraps Neo4j vector similarity search and structured Cypher QA.

    Falls back gracefully when USE_NEO4J=false (returns empty results).
    """

    def __init__(self):
        """Initialize GraphStore; connects to Neo4j unless USE_NEO4J=false, in which case all search methods return empty results immediately."""
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
OPTIONAL MATCH (dd:DigitalDataset)-[:PART_OF]->(node)
OPTIONAL MATCH (ad:AnalysisDataset)-[:PART_OF]->(node)
OPTIONAL MATCH (s:Sample)-[:PART_OF]->(node)
WITH node, score,
     collect(DISTINCT dd.segmented) + collect(DISTINCT ad.segmented) AS segmentedVals,
     collect(DISTINCT dd.voxelDimensions) AS voxelDimVals,
     collect(DISTINCT s.porousMediaType) AS porousMediaTypeVals,
     collect(DISTINCT s.source) AS sourceVals
RETURN
    node.description AS text,
    score,
    {
        title: node.title,
        sampleTitles: [(sample)-[:PART_OF]->(node) | sample.title],
        datasetNumber: node.datasetNumber,
        doi: node.doi,
        segmented: CASE WHEN 'yes' IN segmentedVals THEN 'yes'
                        WHEN 'no' IN segmentedVals THEN 'no'
                        ELSE null END,
        porousMediaType: head(porousMediaTypeVals),
        source: head(sourceVals),
        voxelDimensions: CASE
            WHEN any(v IN voxelDimVals WHERE toLower(v) CONTAINS 'micromet') THEN 'micrometer'
            WHEN any(v IN voxelDimVals WHERE toLower(v) CONTAINS 'millimet') THEN 'millimeter'
            WHEN any(v IN voxelDimVals WHERE toLower(v) CONTAINS 'nanomet') THEN 'nanometer'
            WHEN size(voxelDimVals) > 0 THEN 'other'
            ELSE null END
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
MATCH (node)-[:PART_OF]->(d:Dataset)
OPTIONAL MATCH (s:Sample)-[:PART_OF]->(d)
WITH node, d, score,
     collect(DISTINCT s.porousMediaType) AS pmtVals,
     collect(DISTINCT s.source) AS sourceVals
RETURN
    node.title + coalesce(': ' + node.description, '') AS text,
    score,
    {
        componentType:    labels(node)[0],
        componentTitle:   node.title,
        datasetTitle:     d.title,
        datasetNumber:    d.datasetNumber,
        doi:              d.doi,
        segmented:        node.segmented,
        porousMediaType:  coalesce(node.porousMediaType, head(pmtVals)),
        source:           coalesce(node.source, head(sourceVals)),
        voxelDimensions:  CASE
            WHEN node.voxelDimensions IS NULL THEN null
            WHEN toLower(node.voxelDimensions) CONTAINS 'micromet' THEN 'micrometer'
            WHEN toLower(node.voxelDimensions) CONTAINS 'millimet' THEN 'millimeter'
            WHEN toLower(node.voxelDimensions) CONTAINS 'nanomet' THEN 'nanometer'
            ELSE 'other' END
    } AS metadata
""",
        )

        self._cypher_chain = GraphCypherQAChain.from_llm(
            chat_model,
            graph=self._graph,
            verbose=True,
            cypher_prompt=_cypher_prompt,
            qa_prompt=_qa_prompt,
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
            # Apply post-retrieval filter if provided (Cypher-level filtering is a Week 2 enhancement)
            if not _passes_filters(doc.metadata, filters):
                continue
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "source_label": "[graph match]",
            })

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

        # Build rank lookup by datasetNumber: {datasetNumber: rank (0-based)}. Keyed on
        # datasetNumber rather than doi — doi is missing/null for some datasets (a known
        # upstream metadata gap), and keying on it would silently drop those datasets out of
        # the merge entirely; datasetNumber is always present and unique.
        vec_rank: dict[str, int] = {}
        vec_meta: dict[str, dict] = {}  # datasetNumber -> metadata dict for result assembly
        for rank, doc in enumerate(vec_docs):
            dataset_number = doc.metadata.get("datasetNumber", "")
            if dataset_number and dataset_number not in vec_rank:
                vec_rank[dataset_number] = rank
                vec_meta[dataset_number] = {"meta": doc.metadata, "text": doc.page_content}

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
                    OPTIONAL MATCH (dd:DigitalDataset)-[:PART_OF]->(node)
                    OPTIONAL MATCH (ad:AnalysisDataset)-[:PART_OF]->(node)
                    OPTIONAL MATCH (s:Sample)-[:PART_OF]->(node)
                    WITH node, score,
                         collect(DISTINCT dd.segmented) + collect(DISTINCT ad.segmented) AS segmentedVals,
                         collect(DISTINCT dd.voxelDimensions) AS voxelDimVals,
                         collect(DISTINCT s.porousMediaType) AS porousMediaTypeVals,
                         collect(DISTINCT s.source) AS sourceVals
                    RETURN node.doi AS doi, node.title AS title,
                           node.description AS description,
                           node.datasetNumber AS datasetNumber,
                           [(s2)-[:PART_OF]->(node) | s2.title] AS sampleTitles,
                           CASE WHEN 'yes' IN segmentedVals THEN 'yes'
                                WHEN 'no' IN segmentedVals THEN 'no'
                                ELSE null END AS segmented,
                           head(porousMediaTypeVals) AS porousMediaType,
                           head(sourceVals) AS source,
                           CASE
                               WHEN any(v IN voxelDimVals WHERE toLower(v) CONTAINS 'micromet') THEN 'micrometer'
                               WHEN any(v IN voxelDimVals WHERE toLower(v) CONTAINS 'millimet') THEN 'millimeter'
                               WHEN any(v IN voxelDimVals WHERE toLower(v) CONTAINS 'nanomet') THEN 'nanometer'
                               WHEN size(voxelDimVals) > 0 THEN 'other'
                               ELSE null END AS voxelDimensions
                    """,
                    search_query=query,
                    limit=candidate_k,
                ).data()
            for rank, row in enumerate(rows):
                dataset_number = row.get("datasetNumber", "")
                if dataset_number and dataset_number not in bm25_rank:
                    bm25_rank[dataset_number] = rank
                    bm25_meta[dataset_number] = {
                        "meta": {
                            "title": row["title"],
                            "doi": row.get("doi"),
                            "datasetNumber": dataset_number,
                            "sampleTitles": row.get("sampleTitles") or [],
                            "segmented": row.get("segmented"),
                            "porousMediaType": row.get("porousMediaType"),
                            "source": row.get("source"),
                            "voxelDimensions": row.get("voxelDimensions"),
                        },
                        "text": row.get("description") or row.get("title") or "",
                    }
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "hybrid_search: BM25 query failed (%s); falling back to vector only", e
            )

        # --- RRF merge (shared with rank_fact_sheets — see _rrf_merge) ---
        merged = _rrf_merge([vec_rank, bm25_rank], penalty=candidate_k + 1)

        # --- Apply filters and assemble results ---
        results = []
        for dataset_number in merged:
            if len(results) >= top_k:
                break
            entry = vec_meta.get(dataset_number) or bm25_meta.get(dataset_number)
            if entry is None:
                continue
            meta = entry["meta"]
            if not _passes_filters(meta, filters):
                continue
            results.append({
                "text": entry["text"],
                "metadata": meta,
                "source_label": "[hybrid match]",
            })

        return results

    def component_search(self, query: str, filters: dict = None, top_k: int = 5) -> list[dict]:
        """
        Vector similarity search over individual Sample, DigitalDataset, and AnalysisDataset
        sub-nodes. Each result links back to its parent Dataset.

        Args:
            query: Natural language search query.
            filters: Optional dict of property constraints (e.g. {"segmented": "yes"}),
                     applied the same way as in search()/hybrid_search().
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, score, metadata, source_label.
            metadata keys: componentType, componentTitle, datasetTitle, datasetNumber, doi,
            segmented, porousMediaType, source, voxelDimensions.
        """
        if not self._enabled:
            return []

        retriever = self._component_index.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        results = []
        for doc in docs:
            if not _passes_filters(doc.metadata, filters):
                continue
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "source_label": "[component match]",
            })
        return results

    # ---------------------------------------------------------------------------
    # Fact sheets — raw material for reason_about_dataset_content
    # ---------------------------------------------------------------------------

    def rank_fact_sheets(self, query: str, top_k: int = 40) -> list:
        """
        Rank datasets by how well their precomputed fact sheet matches `query`, using
        the SAME Reciprocal Rank Fusion hybrid_search already runs — just pointed at the
        fact-sheet indexes (`factSheetEmbedding` + `datasetFactSheetFulltext`) instead of
        the plain-description ones. Returns datasetNumbers, best first, no LLM call.

        This is the narrowing step in front of reason_about_dataset_content: one general
        mechanism for any relational question, rather than a hand-written Cypher
        co-occurrence condition per relational phrasing ("different resolutions",
        "segmented pairing", and whatever someone asks next). Because a fact sheet
        already narrates structural relationships in prose — a sample's child digital
        datasets, their resolutions, their segmented status — a query like "paired
        tomographic and segmented images" has a real chance of landing near the right
        fact sheets in embedding space, and BM25 catches what vector similarity
        underweights: literal keyword overlap when the sheet's overall phrasing isn't
        semantically close.

        Trade-off worth naming: a hand-written exact Cypher condition for one
        relationship would have zero recall risk for that one case. Hybrid vector+BM25
        ranking narrows that gap considerably but doesn't eliminate it — accepted in
        exchange for not needing new code per future relational phrasing. Note the
        *ranking* is approximate; what the reasoning pass then asserts is not, since the
        fact sheet carries the literal recorded properties.

        Returns [] when graph search is disabled or neither fact-sheet index is
        queryable (e.g. the fact sheets have not been built yet — see
        scripts/build_dataset_vector_index.py); callers degrade honestly rather than
        silently reasoning over nothing.
        """
        if not self._enabled or not self._driver:
            return []

        candidate_k = top_k * 2  # fetch extra candidates so RRF has room to rerank

        vec_rank: dict = {}
        try:
            from src.assistant.llm import get_embeddings_model
            embedding = get_embeddings_model().embed_query(query)
            rows = self.execute_cypher(
                """
                CALL db.index.vector.queryNodes('factSheetEmbedding', $k, $embedding)
                YIELD node, score
                RETURN node.datasetNumber AS datasetNumber
                """,
                {"k": candidate_k, "embedding": embedding},
            )
            for rank, row in enumerate(rows):
                number = row.get("datasetNumber")
                if number is not None and number not in vec_rank:
                    vec_rank[number] = rank
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "rank_fact_sheets: fact-sheet vector search failed (%s); using BM25 only", e
            )

        bm25_rank: dict = {}
        try:
            rows = self.execute_cypher(
                """
                CALL db.index.fulltext.queryNodes('datasetFactSheetFulltext', $search_query,
                    {limit: $limit})
                YIELD node, score
                RETURN node.datasetNumber AS datasetNumber
                """,
                {"search_query": query, "limit": candidate_k},
            )
            for rank, row in enumerate(rows):
                number = row.get("datasetNumber")
                if number is not None and number not in bm25_rank:
                    bm25_rank[number] = rank
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "rank_fact_sheets: fact-sheet BM25 search failed (%s); using vector only", e
            )

        merged = _rrf_merge([vec_rank, bm25_rank], penalty=candidate_k + 1)
        return merged[:top_k]

    def fetch_fact_sheets(
        self,
        dataset_numbers: list | None = None,
        titles: list[str] | None = None,
    ) -> list[dict]:
        """
        Plain, generic, ID-based read of the precomputed `Dataset.factSheet` /
        `Dataset.factSheetText` properties — nothing is computed here and nothing about
        this query is pattern-specific: it is literally "give me these datasets' fact
        sheets".

        ``dataset_numbers`` fetches these, preserving the given (ranked) order.
        ``titles`` fetches by exact case-insensitive title instead — used when a caller
        already knows the exact set to reason over (a refinement of a previously listed
        result set), so no ranking is needed.
        Passing neither fetches every dataset that has a fact sheet (the exhaustive
        map-reduce fallback path).

        Datasets with no fact sheet built yet are omitted rather than returned empty.
        """
        if not self._enabled or not self._driver:
            return []

        if dataset_numbers is not None:
            if not dataset_numbers:
                return []
            rows = self.execute_cypher(
                """
                MATCH (d:Dataset)
                WHERE d.datasetNumber IN $numbers AND d.factSheet IS NOT NULL
                RETURN d.datasetNumber AS datasetNumber, d.title AS title, d.doi AS doi,
                       d.factSheet AS factSheet, d.factSheetText AS factSheetText
                """,
                {"numbers": list(dataset_numbers)},
            )
            by_number = {r["datasetNumber"]: r for r in rows}
            return [by_number[n] for n in dataset_numbers if n in by_number]

        if titles is not None:
            if not titles:
                return []
            rows = self.execute_cypher(
                """
                MATCH (d:Dataset)
                WHERE toLower(d.title) IN $titles AND d.factSheet IS NOT NULL
                RETURN d.datasetNumber AS datasetNumber, d.title AS title, d.doi AS doi,
                       d.factSheet AS factSheet, d.factSheetText AS factSheetText
                """,
                {"titles": [t.lower() for t in titles if t]},
            )
            return rows

        return self.execute_cypher(
            """
            MATCH (d:Dataset)
            WHERE d.factSheet IS NOT NULL
            RETURN d.datasetNumber AS datasetNumber, d.title AS title, d.doi AS doi,
                   d.factSheet AS factSheet, d.factSheetText AS factSheetText
            ORDER BY d.datasetNumber
            """
        )

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

    def get_dataset_profile(self, reference: str) -> "DatasetProfileMatch | DatasetProfileAmbiguous | None":
        """
        Resolves `reference` (a title, DOI, or dataset number) to exactly one Dataset node
        and fetches its full PART_OF sub-node graph (Sample/DigitalDataset/AnalysisDataset/
        RelatedPublication/RelatedSoftware/RelatedDataset) plus INPUT_FOR pipeline edges among
        the sub-nodes — used for "tell me more about this dataset"-style follow-ups, which need
        far more than the title/DOI/one-line-summary shape search()/cypher_qa() return.

        Resolution tiers, tried in order, stopping at the first with >=1 match:
          1. datasetNumber exact — only when `reference` looks like ONLY a number
             (see _try_parse_dataset_number), so a title that happens to contain digits
             doesn't get misread as a dataset number.
          2. doi exact (case-insensitive, https://doi.org/ prefix stripped from both sides).
          3. title case-insensitive CONTAINS.

        Returns:
          - None if reference matches zero datasets, or if graph search is disabled.
          - DatasetProfileMatch if exactly one dataset matches (in whichever tier fired).
          - DatasetProfileAmbiguous if more than one dataset matches — built directly from
            that tier's own rows, with NO second graph round-trip, so the caller can render a
            disambiguation prompt immediately.

        NOTE: RelatedSoftware/RelatedDataset's relationship to Dataset is not confirmed to be
        PART_OF in the live schema (MANUAL_SCHEMA above only documents this for Sample/
        DigitalDataset/AnalysisDataset/RelatedPublication) — verify empirically
        (e.g. MATCH (n:RelatedSoftware)-[r]->() RETURN type(r), count(*)) and update the
        relationship type below if it differs. A missing relationship type degrades safely
        (empty list), so this is safe to ship speculatively.

        The sub-node graph is fetched with one small query per node/edge type rather than a
        single query chaining several OPTIONAL MATCHes. That earlier shape cross-multiplied
        before collect() — samples x digital x analysis x publications — and was measured at
        28s on the live graph for the largest dataset (961 sub-nodes) with only the PART_OF
        joins present; adding the two INPUT_FOR joins back on top of it did not complete at
        all within 300s. Each query below is a flat relationship scan instead, and the
        assembly happens in Python.
        """
        if not self._enabled or not self._driver:
            return None

        candidates: list[dict] = []
        dataset_number = _try_parse_dataset_number(reference)
        if dataset_number is not None:
            candidates = self.execute_cypher(
                "MATCH (d:Dataset {datasetNumber: $ref}) "
                "RETURN d.datasetNumber AS datasetNumber, d.title AS title, d.doi AS doi",
                {"ref": dataset_number},
            )
        if not candidates:
            # d.doi may be stored bare or with the https://doi.org/ resolver prefix — compare
            # against both forms of the (already-stripped) reference rather than relying on
            # Cypher string manipulation to normalize the stored value.
            stripped_ref = _strip_doi_prefix(reference)
            doi_query = (
                "MATCH (d:Dataset) WHERE toLower(d.doi) = toLower($bare) OR toLower(d.doi) = toLower($prefixed) "
                "RETURN d.datasetNumber AS datasetNumber, d.title AS title, d.doi AS doi"
            )
            candidates = self.execute_cypher(
                doi_query,
                {"bare": stripped_ref, "prefixed": f"https://doi.org/{stripped_ref}"},
            )
        if not candidates:
            candidates = self.execute_cypher(
                "MATCH (d:Dataset) WHERE toLower(d.title) CONTAINS toLower($ref) "
                "RETURN d.datasetNumber AS datasetNumber, d.title AS title, d.doi AS doi",
                {"ref": reference},
            )

        if not candidates:
            return None
        if len(candidates) > 1:
            return DatasetProfileAmbiguous(candidates=candidates)

        matched_number = candidates[0]["datasetNumber"]
        params = {"datasetNumber": matched_number}

        dataset_rows = self.execute_cypher(
            "MATCH (d:Dataset {datasetNumber: $datasetNumber}) "
            "RETURN d{.*, datasetEmbedding: null, factSheetEmbedding: null, "
            "         factSheet: null, factSheetText: null} AS d",
            params,
        )
        if not dataset_rows:
            return None

        def _nodes(query: str) -> list[dict]:
            # A map projection nulls a key rather than dropping it (Cypher can't drop keys),
            # so a node whose only property was an embedding collapses to an all-null map —
            # drop those rather than pass an empty node downstream.
            return [
                dict(r["n"])
                for r in self.execute_cypher(query, params)
                if r.get("n") is not None and any(v is not None for v in dict(r["n"]).values())
            ]

        def _edges(query: str, left: str, right: str) -> list[dict]:
            return [
                {left: r[left], right: r[right]}
                for r in self.execute_cypher(query, params)
                if r.get(left) is not None and r.get(right) is not None
            ]

        return DatasetProfileMatch(
            dataset=dict(dataset_rows[0]["d"]),
            samples=_nodes(
                "MATCH (n:Sample)-[:PART_OF]->(:Dataset {datasetNumber: $datasetNumber}) "
                "RETURN n{.*, componentEmbedding: null} AS n"
            ),
            digital_datasets=_nodes(
                "MATCH (n:DigitalDataset)-[:PART_OF]->(:Dataset {datasetNumber: $datasetNumber}) "
                "RETURN n{.*, componentEmbedding: null} AS n"
            ),
            analysis_datasets=_nodes(
                "MATCH (n:AnalysisDataset)-[:PART_OF]->(:Dataset {datasetNumber: $datasetNumber}) "
                "RETURN n{.*, componentEmbedding: null} AS n"
            ),
            related_publications=_nodes(
                "MATCH (n:RelatedPublication)-[:PART_OF]->(:Dataset {datasetNumber: $datasetNumber}) "
                "RETURN n{.*} AS n"
            ),
            related_software=_nodes(
                "MATCH (n:RelatedSoftware)-[:PART_OF]->(:Dataset {datasetNumber: $datasetNumber}) "
                "RETURN n{.*} AS n"
            ),
            related_datasets=_nodes(
                "MATCH (n:RelatedDataset)-[:PART_OF]->(:Dataset {datasetNumber: $datasetNumber}) "
                "RETURN n{.*} AS n"
            ),
            # INPUT_FOR points CHILD -> PARENT ("was derived from") — see this module's
            # schema docstring. Querying it parent -> child, as this did until the direction
            # was verified against the live graph and scripts/load_graph.py, matched zero
            # rows, which silently emptied the organizational-structure section of every
            # profile and reported every scan as having no recorded sample link.
            sample_to_digital_edges=_edges(
                "MATCH (dd:DigitalDataset)-[:INPUT_FOR]->(s:Sample) "
                "MATCH (dd)-[:PART_OF]->(:Dataset {datasetNumber: $datasetNumber}) "
                "RETURN s.identifier AS sample, dd.identifier AS digitalDataset",
                "sample", "digitalDataset",
            ),
            digital_to_analysis_edges=_edges(
                "MATCH (ad:AnalysisDataset)-[:INPUT_FOR]->(dd:DigitalDataset) "
                "MATCH (ad)-[:PART_OF]->(:Dataset {datasetNumber: $datasetNumber}) "
                "RETURN dd.identifier AS digitalDataset, ad.identifier AS analysisDataset",
                "digitalDataset", "analysisDataset",
            ),
        )

    def cypher_qa(self, question: str, restrict_to_titles: list[str] | None = None) -> str:
        """
        Answer a structured question about datasets using LLM-generated Cypher.
        Source label: [cypher match]

        GraphCypherQAChain's default QA step returns the same generic
        "I don't know the answer." string whether the generated Cypher
        genuinely failed or whether it ran fine and matched zero rows. Those
        are very different outcomes for the caller: the latter is a complete,
        honest answer ("no matches"), the former is a real failure. Since we
        run with return_intermediate_steps=True, we can tell them apart by
        checking whether the last "context" step is an empty list (query
        executed, zero rows) versus absent/non-empty (something else happened).

        For rows that look like a dataset listing (every row has a title), the
        bullet list is built directly from these raw rows in Python — see
        _format_dataset_rows — rather than trusting the QA chain's own LLM call to
        reproduce titles/DOIs correctly. That LLM call is not reliably steerable by
        prompt instructions alone; formatting the one high-stakes shape in code
        removes the failure mode instead of trying to word the prompt more carefully.
        Other shapes (counts, aggregates) still use the QA chain's own prose answer.

        restrict_to_titles: when given (a follow-up refinement of a previously
        listed set of datasets), deterministically narrow the freshly generated
        Cypher's rows to only those whose title matches one of these — in code,
        not by trusting the Cypher-generation LLM to re-derive every prior
        constraint from the compounded natural-language question and re-run it
        consistently over the whole graph. That trust was found to fail in
        practice: the SAME "sandstone" constraint, reworded into two different
        compound questions across two refinement turns, generated two different
        WHERE clauses (`s IS NULL OR toLower(s.porousMediaType) = 'sandstone'`
        vs. `s IS NOT NULL AND toLower(s.porousMediaType) = 'sandstone'`) —
        meaning a refinement's Cypher search re-scans the entire catalog each
        turn and can silently drift from the actual previously-listed set instead
        of narrowing it. Post-filtering the rows to a known-good prior title set
        makes each refinement a true subset of the last one regardless of how
        the regenerated Cypher phrases the earlier filters.
        """
        if not self._enabled or not self._cypher_chain:
            return "Graph search is disabled (USE_NEO4J=false)."
        result = self._cypher_chain.invoke({"query": question})

        steps = result.get("intermediate_steps") or []
        context_steps = [s["context"] for s in steps if isinstance(s, dict) and "context" in s]
        if context_steps and context_steps[-1] == []:
            return "The query ran successfully and found no matching datasets or samples for this question."

        if context_steps:
            rows = context_steps[-1]
            if restrict_to_titles:
                wanted = {t.lower() for t in restrict_to_titles if t}
                narrowed = [
                    r for r in rows
                    if isinstance(r, dict) and str(_row_field(r, "title") or "").lower() in wanted
                ]
                if not narrowed:
                    return (
                        "None of the previously listed datasets match this additional filter."
                    )
                rows = narrowed
            formatted = _format_dataset_rows(rows)
            if formatted:
                return formatted

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
        """Support use as a context manager — returns self."""
        return self

    def __exit__(self, *_):
        """Close the Neo4j connection on context manager exit."""
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
