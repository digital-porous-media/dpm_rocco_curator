# Search & Discovery Assistant

Related: [[00_Project_Overview]], [[ADR_Search_Approach]]

**Assigned to:** Intern A  
**Goal:** Help users find relevant datasets using the metadata that *is* available, while being honest about what is missing.

---

## Problem

The DPM Portal's current UI offers basic keyword filtering. Users who want to find "a micro-CT sandstone sample at high resolution for flow simulation" have no good way to do that. A semantic search layer over the 176 scraped descriptions would immediately improve this.

## What This Assistant Can Do

### Semantic Search
- Query: *"high resolution Berea sandstone"*
- Returns: top-k dataset descriptions ranked by semantic similarity (FAISS + `BAAI/bge-large-en-v1.5`)
- Shows: dataset ID, excerpt from description, similarity score
- Labels low-confidence matches as "partial match" (threshold: cosine similarity < 0.60)

### Metadata Filtering
Filter on fields that are **reliably present** (>80% coverage across datasets):

| Field | Example query |
|-------|--------------|
| Rock / material type | "sandstone", "carbonate", "cement paste" |
| Imaging modality | "micro-CT", "FIB-SEM", "synchrotron" |
| Voxel size / resolution | "voxel size < 2 µm" |
| Facility / scanner | "APS", "Diamond Light Source" |
| Associated publication DOI | "datasets linked to a paper" |
| File format | "TIFF", "raw binary" |

Fields with poor coverage (porosity value, permeability, saturation) return honest canned messages:
> *"Porosity values are not available for most datasets in the portal. I can search by rock type or imaging parameters instead."*

### Combined Search + Filter
User can combine: *"Show me carbonate datasets scanned at synchrotron facilities"* → filter by modality + semantic search for "carbonate".

---

## Architecture

```
assistant_ui.py (Streamlit tab)
    ↓
src/assistant/search_assistant.py   — conversation manager + result merger
    ↓
src/assistant/tools.py
    ├── search_datasets(query, k)         → dataset_index.py → VectorStoreManager (FAISS)
    ├── query_graph(filters)              → graph_store.py → Neo4j Cypher queries
    └── merge_results(faiss, neo4j)       → confidence-aware dedup + ranking
```

### Key Design Decisions

**Hybrid FAISS + Neo4j.** See [[ADR_Search_Approach]]. The two sources are complementary: FAISS works well when descriptions are narrative-rich; Neo4j structured fields (`porousMediaType`, `voxelDimensions`, `imagingEquipmentAndModel`) fill in when descriptions are sparse. Using both together handles the full range of data quality in the portal.

**FAISS handles semantics; Neo4j handles structure.** The previous intern's attempt to use Neo4j's built-in vector index (on keywords or chunks) performed poorly. In the hybrid, all vector search goes through FAISS — Neo4j is queried only via Cypher for structured field filtering.

**Graceful degradation.** A `USE_NEO4J=true/false` flag in `.env` lets the assistant fall back to FAISS-only if Neo4j is unavailable. Results are labeled with their source (`[semantic match]`, `[metadata match]`, `[both]`) so users understand what was searched.

**Honesty still applies.** If both sources return nothing (e.g., "datasets with porosity > 25%"), the honest canned message is returned. The hybrid improves recall; it doesn't create data that isn't there.

### Conversation Flow

1. **Intent classify** — single-call LLM prompt returns `{"intent": "semantic|structured|combined", "params": {...}}`
2. **Tool dispatch** — `search_datasets` (FAISS), `query_graph` (Neo4j Cypher), or both in parallel
3. **Merge & rank** — deduplicate by dataset ID; precedence: Neo4j exact match > FAISS high-confidence (≥0.75) > FAISS partial (0.50–0.75); low FAISS confidence (< 0.60) routes to Neo4j as primary
4. **Synthesize** — LLM assembles response grounded in merged results; cites dataset IDs; does not state metadata values absent from the data
5. **Render** — chat response + expandable "Sources" panel with dataset IDs, sources, similarity scores, portal links

---

## New Files

| File | Purpose |
|------|---------|
| `src/assistant/search_assistant.py` | Conversation manager + result merger for this tab |
| `src/assistant/tools.py` | `search_datasets`, `query_graph`, `merge_results` |
| `src/assistant/dataset_index.py` | Wraps `VectorStoreManager` for the description corpus |
| `src/assistant/graph_store.py` | Neo4j Cypher queries for structured field filtering |
| `src/prompts/search_assistant.yaml` | System prompt with honesty constraints |
| `scripts/build_assistant_index.py` | Embed descriptions → `data/assistant_vector_store/` |

## Reuse

- `src/retriever/retriever.py` → `VectorStoreManager` (used directly in `dataset_index.py`)
- `src/llm/client.py` → `LLMClient.send_prompt`
- `src/prompts/loader.py` → `load_prompt` / `render`
- DPM Portal API → Primary FAISS corpus (176+ dataset descriptions accessible via API)
- Standard Neo4j Python driver → Build `graph_store.py` with connection pooling
- Neo4j documentation → Cypher query examples for structured field filtering (rewrite cleanly for production)

## Tasks

See [[04_Tasks#Intern-A-Search-and-Discovery]].
