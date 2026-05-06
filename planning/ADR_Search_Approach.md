# ADR: Search Approach — Hybrid FAISS + Neo4j + Query Expansion

Related: [[02_General_Assistant]]

**Status:** Decided  
**Date:** 2026-04-23 (updated 2026-04-23 with query expansion and publications)

---

## Context

A previous intern built a Neo4j graph over scraped DPM Portal metadata linked to an LLM. It performed poorly when used as the **sole** search mechanism. We are choosing a search architecture for the new General AI Assistant's discovery capability, including how to handle two key challenges:
1. **Sparse descriptions paired with rich metadata** — the previous attempt missed metadata completely when descriptions were thin
2. **Intent-to-criteria translation** — when a user asks for "LBM-suitable samples," the system needs to infer what properties to search for (binary segmented, voxel size < 5 µm, etc.)

## Decision

**Use a hybrid approach: FAISS semantic search + Neo4j structured retrieval, with results merged by a confidence-aware ranking layer.**

## Rationale

The two sources have complementary strengths that map directly onto the portal's data quality problem:

| Source | Strong when | Weak when |
|--------|------------|-----------|
| FAISS (full-description embedding) | Description is narrative-rich ("Berea sandstone core drilled from the... used to study drainage...") | Description is sparse ("Sandstone sample.") — low cosine similarity scores, poor recall |
| Neo4j (structured Cypher) | Structured fields are populated: `porousMediaType`, `voxelDimensions`, `imagingEquipmentAndModel`, `porosity` | Fields are null — query returns no results or incomplete results |

Critically, **sparse descriptions and sparse metadata are not the same datasets**. Many samples with thin descriptions still have populated `Sample` and `DigitalDataset` node properties (rock type, voxel size, imaging equipment) because those come from a separate metadata form at submission time, not from the description text. The hybrid captures signal from whichever source has it.

## What the Neo4j Graph Actually Contains

The previous intern's graph has the following node types and useful fields:

- **Sample:** `porousMediaType`, `porosity`, `grainSizeAvg/Min/Max`, `geographicOrigin`, `depth`, `collectionMethod`, `onshoreOffshore`
- **DigitalDataset:** `voxelDimensions`, `imagingCenter`, `imagingEquipmentAndModel`, `imageFormat`, `imageDimensions`, `fileTypes`, `segmented`
- **AnalysisDataset:** `type`, `fileTypes`
- **RelatedPublication:** `title`, `authors`, `abstract`, `link`
- **Relationships:** `Sample/DigitalDataset/AnalysisDataset → PART_OF → Dataset`, enabling provenance traversal

This structured data is **not accessible to FAISS** — it lives only in the graph.

## Hybrid Architecture + Query Expansion

```
User query
    ↓
Intent Classifier (LLM, one-shot)
    → intent: "search|filter|educational|workflow"
    ↓
Query Expander (LLM, domain reasoning — IF search/filter intent)
    → expanded_query: "semantic translation of intent using domain knowledge"
    → inferred_filters: {field: criteria, ...} for Neo4j Cypher
    → rationale: "why these properties matter for your goal"
    ↓
Parallel Tool Dispatch
    ├── search_datasets(expanded_query) → FAISS description corpus
    ├── search_publications(expanded_query) → FAISS publication abstract corpus
    └── query_graph(inferred_filters) → Neo4j Cypher structured filtering
    ↓
Result Merger
    - Deduplicate by dataset ID
    - Rank: Neo4j exact match > FAISS description high-conf (≥0.75) > FAISS abstract match > FAISS partial (0.50–0.75)
    - Label source: [metadata], [semantic], [publication], [both], etc.
    ↓
Graceful Degradation (if sparse metadata)
    - If Neo4j yields nothing but FAISS yields results:
      → Surface semantic results with caveat: "not confirmed for your criteria"
      → Blend in educational guidance: tutorial links, example datasets, tools
    ↓
LLM Response Synthesis
    - Grounds in merged results + expansion rationale
    - Cites dataset IDs + publication links
    - Honest about what couldn't be confirmed
```

### Query Expansion: Intent → Criteria via Domain Knowledge

When a user asks "find samples suitable for LBM," the system doesn't search for that phrase directly. Instead, it uses LLM domain knowledge to expand the query:

**Input:** User query + conversation context + intents

**Output:**
```json
{
  "expanded_query": "segmented binary micro-CT pore connectivity high resolution sandstone carbonate",
  "inferred_filters": {"segmented": true, "voxelDimensions": "<5µm"},
  "rationale": "LBM simulations require binary segmented images for pore-scale accuracy; voxels must be sub-5µm to resolve pore structure"
}
```

**Why this matters:** The user's vocabulary ("LBM-suitable") doesn't match the corpus. But the LLM knows what LBM requires, and translates it into searchable terms. This bridges the semantic gap without hallucinating — the domain reasoning is grounded in the user's intent, not in invented data.

**Graceful degradation:** If inferred filters yield no Neo4j results (because `segmented` is sparsely populated), the system doesn't fail. It falls back to semantic-only results, caveats them ("not confirmed segmented"), and surfaces tutorials on how to check or segment images.

### Publication Corpus Integration

Related publications are valuable for two reasons:
1. **Rich abstracts:** A paper about "flow simulation in carbonates" can match even if the dataset description is sparse
2. **Provenance:** Users want to find datasets used in published work; publication links are direct citations

**Implementation:** Extract all `RelatedPublication.abstract` fields from Neo4j during index build. Embed them alongside descriptions in FAISS with metadata tracking which dataset they're linked to. When results are returned, publication matches are labeled and linked.

### Confidence-Aware Routing

When FAISS returns low-confidence scores (< 0.60) for a query, the merger automatically falls back to Neo4j structured fields as the primary signal. This handles sparse-description datasets gracefully without surfacing misleading semantic matches. Publication matches (if any) are surfaced with their confidence scores.

### What Neo4j Is NOT Used For

The previous implementation attempted vector search **inside** Neo4j (embedding keywords or chunks on nodes). This is redundant and performed poorly — Neo4j's vector index on LLM-extracted keywords loses semantic nuance, and 50-word chunk embeddings fragment context.

**Neo4j's role in the hybrid is purely structured retrieval via Cypher.** All vector/semantic work goes through FAISS.

## Consequences

- **Query expansion adds a reasoning step before retrieval.** The LLM translates intent into searchable criteria — this uses LLM domain knowledge freely, but only at the query level, not in result grounding. See [[02_General_Assistant]] for how this integrates with the unified assistant.

- **Publication corpus adds value but maintenance burden.** Extracts from Neo4j require careful linking and embedding. If publications become stale, re-run `scripts/build_assistant_index.py`. The `USE_NEO4J=false` mode still works (descriptions alone) if publication data is unavailable.

- **Intern A takes on more scope** — must understand and query the existing Neo4j graph, extract and embed publications, implement merge logic. Coordinate with supervisor on Neo4j connection credentials using standard credential management practices.

- **Result merging adds complexity** — the merger balances FAISS confidence, Neo4j field coverage, publication matches, and honesty about gaps. See [[02_General_Assistant]] for the precedence rule and degradation logic.

- **Educational backend fills gaps.** When metadata is sparse, educational tools (tutorials, domain explanations, example datasets) provide the fallback. This is not a search problem alone — it's a unified discovery + education architecture. See [[02_General_Assistant]] for the full conversation flow.

## When to Revisit

If Neo4j connection is unavailable or the graph is stale/incomplete, the system should degrade gracefully to FAISS-only. Add a feature flag `USE_NEO4J=true/false` in `.env` so the assistant works in either mode.
