# General AI Assistant for Dataset Discovery & Education

Related: [[00_Project_Overview]], [[ADR_Search_Approach]]

**Assigned to:** Intern A (search backend) + Intern B (educational backend), working on **one unified conversation manager**  
**Goal:** Help users find datasets using metadata *and* understand porous media concepts — gracefully handling incomplete data by blending discovery with education.

---

## The Problems Being Solved

1. **Discovery problem:** Users want "samples suitable for lattice Boltzmann simulation" but the portal offers only keyword search. Many descriptions are sparse, yet structured metadata (rock type, voxel size, segmentation status) is available on the Neo4j graph — we need both.

2. **Education problem:** New researchers know they want to measure porosity or run a flow simulation but don't know where to start. Portal data alone doesn't explain *why* certain datasets matter or *how* to use them.

3. **Metadata gap problem:** When discovery can't confirm results (inferred filters like `segmented=true` are sparsely populated), the system must not fail silently — it must explain the gap and offer alternative paths.

**The unified solution:** One conversational assistant that routes through search, filtering, and educational tools based on intent, with graceful degradation when metadata falls short.

---

## What This Assistant Can Do

### 1. Semantic Search Over Descriptions
- Query: *"high resolution Berea sandstone"*
- Returns: top-k descriptions ranked by semantic similarity (FAISS + `BAAI/bge-large-en-v1.5`)
- Includes matches from dataset descriptions **and related publication abstracts**
- Shows: dataset ID, excerpt, similarity score, source (description vs. publication abstract)

### 2. Metadata Filtering (With Honest Limitations)
Filter on reliably-present fields (>80% coverage):

| Field | Example |
|-------|---------|
| Rock type | "sandstone", "carbonate", "cement paste" |
| Imaging modality | "micro-CT", "FIB-SEM", "synchrotron" |
| Voxel size / resolution | "voxel size < 2 µm" |
| Facility / scanner | "APS", "Diamond Light Source" |
| Publication link | "datasets linked to papers" |
| File format | "TIFF", "raw binary" |
| Segmentation status | "segmented binary images" |

Fields with poor coverage return honest guidance:
> *"Porosity values aren't in most dataset records, but here's how to estimate it from segmented images..."*

### 3. Domain Knowledge Q&A
Answer using LLM training + optional RAG from uploaded papers:
- *"What is a typical porosity range for Berea sandstone?"*
- *"What does voxel size mean for my simulation?"*
- *"What's the difference between micro-CT and FIB-SEM?"*

The LLM draws on domain knowledge freely here — no portal data to invent.

### 4. Workflow Guidance
Map user goals to actionable next steps via handwritten `data/tutorials.yaml`:
- *"I want to measure porosity"* → tutorial link + recommended tools + example datasets known to work
- *"I want to run LBM"* → tools, preprocessing steps, why voxel size matters, example datasets
- *"How do I segment a micro-CT image?"* → tutorial link + software options

### 5. LLM-Mediated Query Expansion
When a user asks "find samples suitable for LBM," the assistant doesn't just search — it:
1. **Expands the query** using domain knowledge: "segmented binary micro-CT pore connectivity high resolution"
2. **Infers filter hints:** `{segmented: true, voxelDimensions: "<5µm"}`
3. **Explains the reasoning:** "LBM requires segmented images and sub-5µm voxels for pore-scale accuracy"
4. **Gracefully handles missing data:** If `segmented` field is sparse, falls back to semantic search + educational guidance

### 6. Graceful Degradation When Data Falls Short

**Degradation chain:**
```
User: "find LBM-suitable samples"
    ↓
[METADATA PATH] query_graph({segmented: true, voxelDimensions: "<5µm"})
    → No/few results (field sparse)
    ↓
[SEMANTIC PATH] search_datasets("segmented binary micro-CT pore connectivity")
    → 3 results, unlabeled for segmentation
    ↓
[EDUCATION PATH] Blend in tutorial guidance:
    "These 3 are close semantically, but I can't confirm they're segmented. 
     Here's how to check: [tutorial link]. 
     These example datasets ARE known segmented: [IDs]. 
     For LBM you'll need voxels < 5µm — here's why: [brief explanation]."
```

The user gets a useful, honest response regardless of metadata completeness.

---

## Architecture

```
assistant_ui.py (Streamlit chat tab in rocco_ui.py)
    ↓
src/assistant/assistant.py    — unified conversation manager
    ↓ intent classify → query expand → tool dispatch
src/assistant/tools.py         — all tool functions (plain Python)
    ├── search_datasets()          → FAISS semantic search (descriptions)
    ├── search_publications()      → FAISS on publication abstracts
    ├── query_graph()              → Neo4j Cypher structured filters
    ├── expand_query()             → LLM domain reasoning (pre-retrieval)
    ├── search_tutorials()         → FAISS on portal tutorial content
    ├── get_educational_context()  → LLM + educational.yaml + domain_workflows.yaml
    ├── get_workflow_guidance()    → tutorials.yaml lookup + LLM synthesis
    └── search_external_literature() → Semantic Scholar / arXiv API
    ↓
Data Sources
    ├── FAISS index: descriptions + publication abstracts (data/assistant_vector_store/)
    ├── FAISS index: portal tutorial content (data/tutorial_vector_store/)
    ├── Neo4j graph: Sample, DigitalDataset, RelatedPublication nodes
    ├── data/tutorials.yaml: handwritten goal → notebook_url + doc_url + tools + example datasets
    ├── data/domain_workflows.yaml: general simulation/analysis workflow descriptions
    ├── data/curated_papers/: foundational domain papers (pre-indexed)
    └── Uploaded papers (per-session FAISS, optional RAG)
```

### Key Design Decisions

**Unified conversation manager, not separate tabs.** One `assistant.py` routes all intents to the appropriate tools. From the user's perspective, it's one coherent chat that blends discovery with education. The intern split is implementation-focused (Intern A: search tools, Intern B: education tools), not user-facing.

**Hybrid FAISS + Neo4j + Educational tools.** See [[ADR_Search_Approach]]. FAISS handles semantics; Neo4j handles structured filtering; education tools provide the fallback and explanation when metadata gaps appear.

**Query expansion before retrieval.** The LLM translates user intent into concrete search criteria using domain knowledge. This bridges the vocabulary gap: "LBM samples" → "segmented binary micro-CT with voxel size < 5 µm."

**Publications embedded in corpus.** `RelatedPublication` nodes' abstracts are embedded and searchable alongside descriptions. A paper about "flow simulation in carbonates" can match even if the dataset description is sparse.

**Portal tutorials are searchable content, not just links.** The DPM Portal has Jupyter tutorial notebooks and HTML documentation pages. These are scraped, chunked, and embedded into a separate `tutorial_vector_store/`. The `search_tutorials()` tool retrieves relevant tutorials for any workflow question. `tutorials.yaml` stores verified URLs alongside example datasets and workflow notes.

**Domain workflows are method-focused, not tool-specific.** `domain_workflows.yaml` covers general porous media workflows — LBM theory and requirements, pore network modeling, image segmentation, DNS, permeability analysis — independent of specific third-party software. If the portal officially features a tool, that tool is referenced here.

**Literature search spans portal and external sources.** Pre-indexed: RelatedPublication abstracts from Neo4j + curated foundational papers. Live: `search_external_literature()` calls Semantic Scholar API at query time for broader coverage. Results labeled `[portal publication]` vs. `[external literature]`, ranked accordingly.

**Graceful degradation is architectural, not just prompt-based.** When inferred filters yield nothing, the system automatically blends results from the semantic path + educational guidance. No "no results found" dead ends.

### Conversation Flow

1. **Intent classify** (single LLM call)
   - Input: user message + conversation history
   - Output: `{"intent": "search|filter|educational|workflow|dataset_explain", "extracted_params": {...}}`

2. **Query expansion** (if search/filter intent)
   - LLM domain reasoning → `{expanded_query, inferred_filters, rationale}`
   - Used to guide both FAISS and Neo4j queries

3. **Parallel tool dispatch**
   - `search_datasets(expanded_query)` → FAISS top-k on descriptions
   - `search_publications(expanded_query)` → FAISS top-k on publication abstracts
   - `query_graph(inferred_filters)` → Neo4j Cypher results
   - `search_tutorials(query)` → FAISS on portal tutorial content (if workflow/educational intent)
   - `search_external_literature(query)` → Semantic Scholar / arXiv (if literature intent)
   - Education intents → `get_educational_context` (backed by domain_workflows.yaml) or `get_workflow_guidance` (backed by tutorials.yaml)

4. **Result ranking & merging**
   - Deduplicate by dataset ID
   - Rank: Neo4j exact match > FAISS description high-conf (≥0.75) > FAISS abstract match > FAISS partial (0.50–0.75)
   - Label source for each result: `[metadata match]`, `[semantic match]`, `[publication match]`, `[both]`

5. **Detect metadata gaps & educate**
   - If search tools yield few results OR inferred filters had no hits:
     - Surface query expansion rationale: "I searched for these criteria because..."
     - Include relevant tutorial links and example datasets
     - Explain the limitation: "Segmentation field isn't populated for most datasets..."

6. **Synthesize & render**
   - LLM assembles response grounded in all available signals
   - Chat response + expandable "Sources" panel with dataset IDs, similarity scores, publication links
   - Optional "Learn more" section with tutorials and tools

---

## New Files to Create

| File | Purpose | Owner |
|------|---------|-------|
| `src/assistant/assistant.py` | Unified conversation manager: intent → expand → tools → synthesis | Intern B |
| `src/assistant/tools.py` | All tool callables: search, graph, expand, educate, literature | Both |
| `src/assistant/dataset_index.py` | FAISS index wrapper for descriptions + publications | Intern A |
| `src/assistant/graph_store.py` | Neo4j Cypher queries for structured field filtering | Intern A |
| `src/assistant/publication_corpus.py` | Extract and embed publication abstracts from Neo4j | Intern A |
| `src/assistant/literature_search.py` | Semantic Scholar / arXiv API wrapper for live external search | Intern B |
| `src/prompts/assistant.yaml` | System + user prompts for conversation manager | Intern B |
| `src/prompts/query_expander.yaml` | LLM prompt for query expansion (domain reasoning) | Intern B |
| `src/prompts/educational.yaml` | Domain Q&A + workflow synthesis prompts | Intern B |
| `data/tutorials.yaml` | Hand-authored: ~20 goals → notebook_url + doc_url + tools + example dataset IDs | Intern B |
| `data/domain_workflows.yaml` | Hand-authored: 10–15 general workflow descriptions (LBM, pore network, segmentation…) | Intern B |
| `data/curated_papers/` | Small set of pre-ingested foundational domain papers | Intern B |
| `assistant_ui.py` | Streamlit chat UI (imported into rocco_ui.py as a tab) | Intern B |
| `scripts/build_assistant_index.py` | Embed descriptions + publication abstracts → `data/assistant_vector_store/` | Intern A |
| `scripts/build_tutorial_index.py` | Scrape portal tutorial content → `data/tutorial_vector_store/` | Intern B |

## Files to Modify

| File | Change |
|------|--------|
| `rocco_ui.py` | Add `st.tabs(["Dataset Curator", "General Assistant"])` at top; namespace all session state with `curator_` prefix; import `assistant_ui` module and wire into tab 2 |

---

## Reuse (Do Not Reinvent)

**Search backend:**
- `src/retriever/retriever.py` → `VectorStoreManager.load()` + `similarity_search_with_score()` → direct parent of `dataset_index.py`
- DPM Portal API → Primary description corpus (176+ datasets accessible via API)
- Standard Neo4j Python driver → Build `graph_store.py` with best-practice connection pooling
- Cypher query examples from Neo4j documentation → Rewrite cleanly for production use

**Education backend:**
- `src/ingestor/document_ingestor.py` → PDF chunking (reuse for uploaded papers)
- `src/retriever/retriever.py` → per-session FAISS for papers

**Shared:**
- `src/llm/client.py` → `LLMClient.send_prompt()`
- `src/prompts/loader.py` → `load_prompt()` + Jinja2 render
- All new YAMLs follow the existing `src/prompts/` schema and versioning

---

## Intern Coordination

| Milestone | What | Status Check |
|-----------|------|--------------|
| **Week 2 end** | Data audit (Intern A); UI prototype + prompt drafts (Intern B) | Intern A confirms which metadata fields are reliably populated; Intern B validates chat layout with supervisor |
| **Week 4 end** | `dataset_index.py`, `graph_store.py`, search tools (A); `assistant.py` conversation manager (B); both can call each other's tools in isolation | Run `scripts/build_assistant_index.py`; test each tool independently |
| **Week 6 end** | Full integration: unified `assistant.py` routes all intents; `assistant_ui.py` connected to backend; session state namespaced in `rocco_ui.py`; no collisions between curator + assistant | `streamlit run rocco_ui.py` → both tabs load; run 10 queries mixing search + education intents; verify session state doesn't bleed |
| **Week 8 end** | Polish: sidebar examples, publication links working, tutorials loading, documentation written | Final demo; final index rebuild; run full 20–30 query test suite |

---

## Why Unified, Not Separate Tabs?

Early design considered two separate tabs (search tab + education tab). We unified for three reasons:

1. **Natural interaction:** Users often want both at once — "find LBM samples AND explain LBM" is one thought, not two.
2. **Handles metadata gaps:** When discovery can't confirm results, educational tools provide immediate context. Separate tabs force context-switching.
3. **Simpler implementation:** One conversation manager + intent classifier handles all cases. Two managers would duplicate conversation history, session state, and LLM synthesis logic.

The intern split remains for implementation (Intern A builds search tools, Intern B builds education tools), but they converge in one `assistant.py`.

---

## See Also

- **For search architecture details** (hybrid FAISS + Neo4j rationale): [[ADR_Search_Approach]]
- **For Rocco extensions** (separate work stream): [[01_Rocco_Extension]]
- **For week-by-week tasks**: [[04_Tasks]]
