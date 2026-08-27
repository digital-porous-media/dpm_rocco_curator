# General Assistant — Pre-Sprint Tasks

Tasks Bernie must complete before intern Week 1. Track progress here.

> **Historical.** The sections below (Environment & Infrastructure, Data Preparation, Domain
> Content, Repo Skeleton, Intent Classifier Testing) are the pre-sprint checklist from before
> Week 1 (ended Jun 6, 2026). Some boxes were never checked off even though the work was done —
> the graph has been loaded and queryable for months (184 datasets; see `docs/neo4j_schema.md`
> and CLAUDE.md's "Remaining Work" section). Don't read an unchecked box here as "not done yet."

---

## Environment & Infrastructure

- [x] Install Neo4j via Homebrew (`neo4j 2026.04.0`)
- [x] Create the `rocco` conda environment with all dependencies (includes `langchain-neo4j`)
- [ ] Install graph dependencies:
  ```bash
  pip install -e ".[graph]"  # Installs neo4j, langchain-neo4j, langchain-openai
  ```
- [ ] Start Neo4j and set password
  ```bash
  neo4j start
  # Then open http://localhost:7474 — login neo4j/neo4j, set new password
  ```
- [ ] Add Neo4j credentials to `.env`:
  ```
  USE_NEO4J=true
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=<your-password>
  ```
- [ ] Verify Neo4j connection:
  ```bash
  conda activate rocco
  python -c "
  from neo4j import GraphDatabase
  import os; from dotenv import load_dotenv; load_dotenv()
  driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')))
  driver.verify_connectivity()
  print('Connected!')
  driver.close()
  "
  ```

---

## Data Preparation

- [ ] Download DRP metadata from TACC Corral:
  ```bash
  conda activate rocco
  python scripts/scrape_metadata.py --output data/metadata/
  ```
- [ ] Load metadata into Neo4j:
  ```bash
  conda run -n rocco python scripts/load_graph.py --mode rebuild
  ```
  Verify in Neo4j Browser: `MATCH (d:Dataset) RETURN count(d)` — expect 176
- [x] Build vector indexes in Neo4j:
  ```bash
  conda run -n rocco python scripts/build_dataset_vector_index.py
  ```
  Confirm in Neo4j Browser: `SHOW INDEXES` → look for `datasetEmbedding` and `componentEmbedding`
  - `datasetEmbedding` — one vector per Dataset node (aggregated metadata), dim=4096
  - `componentEmbedding` — one vector per DatasetComponent (Sample/DigitalDataset/AnalysisDataset), dim=4096
  - Re-running is safe (upserts); only required again if datasets change or embedding model changes
- [ ] Verify graph completeness and property correctness:
  ```bash
  conda run -n rocco python scripts/audit_schema.py --folder data/metadata/ --verify --output docs/neo4j_schema.md
  ```
  Expected: 176/176 datasets loaded, no property mismatches, sub-node counts match

---

## Domain Content (Only Bernie Can Write These)

- [ ] Author `data/domain_workflows.yaml` — 15 DRP workflows
  - Format: name, description, steps, inputs, outputs, software, references
  - Do not edit without domain review once written
- [ ] Populate `data/tutorials.yaml` — 20+ portal tutorial URLs
  - Each entry: goal, url, keywords
  - Verify all URLs are live before intern Week 1

---

## Repo Skeleton (Done ✅)

- [x] Create `src/assistant/` with all stub/working files
- [x] Create `scripts/scrape_metadata.py`, `scripts/build_dataset_vector_index.py`
- [x] Create `src/prompts/assistant.yaml`, `query_expander.yaml`, `educational.yaml`
- [x] Create `data/domain_workflows.yaml` and `data/tutorials.yaml` templates
- [x] Update `.gitignore` — `data/metadata/`, `data/curated_papers/`, `data/publication_vector_store/`
- [x] Update `.env.example` — Neo4j vars, embedding vars, Semantic Scholar key
- [x] Update `pyproject.toml` — added `langchain-openai` to `[graph]` extra
- [x] Port `Chatbot/` working code into `src/assistant/`:
  - [x] `graph_store.py` — full Neo4j vector + Cypher QA, schema documented
  - [x] `tools.py` — `search_datasets`, `get_dataset_details`, `general_chat` working; Intern B stubs added
  - [x] `conversation_manager.py` — LangGraph ReAct agent with MemorySaver
  - [x] `assistant_ui.py` — working Streamlit tab, `assistant_`-prefixed session keys
  - [x] `llm.py` — `ChatOpenAI` + `OpenAIEmbeddings` via `.env`
- [x] Fix bugs from `Chatbot/`:
  - [x] "movies" stale description in tool → fixed to "datasets"
  - [x] `passwords.py` credential pattern → replaced with `.env`
  - [x] AgentExecutor (removed in langchain 1.x) → replaced with `langgraph.prebuilt.create_react_agent`
  - [x] APOC constraint added to Cypher prompt — no plugin required

---

## Intent Classifier Testing (Issue #10 + Verification)

**Note:** `assistant.yaml` is not used in the live routing path — routing is the ReAct agent
matching tool descriptions (see CLAUDE.md's "Search Architecture"). This classifier is retained
for offline prompt-quality testing only.

- [x] Draft `src/prompts/assistant.yaml` (v0.2.0) — comprehensive intent classifier
  - Intent definitions: semantic_search, metadata_filter, domain_qa, workflow_guidance, query_expansion, literature_search
  - Examples per intent with porous media terminology (porosity, permeability, micro-CT, wettability, etc.)
  - Decision rules to avoid boundary cases
- [x] Create `test_intent_classifier.py` (Option 1 — quick test script)
  - Run manually with: `python test_intent_classifier.py`
  - Tests all 6 intents + edge cases
  - Good for rapid iteration on the prompt
- [x] Create `tests/assistant/test_intent_classifier.py` (Option 2 — formal test suite)
  - Run with: `pytest tests/assistant/test_intent_classifier.py -v`
  - Parametrized test cases with expected intents and confidence thresholds
  - JSON output validation
  - Can be integrated into CI/CD

---

## Pre-Sprint Verification

Run these before intern Week 1 to confirm everything works end-to-end:

```bash
conda activate rocco

# 1. Imports clean
python -c "from src.assistant.tools import build_langchain_tools; print('OK')"
python -c "import os; os.environ['USE_NEO4J']='false'; from src.assistant.graph_store import GraphStore; GraphStore(); print('OK')"

# 2. Curator tab still works
streamlit run rocco_ui.py

# 3. With Neo4j running — test vector search
python -c "
from src.assistant.graph_store import GraphStore
g = GraphStore()
results = g.search('sandstone permeability', top_k=3)
for r in results: print(r['metadata'].get('title'))
comp = g.component_search('carbonate grain size', top_k=3)
for r in comp: print(r['metadata'].get('componentTitle'), '->', r['metadata'].get('datasetTitle'))
"
```

---

## Pre-Sprint: Bernie Before Week 2 (May 27 – Jun 6) — **COMPLETE**

Intern started Jun 7. Bernie was away Weeks 2–3. All items completed before Jun 7.

### Completed this week (May 27–30)

- [x] ~~Download curated papers into `data/curated_papers/` (#13)~~ — **Cancelled.** Local PDF corpus dropped due to copyright risk; literature search uses Semantic Scholar API only. (#13 closed)
- [x] Write intern onboarding doc (#16)
- [x] Record or write Rocco codebase walkthrough (#19)
- [x] Write `CONTRIBUTING.md` (#14)
- [x] Add pytest scaffolding `tests/assistant/conftest.py` (#15)
- [x] Confirm intern has repo access, `rocco` env setup instructions, `.env.example`, Neo4j

### Completed during Week 1 (May 31 – Jun 6)

- [x] Author `data/domain_workflows.yaml` — 24 DRP workflows (#11) *(exceeds 15 target)*
- [x] Draft `data/tutorials.yaml` skeleton — 20+ verified entries (#12)
- [x] Catalogue DPM Portal tutorials; extend `tutorials.yaml` to 20+ entries (#23)
- [x] Implement `src/assistant/literature_search.py` (#17)
- [x] Draft `src/prompts/query_expander.yaml` + `educational.yaml` (#24)
- [x] Fix `conversation_manager.py` system prompt — implement tiered knowledge policy
- [x] Fix `general_chat` tool in `tools.py` — remove "answer only from provided context" restriction
- [x] Implement `expand_query`, `get_educational_context`, `get_workflow_guidance` in `tools.py` (#29)
- [x] Implement `search_literature` in `tools.py` — direct Semantic Scholar call via `LiteratureSearch` (#30)
- [x] Define 20 representative test queries (#18)
- [x] Finish infra: load graph, verify 176 datasets (see Environment & Infrastructure above)

---

## Week 4 (Jun 21–27) — COMPLETE

### Bernie's Tasks

- [x] Integrate JRS's `julia/graph-store` branch into `src/assistant/graph_store.py`
  - [x] Merge branch with full commit history
  - [x] Fold raw-driver primitives (`semantic_search`, `filter_by_metadata`, `search_datasets`, `execute_cypher`, `get_schema_blueprint`) into existing `GraphStore`
  - [x] Add injection-safe helpers (`_SAFE_KEY_RE`, `_validate_keys`, `_build_where_clause`)
  - [x] Add `SearchResult` dataclass for low-level search methods
  - [x] Fix bugs: correct default `index_name` to `"datasetEmbedding"`, fix test patch paths
  - [x] Migrate tests from top-level to `tests/assistant/test_graph_store.py` (22 tests passing)
  - [x] Delete duplicate top-level files
  - [x] Commit with full integration message
- [x] Review `conversation_manager.py` — add docstrings for cross-intent routing (#37, module docstring documents intent→tool routing table + cross-intent handling)
- [x] Review `domain_workflows.yaml` — fill any structural gaps; verify 24 workflows cover the core DRP pipeline (#31)
- [x] Prepare documentation for JRS's Week 5 work — `search_datasets()` fully implemented with docstring at `graph_store.py:626`

### JRS's Tasks (Week 4–5)

- [x] `graph_store.py`: finalize `semantic_search()` + `filter_by_metadata()` based on integrated raw-driver methods
- [x] `search_datasets()` combined query + source labels (hybrid vector+metadata in one Cypher call) — implemented (#52); **board still shows this as Todo, needs to be closed to unblock #42**

---

## Week 5–6 — COMPLETE (code), board reconciliation needed

- [x] Cross-intent queries; distinguish paper sources; docstrings for `conversation_manager.py` (#37) — done, board still shows Open
- [x] Review Semantic Scholar edge cases; unblock integration issues (#38) — done via `_get_with_retry` (429 backoff, throttling) in `literature_search.py`; board still shows Open
- [x] Modify `rocco_ui.py` to add General Assistant tab (#39) — done, `_PAGES` nav + `render_assistant_tab()` wired; board still shows Open
- [x] Surface source labels in UI; verify search stack in tabbed UI (#40)
- [x] Connect `assistant_ui.py` into General Assistant tab (#41) — done, board still shows In Progress

**Action item:** see CLAUDE.md's "Remaining Work Before Project Conclusion" section for the
current board-reconciliation status — code for all five listed issues is merged; only the board
itself is stale.

---

## Remaining Work Before Project Conclusion (Week 6–9)

### Week 6–7 (Jul 5–18)

- [ ] **#42** — Run the full acceptance suite through the tabbed `rocco_ui.py`; demo to BCC, MP, and ME
  - All queries already exist as automated tests in `tests/assistant/test_search_integration.py` (S-1..4, M-1..3, D-1..6, W-1..5, Q-1..3, L-1..4 — 25 total; see that file for the current list rather than this range, which will drift as queries are added) — this task is *execution*, not authoring
  - Demo must show both tabs (Curator + General Assistant) working independently with no session-state collision
- [x] **Investigate hanging test suite** — **resolved.** The cause was live network tests running unintentionally. They now carry a `live` marker and `pytest.ini` sets `addopts = -m "not live"`, so the default run excludes them. Verified Aug 2026: `pytest tests/ -v` → **358 passed, 51 deselected**, reproducible across runs (wall-clock varies by machine). Run the live tier explicitly with `pytest tests/ -m live -v` (needs real credentials + a running Neo4j). #43 can rely on a clean default run.
- [ ] **#43** — Final index rebuild (`python scripts/build_dataset_vector_index.py`) + evaluation.
  Assistant documentation is **done** — it shipped as `docs/user_guide/assistant.rst` plus nine
  per-capability pages, not the originally-planned single `docs/assistant.md`. There is no
  `build_publication_index.py`; the publication corpus was dropped with #27/#32.
- [ ] **#45** — README updates, handoff doc, tag `v2.0.0`, record demo video

### Week 8–9 (Jul 19–Aug 1)

- [ ] **#46** — Write and submit poster (Intern-A)
- [ ] **#48** — Review poster draft (Bernie)
- [ ] **#49** — Write and submit paper (Intern-A)
- [ ] **#51** — Review paper draft (Bernie)

---

## GitHub Project Board — Updated Jun 23, 2026

Board updated to reflect revised schedule (Bernie front-loads to Week 1; away Weeks 2–3).

### Reassigned to Bernie
- `#23` Catalogue `tutorials.yaml` → Bernie, week=1
- `#25` Prototype `assistant_ui.py` → Bernie, week=6

### Rescheduled
- `#24` `query_expander.yaml` + `educational.yaml` → week=1 (front-loaded)
- `#29` `expand_query`, educational tools → week=1 (front-loaded; was week=3)
- `#30` Literature routing → week=1 (simplified: direct Semantic Scholar call only)
- `#31` Review intern search output + domain_workflows gaps → week=4 (was week=3; Bernie away)
- `#36` `USE_NEO4J=false` fallback + docstrings → week=6 (was week=5; matches revised intern scope)

### Closed
- `#28` Source labels + 10 search queries → closed; scope folded into `#52`
- `#13` Collect curated papers → **closed** (local PDF corpus dropped; copyright risk)
- `#27` [JRS] Implement publication_corpus.py → **closed** (task dropped; intern freed for Week 3)
- `#32` [BCC] Integrate publication corpus into search → **closed** (dropped with corpus)

### Created
- `#52` [JRS] Week 5: `search_datasets()` combined query + source labels (Intern-A, P0)

---

## langchain-community Migration (Future — Separate Branch)

`langchain-community` has been sunset (no new features; maintenance-only). See https://github.com/langchain-ai/langchain-community/issues/674.
The package is pinned to `>=0.4.1,<0.5.0` in `pyproject.toml` to prevent silent breakage from future removals.

Create a dedicated branch (`chore/migrate-langchain-community`) when the replacement packages stabilize. Work required:

- [ ] **`src/ingestor/document_ingestor.py`** — replace `langchain_community.document_loaders.PyPDFLoader`
  with `langchain_pypdf.PyPDFLoader` (install: `pip install langchain-pypdf`); replace
  `Docx2txtLoader` with native `python-docx` loading or the equivalent standalone package
- [ ] **`src/retriever/retriever.py`** — replace `langchain_community.vectorstores.FAISS`
  once a dedicated `langchain-faiss` package exists; alternatively wrap `faiss-cpu` directly
  since `VectorStoreManager` already abstracts the interface
- [ ] **`tests/test_vector_store.py`** — update matching import after retriever is migrated
- [ ] **`pyproject.toml`** — remove `langchain-community` dep; add replacement packages
- [ ] Run full test suite (`pytest tests/ -v`) after migration; verify RAG pipeline end-to-end

Track the FAISS situation at https://github.com/langchain-ai/langchain-community/issues/674 —
a dedicated standalone package may be released; migrate then rather than reimplementing from scratch.

---

## Future Feature: Dataset Detail Follow-Up Queries — IMPLEMENTED

**Status (implemented on `feature/dataset-details`):** Done, with a broader scope than
originally proposed below. `get_dataset_profile(dataset_reference, question)` was added to
`src/assistant/tools.py`, backed by a new `GraphStore.get_dataset_profile()` (resolves
datasetNumber/DOI/title, fetches the full `PART_OF` sub-node graph + `INPUT_FOR` pipeline
edges). It also handles organizational-structure questions, file-type/"how do I read this in
Python" reasoning (including a real TACC Corral archive URL derived from `datasetNumber`),
reuse-suitability reasoning, and multi-dataset comparisons (by calling the tool once per
dataset). Classified in `_SELF_CONTAINED_TOOLS`, **not** `_VERBATIM_TOOLS` as originally
proposed below — it needs its own grounded LLM synthesis pass (via new
`src/prompts/dataset_profile.yaml`) to reason over the data rather than just splice it verbatim,
and to give a concise high-level overview for general "tell me more" questions instead of an
exhaustive field dump. Also required two `conversation_manager.py` fixes not anticipated below:
extending `_TOOL_PARAM_KEYS`/`_extract_tool_calls_from_text`/`_extract_tool_calls_from_error` to
support a tool with two required args, and extending `_FOLLOWUP_TOOL_GATE_SYSTEM_PROMPT` so a
comparison (same tool, second dataset) isn't short-circuited after the first profile call.
Documented at `docs/user_guide/dataset_profiles.rst`. See `HANDOFF.md` for the original
investigation this expanded on.

**Problem:** After `search_datasets` (or `get_dataset_details`) returns results, a natural
follow-up like *"tell me more about this dataset"* / *"give me more details on the first one"*
currently has no dedicated handling. It re-triggers the same search/lookup path and returns the
same title/DOI/summary shape as the original result — not a fuller profile of the one dataset
the user is actually asking about.

**Why current tools don't cover this:**
- `search_datasets` is tuned for discovery across many datasets (semantic/hybrid search, one
  short LLM-summarized sentence per result) — not a deep single-dataset profile.
- `get_dataset_details` answers structured questions via generated Cypher — it doesn't have a
  "give me everything you have on dataset X" mode, and isn't naturally triggered by a pronoun
  reference ("this dataset") with no named property.
- `GraphStore.get_dataset(dataset_id)` (`src/assistant/graph_store.py`) already exists and
  fetches full `Dataset` node properties by `datasetNumber` — but it's dead code from the
  assistant's perspective: not registered as a LangChain tool in `tools.py`, and nothing calls
  it today.

**Original proposed approach (superseded by the broader scope noted above — kept for history):**
- [x] Add a new tool (`get_dataset_profile`) in `src/assistant/tools.py` that pulls the full
  `Dataset` node plus its `Sample`/`DigitalDataset`/`AnalysisDataset`/`RelatedPublication`
  sub-nodes — implemented as a sibling method, `GraphStore.get_dataset_profile()`, also fetching
  `INPUT_FOR` pipeline edges and (speculatively, pending live-schema verification)
  `RelatedSoftware`/`RelatedDataset`.
- [x] Update `conversation_manager.py`'s `SYSTEM_PROMPT` tool-selection rules so follow-ups
  route here instead of re-running `search_datasets`/`get_dataset_details`, with the reference
  resolved from conversation history before calling.
- [x] Decide response-assembly classification — landed on `_SELF_CONTAINED_TOOLS`, not
  `_VERBATIM_TOOLS` as originally guessed here (see the IMPLEMENTED note above for why).
- [x] Add test coverage — `tests/assistant/test_tools.py`, `tests/assistant/test_graph_store.py`,
  `tests/assistant/test_conversation_manager.py`, and a two-turn/comparison flow in
  `tests/assistant/test_search_integration.py`.
- [x] Document the new capability — `docs/user_guide/dataset_profiles.rst` (own capability page,
  not folded into `structured_queries.rst`).

---

## Feature: Honest Content/Relationship Reasoning — IMPLEMENTED

**Status (implemented on `feature/dataset-details`):** Done. Closes the residual honesty gap
left by Bug 4 in `HANDOFF.md` — that fix stopped "paired tomographic and segmented images" from
crashing/falling back to weak semantic search, but the resulting answer (a bare
`segmented='yes'` list) still presented a generic "has some segmented data" result as if it had
verified "paired", which it never checked.

**What landed:**
- [x] `reason_about_dataset_content(question)` in `src/assistant/tools.py` — a single general
  mechanism for any question no literal field can settle (a relationship, a comparison across a
  dataset's sub-nodes, or a property that only appears in free text). Registered in
  `build_langchain_tools()` and in `_SELF_CONTAINED_TOOLS`/`_TOOL_PARAM_KEYS`.
- [x] Precomputed fact sheets — `Dataset.factSheet` (JSON) + `Dataset.factSheetText` (rendered
  prose), built by a new step in `scripts/build_dataset_vector_index.py`, with its own
  edge-preserving assembly (NOT `_build_embedding_text`, which flattens away which
  `DigitalDataset` belongs to which `Sample`). `--only fact-sheets` rebuilds just this stage.
- [x] `factSheetEmbedding` vector index + `datasetFactSheetFulltext` BM25 index.
- [x] `GraphStore.rank_fact_sheets()` / `fetch_fact_sheets()` — ranking reuses the *existing*
  `hybrid_search` RRF fusion, extracted into a shared `_rrf_merge()`; no new fusion mechanism
  and no per-relationship Cypher pattern to author.
- [x] `src/prompts/corpus_reasoning.yaml` — the cited reasoning pass, plus the map-reduce
  batch-screening prompt for exhaustive questions.
- [x] Deterministic `_needs_content_reasoning()` gate wired into BOTH `get_dataset_details` and
  `search_datasets`, so the split is correct regardless of which tool the agent picked.
- [x] Grounding enforced in code, not prompt: uncited candidates dropped; candidates not in the
  shortlist actually sent dropped; titles/DOIs taken from graph records.
- [x] Tests — `tests/assistant/test_tools.py` (gate, three worked cases, grounding guards,
  budget, map-reduce, restrict-to-titles), `test_graph_store.py` (RRF, ranking, fetch),
  `test_fact_sheet_builder.py` (new file), `test_prompts.py`, `test_conversation_manager.py`.
- [x] Docs — `docs/user_guide/content_reasoning.rst` (new capability page), plus updates to
  `assistant.rst`, `architecture.rst`, `neo4j_schema.md`, `index.rst`, and `CLAUDE.md`.

**Two bugs found while implementing, both fixed** (see `HANDOFF.md` for the full bug-by-bug
investigation narrative, including live-verification detail omitted here):
1. **`INPUT_FOR` was documented and queried backwards everywhere.** The live graph has
   `(DigitalDataset)-[:INPUT_FOR]->(Sample)` — child → parent, "was derived from" — confirmed by
   edge counts (1893 / 983 / 55) and by `load_graph.py`'s `_establish_connection`. Because
   `get_dataset_profile()` queried it the other way, **every** profile's organizational-structure
   section was silently empty and every scan was reported as unlinked. `MANUAL_SCHEMA` (which
   grounds all generated Cypher) and `docs/neo4j_schema.md` were wrong too.
2. **`get_dataset_profile()`'s Cypher was pathologically slow** — chained `OPTIONAL MATCH`es
   cross-multiplying before `collect()`: 28s on the largest live dataset with only the `PART_OF`
   joins, and no completion within 300s once the `INPUT_FOR` joins were restored. Decomposed into
   one flat query per node/edge type → **0.8s** for the same dataset.

**Remaining:** the fact sheets themselves have not been built against the live Neo4j yet
(`python scripts/build_dataset_vector_index.py --only fact-sheets`), so end-to-end live
verification of the three example queries is still outstanding.

---

## Notes

- **Conda env:** Always `conda activate rocco` before running anything
- **Neo4j notebook:** Use `CurationTools/JsonToNeo4jwKeywords.ipynb` (not the chunking one) to load data
- **`langchain-neo4j`** is installed via `pip install -e ".[graph]"` and provides `Neo4jVector` for semantic search abstractions
- **`langchain_sambanova`** is not installed — embeddings use `OpenAIEmbeddings` with custom base URL instead; if SambaNova's embedding endpoint is not OpenAI-compatible, update `src/assistant/llm.py`
- **`Chatbot/` and `CurationTools/`** — kept as historical reference; do not delete before intern Week 1
