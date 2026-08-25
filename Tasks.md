# General Assistant — Pre-Sprint Tasks

Tasks Bernie must complete before intern Week 1. Track progress here.

---

## Environment & Infrastructure

- [x] Install Neo4j via Homebrew (`neo4j 2026.04.0`)
- [x] Create `rocco_ai` conda environment with all dependencies (includes `langchain-neo4j`)
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
  conda activate rocco_ai
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
  conda activate rocco_ai
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
conda activate rocco_ai

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
- [x] Confirm intern has repo access, `rocco_ai` env setup instructions, `.env.example`, Neo4j

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

**Action item:** close #37, #38, #39, #41, #52 on the GitHub project board — code for all five is merged, but the board hasn't been updated to reflect it. This is blocking #42 from being unblocked in the tracker even though there's no real code dependency left.

---

## Remaining Work Before Project Conclusion (Week 6–9)

### Week 6–7 (Jul 5–18)

- [ ] **#42** — Run the full 20-query acceptance suite through the tabbed `rocco_ui.py`; demo to BCC, MP, and ME
  - All 20 queries already exist as automated tests in `tests/assistant/test_search_integration.py` (S-1..4, M-1..3, D-1..4, W-1..4, Q-1..2, L-1..3) — this task is *execution*, not authoring
  - Demo must show both tabs (Curator + General Assistant) working independently with no session-state collision
- [ ] **Investigate hanging test suite** — `pytest tests/assistant/test_graph_store.py` passes cleanly (22/22, ~6s), but `pytest tests/assistant/` (full directory) hangs indefinitely. Likely a live network call (Semantic Scholar or Neo4j) in `test_search_integration.py` or `test_tools.py` that isn't mocked/skipped when it should be. Must be fixed before #43's "final index rebuild, evaluation" can rely on `pytest tests/ -v` passing cleanly.
- [ ] **#43** — Final index rebuild (`build_dataset_vector_index.py`, `build_publication_index.py`), evaluation, write `docs/assistant.md`
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

## Future Feature: Dataset Detail Follow-Up Queries

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

**Proposed approach (not yet implemented):**
- [ ] Add a new tool (e.g. `get_dataset_profile(doi_or_title)`) in `src/assistant/tools.py` that
  pulls the full `Dataset` node plus its `Sample`/`DigitalDataset`/`AnalysisDataset`/
  `RelatedPublication` sub-nodes (extend `GraphStore.get_dataset()` or add a sibling method —
  the existing one only fetches the `Dataset` node itself, not its `PART_OF` sub-nodes) and
  renders a fuller profile than a search-result summary line.
- [ ] Update `conversation_manager.py`'s `SYSTEM_PROMPT` tool-selection rules so a follow-up
  referencing a specific already-shown result ("tell me more about *this*/*the first*/*the
  sandstone one*") routes here instead of re-running `search_datasets`. This needs the DOI/title
  to be resolved from conversation history (already passed in as `history`), since the tool
  itself is stateless per call.
- [ ] Decide response-assembly classification for the new tool — likely `_VERBATIM_TOOLS` (real
  dataset metadata that must reach the user unmodified), same as `search_datasets`/
  `get_dataset_details`.
- [ ] Add test coverage (`tests/assistant/test_tools.py` and/or
  `tests/assistant/test_search_integration.py`) for a two-turn "search → tell me more" flow.
- [ ] Document the new capability — either fold into `docs/user_guide/structured_queries.rst` or
  give it its own capability page (matches the existing pattern; see e.g. `dataset_discovery.rst`).

## Notes

- **Conda env:** Always `conda activate rocco_ai` before running anything
- **Neo4j notebook:** Use `CurationTools/JsonToNeo4jwKeywords.ipynb` (not the chunking one) to load data
- **`langchain-neo4j`** is installed via `pip install -e ".[graph]"` and provides `Neo4jVector` for semantic search abstractions
- **`langchain_sambanova`** is not installed — embeddings use `OpenAIEmbeddings` with custom base URL instead; if SambaNova's embedding endpoint is not OpenAI-compatible, update `src/assistant/llm.py`
- **`Chatbot/` and `CurationTools/`** — kept as historical reference; do not delete before intern Week 1
