# Contributor Onboarding Guide

Welcome to the Rocco project. This document covers project context, environment setup, a
walkthrough of the codebase, and a set of orientation exercises. Read `CONTRIBUTING.md` alongside
this for workflow rules (branching, commit style, PR process).

---

## Project Context

**Rocco** is an AI assistant framework for the
[Digital Porous Media (DPM) Portal](https://digitalporousmedia.org/), a community repository for
micro-CT and related datasets from porous media samples (sandstones, carbonates, coals, beads,
fibrous media, soils). Two modules ship as two tabs of the same Streamlit app:

| Module | What it does |
|--------|--------------|
| **Description Curator** | Evaluates and enhances dataset descriptions using a 10-criterion rubric + RAG over uploaded papers |
| **General Assistant** | Conversational dataset search, relationship reasoning, domain Q&A, workflow guidance, portal-doc and literature search |

Both are implemented and running. See `CHANGELOG.md` for what shipped when, and `Tasks.md` for
what's still open.

---

## Prerequisites

Before starting, confirm you have:

- [ ] Python 3.11 via `conda` (Miniforge recommended)
- [ ] `git` and access to `github.com/digital-porous-media/dpm_rocco_curator`
- [ ] Neo4j ≥ 5.x installed locally (Homebrew: `brew install neo4j`)
- [ ] LLM API credentials from Bernie (added to `.env` — see below)
- [ ] Semantic Scholar API key from Bernie (optional; unauthenticated requests work, just
      rate-limited)

---

## Deployment Context

| Environment | Neo4j location | Notes |
|-------------|---------------|-------|
| **Local development** | Your laptop | Default for contributor work |
| **Production** | TACC VM | Same schema; different `NEO4J_URI` and credentials. See `DEPLOYMENT.md` |

Never commit credentials. Always use `.env` (gitignored).

`USE_NEO4J=false` degrades the assistant gracefully rather than breaking it: all dataset search
(discovery, structured queries, profiles, content reasoning) goes dark and returns empty results
without even importing the Neo4j driver, while domain Q&A, workflow guidance, portal-doc search,
and literature search keep working. There is no alternative dataset-search backend — that is
what "degrade" means here.

---

## First-Day Setup

```bash
# 1. Clone and enter the repo
git clone git@github.com:digital-porous-media/dpm_rocco_curator.git
cd dpm_rocco_curator

# 2. Create and activate the conda environment
#    NOTE: the env on Bernie's machine is named `rocco`. Match it to avoid confusion.
conda create -n rocco python=3.11
conda activate rocco

# 3. Install the package with all dependencies
pip install -e ".[dev,graph]"
# [graph] adds neo4j, langchain-neo4j, langchain-openai, langgraph

# 4. Copy and fill in credentials
cp .env.example .env
# Open .env and fill in:
#   LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
#   EMBEDDING_API_KEY, EMBEDDING_URL, EMBEDDING_MODEL
#   NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  (after step 5)
#   SEMANTIC_SCHOLAR_API_KEY

# 5. Start Neo4j and set a password
neo4j start
# Open http://localhost:7474
# Login: neo4j / neo4j → set a new password → add it to .env

# 6. Populate the graph (see "Loading the Graph" below — this is three steps, not one)
python scripts/scrape_metadata.py
python scripts/load_graph.py --mode rebuild
python scripts/build_dataset_vector_index.py

# 7. Verify imports
python -c "from src.assistant.tools import build_langchain_tools; print(len(build_langchain_tools()), 'tools')"
python -c "import os; os.environ['USE_NEO4J']='false'; from src.assistant.graph_store import GraphStore; GraphStore(); print('OK')"

# 8. Run tests — expect "381 passed, 64 deselected"
pytest tests/ -v

# 9. Launch the Streamlit UI — both tabs
streamlit run rocco_ui.py
```

> `pytest.ini` sets `addopts = -m "not live"`, so tests making real network calls are excluded
> by default — a clean run reports 381 passed, 64 deselected. Run the live tier explicitly with
> `pytest tests/ -m live -v` when you have credentials and a running Neo4j; those are slow and
> can block on a rate-limited endpoint, which is why they're opt-in.

### Loading the Graph

Three distinct steps, and skipping the third is the most common setup mistake:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `scripts/scrape_metadata.py` | Downloads DRP metadata JSONs from TACC Corral into `data/metadata/` (gitignored) |
| 2 | `scripts/load_graph.py --mode rebuild` | Loads nodes + relationships into Neo4j. **Nothing else.** |
| 3 | `scripts/build_dataset_vector_index.py` | Embeddings, fact sheets, and all five vector/fulltext indexes |

A graph loaded without step 3 answers structured Cypher questions but returns nothing from
semantic search or content reasoning — and does so silently. Use `--mode upsert` for incremental
loads (it preserves derived properties on unchanged nodes) and
`scripts/reembed_single_dataset.py --doi ...` to patch one dataset.

Verify with `python scripts/audit_schema.py --folder data/metadata/ --neo4j --verify`.

---

## Codebase Walkthrough

### Top-Level Files

| File | Purpose |
|------|---------|
| `rocco_ui.py` | Streamlit entry point — page nav + `render_curator_tab()` / `render_assistant_tab()` |
| `pyproject.toml` | Package dependencies; `[graph]` extra adds Neo4j/LangChain |
| `.env.example` | Template for all environment variables |
| `CONTRIBUTING.md` | Branching, commit style, code quality, PR process |
| `DEPLOYMENT.md` | TACC VM deployment runbook |
| `CLAUDE.md` | Implementation patterns, design constraints, hard-won gotchas |
| `Tasks.md` | Task tracker |
| `HANDOFF.md` | Detailed handoff notes for the most recent feature work |

### `src/` Module Map

```
src/
├── llm/           LLM client, embeddings factory, schemas, content screening
├── evaluator/     Rubric evaluation (Curator)
├── editor/        Description enhancement with citations (Curator)
├── ingestor/      PDF/DOCX → chunked documents
├── retriever/     FAISS vector store queries (Curator RAG)
└── assistant/     General Assistant
    ├── conversation_manager.py   orchestrator: gates, routing, response assembly, cross-turn state
    ├── tools.py                  all 8 callable tools + their routing gates
    ├── graph_store.py            Neo4j: hybrid search, component search, fact-sheet ranking, Cypher QA, profiles
    ├── literature_search.py      Semantic Scholar API wrapper
    ├── portal_docs_retrieval.py  heading-tree retrieval over dpm_docs
    ├── portal_docs_tree.py       markdown → heading tree
    ├── llm.py                    RoccoClient + embeddings singletons
    ├── assistant_ui.py           Streamlit tab
    └── assistant.py              re-export of ConversationManager (back-compat)
```

#### `src/llm/`

`client.py` contains `RoccoClient` — the single LLM call interface used by every module. It
inherits from both `LLMClient` (provider routing) and LangChain's `BaseChatModel` (LangGraph
compatibility). Provider is configured via `.env`; you do not need to touch this file.

`embeddings.py` picks an embedding model/endpoint from `LLM_PROVIDER` unless `EMBEDDING_*` is set
explicitly. `schemas.py` defines Pydantic output schemas for the curator's structured responses.

#### `src/prompts/`

All LLM prompts are externalized as versioned YAML and rendered with Jinja2. Never embed a prompt
in Python — add a YAML file and load it with `load_prompt("filename")`.

| Prompt file | Used by |
|-------------|---------|
| `evaluator.yaml` | Curator — rubric scoring |
| `editor.yaml` | Curator — description enhancement |
| `content_screener.yaml` | Curator — feedback validation |
| `query_expander.yaml` | Assistant — query expansion + filter inference |
| `educational.yaml` | Assistant — domain Q&A **and** workflow guidance (shared) |
| `dataset_profile.yaml` | Assistant — single-dataset deep dive |
| `corpus_reasoning.yaml` | Assistant — relationship reasoning over fact sheets (+ batch screening) |
| `portal_docs.yaml` | Assistant — portal documentation answers |
| `assistant.yaml` | **Not called at runtime.** Standalone intent classifier, kept for offline analysis and tests |

Routing is *not* driven by `assistant.yaml`. It is implicit: the ReAct agent picks tools by
matching each tool's own description in `tools.py`, so that description is where a per-tool
routing rule goes. `SYSTEM_PROMPT` in `conversation_manager.py` carries only the knowledge tiers,
the cross-tool boundaries, and the response contract. See
[the Prompt Reference](https://digital-porous-media.github.io/dpm_rocco_curator/developer_guide/prompts.html).

#### `src/assistant/`

**`tools.py`** — Every function the agent can call. Eight tools are registered by
`build_langchain_tools()`: `search_datasets`, `get_dataset_details`, `get_dataset_profile`,
`reason_about_dataset_content`, `search_portal_docs`, `get_workflow_guidance`,
`get_educational_context`, `search_literature`. Do not change a tool's name or signature without
coordinating — the system prompt and the manual-dispatch recovery path both reference them.

Note the **deterministic routing gates** in this file (`_needs_content_reasoning`,
`_is_plain_property_query`, `_mentions_named_person`). These exist because prompt-level routing
proved unreliable for fine distinctions — "segmented and porosity above 0.3" is a plain field
query, "segmented and imaged the same way" is relational, and the agent could not be relied on to
tell them apart. Gating in code makes the split correct regardless of which tool the agent picked.

**`conversation_manager.py`** — The orchestrator. Cheap tools-unbound gate calls, then a
LangGraph ReAct agent, then response assembly that depends on which *kind* of tool ran (verbatim
splice vs. self-contained passthrough vs. outer-agent synthesis). Also holds the cross-turn
result-set state that makes "of these…" and "the second one" work.

**`graph_store.py`** — Search methods used by `tools.py`: `search()`, `hybrid_search()`,
`component_search()`, `cypher_qa()`, `get_dataset_profile()`, `rank_fact_sheets()`,
`fetch_fact_sheets()`. `execute_cypher()` runs raw parameterized Cypher over the `neo4j`
driver and backs `hybrid_search()`'s BM25 half, the fact-sheet methods and
`get_dataset_profile()`. Accepts `filters: dict` rather than hardcoded field names, per the
Croissant extensibility constraint.

**`literature_search.py`** — Semantic Scholar wrapper with a 1 req/s throttle and 429 backoff.

**`portal_docs_retrieval.py` / `portal_docs_tree.py`** — PageIndex-style heading-tree retrieval
over `data/portal_docs/docs/`. No chunking, no embeddings; the tree is parsed at import time, so
re-syncing the markdown and restarting is the whole update procedure.

**`llm.py`** — Singletons: `get_llm()` → `RoccoClient`, `get_embeddings()` → `OpenAIEmbeddings`.

**`assistant_ui.py`** — Streamlit tab. Session keys are `assistant_`-prefixed; the curator's are
unprefixed, so that prefix is the only thing keeping the tabs from colliding.

### Data and Scripts

| Path | What it is |
|------|-----------|
| `data/domain_workflows.yaml` | 15 DRP workflows; ground truth for workflow/educational responses. Do not edit without domain review. |
| `data/tutorials.yaml` | Portal tutorial notebook paths mapped to user goals. Treated as strictly as dataset DOIs — never fabricate a path. |
| `data/portal_docs/docs/` | Synced copy of the [dpm_docs](https://github.com/digital-porous-media/dpm_docs) repo. Update with `scripts/sync_dpm_docs.py`. |
| `data/metadata/` | **Gitignored.** Scraped DRP metadata JSONs (one per dataset). |
| `scripts/scrape_metadata.py` | Downloads metadata JSONs from TACC Corral. |
| `scripts/load_graph.py` | Loads metadata JSONs into Neo4j (nodes + relationships only). |
| `scripts/build_dataset_vector_index.py` | Embeddings, fact sheets, and all five indexes. |
| `scripts/reembed_single_dataset.py` | Patches one dataset's embeddings after an upsert. |
| `scripts/audit_schema.py` | Audits graph completeness; **generates `docs/neo4j_schema.md`**. |
| `scripts/sync_dpm_docs.py` | Pulls portal documentation updates. `--check` compares against the last-synced SHA. |
| `scripts/check_embedding_health.py` | Diagnoses embedding-endpoint failures. |
| `docs/neo4j_schema.md` | Your Cypher reference: schema, coverage stats, starter queries. Read before writing any Cypher. |

> `docs/neo4j_schema.md` is **generated**. Fix inaccuracies in `scripts/audit_schema.py` and
> regenerate, or your edit disappears on the next run:
> `python scripts/audit_schema.py --folder data/metadata/ --output docs/neo4j_schema.md`

### Tests

```
tests/
├── conftest.py                     # Curator fixtures
├── test_curator_integration.py     # Evaluator + editor + screener
├── test_llm_client.py              # LLM provider routing
├── test_vector_store.py            # FAISS embedding and alignment
└── assistant/
    ├── conftest.py                     # mock Neo4j driver, mock GraphStore
    ├── test_graph_store.py             # search, filters, profiles, fact-sheet ranking
    ├── test_tools.py                   # tool behavior, routing gates, grounding guards
    ├── test_conversation_manager.py    # response assembly, cross-turn state, reference resolution
    ├── test_fact_sheet_builder.py      # fact-sheet assembly, caps, char-budget batching
    ├── test_portal_docs_retrieval.py   # heading-tree selection and synthesis
    ├── test_literature_search.py       # Semantic Scholar wrapper, throttle, backoff
    ├── test_search_integration.py      # the 20-query acceptance suite
    ├── test_assistant_ui.py            # badges, DOI/URL linkifying, LaTeX normalization
    ├── test_prompts.py                 # query_expander/educational/corpus_reasoning load+render
    └── test_intent_classifier.py       # assistant.yaml (offline only)
```

`tests/assistant/conftest.py` mocks the Neo4j driver, so Neo4j is not required for unit tests.

**Always run tests before pushing.** The curator integration tests catch interface regressions
even in assistant code.

---

## Neo4j Graph Overview

The DPM catalog is a property graph with **two** relationship types. Both point child → parent:

```
(Sample)-[:PART_OF]->(Dataset)
(DigitalDataset)-[:PART_OF]->(Dataset)
(AnalysisDataset)-[:PART_OF]->(Dataset)
(RelatedPublication)-[:PART_OF]->(Dataset)
(RelatedSoftware)-[:PART_OF]->(Dataset)
(RelatedDataset)-[:PART_OF]->(Dataset)

(DigitalDataset)-[:INPUT_FOR]->(Sample)            # this scan was taken of that sample
(AnalysisDataset)-[:INPUT_FOR]->(DigitalDataset)   # this analysis was computed from that scan
(AnalysisDataset)-[:INPUT_FOR]->(Sample)           # no intermediate scan
```

> ⚠️ **`INPUT_FOR` means "was derived from" and runs the same direction as `PART_OF`, despite the
> name.** A scan points *at* its sample. Writing it the intuitive way round —
> `(s:Sample)-[:INPUT_FOR]->(dd:DigitalDataset)` — matches **zero rows and fails silently**. This
> was live in production for a while: every dataset profile's organizational-structure section
> was empty and nobody noticed, because an empty result looks the same as "no data".

There are five indexes, all built by `scripts/build_dataset_vector_index.py`:

| Index | Node | Purpose |
|---|---|---|
| `datasetEmbedding` | `Dataset` | Dataset-level semantic search |
| `componentEmbedding` | `DatasetComponent` | Per-sub-node semantic search |
| `factSheetEmbedding` | `Dataset` | Fact-sheet ranking for content reasoning |
| `datasetDescriptionFulltext` | `Dataset` | BM25 half of hybrid search |
| `datasetFactSheetFulltext` | `Dataset` | BM25 half of fact-sheet ranking |

`DatasetComponent` is a secondary label added to sub-nodes at embed time — `load_graph.py` does
not set it. Vector indexes are 4096-dim (E5-Mistral-7B-Instruct).

**Two things to internalize before writing Cypher:**

1. **"The field exists" ≠ "the field has data."** Several properties are populated on well under
   10% of nodes. A filter on a 4%-populated field returns a few rows and silently drops the rest
   of the catalog, which is more dangerous than returning nothing. See the Graceful Degradation
   Tiers in `docs/neo4j_schema.md`.
2. **Never `RETURN d` on a `Dataset`.** It carries a 4096-float embedding and the full fact-sheet
   text; dumping one into an LLM context has already caused a production context-window failure.
   Use a map projection:
   `RETURN d{.*, datasetEmbedding: null, factSheetEmbedding: null, factSheetText: null}`.

---

## Orientation Exercises

1. **Verify your environment** — run the import checks above; confirm 8 tools register.

2. **Explore the graph in the browser** (http://localhost:7474):
   ```cypher
   MATCH (d:Dataset) RETURN count(d)
   SHOW INDEXES
   MATCH (dd:DigitalDataset)-[:INPUT_FOR]->(s:Sample) RETURN s.title, collect(dd.title) LIMIT 5
   ```
   Then run the schema audit:
   ```bash
   conda activate rocco
   python scripts/audit_schema.py --folder data/metadata/ --verify
   ```

3. **Read `docs/neo4j_schema.md` in full.** Try 3–5 of the starter queries in the browser. Then
   deliberately run one of them with `INPUT_FOR` reversed and watch it return zero rows without
   erroring — that failure mode is worth feeling once.

4. **Trace one query end to end.** Pick "sandstone datasets with porosity above 0.3" and follow
   it: `conversation_manager.chat()` → gates → ReAct agent → `tools.get_dataset_details` →
   `_needs_content_reasoning` (does not fire) → `graph_store.cypher_qa()` → response assembly.
   Then do the same for "are there paired tomographic and segmented images?" and note where the
   two paths diverge.

5. **Use the app.** `streamlit run rocco_ui.py`, then run a search, ask "tell me more about the
   second one", and follow up with "of these, which are coal?". Watch the logs to see the
   result-set restriction fire.

---

## Development Workflow Quick Reference

| Step | Command |
|------|---------|
| Activate env | `conda activate rocco` |
| Branch from | `main` |
| Branch name | `feature/<short-description>` or `fix/<short-description>` |
| Format code | `black . --line-length 100 && isort .` |
| Run tests | `pytest tests/ -v` |
| PR target | `main` |

See `CONTRIBUTING.md` for commit message format, protected branches, and the PR review process.

---

## Key References

| Resource | Location |
|----------|---------|
| Full documentation | https://digital-porous-media.github.io/dpm_rocco_curator/ |
| Development workflow | `CONTRIBUTING.md` |
| Implementation patterns and gotchas | `CLAUDE.md` |
| Neo4j schema + Cypher | `docs/neo4j_schema.md` |
| Deployment runbook | `DEPLOYMENT.md` |
| DRP domain knowledge | `data/domain_workflows.yaml` |
| Portal tutorials | `data/tutorials.yaml` |
| Task tracker | `Tasks.md` |
| GitHub Project board | https://github.com/orgs/digital-porous-media/projects/3 |
