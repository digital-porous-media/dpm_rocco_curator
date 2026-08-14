# Intern Onboarding Guide

Welcome to the Rocco project. This document covers everything you need to get started: project context, environment setup, a walkthrough of the codebase, and your first-week tasks. Read `CONTRIBUTING.md` alongside this for workflow rules (branching, commit style, PR process).

---

## Project Context

**Rocco** is an AI assistant framework for the [Digital Porous Media (DPM) Portal](https://www.digitalrocksportal.org/), a community repository for micro-CT datasets from porous rock samples (sandstones, carbonates, shales, etc.). Two modules are in the codebase:

| Module | Status | What it does |
|--------|--------|--------------|
| **Description Curator** | Working | Evaluates and enhances dataset descriptions using a rubric + RAG |
| **General Assistant** | In development | Conversational search, domain Q&A, and workflow guidance over the dataset catalog |

You are working on the **General Assistant**. Your primary deliverable is `src/assistant/graph_store.py` and `src/assistant/publication_corpus.py` — the search layer that the assistant's tools call into.

**Contact:** Bernie Chang (async OK via Slack or email). Check the [GitHub Project board](https://github.com/orgs/digital-porous-media/projects/3) for your assigned issues and weekly milestones.

---

## Prerequisites

Before starting, confirm you have:

- [ ] Python 3.11 via `conda` (Miniforge recommended)
- [ ] `git` and access to `github.com/digital-porous-media/dpm_rocco_curator`
- [ ] Neo4j ≥ 5.x installed locally (Homebrew: `brew install neo4j`)
- [ ] LLM API credentials from Bernie (added to `.env` — see below)
- [ ] Semantic Scholar API key from Bernie (free; added to `.env`)

---

## Deployment Context

| Environment | Neo4j location | Notes |
|-------------|---------------|-------|
| **Local development** | Your laptop via Homebrew | Default for intern work |
| **Production (planned)** | TACC VM | Same schema; different `NEO4J_URI` and credentials |

Never commit credentials. Always use `.env` (gitignored). The `USE_NEO4J=false` flag lets the assistant fall back to FAISS-only search if Neo4j is unavailable.

---

## First-Day Setup

```bash
# 1. Clone and enter the repo
git clone git@github.com:digital-porous-media/dpm_rocco_curator.git
cd dpm_rocco_curator

# 2. Create and activate the conda environment
conda create -n rocco_ai python=3.11
conda activate rocco_ai

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

# 6. Verify imports
python -c "from src.assistant.tools import build_langchain_tools; print('OK')"
python -c "import os; os.environ['USE_NEO4J']='false'; from src.assistant.graph_store import GraphStore; GraphStore(); print('OK')"

# 7. Run tests — all should pass before you write any code
pytest tests/ -v

# 8. Launch the Streamlit UI (curator tab only until Week 7)
streamlit run rocco_ui.py
```

---

## Codebase Walkthrough

### Top-Level Files

| File | Purpose |
|------|---------|
| `rocco_ui.py` | Streamlit entry point — all UI tabs live here |
| `pyproject.toml` | Package dependencies; `[graph]` extra adds Neo4j/LangChain |
| `.env.example` | Template for all required environment variables |
| `CONTRIBUTING.md` | Branching, commit style, code quality, PR process |
| `Tasks.md` | Pre-sprint task tracker (Bernie's planning doc) |

### `src/` Module Map

```
src/
├── llm/           LLM client, schemas, content screening
├── evaluator/     Rubric evaluation (Curator module)
├── editor/        Description enhancement with citations (Curator module)
├── ingestor/      PDF/DOCX → chunked FAISS documents
├── retriever/     FAISS vector store queries
├── prompts/       All YAML prompt templates
└── assistant/     General Assistant — your workspace
```

#### `src/llm/`

`client.py` contains `RoccoClient` — the single LLM call interface used by every module (curator and assistant). It inherits from both `LLMClient` (provider routing) and LangChain's `BaseChatModel` (LangGraph compatibility). Provider is configured via `.env`; you do not need to touch this file.

`schemas.py` defines Pydantic output schemas for structured LLM responses (evaluator, editor, screener).

#### `src/prompts/`

All LLM prompts are externalized as versioned YAML files and rendered with Jinja2. Never embed prompts in Python code — add a new YAML file and load it with `load_prompt("filename")` from `src/prompts/loader.py`.

| Prompt file | Used by |
|-------------|---------|
| `evaluator.yaml` | Description Curator |
| `editor.yaml` | Description Curator |
| `content_screener.yaml` | Description Curator |
| `assistant.yaml` | Intent classifier (General Assistant) |
| `query_expander.yaml` | Query expansion (General Assistant) |
| `educational.yaml` | Domain Q&A + workflow guidance (General Assistant) |

#### `src/assistant/` — Your Primary Workspace

This is where you spend most of your time.

**`tools.py`** — The shared interface. Every function the LangGraph agent can call is defined here as a LangChain `Tool`. Both interns code to this interface. Do not change a tool's name or signature without coordinating with Bernie — the agent prompt references tool names.

**`conversation_manager.py`** — The LangGraph ReAct agent. Receives a user message, classifies intent (via `assistant.yaml`), dispatches to the appropriate tool, synthesizes a response. Bernie builds this; you do not own it, but you need to understand it.

**`graph_store.py`** — **You own this.** Neo4j vector index + Cypher search. Two methods Bernie already scaffolded:
- `search(query, top_k)` — semantic search over `Dataset` nodes via `datasetEmbedding` index
- `component_search(query, top_k)` — fine-grained search over `DatasetComponent` nodes (Sample, DigitalDataset, AnalysisDataset)

Your Week 4 task adds `semantic_search()` and `filter_by_metadata()`. Week 5 adds `search_datasets()` (combined query + source labels). Week 6 adds `component_search()` and the `USE_NEO4J=false` fallback.

**`publication_corpus.py`** — **You own this.** FAISS index over chunked PDFs from `data/curated_papers/`. Currently a stub. You implement it in Week 3: chunk PDFs with `DocumentIngestor`, tag chunks with dataset IDs from `RelatedPublication` graph nodes, persist as a FAISS index.

**`literature_search.py`** — Semantic Scholar API wrapper. Bernie pre-built this. You call it from `tools.py` for the literature routing fallback.

**`llm.py`** — Returns singletons: `get_llm()` → `RoccoClient`, `get_embeddings()` → `OpenAIEmbeddings`. The embeddings singleton is what `GraphStore` uses for vector search.

**`assistant_ui.py`** — Streamlit tab for the General Assistant. Bernie builds this in Week 6. You connect `graph_store.py` to it in Week 7.

### Data and Scripts

| Path | What it is |
|------|-----------|
| `data/domain_workflows.yaml` | 15 DRP workflows; ground truth for `workflow_guidance` responses. Read-only — do not edit without domain review. |
| `data/tutorials.yaml` | 20+ portal tutorial notebook paths mapped to user goals. |
| `data/metadata/` | **Gitignored.** Scraped DRP metadata JSONs (one per dataset). Download with `scripts/scrape_metadata.py`. |
| `data/curated_papers/` | **Gitignored.** Licensed PDFs. Do not commit. |
| `data/publication_vector_store/` | **Gitignored.** Generated FAISS index from `scripts/build_publication_index.py`. |
| `scripts/scrape_metadata.py` | Downloads metadata JSONs from TACC Corral. |
| `scripts/build_dataset_vector_index.py` | Embeds 176 Dataset nodes + 3273 DatasetComponent nodes into Neo4j vector indexes. |
| `scripts/audit_schema.py` | Audits Neo4j graph completeness; generates `docs/neo4j_schema.md`. |
| `docs/neo4j_schema.md` | Your Cypher reference: full schema, coverage stats, starter queries. Read this before writing any Cypher. |

### Tests

```
tests/
├── conftest.py                        # Global fixtures (mock Neo4j, in-memory FAISS)
├── test_curator_integration.py        # Evaluator + editor + screener
├── test_llm_client.py                 # LLM provider routing
├── test_vector_store.py               # FAISS embedding and alignment
└── assistant/
    ├── __init__.py
    └── test_intent_classifier.py      # Intent classifier prompt tests
```

The `conftest.py` fixtures mock the Neo4j driver and provide a small in-memory FAISS index — you do not need Neo4j running to run unit tests. You will add `tests/assistant/test_graph_store.py` and `tests/assistant/test_publication_corpus.py` as you implement those modules (Week 6).

**Always run `pytest tests/ -v` before pushing.** The curator integration tests will catch interface regressions even in assistant code.

---

## Neo4j Graph Overview

The DRP Portal catalog is stored as a property graph. Nodes and relationships:

```
(Dataset)-[:HAS_SAMPLE]->(Sample)
(Dataset)-[:HAS_DIGITAL_DATASET]->(DigitalDataset)
(Dataset)-[:HAS_ANALYSIS_DATASET]->(AnalysisDataset)
(Dataset)-[:HAS_PUBLICATION]->(Publication)
(Sample)-[:HAS_DIGITAL_DATASET]->(DigitalDataset)
(DigitalDataset)-[:HAS_ANALYSIS_DATASET]->(AnalysisDataset)
```

176 `Dataset` nodes; 3,273 `DatasetComponent` nodes (Samples + DigitalDatasets + AnalysisDatasets).

Two vector indexes (built by `scripts/build_dataset_vector_index.py`):
- `datasetEmbedding` on `Dataset` nodes — aggregated metadata blob, dim=4096
- `componentEmbedding` on `DatasetComponent` nodes — per sub-node blob with parent context, dim=4096

Read `docs/neo4j_schema.md` for full property coverage, enum values, and starter Cypher queries. Key constraint: **all imaging metadata fields (`imagingCenter`, `imagingEquipmentAndModel`, etc.) are 0% populated** — do not query or assert these exist.

---

## Week 2 Orientation Tasks

Your first week is onboarding; no deliverable code is expected. Complete these in order:

1. **Verify your environment** — run the import checks above and confirm `pytest tests/ -v` passes.

2. **Start Neo4j and verify the graph:**
   ```bash
   neo4j start
   # Open http://localhost:7474
   # Run: MATCH (d:Dataset) RETURN count(d)   → expect 176
   # Run: SHOW INDEXES                         → look for datasetEmbedding, componentEmbedding
   ```
   Then run the schema audit:
   ```bash
   conda activate rocco_ai
   python scripts/audit_schema.py --folder data/metadata/ --verify
   ```

3. **Read `docs/neo4j_schema.md` in full.** It is your Cypher reference for the whole project. Try 3–5 of the starter queries in the browser.

4. **Read `src/assistant/graph_store.py` in full.** Understand `search()` and `component_search()`. Trace how a query flows from `tools.py` → `graph_store.py` → Neo4j → returned result dict.

5. **Run `pytest tests/ -v`** — all tests should pass. If one fails, investigate before writing any code.

6. **Open `src/assistant/publication_corpus.py`** — it is a stub. This is what you implement in Week 3. Read the docstring and the imports to understand what it should do.

---

## Development Workflow Quick Reference

| Step | Command |
|------|---------|
| Activate env | `conda activate rocco_ai` |
| Branch from | `feature/general-assistant` (not `main`) |
| Branch name | `feature/<task-id>` or `fix/<task-id>` |
| Format code | `black . --line-length 100 && isort .` |
| Run tests | `pytest tests/ -v` |
| PR target | `feature/general-assistant` |

See `CONTRIBUTING.md` for full details on commit message format, protected branches, and the PR review process.

---

## Key References

| Resource | Location |
|----------|---------|
| Development workflow | `CONTRIBUTING.md` |
| Neo4j schema + Cypher | `docs/neo4j_schema.md` |
| DRP domain knowledge | `data/domain_workflows.yaml` |
| Portal tutorials | `data/tutorials.yaml` |
| Task tracker | `Tasks.md` |
| GitHub Project board | https://github.com/orgs/digital-porous-media/projects/3 |
