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

## Pre-Sprint: Bernie Before Week 2 (May 27 – Jun 6)

Intern starts Jun 7. Bernie is away Weeks 2–3. Everything below must be done before Jun 7.

### Must be done this week (May 27–30) — intern is blocked without these

- [ ] Download curated papers into `data/curated_papers/` (#13) — intern needs these for Week 3
- [ ] Write intern onboarding doc (#16)
- [ ] Record or write Rocco codebase walkthrough (#19)
- [ ] Write `CONTRIBUTING.md` (#14)
- [ ] Add pytest scaffolding `tests/assistant/conftest.py` (#15)
- [ ] Confirm intern has repo access, `rocco_ai` env setup instructions, `.env.example`, Neo4j

### Must be done during Week 1 (May 31 – Jun 6) — intern is in orientation; Bernie works self-directed

- [ ] Author `data/domain_workflows.yaml` — 15 DRP workflows (#11)
- [ ] Draft `data/tutorials.yaml` skeleton — 10 verified entries (#12)
- [ ] Catalogue DPM Portal tutorials; extend `tutorials.yaml` to 20+ entries (#23)
- [ ] Implement `src/assistant/literature_search.py` (#17)
- [ ] Draft `src/prompts/query_expander.yaml` + `educational.yaml` (#24)
- [ ] Implement `expand_query`, `get_educational_context`, `get_workflow_guidance` in `tools.py` (#29)
- [ ] Wire two-tier literature routing into `tools.py` (#30)
- [ ] Define 20 representative test queries (#18)
- [ ] Finish infra: load graph, verify 176 datasets (see Environment & Infrastructure above)

---

## GitHub Project Board — Updated May 27, 2026

Board updated to reflect revised schedule (Bernie front-loads to Week 1; away Weeks 2–3).

### Reassigned to Bernie
- `#23` Catalogue `tutorials.yaml` → Bernie, week=1
- `#25` Prototype `assistant_ui.py` → Bernie, week=6
- `#32` Integrate publication corpus into search; handle dedup → Bernie, week=6

### Rescheduled
- `#24` `query_expander.yaml` + `educational.yaml` → week=1 (front-loaded)
- `#29` `expand_query`, educational tools → week=1 (front-loaded; was week=3)
- `#30` Literature routing → week=1 (front-loaded; was week=3)
- `#31` Review intern search output + domain_workflows gaps → week=4 (was week=3; Bernie away)
- `#36` `USE_NEO4J=false` fallback + docstrings → week=6 (was week=5; matches revised intern scope)

### Closed
- `#28` Source labels + 10 search queries → closed; scope folded into `#52`

### Created
- `#52` [JRS] Week 5: `search_datasets()` combined query + source labels (Intern-A, P0)

---

## Notes

- **Conda env:** Always `conda activate rocco_ai` before running anything
- **Neo4j notebook:** Use `CurationTools/JsonToNeo4jwKeywords.ipynb` (not the chunking one) to load data
- **`langchain-neo4j`** is installed via `pip install -e ".[graph]"` and provides `Neo4jVector` for semantic search abstractions
- **`langchain_sambanova`** is not installed — embeddings use `OpenAIEmbeddings` with custom base URL instead; if SambaNova's embedding endpoint is not OpenAI-compatible, update `src/assistant/llm.py`
- **`Chatbot/` and `CurationTools/`** — kept as historical reference; do not delete before intern Week 1
