# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Active task board:** https://github.com/orgs/digital-porous-media/projects/3

---

# Rocco: Domain-Agnostic AI Research Assistant Framework

## Project Overview

**Rocco** is a domain-agnostic AI framework for automating research data curation, enhancement, discovery, and educational support. It combines rubric-based evaluation, retrieval-augmented generation (RAG), and interactive feedback workflows to help researchers improve dataset documentation and facilitate knowledge discovery.

**Current Implementation:** Rocco is deployed for porous media datasets in the Digital Porous Media (DPM) Portal, but the framework is designed for extension to any research domain. All domain-specific elements (evaluation rubric, RAG context, metadata schema) are customizable.

---

## Vision & Roadmap

### Phase 1: General AI Research Assistant (In Development — Q3 2026)

Rocco is expanding from description curation to a unified research assistant with three integrated modules:

1. **Description Curator** (Current)
   - Rubric-based evaluation of dataset descriptions
   - RAG-powered enhancement with automatic citation generation
   - Interactive refinement via multi-turn conversation
   - Domain-customizable evaluation criteria

2. **Dataset Discovery Assistant** (Planned)
   - Semantic search + metadata filtering over research datasets
   - Hybrid FAISS + Neo4j for sparse, structured metadata
   - Natural language queries across dataset catalogs
   - See `planning/ADR_Search_Approach.md` for architecture

3. **Educational Support Assistant** (Planned)
   - Domain knowledge Q&A and workflow guidance
   - Dataset best practices and tutorials
   - Research method explanations
   - See `planning/02_General_Assistant.md` for unified architecture

All modules will be accessible via a unified Streamlit interface with shared session history. Development: 2 interns, ~8 weeks (detailed in `planning/04_Tasks.md`).

### Phase 2: Long-Term Roadmap
- **Custom rubric templates** — domain-specific evaluation criteria per research group
- **Bulk processing** — evaluate/enhance entire dataset collections at once
- **External vector stores** — scale RAG with Pinecone, Weaviate (current: local FAISS only)
- **Audit trail & versioning** — full provenance of all description edits
- **Portal integration** — direct API hooks instead of Streamlit-only deployment
- **Multi-language support** — translation-aware evaluation and RAG

---

## Architecture

```
Streamlit UI (rocco_ui.py)
    ↓
DescriptionEvaluator (evaluator.py) → Scores against 10-criterion rubric
    ↓
DescriptionEditor (editor.py) → Improves description with RAG context
    ↓
ContentScreener (content_screener.py) → Validates user feedback
    ↓
Vector Store (FAISS) ← DocumentIngestor (ingestor.py) ← PDF/DOCX uploads
```

## Core Components

### Evaluation
- **10-criterion rubric** (`src/evaluator/rubric.json`) covering completeness, methodology clarity, data organization, quality control, and accessibility
- **Few-shot examples** (`src/evaluator/examples_v3.json`) for few-shot learning
- Scores descriptions on a 0-10 point scale

### Enhancement
- Uses RAG to retrieve relevant excerpts from uploaded papers
- Integrates reviewer feedback from evaluation
- Supports multi-turn refinement via conversation history
- Requires citations for all added/modified information

### Content Screening
- Validates user feedback for relevance, accuracy, and coherence
- Recommends accepting, rejecting, or flagging feedback for review

## LLM Backend

**Provider-Agnostic:** Rocco supports any OpenAI-compatible LLM provider via the `LLMClient` in `src/llm/client.py`.

Supported providers (configure via `.env`):
- **OpenAI** — `gpt-4o`, `gpt-4o-mini` (default)
- **Anthropic** — `claude-opus-4-7`, `claude-sonnet-4-6`
- **Google Gemini** — `gemini-2.0-flash`, `gemini-1.5-pro`
- **DeepSeek** — `deepseek-chat`, `deepseek-reasoner`
- **HuggingFace Inference Providers** — 15+ unified providers (Kimi, Together, etc.) via OpenAI-compatible router
- **Ollama** (local) — `llama3`, `mistral`, etc.
- **SambaNova/TACC** — `Llama-4-Maverick-17B-128E-Instruct`
- **Any OpenAI-compatible endpoint** — Custom base URL + API key

Environment variables:
- `LLM_PROVIDER` — Convenience alias (`openai`, `anthropic`, `huggingface`, `ollama`, etc.) — optional
- `LLM_API_KEY` — Your API key (or `ollama` for local Ollama, `HF_TOKEN` for HuggingFace)
- `LLM_BASE_URL` — Custom endpoint URL (optional if `LLM_PROVIDER` is set)
- `LLM_MODEL` — Model name (defaults to `gpt-4o-mini`)

Example HuggingFace setup:
```bash
LLM_PROVIDER=huggingface
LLM_API_KEY=hf_yourtoken
LLM_MODEL=moonshotai/Kimi-K2-Instruct-0905
# or any model ID from https://huggingface.co/models?pipeline_tag=text-generation
```

All calls route through `RoccoClient` in `src/llm/client.py`, which wraps the `openai` SDK.

## Prompt Management

All LLM prompts are externalized to versioned YAML files in `src/prompts/`, managed via `PromptLoader`:

```yaml
src/prompts/
  evaluator.yaml          # Rubric evaluation prompt
  editor.yaml             # Description enhancement prompt
  content_screener.yaml   # User feedback validation prompt
```

**Versioning:** Semantic versioning (`major.minor.patch`) in each YAML file. Git history is the version control.
- `major` = breaking output format change
- `minor` = new template variables added
- `patch` = wording/clarity tweaks

To use a prompt: `prompt_data = load_prompt("evaluator")` → render with variables via Jinja2 templates.

## Document Chunking & Citations

- **Chunking:** LangChain `RecursiveCharacterTextSplitter` (500 chars, 100 overlap)
- **Supported formats:** PDF, DOCX
- **Metadata enriched at ingest time:**
  - `doc_title`: Filename without extension
  - `page`: Page number (PDF only)
  - `chunk_index`: Sequential chunk number
  - `source`: File path (from LangChain)

**Citation output:** Each statement in enhanced descriptions is traced to its source (original description, uploaded document, or user feedback) with exact quotes.

## Key Data Files

- `src/evaluator/rubric.json` — 10 evaluation criteria (1 point each)
- `src/evaluator/examples_v3.json` — 3 few-shot examples for evaluator
- `src/llm/schemas.py` — Pydantic/dataclass schemas for structured outputs
- `pyproject.toml` — Python dependencies

## Development Setup

### Prerequisites
- Python 3.9+
- LLM API credentials (set `LLM_API_KEY` in `.env`) — see `.env.example` for all supported providers
- For PDF/DOCX processing: `pypdf` and `python-docx` are included

### Initial Setup
```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify environment
python -c "import src; print('Setup successful')"
```

### Code Quality
```bash
# Format code
black . --line-length 100

# Sort imports
isort .

# Both together
black . && isort .
```

## How to Run

### Streamlit UI
```bash
streamlit run rocco_ui.py
```
Starts the web interface for evaluating and editing descriptions with document upload and RAG support.

### CLI Evaluation
```bash
python evaluate_description.py <description_text>
```
Example: `python evaluate_description.py "This is a porous media dataset..."`

### Testing

**IMPORTANT: Always run tests before committing changes.** This catches refactoring regressions early.

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_curator_integration.py -v  # Tests evaluator, editor, screener
pytest tests/test_vector_store.py -v         # Tests embedding and vector store
pytest tests/test_llm_client.py -v           # Tests LLM client configuration

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

**Key Test Coverage:**
- **`test_curator_integration.py`** — Verifies that RoccoClient interface changes don't break evaluator, editor, or screener. This catches the `send_prompt()` AttributeError bug.
- **`test_vector_store.py`** — Tests embedding batch handling, document alignment, and graceful failure recovery. Catches issues from API changes (e.g., switching to SambaNova embeddings).
- **`test_llm_client.py`** — Tests provider routing, configuration, and backward compatibility.

**Before Refactoring:**
1. Make sure all tests pass: `pytest tests/ -v`
2. After refactoring a public method/interface (like `RoccoClient`), verify tests still pass
3. If your change affects evaluator, editor, or screener, the curator integration tests will catch it

### Manual Testing
```bash
# Test RAG pipeline
python test_rag_pipeline.py

# Test evaluation without Streamlit
python -c "from src.evaluator import DescriptionEvaluator; eval = DescriptionEvaluator(); print(eval.evaluate('test description'))"
```

## Output Formats

### Evaluator Output
```json
{
  "rubric_breakdown": [
    {"criterion": "Self-Contained Description", "score": 1, "explanation": "..."},
    ...
  ]
}
```

### Editor Output
```json
{
  "updated_description": "Improved text",
  "rationale": "Key changes made",
  "citations": [
    {
      "statement": "Added fact",
      "source": "uploaded_document",
      "quote": "Original quote from source",
      "doc_title": "paper_name",
      "page": 3,
      "chunk_index": 5
    }
  ]
}
```

### Content Screener Output
```json
{
  "is_relevant": true,
  "is_accurate": true,
  "is_respectful": true,
  "is_coherent": true,
  "issues": [],
  "confidence": 0.95,
  "recommendation": "accept"
}
```

## Session & Context Management

### Session Persistence
- Editing sessions are saved to JSON files (timestamp-based)
- Sessions preserve conversation history, original/current descriptions, and configuration
- Reload sessions to continue iterative refinement

### Multi-Turn Context Management (Streamlit UI)
- **"Manage Context (Prior Turns)"** expander displays after accepting an enhancement
- Users can selectively include/exclude prior feedback rounds for the next enhancement pass
- For each prior turn, users can:
  - **Uncheck** to exclude from next prompt (useful when constraints are no longer relevant)
  - **Edit** feedback text inline (edits persist within the session)
  - **Review context** — see which document chunks were retrieved (title, page, snippet)
  - **Review result** — preview the enhanced description from that turn
- **"Clear history"** button wipes entire enhancement thread and starts fresh
- Conversation history is automatically carried across `enhance()` calls in `DescriptionEditor`

## Configuration

See `src/llm/client.py` for LLM backend configuration (API key, endpoint, model).

Vector store location is configurable in `rocco_ui.py` and `pdf_rag_pipeline.py` (defaults to local FAISS index).

## Key Implementation Patterns

### LLM Client Usage
All LLM calls go through `RoccoClient` (`src/llm/client.py`). This is a thin wrapper around OpenAI-compatible API calls via `langchain-sambanova`. Example usage:

```python
from src.llm.client import RoccoClient

client = RoccoClient()
response = client.call(
    system="You are an evaluator",
    user="Evaluate this description: ...",
    temperature=0.7,
    max_tokens=500,
    model="Qwen3-32B"  # Optional, defaults to config
)
```

### Prompt Loading & Templating
Prompts are loaded via `PromptLoader` and rendered with Jinja2:

```python
from src.prompts.loader import load_prompt

prompt_data = load_prompt("evaluator")  # Loads src/prompts/evaluator.yaml
# prompt_data = {"version": "1.0.0", "system": "...", "user": "..."}

# Render with variables
system = prompt_data["system"].format(rubric_json=rubric_str)
```

Prompt YAML structure:
```yaml
version: "1.0.0"
description: "What this prompt does"
system: |
  You are an expert evaluator...
user: |
  Evaluate: {{ description }}
```

### Data Flow: Description → Evaluation → Enhancement
1. **Input:** Raw description text + optional RAG context
2. **Evaluate** (`src/evaluator/evaluator.py`): Score against rubric, return structured breakdown
3. **Retrieve Context** (`src/retriever/retriever.py`): Query vector store with description, get top-k chunks
4. **Edit** (`src/editor/editor.py`): Use LLM to improve description with context, produce citations
5. **Screen Feedback** (`src/llm/content_screener.py`): Validate user feedback before accepting
6. **Save Session** (`rocco_ui.py`): Preserve conversation history and current state

### Citation System
- Each citation links a statement to its source (document metadata + exact quote)
- Sources: `original_description`, `uploaded_document`, or `user_feedback`
- For uploaded documents: `doc_title`, `page`, `chunk_index` enable tracing back to source PDF/DOCX
- User-friendly naming: "uploaded_document" is more intuitive than technical "context_chunk"

### Session Persistence
Sessions are JSON files with this structure:
```json
{
  "timestamp": "2026-04-23T10:30:00",
  "original_description": "...",
  "current_description": "...",
  "conversation_history": [{"role": "user|assistant", "content": "..."}],
  "evaluations": [{"total_score": 8, "rubric_breakdown": [...]}]
}
```

## Important Notes

### Before Modifying Components
- **Evaluator rubric** (`src/evaluator/rubric.json`): The rubric is domain-customizable (currently porous media). Changing criteria affects the 10-point scale; update examples if you change criteria. For other domains, create a new rubric file and load it via configuration.
- **Few-shot examples** (`src/evaluator/examples_v3.json`): Directly influence evaluator output quality; test thoroughly after changes. Domain-specific examples should be created when deploying to new research areas.
- **Prompt versions** (`src/prompts/*.yaml`): Use semantic versioning; major bumps indicate breaking output format changes
- **Vector store** (FAISS): Rebuilding requires re-ingesting all documents; preserve old indexes during transition
- **LLM calls**: All use the SambaNova endpoint; model names differ between CLI and UI; check `src/llm/client.py` for defaults

### Common Gotchas
- The vector store is persisted locally (FAISS index files). If you change chunking strategy, old indexes won't work with new documents.
- Prompt variables must match the fields used in rendering (e.g., `{{ description }}` in template needs `.format(description=...)`)
- Session files use timestamps; don't manually edit them or session loading may break.
- The `.env` file is in `.gitignore`; API credentials won't be committed.

---

---

## General Assistant Development (Q3 2026)

### Overview

Rocco is being extended with a General Assistant tab — a unified conversational interface for dataset discovery, domain Q&A, and workflow guidance. Development is tracked on the [GitHub Project board](https://github.com/orgs/digital-porous-media/projects/3).

**Branch:** `feature/general-assistant` off `main`. Interns branch off this per task; merges to `main` + tag `v2.0.0` at Week 7.

### New Repo Structure

```
src/assistant/
├── __init__.py
├── tools.py                  # all callable tools — shared interface both interns code to
├── conversation_manager.py   # LangGraph ReAct agent (intent → dispatch → synthesize)
├── assistant.py              # one-line re-export of ConversationManager (backwards compat)
├── graph_store.py            # Neo4j vector index + structured Cypher search
├── literature_search.py      # Semantic Scholar API wrapper (Bernie pre-builds)
├── llm.py                    # RoccoClient + OpenAIEmbeddings singletons (unified with curator)
└── assistant_ui.py           # Streamlit UI — added as new tab in rocco_ui.py Week 6

src/prompts/                  # new prompts alongside existing ones
├── assistant.yaml            # intent classifier (semantic_search / metadata_filter /
│                             #   domain_qa / workflow_guidance / query_expansion / literature_search)
├── query_expander.yaml       # user query → expanded_query + inferred_filters + rationale
└── educational.yaml          # domain Q&A + workflow synthesis

data/
├── tutorials.yaml                 # 20+ user goals → verified portal tutorial URLs
├── domain_workflows.yaml          # 15 DRP workflows (Bernie authors; do not edit without domain review)
└── metadata/                      # scraped DRP metadata JSONs — GITIGNORED

scripts/
├── scrape_metadata.py             # downloads DRP metadata JSONs from TACC Corral
└── build_dataset_vector_index.py  # embeds dataset nodes → Neo4j vector index

tests/assistant/
├── conftest.py                    # fixtures: mock Neo4j driver, small FAISS index
├── test_graph_store.py
└── test_search_integration.txt
```

### Search Architecture

Two-layer search — all handled in `graph_store.py`:

1. **Neo4j vector index** — semantic similarity over dataset nodes (embeddings stored as node properties via `db.index.vector`), queried via `langchain-neo4j` abstractions for vector retrieval. Replaces the old separate FAISS-over-descriptions approach.
2. **Neo4j structured filtering** — exact Cypher queries on `Sample`/`DigitalDataset` properties. Combined with vector search in a single query where possible.

Literature search uses the **Semantic Scholar API** only (see `literature_search.py`). A local full-text PDF corpus was considered but dropped due to copyright concerns — publisher PDFs cannot be legally chunked and stored even under institutional access licenses. Semantic Scholar provides titles, abstracts, DOIs, and citation counts, which is sufficient for the assistant's use cases.

Source labels on all results: `[graph match]`, `[semantic match]`, `[semantic scholar]`.

### Literature Strategy

- **Semantic Scholar** — free API, one `SEMANTIC_SCHOLAR_API_KEY` in server `.env` (no per-user keys); provides titles, abstracts, DOIs, citation counts
- A local full-text PDF corpus was considered and dropped — see §Search Architecture for rationale

### Key Design Constraints

- `graph_store.py` must accept `filters: dict` (not hardcoded field names) — required for future Croissant file-level metadata extension
- `USE_NEO4J=false` env flag must allow the assistant to degrade gracefully (Semantic Scholar still works)
- Session state for the new tab must be namespaced (Bernie adds `curator_` prefix to all existing keys in Week 6)
- Never assert a dataset property that isn't present in the graph — honest gap responses required

### Knowledge Source Policy (conversation_manager.py + educational.yaml)

The system prompt must **not** blanket-restrict the LLM to tool-only knowledge. The right policy is tiered:

| Question type | Policy | Example |
|--------------|--------|---------|
| Dataset facts / portal content | **Tools only** — no pre-trained fallback. Hallucinated dataset properties erode researcher trust. | "How many sandstone datasets have φ > 0.2?" |
| Domain Q&A / workflows | **Tools first** (`domain_workflows.yaml`, Semantic Scholar). Fall back to pre-trained with explicit disclaimer: *"I don't have portal-specific data on this, but generally…"* | "How do I compute relative permeability?" |
| Foundational concepts | **Pre-trained knowledge is fine** — these are stable and well-established. | "What is porosity?" |

**Applies to:**
- `conversation_manager.py` system prompt — replace "do not answer from pre-trained knowledge" with "prefer tool results; for general domain knowledge you may draw on your expertise but make the source clear"
- `educational.yaml` system prompt — same tiered framing; instruct the LLM to distinguish between portal knowledge base results vs. general domain knowledge
- `general_chat` tool in `tools.py` — currently broken because it also forbids pre-trained knowledge while receiving no context; fix by removing that restriction (this tool is a placeholder until `get_educational_context` is wired up)

### Prompts for New Modules

Follow the existing versioned YAML pattern in `src/prompts/`. New files:
- `assistant.yaml` — intent classifier
- `query_expander.yaml` — LLM query expansion
- `educational.yaml` — domain Q&A and workflow synthesis

### Equation Rendering

`data/domain_workflows.yaml` uses plain Unicode/ASCII math (e.g., `k = Q μ L / (A ΔP)`) — this
is intentional and fine for LLM ingestion. Do **not** convert the YAML to LaTeX.

The fix is at the **prompt + rendering layer**:

1. **`educational.yaml` system prompt** must instruct the LLM to output equations in LaTeX
   delimiters: inline `$...$`, block `$$...$$`. Example instruction:
   ```
   When including mathematical expressions, always use LaTeX delimiters:
   inline: $k = Q \mu L / (A \Delta P)$  — block: $$P_c = \frac{2\sigma\cos\theta}{r}$$
   Do not use plain-text math notation in responses.
   ```
2. **`assistant_ui.py`** must render assistant responses via `st.markdown()` (KaTeX renders
   `$...$` automatically). If the model emits `\(...\)` / `\[...\]` style instead, add a
   pre-render substitution:
   ```python
   text = text.replace(r'\(', '$').replace(r'\)', '$')
   text = text.replace(r'\[', '$$').replace(r'\]', '$$')
   st.markdown(text)
   ```

### Vector Indexes

Two indexes are built by `scripts/build_dataset_vector_index.py`:

| Index | Node | Property | Purpose |
|-------|------|----------|---------|
| `datasetEmbedding` | `Dataset` | `datasetEmbedding` | Aggregated dataset-level vector (title + description + sub-node metadata). Used by `GraphStore.search()` and `GraphCypherQAChain`. |
| `componentEmbedding` | `DatasetComponent` | `componentEmbedding` | One vector per `Sample`/`DigitalDataset`/`AnalysisDataset` sub-node. Each blob includes the parent Dataset title + description as a context header so sparse sub-nodes inherit parent signal. Used by `GraphStore.component_search()`. |

`DatasetComponent` is a secondary label added to sub-nodes at embed time — it is not set by `load_graph.py`.

### Index Rebuild

```bash
# Rebuild both Neo4j vector indexes (dataset + component)
python scripts/build_dataset_vector_index.py

# Rebuild publication FAISS
python scripts/build_publication_index.py
```

Re-running is safe — both indexes use `CREATE ... IF NOT EXISTS` and embeddings are upserted with `SET`.
Rebuild required when: new datasets are added, the embedding model changes, or text assembly logic changes.
If switching embedding models, drop old indexes first:
```cypher
DROP INDEX datasetEmbedding IF EXISTS;
DROP INDEX componentEmbedding IF EXISTS;
```

Both are also triggerable via the stretch-goal index management API (`src/assistant/index_api.py`) once implemented.

### LLM / Agent Stack for the Assistant

The assistant module shares the **unified** LLM client with the curator:

| Component | Class | Module | Notes |
|-----------|-------|--------|-------|
| Chat LLM | `RoccoClient` | `src.llm.client` | Inherits from `BaseChatModel`; works with all providers (OpenAI, SambaNova, etc.) |
| Embeddings | `OpenAIEmbeddings` | `langchain_openai` | Custom `EMBEDDING_URL` for provider-specific embedding endpoint |
| Agent | `create_react_agent` | `langgraph.prebuilt` | LangGraph ReAct; replaces legacy `AgentExecutor` (removed in langchain 1.x) |
| Memory | `MemorySaver` | `langgraph.checkpoint.memory` | In-process per-session history; resets on restart |
| Neo4j Vector Search | `Neo4jVector` | `langchain_neo4j` | Vectorstore abstraction for semantic search over dataset embeddings |

Both curator and assistant use `RoccoClient` from `src/llm/client.py`.

**Installation:** langchain-neo4j is in the optional `[graph]` extra:
```bash
pip install -e ".[graph]"  # Includes neo4j, langchain-neo4j, langchain-openai
```

**Conda environment:** All development uses `conda activate rocco_ai`.

### APOC Note

APOC is **not required**. The Cypher generation prompt in `graph_store.py` explicitly forbids `apoc.*` calls, making the code portable across local Neo4j (Homebrew), the TACC VM, and AuraDB.

`langchain-neo4j` provides vectorstore abstractions for semantic search but does not introduce external dependencies — all Cypher queries remain within the Neo4j driver and are portable.

### Week-by-Week Plan (Intern Sprint)

> **Note (May 2026):** Revised for one intern with realistic ramp-up. Week 1 is program orientation (no project work). Bernie is away Weeks 2–3; intern works self-directed. Weeks 3 and 5 are short (4-day). Intern has light capacity in Weeks 8–9 alongside poster/paper.

| Week | Dates | Intern | Bernie |
|------|-------|--------|--------|
| 1 | May 31–Jun 6 | Program orientation — no project work | Pre-sprint prep + front-load: `literature_search.py`, `query_expander.yaml`/`educational.yaml`, `expand_query`/`get_educational_context`/`get_workflow_guidance` in `tools.py`, literature routing, `domain_workflows.yaml`, `tutorials.yaml` cataloguing |
| 2 | Jun 7–13 | Onboarding + Neo4j coverage audit (`audit_schema.py`, Cypher in browser) | Away |
| 3 *(short)* | Jun 14–20 | `publication_corpus.py` — chunk PDFs, tag chunks with dataset IDs from `RelatedPublication` nodes | Away |
| 4 | Jun 21–27 | `graph_store.py`: `semantic_search()` + `filter_by_metadata()` | Back; `conversation_manager.py`; review intern pub corpus output; fill `domain_workflows.yaml` gaps |
| 5 *(short)* | Jun 28–Jul 4 | `graph_store.py`: `search_datasets()` combined query + source labels | Cross-intent queries + docstrings; Semantic Scholar edge cases |
| 6 | Jul 5–11 | `component_search()` + `USE_NEO4J=false` fallback + `tests/assistant/test_graph_store.py` | `assistant_ui.py` shell + `rocco_ui.py` tab integration + pub corpus dedup (#32) |
| 7 | Jul 12–18 | Connect `graph_store.py` to UI; 10-query smoke tests; bug fixes | Prompt review + polish; unblock integration issues |
| 8–9 | Jul 19–Aug 1 | `docs/search_layer.md` + poster/paper | Review drafts |

**Intern owns:** `publication_corpus.py`, `graph_store.py`, `tests/assistant/test_graph_store.py`, `docs/search_layer.md`
**Bernie owns:** `assistant_ui.py`, `conversation_manager.py`, `expand_query`/`get_educational_context`/`get_workflow_guidance` in `tools.py`, `literature_search.py`, all three new prompts (`assistant.yaml`, `query_expander.yaml`, `educational.yaml`), `domain_workflows.yaml`, `tutorials.yaml`, `rocco_ui.py` tab integration
**Deferred post-v2.0.0:** `build_tutorial_index.py` + `tutorial_vector_store/`; separate `docs/assistant_architecture.md` and `docs/adding_tutorials.md`

Project board: https://github.com/orgs/digital-porous-media/projects/3

---

## Recent Changes

### Schema Audit & Reference Doc (May 2026)
- Added `scripts/audit_schema.py` — scans `data/metadata/*.json` to compute node counts,
  % non-null coverage, and distinct enum values for all 7 node labels. Run offline (no Neo4j
  required) or with `--neo4j` for live coverage verification.
- Added `--verify` flag: cross-checks a loaded Neo4j graph for completeness (176 datasets),
  property correctness (title, doi, description, authors, publicationDate), sub-node counts,
  and relationship counts.
- Generated `docs/neo4j_schema.md` — intern Cypher reference doc with full schema, coverage
  percentages, enum value lists, vector index info, starter Cypher queries, and a
  **Graceful Degradation Tiers** guide (always attempt queries; tier governs response messaging
  when results are empty).
- Key finding: all imaging metadata fields (`imagingCenter`, `imagingEquipmentAndModel`, etc.)
  are 0% in current data — assistant must not assume these exist.
- `Tasks.md` updated: `load_graph.py` replaces the old notebook for data loading;
  `audit_schema.py --verify` is now the verification step.

### Vector Index Implementation (May 2026)
- Implemented `scripts/build_dataset_vector_index.py` (was a stub)
- Two indexes: `datasetEmbedding` (Dataset-level aggregation) and `componentEmbedding` (per sub-node)
- Sub-nodes tagged with `DatasetComponent` secondary label; all three types share one index
- `GraphStore.component_search()` added for fine-grained sub-node retrieval
- Fixed `check_embedding_ctx_length=False` on `OpenAIEmbeddings` — TACC/LiteLLM expects raw strings
- Fixed `refresh_schema=False` on `Neo4jGraph` — skips `apoc.meta.data()` (APOC not installed)
- Fixed `src.llm.embeddings` factory: `EMBEDDING_API_KEY` must be the actual API key, not a model name
- 176 Dataset nodes + 3,273 DatasetComponent nodes embedded at dim=4096 (E5-Mistral-7B-Instruct)

### General Assistant Skeleton (May 2026)
- Created `src/assistant/` with working implementations ported from legacy `Chatbot/` folder
- `conversation_manager.py` is the top-level orchestrator (renamed from `assistant.py` to avoid confusion with Intern B's educational work); `assistant.py` is a one-line re-export
- Agent upgraded from legacy `AgentExecutor` (removed in langchain 1.x) to `langgraph.prebuilt.create_react_agent` + `MemorySaver`
- LLM/embeddings for assistant use `ChatOpenAI` + `OpenAIEmbeddings` from `langchain_openai` (provider-agnostic via `.env`; `langchain_sambanova` not required)
- `graph_store.py` documents full Neo4j schema; Neo4j imports are lazy so `USE_NEO4J=false` works without the driver loading
- `scripts/scrape_metadata.py` ported from `CurationTools/ScrapesMetadata.py`
- Credential files (`Chatbot/passwords.py`, `CurationTools/credentials.py`) replaced with `.env` pattern throughout

### Citation Naming (May 2026)
- Renamed `"context_chunk"` source type to **`"uploaded_document"`** for better UX clarity
- Updated everywhere: `src/llm/schemas.py`, `src/prompts/editor.yaml`, all documentation
- No code logic change — just more intuitive naming for end users

### langchain-community Sunset — Migration Planned (May 2026)
- `langchain-community` is being sunset (no new integrations; maintenance-only). See https://github.com/langchain-ai/langchain-community/issues/674.
- **Pinned** to `>=0.4.1,<0.5.0` in `pyproject.toml` to prevent silent breakage.
- **Migration is deferred** to a dedicated branch (`chore/migrate-langchain-community`) once replacement packages stabilize. Do not remove the pin or bump the package without completing the migration below.
- Files that need to change when migrating:
  - `src/ingestor/document_ingestor.py` — `PyPDFLoader` → `langchain-pypdf`; `Docx2txtLoader` → native `python-docx` or standalone package
  - `src/retriever/retriever.py` — `FAISS` → standalone `langchain-faiss` (not yet released as of May 2026) or direct `faiss-cpu` wrapper
  - `tests/test_vector_store.py` — update imports to match retriever
- See `Tasks.md` §langchain-community Migration for the full checklist.

### Unified LLM Client Architecture (May 2026)
- **Refactored `RoccoClient`** to inherit from both `LLMClient` and `BaseChatModel`
  - Eliminates separate `ChatOpenAI` layer in `src/assistant/llm.py`
  - Single source of truth for all LLM configuration (curator + assistant)
  - Works directly with LangChain/LangGraph (implements `BaseChatModel` interface)
- **Deleted `src/assistant/rocco_chat_model.py`** — adapter pattern no longer needed
- **Updated `src/assistant/llm.py`** — now instantiates `RoccoClient` directly
- **Pydantic integration** — declared all `RoccoClient` attributes as Pydantic fields to work with `BaseChatModel`
  - Fields: `provider`, `api_key`, `api_url`, `model`, `timeout`, `temperature`
  - `client` and `logger` use `exclude=True` (instance-only, not serialized)
- **Pattern**: Inherit from both parent classes, call `BaseChatModel.__init__` first (Pydantic setup), then `LLMClient.__init__` (environment config)
- **Benefit**: Curator and assistant now share identical LLM configuration with zero duplication

### GraphStore Raw-Driver Integration (June 2026, Week 4)
- **Merged `julia/graph-store` branch** — JRS's standalone `Neo4jGraphStore` implementation folded into `src/assistant/graph_store.py`
- **Unified architecture** — `GraphStore` now has two code layers:
  - **High-level** (existing, Week 1–3): `search()`, `component_search()`, `cypher_qa()` via langchain-neo4j — used by `tools.py`
  - **Low-level** (new, Week 4–5): `semantic_search()`, `filter_by_metadata()`, `search_datasets()`, `execute_cypher()`, `get_schema_blueprint()` — raw `neo4j.GraphDatabase.driver` for JRS's Week 5 hybrid queries
- **Injection safety** — added `_SAFE_KEY_RE`, `_validate_keys()`, `_build_where_clause()` helpers from JRS
- **SearchResult dataclass** — low-level methods return `SearchResult` (dataset_id, score, properties) while high-level methods continue returning dicts
- **Bug fixes on merge**:
  - Fixed default `index_name` from `"dataset-embeddings"` → `"datasetEmbedding"` (matches Neo4j index name from `build_dataset_vector_index.py`)
  - Fixed test patch paths from `"neo4j_graph_store.*"` → `"src.assistant.graph_store.*"`
- **Tests** — migrated `test_graph_store.py` from top-level to `tests/assistant/test_graph_store.py` (22 tests, all passing)
- **Zero churn** — kept `GraphStore` class name (not renamed to `Neo4jGraphStore`), no changes to `tools.py` imports, existing high-level interface unchanged

