# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# Rocco: AI Curator for Digital Porous Media Portal

## Project Overview

**Rocco** is an AI-powered description curator and evaluator for the Digital Porous Media (DPM) Portal. It helps researchers improve dataset descriptions using a rubric-based evaluation framework, retrieval-augmented generation (RAG) from uploaded research papers, and an interactive user feedback loop.

---

## Expansion: General AI Assistant (In Development)

Rocco is being expanded into a broader AI assistant suite with two new capabilities:

1. **Dataset Discovery** — semantic search + metadata filtering over 176 datasets using hybrid FAISS + Neo4j
2. **Educational Support** — domain knowledge Q&A, workflow guidance, and tutorials

These will be integrated as a new tab in `rocco_ui.py`, alongside the existing Curator. See `planning/` for detailed planning:
- **`planning/02_General_Assistant.md`** — unified architecture (discovery + education as one conversational assistant)
- **`planning/ADR_Search_Approach.md`** — why hybrid FAISS + Neo4j works for sparse metadata
- **`planning/04_Tasks.md`** — week-by-week intern work breakdown

The expansion is in **planning phase** (2 interns, ~8 weeks). No code changes to the existing Curator yet.

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

**Citation output:** Each statement in enhanced descriptions is traced to its source (original description, context chunk, or user feedback) with exact quotes.

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
```bash
# All tests
pytest tests/

# Single test file
pytest tests/test_file.py

# With verbose output
pytest -v
```

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
      "source": "context_chunk",
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

## Session Management

- Editing sessions are saved to JSON files (timestamp-based)
- Sessions preserve conversation history, original/current descriptions, and configuration
- Reload sessions to continue iterative refinement

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
- Sources: `original_description`, `context_chunk`, or `user_feedback`
- For context chunks: `doc_title`, `page`, `chunk_index` enable tracing back to source PDF/DOCX

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
- **Evaluator rubric** (`src/evaluator/rubric.json`): Changing criteria affects the 10-point scale; update examples if you change criteria
- **Few-shot examples** (`src/evaluator/examples_v3.json`): Directly influence evaluator output quality; test thoroughly after changes
- **Prompt versions** (`src/prompts/*.yaml`): Use semantic versioning; major bumps indicate breaking output format changes
- **Vector store** (FAISS): Rebuilding requires re-ingesting all documents; preserve old indexes during transition
- **LLM calls**: All use the SambaNova endpoint; model names differ between CLI and UI; check `src/llm/client.py` for defaults

### Common Gotchas
- The vector store is persisted locally (FAISS index files). If you change chunking strategy, old indexes won't work with new documents.
- Prompt variables must match the fields used in rendering (e.g., `{{ description }}` in template needs `.format(description=...)`)
- Session files use timestamps; don't manually edit them or session loading may break.
- The `.env` file is in `.gitignore`; API credentials won't be committed.
