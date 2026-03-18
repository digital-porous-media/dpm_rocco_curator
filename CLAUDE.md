# Rocco: AI Curator for Digital Porous Media Portal

## Project Overview

**Rocco** is an AI-powered description curator and evaluator for the Digital Porous Media (DPM) Portal. It helps researchers improve dataset descriptions using a rubric-based evaluation framework, retrieval-augmented generation (RAG) from uploaded research papers, and an interactive user feedback loop.

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

**Provider:** SambaNova/TACC (OpenAI-compatible API)
- **Endpoint:** `https://ai.tejas.tacc.utexas.edu/v1`
- **Default model (CLI):** `Llama-4-Maverick-17B-128E-Instruct`
- **UI model:** `Qwen3-32B`

All calls route through `RoccoClient` in `src/llm/client.py`.

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

## How to Run

### Streamlit UI
```bash
streamlit run rocco_ui.py
```

### CLI Evaluation
```bash
python evaluate_description.py <description_text>
```

### Development
```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/
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
