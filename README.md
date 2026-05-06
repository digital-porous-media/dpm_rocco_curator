# Rocco: AI Curator for Dataset Descriptions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Rocco is an AI-powered description curator and evaluator for the [Digital Porous Media (DPM) Portal](https://digitalporousmedia.org/). It helps researchers improve dataset descriptions using a rubric-based evaluation framework, retrieval-augmented generation (RAG) from uploaded research papers, and an interactive user feedback loop.

## Features

✨ **Description Evaluation**: Score descriptions against a 10-criterion rubric covering completeness, clarity, data organization, quality control, and accessibility.

✨ **RAG-Powered Enhancement**: Automatically improve descriptions using relevant excerpts from your uploaded research papers and documents.

✨ **Interactive Feedback**: Validate and integrate user feedback into refined descriptions with full citation tracking.

✨ **Multi-LLM Support**: Use OpenAI, Anthropic, Ollama, DeepSeek, Gemini, or any OpenAI-compatible LLM provider.

✨ **Session Persistence**: Save and reload description refinement sessions to continue iterative improvement.

## Quick Start

### Prerequisites
- Python 3.9 or higher
- An API key from one of the supported LLM providers (or a local Ollama instance)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/USER/dpm-rocco-curator.git
   cd dpm-rocco-curator
   ```

2. **Install in editable mode:**
   ```bash
   pip install -e .
   ```

3. **Configure your LLM provider:**
   ```bash
   cp .env.example .env
   # Edit .env with your API key and preferred LLM provider
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run rocco_ui.py
   ```

   The app will open at `http://localhost:8501`

### Configuration

Rocco supports any **OpenAI-compatible** LLM provider. Configure your preferred provider in `.env`:

| Provider | LLM_PROVIDER | Example API Key | Default Model |
|----------|------------|-----------------|----------------|
| **OpenAI** | `openai` | `sk-proj-...` | `gpt-4o-mini` |
| **Anthropic** (Claude) | `anthropic` | `sk-ant-...` | `claude-opus-4-7` |
| **Google Gemini** | `gemini` | `AIza...` | `gemini-2.0-flash` |
| **DeepSeek** | `deepseek` | `sk-...` | `deepseek-chat` |
| **HuggingFace** | `huggingface` | `hf_...` | `meta-llama/Llama-3.1-8B-Instruct` |
| **Ollama (Local)** | `ollama` | (auto-set) | `llama3` or other local model |
| **SambaNova (TACC)** | `sambanova` | `sk-...` | `Llama-4-Maverick-17B-128E-Instruct` |
| **Custom OpenAI-compatible** | `openai_compatible` | Your key | Your model |

**Note:** Rocco requires OpenAI-compatible `/v1/chat/completions` endpoints. For other APIs (e.g., native HuggingFace), wrap with an adapter or use a proxy like Text Generation Inference (TGI).

**Example `.env` configuration for OpenAI:**
```
LLM_PROVIDER=openai
LLM_API_KEY=sk-proj-your-key-here
LLM_MODEL=gpt-4o-mini
```

**Example `.env` configuration for Ollama (local):**
```
LLM_PROVIDER=ollama
# api_key is auto-set to "ollama"
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3
```

See [`.env.example`](.env.example) for all supported providers and their base URLs.

## Usage Workflow

1. **Enter Description**: Paste your dataset description in the text area.

2. **Evaluate**: Click "Evaluate Description" to score it against Rocco's 10-criterion rubric.

3. **Upload Context Documents**: Upload relevant PDFs or DOCX files (optional). Rocco will build a vector index for RAG.

4. **Provide Feedback**: Write specific suggestions for improvement.

5. **Enhance**: Click "Enhance with Rocco" to generate an improved description using RAG context and your feedback.

6. **Review & Refine**: See the enhanced description with citations. Save or discard, and iterate.

7. **Export**: Download the refined description along with evaluation results and session history.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            Streamlit UI (rocco_ui.py)               │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌────────────────────────────────────────────────┐ │
│  │  DescriptionEvaluator (evaluator.py)          │ │
│  │  → Scores against 10-criterion rubric          │ │
│  └────────────────────────────────────────────────┘ │
│                        ↓                             │
│  ┌────────────────────────────────────────────────┐ │
│  │  DocumentIngestor + VectorStore (FAISS)        │ │
│  │  → Chunks PDFs/DOCX, embeds for RAG            │ │
│  └────────────────────────────────────────────────┘ │
│                        ↓                             │
│  ┌────────────────────────────────────────────────┐ │
│  │  DescriptionEditor (editor.py)                 │ │
│  │  → Improves description with RAG context       │ │
│  └────────────────────────────────────────────────┘ │
│                        ↓                             │
│  ┌────────────────────────────────────────────────┐ │
│  │  ContentScreener (content_screener.py)         │ │
│  │  → Validates user feedback                     │ │
│  └────────────────────────────────────────────────┘ │
│                        ↓                             │
│  ┌────────────────────────────────────────────────┐ │
│  │  RoccoClient (llm/client.py)                   │ │
│  │  → Provider-agnostic OpenAI SDK wrapper        │ │
│  └────────────────────────────────────────────────┘ │
│                        ↓                             │
│  ┌────────────────────────────────────────────────┐ │
│  │  LLM Provider (OpenAI, Anthropic, Ollama, etc) │ │
│  └────────────────────────────────────────────────┘ │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## Core Components

### Evaluation Rubric (src/evaluator/)
- **rubric.json**: 10 criteria (1 point each) evaluating completeness, clarity, methodology, organization, data access, quality control, and more.
- **evaluator.py**: Scores descriptions and returns structured feedback.
- **examples_v3.json**: Few-shot examples for improved evaluation consistency.

### RAG Pipeline (src/ingestor/ + src/retriever/)
- **DocumentIngestor**: Chunks PDFs and DOCX files into 500-character segments with metadata.
- **DocumentEmbedder**: Uses `sentence-transformers` (BAAI/bge-large-en-v1.5) for semantic embeddings.
- **VectorStoreManager**: FAISS-based vector store for similarity search and context retrieval.

### Enhancement & Screening (src/editor/ + src/llm/)
- **DescriptionEditor**: Takes original description + RAG context + user feedback → improved description with citations.
- **ContentScreener**: Validates feedback for relevance, accuracy, and coherence before integration.
- **RoccoClient**: Unified interface to OpenAI-compatible LLM endpoints.

### Prompts (src/prompts/)
All LLM prompts are externalized as YAML files with semantic versioning:
- **evaluator.yaml**: Rubric scoring prompt
- **editor.yaml**: Description enhancement prompt
- **content_screener.yaml**: Feedback validation prompt

## Installation for Development

To contribute to Rocco:

```bash
git clone https://github.com/USER/dpm-rocco-curator.git
cd dpm-rocco-curator
pip install -e ".[dev]"
```

### Code Style

Format code before committing:

```bash
black . --line-length 100
isort .
```

### Running Tests

```bash
pytest tests/
pytest -v tests/test_file.py  # Single file with verbose output
```

## Documentation

For detailed architecture, configuration, and development guidance, see:

- **[CLAUDE.md](CLAUDE.md)** — Developer guide (components, patterns, implementation details)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines and code standards
- **[.env.example](.env.example)** — Detailed LLM provider configuration reference

## Output Formats

### Evaluation Output
```json
{
  "rubric_breakdown": [
    {
      "criterion": "Self-Contained Description",
      "score": 1,
      "explanation": "..."
    }
  ],
  "total_score": 8,
  "feedback": "..."
}
```

### Enhanced Description Output
```json
{
  "updated_description": "Improved text...",
  "rationale": "Key changes made...",
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

## Citation

If you use Rocco in your research, please cite it using the DOI from Zenodo:

[![DOI](https://zenodo.org/badge/...)](https://zenodo.org/record/...)

```bibtex
@software{rocco2024,
  author = {DPM Rocco Contributors},
  title = {Rocco: AI Curator for Dataset Descriptions},
  year = {2024},
  doi = {10.5281/zenodo.YOUR-DOI},
  url = {https://github.com/USER/dpm-rocco-curator}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome bug reports, feature requests, and pull requests! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/USER/dpm-rocco-curator/issues)
- **Questions**: Open a discussion on [GitHub Discussions](https://github.com/USER/dpm-rocco-curator/discussions)

## Acknowledgments

Rocco is developed as part of the Digital Porous Media (DPM) Portal initiative to support dataset discovery and curation in geoscience and engineering research.

---

**Ready to improve your dataset descriptions?** Start by following the [Quick Start](#quick-start) guide above!
