# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### General Assistant
- **Conversational research assistant** (`src/assistant/`): a new Streamlit tab alongside the
  Description Curator, powered by a LangGraph ReAct agent (`ConversationManager`)
  - **Dataset Discovery**: semantic search over a Neo4j vector index (dataset- and
    component-level embeddings) combined with structured Cypher filtering for exact/numeric
    property queries (rock type, porosity, voxel size, named authors, etc.)
  - **Domain Q&A & Workflow Guidance**: curated digital rock physics workflows
    (`data/domain_workflows.yaml`) and portal tutorials (`data/tutorials.yaml`), with a
    tiered knowledge-source policy (tools-only for dataset facts, tools-first-with-disclaimer
    for domain Q&A, pre-trained knowledge for foundational concepts)
  - **Literature Search**: Semantic Scholar API integration (titles, abstracts, DOIs,
    citation counts)
  - **Portal Documentation Search**: PageIndex-style heading-tree retrieval over the DPM
    Portal's user documentation
  - **Source-labeled responses**: every answer is tagged with its origin (`[graph match]`,
    `[cypher match]`, `[semantic scholar]`, `[portal docs]`, etc.), rendered as colored badges
  - **Graceful degradation**: `USE_NEO4J=false` disables dataset graph search only; all other
    capabilities keep working without a Neo4j connection

#### Unified LLM/Agent Stack
- `RoccoClient` now inherits from both `LLMClient` and `BaseChatModel`, giving the curator and
  assistant identical, zero-duplication LLM configuration
- Agent stack: `langgraph.prebuilt.create_react_agent` + `MemorySaver` (replaces the legacy
  `AgentExecutor`, removed in LangChain 1.x)
- Neo4j vector search via `langchain-neo4j` (optional `graph` extra:
  `pip install -e ".[graph]"`)

### Documentation
- New `docs/user_guide/assistant.rst` — General Assistant user guide (capabilities, source
  badge reference, knowledge-source policy)
- New General Assistant sections in `README.md`, `docs/index.rst`,
  `docs/developer_guide/architecture.rst`, `docs/developer_guide/api_reference.rst`,
  `docs/user_guide/quickstart.rst`, `docs/user_guide/installation.rst`, and
  `docs/user_guide/configuration.rst`

## [1.0.0] - 2026-05-13

### Added

#### Core Evaluation System
- **Description Evaluator**: 10-criterion rubric-based evaluation framework for research dataset descriptions
  - Customizable rubric for domain-specific evaluation criteria
  - Few-shot learning with example-based scoring
  - Structured scoring breakdown (0-10 point scale) with detailed explanations
  - Porous Media domain implementation included

#### RAG-Powered Enhancement Pipeline
- **Description Editor**: LLM-powered description improvement with citation tracking
  - Retrieves relevant context from uploaded research documents (PDF, DOCX)
  - Integrates user feedback and prior refinement iterations
  - Multi-turn conversational enhancement workflow
  - Full citation system with exact quotes and source metadata

#### Document Processing & Retrieval
- **Document Ingestion**: Support for PDF and DOCX file uploads
  - Chunking via LangChain `RecursiveCharacterTextSplitter`
  - Rich metadata enrichment (title, page number, chunk index)
- **Vector Store**: FAISS-based similarity search for context retrieval
  - Semantic search over document corpus
  - Configurable chunk size and overlap

#### Content Validation
- **Content Screener**: Validates user feedback before acceptance
  - Relevance, accuracy, coherence, and respectfulness checks
  - Structured feedback validation output
  - Confidence scoring and recommendation system

#### User Interface
- **Streamlit Web Application** (`rocco_ui.py`): Interactive evaluation and enhancement workflow
  - Upload documents and provide descriptions
  - Real-time evaluation with rubric breakdown
  - Iterative description refinement with conversation history
  - Session management and persistence (JSON-based)
  - Context management for multi-turn refinement
  - Reviewer feedback integration

#### Configuration & Extensibility
- **Provider-Agnostic LLM Backend**: Support for multiple LLM providers
  - OpenAI (GPT-4o, GPT-4o-mini)
  - Anthropic (Claude Opus, Claude Sonnet)
  - Google Gemini
  - DeepSeek
  - HuggingFace Inference (15+ providers)
  - Ollama (local)
  - SambaNova
  - Any OpenAI-compatible endpoint

- **Externalized Prompt Management**: YAML-based prompt versioning
  - Semantic versioning for prompts
  - Jinja2 templating support
  - Easy customization for different domains

#### Documentation & Development
- **Sphinx Documentation**: Complete developer and user guides
  - Architecture diagrams
  - API reference
  - Contribution guidelines
  - Setup and configuration instructions
- **Comprehensive Test Suite**: Unit and integration tests for all components
- **Development Tools**: Code formatting (black), import sorting (isort), testing (pytest)

### Technical Details

- **Python 3.9+** compatibility
- **Domain-Agnostic Design**: All domain-specific elements (rubric, RAG context, schemas) are customizable for extension to other research domains
- **Session Persistence**: JSON-based session storage with full conversation history
- **Citation Tracing**: Every enhanced statement links to its source with exact quotes and metadata

### Known Limitations

- Vector store is local FAISS only
- No built-in support for bulk document processing
- No audit trail or versioning of description edits
- Single-language support

### Future Work

Planned features are tracked in `CLAUDE.md` under "Vision & Roadmap":
- **Dataset Discovery Assistant**: Semantic search and metadata filtering over research datasets
  *(shipped — see `[Unreleased]` above)*
- **Educational Support Assistant**: Domain knowledge Q&A and workflow guidance
  *(shipped — see `[Unreleased]` above)*
- **Custom Rubric Templates**: Per-domain and per-research-group evaluation criteria
- **Bulk Processing**: Evaluate/enhance entire dataset collections at once
- **External Vector Stores**: Pinecone, Weaviate integration for production scale
- **Audit Trail & Versioning**: Full provenance of all description edits
- **Portal Integration**: Direct API hooks instead of Streamlit-only deployment
- **Multi-Language Support**: Translation-aware evaluation and RAG

---

For detailed information about components, architecture, and development setup, see [CLAUDE.md](CLAUDE.md).
