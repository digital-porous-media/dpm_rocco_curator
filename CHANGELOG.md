# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### General Assistant
- **Conversational research assistant** (`src/assistant/`): a new Streamlit tab alongside the
  Description Curator, powered by a LangGraph ReAct agent (`ConversationManager`). Eight tools,
  routed implicitly by the agent from their descriptions — there is no hardcoded intent
  dispatcher — behind two cheap tools-unbound classifier gates.
  - **Dataset Discovery**: hybrid retrieval over a Neo4j vector index — vector similarity plus
    BM25 full-text combined by reciprocal rank fusion — with a component-level second pass over
    per-sub-node embeddings, and structured Cypher filtering for exact/numeric property queries
    (rock type, porosity, voxel size, named authors, etc.)
  - **Dataset Profiles & Comparisons** (`get_dataset_profile`): deep dive on one already-identified
    dataset — organizational structure along `PART_OF`/`INPUT_FOR` edges, file-format and
    data-location reasoning, reuse suitability. Comparisons call it once per dataset and let the
    outer agent synthesize, rather than adding a separate comparison tool.
  - **Relationship & Content Reasoning** (`reason_about_dataset_content`): answers questions no
    literal field can settle ("paired tomographic and segmented images", "the same sample at
    different resolutions", "imaged on an Xradia scanner") by ranking precomputed per-dataset
    fact sheets and running one cited reasoning pass. Two grounding guards run in code: a
    candidate without a citation is dropped, and so is one whose title wasn't in the shortlist
    actually sent.
  - **Multi-turn refinement**: result-set tracking across turns — narrow a listing ("of these,
    which are coal?"), refer back by position or name ("the second one"), or add a bare
    superseding constraint ("how about any below 0.25?") without re-searching the catalog
  - **Deterministic routing gates**: `_needs_content_reasoning()` and `_is_plain_property_query()`
    run in code before a tool commits to a Cypher answer, because prompt-level routing proved
    unreliable for distinctions this fine
  - **Domain Q&A & Workflow Guidance**: curated digital rock physics workflows
    (`data/domain_workflows.yaml`) and portal tutorials (`data/tutorials.yaml`), with a
    tiered knowledge-source policy (tools-only for dataset facts, tools-first-with-disclaimer
    for domain Q&A, pre-trained knowledge for foundational concepts)
  - **Literature Search**: Semantic Scholar API integration (titles, abstracts, DOIs,
    citation counts), with request throttling and 429 backoff
  - **Portal Documentation Search**: PageIndex-style heading-tree retrieval over the DPM
    Portal's user documentation
  - **Source-labeled responses**: every answer is tagged with its origin (`[hybrid match]`,
    `[cypher match]`, `[dataset profile]`, `[content reasoning]`, `[semantic scholar]`,
    `[portal docs]`, etc.), rendered as colored badges
  - **Graceful degradation**: `USE_NEO4J=false` disables dataset graph search only; all other
    capabilities keep working without a Neo4j connection

#### Graph and Index Build
- `scripts/load_graph.py` — loads DRP metadata JSONs into Neo4j (`--mode rebuild` / `--mode upsert`)
- `scripts/build_dataset_vector_index.py` — dataset, component, and fact-sheet embeddings plus
  all five vector/fulltext indexes. Batches fact-sheet embedding by character budget and caps
  per-item length, working around a total-characters-per-request limit on the embedding endpoint
- `scripts/audit_schema.py` — coverage audit and generator for `docs/neo4j_schema.md`
- `scripts/sync_dpm_docs.py` — pulls portal documentation updates from the `dpm_docs` repo
- `Dataset.factSheet` / `factSheetText` — precomputed, edge-preserving per-dataset summaries;
  the raw material for content reasoning. Derived from published metadata, which is never modified.

### Fixed
- **`INPUT_FOR` was documented and queried backwards.** The live graph has
  `(DigitalDataset)-[:INPUT_FOR]->(Sample)` — child → parent, "was derived from" — not the
  reverse. Every dataset profile's organizational-structure section was silently empty as a
  result, since the wrong direction matches zero rows without erroring.
- **`get_dataset_profile()`'s query was pathologically slow** — four chained `OPTIONAL MATCH`es
  cross-multiplied before `collect()` (28s on the largest dataset, not completing at all once
  `INPUT_FOR` joins were restored). Decomposed into one flat query per node/edge type: 0.8s.
- **Context-window overflow** from returning embedding-carrying nodes wholesale; all such queries
  now use map projections.
- **Follow-up turns lost the content-reasoning result set** — `reason_about_dataset_content` was
  missing from `_DATASET_LISTING_TOOLS`, leaving a stale result set from an earlier turn in place
  looking current, so a refinement silently narrowed the wrong set.
- **A refinement could dispatch with an empty `restrict_to_titles`**, which `cypher_qa` treats as
  no restriction at all — running the query over the whole catalog while logging a restricted search.
- **`[content reasoning]` rendered as literal bracketed text** instead of a colored badge.
- **`load_graph.py` created a dead vector index** (`datasetDescription`) on a property
  (`descriptionEmbedding`) that nothing has ever written; `audit_schema.py` audited the same
  phantom property, propagating it into `docs/neo4j_schema.md`. Index creation now belongs solely
  to `build_dataset_vector_index.py`, which knows the embedding dimension.
- **`docs/neo4j_schema.md`'s degradation tiers were hardcoded** and had drifted — still claiming
  all imaging metadata was 0% populated after several fields gained (sparse) data. Now computed
  from the same coverage numbers as the rest of the document.

#### Unified LLM/Agent Stack
- `RoccoClient` now inherits from both `LLMClient` and `BaseChatModel`, giving the curator and
  assistant identical, zero-duplication LLM configuration
- Agent stack: `langgraph.prebuilt.create_react_agent` (replaces the legacy `AgentExecutor`,
  removed in LangChain 1.x). No checkpointer — conversation history is replayed per call by the
  UI layer
- Neo4j vector search via `langchain-neo4j` (optional `graph` extra:
  `pip install -e ".[graph]"`)

### Documentation
- New `docs/user_guide/assistant.rst` — General Assistant overview (request lifecycle, source
  badge reference, knowledge-source policy)
- New per-capability pages: `dataset_discovery`, `structured_queries`, `dataset_profiles`,
  `content_reasoning`, `multi_turn`, `portal_docs`, `domain_qa`, `workflow_guidance`,
  `literature_search`
- New `docs/user_guide/quickstart_assistant.rst`; the existing quickstart was renamed to
  `quickstart_curator.rst`
- `docs/developer_guide/prompts.rst` extended to cover all six assistant prompts, plus the
  prompts that live as code constants rather than YAML
- `docs/developer_guide/onboarding.md` rewritten and added to the toctree
- New General Assistant sections in `README.md` (including an assistant architecture diagram),
  `docs/index.rst`, `docs/developer_guide/architecture.rst`,
  `docs/developer_guide/api_reference.rst`, `docs/user_guide/installation.rst`, and
  `docs/user_guide/configuration.rst`
- New `DEPLOYMENT.md` — TACC VM deployment and maintenance runbook

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
