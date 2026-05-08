Architecture
=============

Rocco's modular design separates concerns across evaluation, enhancement, RAG, and LLM integration.

System Diagram
--------------

.. code-block:: text

   ┌──────────────────────────────────────────────────────────┐
   │            Streamlit UI (rocco_ui.py)                    │
   │   - Description input                                    │
   │   - Evaluation display                                  │
   │   - Document upload                                     │
   │   - Enhancement workflow                                │
   └──────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────┐
   │  Evaluation Pipeline (src/evaluator/)                   │
   │  ─────────────────────────────────────────────────────  │
   │  Input: description text                                 │
   │  Process: rubric scoring + few-shot learning            │
   │  Output: 10-criterion breakdown, 0–10 score            │
   └─────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────┐
   │  Document Ingestion & RAG (src/ingestor/, src/retriever/)
   │  ─────────────────────────────────────────────────────  │
   │  Input: PDFs, DOCX files                                 │
   │  Process: chunking (500 char, 100 overlap)              │
   │           sentence-transformers embeddings              │
   │           FAISS vector store                            │
   │  Output: vector index + chunk metadata                  │
   └─────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────┐
   │  Content Screening (src/llm/content_screener.py)        │
   │  ─────────────────────────────────────────────────────  │
   │  Input: user feedback text                              │
   │  Process: relevance + accuracy + tone validation        │
   │  Output: accept/reject/flag for review                  │
   └─────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────┐
   │  Enhancement Pipeline (src/editor/)                     │
   │  ─────────────────────────────────────────────────────  │
   │  Input: orig description + RAG context + feedback       │
   │  Process: prompt rendering + LLM call                   │
   │  Output: enhanced description + citations               │
   └─────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────┐
   │  LLM Client (src/llm/client.py)                         │
   │  ─────────────────────────────────────────────────────  │
   │  Provider-agnostic OpenAI SDK wrapper                   │
   │  Supports: OpenAI, Anthropic, Ollama, DeepSeek, etc.    │
   └─────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────┐
   │  External LLM Provider                                  │
   │  (OpenAI API, Anthropic API, Ollama, etc.)              │
   └─────────────────────────────────────────────────────────┘

Core Modules
------------

**src/evaluator/** — Rubric Evaluation

- ``evaluator.py`` — ``DescriptionEvaluator`` class
  - Scores descriptions against 10 criteria
  - Uses few-shot examples for consistency
  - Returns structured breakdown + total score

- ``rubric.json`` — Evaluation criteria definition
  - 10 criteria, 1 point each
  - Criterion name, description, scoring guidance

- ``examples_v3.json`` — Few-shot examples
  - 3 example (description, score, explanation) tuples
  - Improves evaluator consistency

**src/ingestor/** — Document Chunking & Embedding

- ``document_ingestor.py`` — ``DocumentIngestor`` class
  - Chunks PDFs/DOCX using LangChain's RecursiveCharacterTextSplitter
  - Config: 500 char chunks, 100 char overlap
  - Enriches chunks with metadata (filename, page, chunk index)

- ``embedder.py`` — ``DocumentEmbedder`` class
  - Uses ``sentence-transformers`` (BAAI/bge-large-en-v1.5)
  - Generates semantic embeddings for retrieval

- ``base.py`` — Abstract base class
  - Common interface for pluggable ingestors

**src/retriever/** — Vector Storage & Search

- ``retriever.py`` — ``VectorStoreManager`` class
  - FAISS-backed vector store
  - Methods: ``add_documents()``, ``similarity_search_with_score()``
  - Supports save/load to disk

**src/editor/** — Description Enhancement

- ``editor.py`` — ``DescriptionEditor`` class
  - Input: original description, RAG context, user feedback
  - Process: prompt rendering + LLM call
  - Output: improved description + citations

**src/llm/** — LLM Integration

- ``client.py`` — ``LLMClient`` and ``RoccoClient``
  - Wraps OpenAI SDK for provider-agnostic usage
  - Supports: OpenAI, Anthropic, Ollama, DeepSeek, Gemini, HuggingFace, SambaNova
  - Environment-driven configuration (LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL)

- ``content_screener.py`` — ``ContentScreener`` class
  - Validates user feedback for relevance, accuracy, tone, coherence
  - Returns recommendation (accept/reject/flag)

- ``schemas.py`` — Pydantic models
  - Structured output schemas for all LLM calls

**src/prompts/** — Prompt Management

- ``loader.py`` — ``PromptLoader`` class
  - Loads YAML prompt files
  - Renders with Jinja2 template variables

- YAML prompt files:
  - ``evaluator.yaml`` — Rubric scoring prompt
  - ``editor.yaml`` — Description enhancement prompt
  - ``content_screener.yaml`` — Feedback validation prompt

Data Flow
---------

**Evaluation Path**

.. code-block:: text

   user input (description text)
       ↓
   DescriptionEvaluator.evaluate(description)
       ↓
   load_prompt("evaluator") → render with rubric_json + description
       ↓
   RoccoClient.send_prompt(system, user)
       ↓
   LLM API call
       ↓
   parse structured output (rubric_breakdown, total_score, feedback)
       ↓
   return EvaluationResult

**Enhancement Path**

.. code-block:: text

   user input (description + feedback + optional uploaded files)
       ↓
   DocumentIngestor.ingest(files) → DocumentEmbedder.embed_documents()
       ↓
   VectorStoreManager.add_documents() → FAISS index
       ↓
   (user provides feedback)
       ↓
   ContentScreener.screen(feedback)
       ↓
   if accept:
       VectorStoreManager.similarity_search(description, top_k=5)
           ↓
       DescriptionEditor.enhance(description, context_chunks, feedback)
           ↓
       load_prompt("editor") → render with description + context + feedback
           ↓
       RoccoClient.send_prompt()
           ↓
       parse structured output (updated_description, rationale, citations)
           ↓
       return EditorResult

Configuration
-------------

**Environment Variables** (via .env)

- ``LLM_PROVIDER`` — Shortcut to endpoint (openai, anthropic, ollama, etc.)
- ``LLM_API_KEY`` — API key or "ollama" for local
- ``LLM_BASE_URL`` — Custom endpoint URL (optional)
- ``LLM_MODEL`` — Model name (defaults to gpt-4o-mini)

**Session State** (Streamlit)

Stored in ``st.session_state``:
- ``description_text`` — current description
- ``evaluation`` — latest evaluation result
- ``vector_store_manager`` — loaded FAISS index
- ``enhanced_description`` — improved version
- ``user_feedback`` — feedback text
- ``screening_result`` — content screener result
- And more...

Extension Points
----------------

**Adding a New LLM Provider**

1. Add provider → base URL mapping to ``PROVIDER_URLS`` in ``src/llm/client.py``
2. Update ``.env.example`` with provider config
3. No code change needed (OpenAI SDK handles compatibility)
4. Document in README and configuration guide

**Adding New Evaluation Criteria**

1. Add criterion to ``src/evaluator/rubric.json``
2. Update ``src/evaluator/examples_v3.json`` with new examples
3. Update ``src/prompts/evaluator.yaml`` to reference new criteria
4. Bump version in evaluator.yaml (major if score scale changes)

**Adding a New Document Type**

1. Create ``CustomIngestor`` extending ``DocumentIngestor``
2. Implement custom chunking logic
3. Register in ``rocco_ui.py``

Testing
-------

Run tests:

.. code-block:: bash

   pytest tests/

Key test patterns:

- **Evaluator tests** — verify rubric scoring consistency
- **Retriever tests** — verify FAISS indexing and search
- **Editor tests** — verify prompt rendering and citation tracking
- **Integration tests** — end-to-end workflow (evaluate → enhance → screen)

Performance Considerations
--------------------------

- **Evaluation**: ~2–5 seconds (LLM call)
- **Document Ingestion**: ~1–3 seconds per MB (chunking + embedding)
- **Enhancement**: ~5–10 seconds (LLM call + RAG retrieval)
- **Memory**: ~2GB base + ~1GB per uploaded document

For large document sets (100+ MB), consider batch ingestion or external vector stores (Pinecone, Weaviate).

See Also
--------

- :doc:`../user_guide/streamlit_app` — User-facing workflow
- :doc:`../developer_guide/contributing` — Development guidelines
- ``CLAUDE.md`` — Detailed implementation patterns
