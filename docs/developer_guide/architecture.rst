Architecture
=============

Rocco is designed to be modular, separating concerns across evaluation, enhancement, RAG, and LLM integration.

System Diagram
--------------

.. **LLM Infrastructure Layer**

.. .. graphviz::

..    digraph LLM {

..        rankdir=TB;
..        fontsize=16;
..        fontname="Helvetica";
..        bgcolor="transparent";

..        node [
..            shape=box,
..            style="rounded,filled",
..            fontname="Helvetica",
..            fontsize=12,
..            margin="0.35,0.25",
..            penwidth=2
..        ];

..        LLM [
..            label=<
..                <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
..                    <TR><TD><B><FONT POINT-SIZE="15">LLM Client</FONT></B></TD></TR>
..                    <TR><TD><FONT FACE="monospace" POINT-SIZE="12">src/llm/client.py</FONT></TD></TR>
..                    <TR><TD><FONT POINT-SIZE="13">Provider-agnostic OpenAI SDK wrapper</FONT></TD></TR>
..                </TABLE>
..            >,
..            fillcolor="#ffe0b2",
..            height=0.9,
..            width=7.0
..        ];
..    }

**Core Processing Pipeline**

.. graphviz::

   digraph Pipeline {

       rankdir=TB;
       fontsize=16;
       fontname="Helvetica";
       bgcolor="transparent";

       node [
           shape=box,
           style="rounded,filled",
           fontname="Helvetica",
           fontsize=14,
           margin="0.35,0.25",
           penwidth=2
       ];

       edge [
           fontname="Helvetica",
           fontsize=12,
           penwidth=2
       ];

       INPUT [
           label=<
               <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                   <TR><TD><B><FONT POINT-SIZE="15">Draft Description</FONT></B></TD></TR>
                   <TR><TD><FONT POINT-SIZE="13">User-provided description</FONT></TD></TR>
               </TABLE>
           >,
           fillcolor="#e3f2fd",
           height=1.4,
           width=3.5
       ];

       EVAL [
           label=<
               <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                   <TR><TD><B><FONT POINT-SIZE="15">Evaluator</FONT></B></TD></TR>
                   <TR><TD><FONT FACE="monospace" POINT-SIZE="12">src/evaluator</FONT></TD></TR>
                   <TR><TD><FONT POINT-SIZE="13">10-point rubric scoring</FONT></TD></TR>
               </TABLE>
           >,
           fillcolor="#fff3e0",
           height=1.4,
           width=3.5
       ];

       RAG [
           label=<
               <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                   <TR><TD><B><FONT POINT-SIZE="15">Retriever</FONT></B></TD></TR>
                   <TR><TD><FONT FACE="monospace" POINT-SIZE="12">src/retriever</FONT></TD></TR>
                   <TR><TD><FONT POINT-SIZE="13">RAG pipeline</FONT></TD></TR>
               </TABLE>
           >,
           fillcolor="#f3e5f5",
           height=1.4,
           width=3.5
       ];

       SCREEN [
           label=<
               <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                   <TR><TD><B><FONT POINT-SIZE="15">Content Screener</FONT></B></TD></TR>
                   <TR><TD><FONT FACE="monospace" POINT-SIZE="12">src/llm</FONT></TD></TR>
                   <TR><TD><FONT POINT-SIZE="13">Validate feedback</FONT></TD></TR>
               </TABLE>
           >,
           fillcolor="#f3e5f5",
           height=1.4,
           width=3.5
       ];

       EDIT [
           label=<
               <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                   <TR><TD><B><FONT POINT-SIZE="15">Editor</FONT></B></TD></TR>
                   <TR><TD><FONT FACE="monospace" POINT-SIZE="12">src/editor</FONT></TD></TR>
                   <TR><TD><FONT POINT-SIZE="13">Apply feedback + context</FONT></TD></TR>
               </TABLE>
           >,
           fillcolor="#fff3e0",
           height=1.4,
           width=3.5
       ];

       OUTPUT [
           label=<
               <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                   <TR><TD><B><FONT POINT-SIZE="15">Refined Description</FONT></B></TD></TR>
                   <TR><TD><FONT POINT-SIZE="12">(Output)</FONT></TD></TR>
                   <TR><TD><FONT POINT-SIZE="13">With citations</FONT></TD></TR>
               </TABLE>
           >,
           fillcolor="#c8e6c9",
           height=1.4,
           width=3.5
       ];

       INPUT -> EVAL -> EDIT -> OUTPUT;
       RAG -> EDIT;
       SCREEN -> EDIT;
   }

Core Modules
------------

.. dropdown:: src/llm/ — LLM Integration
   :icon: file-directory-fill

   - ``client.py`` — ``LLMClient`` and ``RoccoClient``

     - Wraps OpenAI SDK for provider-agnostic usage
     - Supports: OpenAI, Anthropic, Ollama, DeepSeek, Gemini, HuggingFace, SambaNova
     - Environment-driven configuration (LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL)

   - ``content_screener.py`` — ``ContentScreener`` class

     - Validates user feedback for relevance, accuracy, tone, coherence
     - Returns recommendation (accept/reject/flag)

   - ``schemas.py`` — Pydantic models

     - Structured output schemas for all LLM calls


.. dropdown:: src/evaluator/ — Rubric Evaluation
   :icon: file-directory-fill

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

.. dropdown:: src/ingestor/ — Document Chunking & Embedding
   :icon: file-directory-fill

   - ``document_ingestor.py`` — ``DocumentIngestor`` class

     - Chunks PDFs/DOCX using LangChain's RecursiveCharacterTextSplitter
     - Config: 500 char chunks, 100 char overlap
     - Enriches chunks with metadata (filename, page, chunk index)

   - ``embedder.py`` — ``DocumentEmbedder`` class

     - Uses ``sentence-transformers`` (BAAI/bge-large-en-v1.5)
     - Generates semantic embeddings for retrieval

   - ``base.py`` — Abstract base class

     - Common interface for pluggable ingestors

.. dropdown:: src/retriever/ — Vector Storage & Search
   :icon: file-directory-fill

   - ``retriever.py`` — ``VectorStoreManager`` class

     - FAISS-backed vector store
     - Methods: ``add_documents()``, ``similarity_search_with_score()``
     - Supports save/load to disk

.. dropdown:: src/editor/ — Description Enhancement
   :icon: file-directory-fill

   - ``editor.py`` — ``DescriptionEditor`` class

     - Input: original description, RAG context, user feedback
     - Process: prompt rendering + LLM call
     - Output: improved description + citations


.. dropdown:: src/prompts/ — Prompt Management
   :icon: file-directory-fill

   - ``loader.py`` — ``PromptLoader`` class

     - Loads YAML prompt files
     - Renders with Jinja2 template variables

   - YAML prompt files:

     - ``evaluator.yaml`` — Rubric scoring prompt
     - ``editor.yaml`` — Description enhancement prompt
     - ``content_screener.yaml`` — Feedback validation prompt

Data Flow
---------

.. dropdown:: Evaluation Path

   .. graphviz::

      digraph EvaluationPath {

          rankdir=TB;
          fontsize=14;
          fontname="Helvetica";
          bgcolor="transparent";

          node [
              shape=box,
              style="rounded,filled",
              fontname="Helvetica",
              fontsize=12,
              margin="0.35,0.25",
              penwidth=2
          ];

          edge [
              fontname="Helvetica",
              fontsize=12,
              penwidth=2
          ];

          INPUT [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="13">Draft Description</FONT></B></TD></TR>
                  </TABLE>
              >,
              fillcolor="#e3f2fd",
              height=1.0,
              width=2.5
          ];

          OUTPUT [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="13">Evaluation Result</FONT></B></TD></TR>
                      <TR><TD><FONT POINT-SIZE="11">Structured scoring breakdown and reasoning</FONT></TD></TR>
                  </TABLE>
              >,
              fillcolor="#c8e6c9",
              height=1.0,
              width=2.5
          ];

          subgraph cluster_evaluate {
              style=dashed;
              label="";

              EVAL [
                  label=<
                      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                          <TR><TD><B><FONT POINT-SIZE="13">Load Prompt</FONT></B></TD></TR>
                          <TR><TD><FONT FACE="monospace" POINT-SIZE="11">load_prompt("evaluator")</FONT></TD></TR>
                          <TR><TD><FONT POINT-SIZE="11">Build prompt with the draft description</FONT></TD></TR>
                      </TABLE>
                  >,
                  fillcolor="#f3e5f5",
                  height=1.0,
                  width=2.5
              ];

              CALL [
                  label=<
                      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                          <TR><TD><B><FONT POINT-SIZE="13">LLM API Call</FONT></B></TD></TR>
                          <TR><TD><FONT FACE="monospace" POINT-SIZE="11">RoccoClient.send_prompt()</FONT></TD></TR>
                      </TABLE>
                  >,
                  fillcolor="#ffe0b2",
                  height=1.0,
                  width=2.5
              ];

              PARSE [
                  label=<
                      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                          <TR><TD><B><FONT POINT-SIZE="13">Parse Output</FONT></B></TD></TR>
                          <TR><TD><FONT POINT-SIZE="11">Extract scoring breakdown &amp; reasoning</FONT></TD></TR>
                      </TABLE>
                  >,
                  fillcolor="#f3e5f5",
                  height=1.0,
                  width=2.5
              ];

              EVAL -> CALL -> PARSE;
          }

          cluster_eval_label [
              shape=none,
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="13">DescriptionEvaluator.evaluate()</FONT></B></TD></TR>
                  </TABLE>
              >,
              fillcolor="#ffffff",
              margin="0,0"
          ];


          INPUT -> EVAL;
          PARSE -> OUTPUT;
          { rank=same; CALL; cluster_eval_label; }
      }

.. dropdown:: Enhancement Path

   .. graphviz::

      digraph EnhancementPath {

          rankdir=TB;
          fontsize=14;
          fontname="Helvetica";
          bgcolor="transparent";

          node [
              shape=box,
              style="rounded,filled",
              fontname="Helvetica",
              fontsize=12,
              margin="0.35,0.25",
              penwidth=2
          ];

          edge [
              fontname="Helvetica",
              fontsize=12,
              penwidth=2
          ];

          FILES [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="15">File Upload</FONT></B></TD></TR>
                  </TABLE>
              >,
              fillcolor="#e3f2fd",
              height=1.2,
              width=3.5
          ];

          INGEST [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="15">DocumentIngestor</FONT></B></TD></TR>
                      <TR><TD><FONT FACE="monospace" POINT-SIZE="12">.ingest() &amp; .embed_documents()</FONT></TD></TR>
                      <TR><TD><FONT POINT-SIZE="13">Chunk &amp; embed documents</FONT></TD></TR>
                  </TABLE>
              >,
              fillcolor="#f3e5f5",
              height=1.4,
              width=3.5
          ];

          ADD [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="15">VectorStoreManager</FONT></B></TD></TR>
                      <TR><TD><FONT FACE="monospace" POINT-SIZE="12">.add_documents() → FAISS</FONT></TD></TR>
                      <TR><TD><FONT POINT-SIZE="13">Build vector database</FONT></TD></TR>
                  </TABLE>
              >,
              fillcolor="#f3e5f5",
              height=1.4,
              width=3.5
          ];

          DESC [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="15">Structured Description Evaluation</FONT></B></TD></TR>
                  </TABLE>
              >,
              fillcolor="#e3f2fd",
              height=1.2,
              width=3.5
          ];

          FEEDBACK [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="15">User Feedback</FONT></B></TD></TR>
                  </TABLE>
              >,
              fillcolor="#e3f2fd",
              height=1.2,
              width=3.5
          ];

          SCREEN [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="15">ContentScreener</FONT></B></TD></TR>
                      <TR><TD><FONT FACE="monospace" POINT-SIZE="12">.screen(feedback)</FONT></TD></TR>
                      <TR><TD><FONT POINT-SIZE="13">Validate feedback</FONT></TD></TR>
                  </TABLE>
              >,
              fillcolor="#f3e5f5",
              height=1.4,
              width=3.5
          ];

          DECISION [
              label="Accept?",
              shape=diamond,
              fillcolor="#fff9c4",
              height=1.0,
              width=1.8
          ];

          OUTPUT [
              label=<
                  <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                      <TR><TD><B><FONT POINT-SIZE="15">EditorResult</FONT></B></TD></TR>
                      <TR><TD><FONT POINT-SIZE="13">Enhanced description</FONT></TD></TR>
                  </TABLE>
              >,
              fillcolor="#c8e6c9",
              height=1.2,
              width=3.5
          ];

          ENHANCE [
                  label=<
                      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                          <TR><TD><B><FONT POINT-SIZE="15">Retrieve Context</FONT></B></TD></TR>
                          <TR><TD><FONT POINT-SIZE="13">RAG + vector search</FONT></TD></TR>
                      </TABLE>
                  >,
                  fillcolor="#f3e5f5",
                  height=1.2,
                  width=3.5
          ];

          subgraph cluster_enhance {
              style=dashed;
              label="";

              LOAD [
                  label=<
                      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                          <TR><TD><B><FONT POINT-SIZE="15">Load Prompt</FONT></B></TD></TR>
                          <TR><TD><FONT FACE="monospace" POINT-SIZE="12">load_prompt("editor")</FONT></TD></TR>
                          <TR><TD><FONT POINT-SIZE="12">Build prompt with retrieved context, user feedback, and evaluation results</FONT></TD></TR>
                      </TABLE>
                  >,
                  fillcolor="#f3e5f5",
                  height=1.2,
                  width=3.5
              ];

              CALL [
                  label=<
                      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                          <TR><TD><B><FONT POINT-SIZE="15">LLM API Call</FONT></B></TD></TR>
                          <TR><TD><FONT FACE="monospace" POINT-SIZE="12">RoccoClient.send_prompt()</FONT></TD></TR>
                      </TABLE>
                  >,
                  fillcolor="#ffe0b2",
                  height=1.2,
                  width=3.5
              ];

              PARSE [
                  label=<
                      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                          <TR><TD><B><FONT POINT-SIZE="15">Parse Output</FONT></B></TD></TR>
                          <TR><TD><FONT POINT-SIZE="13">Extract enhanced description, citations, and rationale for changes</FONT></TD></TR>
                      </TABLE>
                  >,
                  fillcolor="#f3e5f5",
                  height=1.2,
                  width=3.5
              ];

              LOAD -> CALL -> PARSE;
          }
           cluster_enhance_label [
               shape=none,
               label=<
                   <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                       <TR><TD><B><FONT POINT-SIZE="13">DescriptionEditor.enhance()</FONT></B></TD></TR>
                   </TABLE>
               >,
               fillcolor="#ffffff",
               margin="0,0"
          ];

          { rank=same; FILES; DESC; FEEDBACK; }

          FILES -> INGEST -> ADD -> ENHANCE;
          FEEDBACK -> SCREEN -> DECISION;
          DECISION -> FEEDBACK [label="no (revise)"];
          DESC -> LOAD;
          ENHANCE -> LOAD;
          DECISION -> LOAD [label="yes"];
          PARSE -> OUTPUT;
          { rank=same; CALL; cluster_enhance_label; }
      }


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

See Also
--------

- :doc:`../user_guide/streamlit_app` — User-facing workflow
- :doc:`../developer_guide/contributing` — Development guidelines
- ``CLAUDE.md`` — Detailed implementation patterns
