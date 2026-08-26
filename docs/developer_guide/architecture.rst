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

General Assistant Architecture
--------------------------------

The General Assistant is a second, independent module (``src/assistant/``) sharing only the
LLM/embedding layer (``src/llm/client.py``) with the curator described above.

There is **no hardcoded intent dispatcher**. A message passes through a short chain of cheap,
tools-unbound gate calls, then (if needed) a LangGraph ReAct agent
(``langgraph.prebuilt.create_react_agent``) that picks a tool by matching its description
against the system prompt's routing rules — not a lookup table. How the final response is
assembled then depends on *which kind* of tool ran. See :doc:`../user_guide/assistant` for the
user-facing version of this same flow, and the per-capability pages linked from its table for
each tool's own internals.

**Request Lifecycle**

.. graphviz::

   digraph AssistantLifecycle {

       rankdir=TB;
       fontsize=16;
       fontname="Helvetica";
       bgcolor="transparent";

       node [
           shape=box,
           style="rounded,filled",
           fontname="Helvetica",
           fontsize=13,
           margin="0.3,0.2",
           penwidth=2
       ];

       edge [
           fontname="Helvetica",
           fontsize=11,
           penwidth=2
       ];

       QUERY [label="User message\n+ prior history", fillcolor="#e3f2fd", width=2.6, height=0.9];

       OFFDOMAIN [
           label="Off-domain gate\n_classify_off_domain()\n(tools-unbound LLM call)",
           shape=diamond, fillcolor="#fff9c4", width=2.6, height=1.3
       ];

       STEERBACK [
           label="Fixed steer-back message\n(no further LLM calls)",
           fillcolor="#ffcdd2", width=2.6, height=0.9
       ];

       TOOLGATE [
           label="Tool-need gate\n_classify_needs_tool()\n(tools-unbound LLM call)",
           shape=diamond, fillcolor="#fff9c4", width=2.6, height=1.3
       ];

       DIRECT [
           label="_answer_direct()\nSYSTEM_PROMPT, no tools bound\n(greetings, small talk, Tier 3 concepts)",
           fillcolor="#f3e5f5", width=3.0, height=1.1
       ];

       REACT [
           label="ReAct agent\ncreate_react_agent()\nSYSTEM_PROMPT + all 7 tools bound\n(model picks tool(s) from descriptions)",
           fillcolor="#fff3e0", width=3.4, height=1.3
       ];

       TOOLS [
           label="search_datasets · get_dataset_details · get_dataset_profile\nsearch_portal_docs\nget_educational_context · get_workflow_guidance\nsearch_literature",
           fillcolor="#e1f5fe", width=3.8, height=1.5
       ];

       ASSEMBLE [
           label="Response assembly\n(by which tool(s) ran this turn)",
           shape=diamond, fillcolor="#fff9c4", width=2.8, height=1.3
       ];

       VERBATIM [
           label="Verbatim splice\nsearch_datasets / get_dataset_details\nLLM lead-in only + fixed disclaimer",
           fillcolor="#d1c4e9", width=3.0, height=1.1
       ];

       SELFCONTAINED [
           label="Self-contained passthrough\nget_workflow_guidance / get_educational_context\nsearch_portal_docs / get_dataset_profile — already cited, not re-synthesized",
           fillcolor="#d1c4e9", width=3.6, height=1.3
       ];

       SYNTHESIZE [
           label="Outer-agent synthesis\ncross-intent / multi-tool turns\n(preserves source labels + DOIs)",
           fillcolor="#d1c4e9", width=3.0, height=1.1
       ];

       OUTPUT [label="Response to user", fillcolor="#c8e6c9", width=2.6, height=0.9];

       QUERY -> OFFDOMAIN;
       OFFDOMAIN -> STEERBACK [label="off-domain"];
       OFFDOMAIN -> TOOLGATE [label="in-domain"];
       TOOLGATE -> DIRECT [label="direct"];
       TOOLGATE -> REACT [label="tool"];
       REACT -> TOOLS;
       TOOLS -> ASSEMBLE;
       ASSEMBLE -> VERBATIM;
       ASSEMBLE -> SELFCONTAINED;
       ASSEMBLE -> SYNTHESIZE;
       STEERBACK -> OUTPUT;
       DIRECT -> OUTPUT;
       VERBATIM -> OUTPUT;
       SELFCONTAINED -> OUTPUT;
       SYNTHESIZE -> OUTPUT;
   }

A manual-dispatch fallback (not pictured) handles a known tool-call-format issue with one
supported model (Llama-4-Maverick via SambaNova/TACC): if the backend rejects the model's native
tool-call syntax with a 400 error, the intended call is parsed out of the error text and
dispatched directly, following the same verbatim/self-contained rules above rather than falling
back to an ungrounded direct answer. That extraction path (``_TOOL_PARAM_KEYS`` +
``_extract_tool_calls_from_text``/``_extract_tool_calls_from_error``) supports tools with more
than one required argument — ``get_dataset_profile`` is currently the only one, taking both
``dataset_reference`` and ``question``.

Multi-dataset comparisons ("compare dataset A and dataset B") route through the same SYNTHESIZE
node as cross-intent queries: the agent calls ``get_dataset_profile`` once per dataset, and
since that's more than one tool call in the turn, the single-call short-circuit
(VERBATIM/SELFCONTAINED) never fires — the outer agent's own synthesis combines both profiles.

**Core Modules**

For what each tool actually does internally (prompts, matching logic, data schemas), see the
capability pages: :doc:`../user_guide/dataset_discovery`, :doc:`../user_guide/structured_queries`,
:doc:`../user_guide/dataset_profiles`, :doc:`../user_guide/content_reasoning`,
:doc:`../user_guide/portal_docs`, :doc:`../user_guide/domain_qa`,
:doc:`../user_guide/workflow_guidance`, :doc:`../user_guide/literature_search`. The dropdowns
below are the module-level (class/file) reference.

.. dropdown:: src/assistant/conversation_manager.py — Orchestrator
   :icon: file-directory-fill

   - ``ConversationManager`` class — built on ``langgraph.prebuilt.create_react_agent`` with
     ``MemorySaver`` for per-session (in-process) history
   - ``_classify_off_domain()`` / ``_classify_needs_tool()`` / ``_needs_followup_tool_call()`` —
     the tools-unbound gate calls in the Request Lifecycle diagram above
   - ``_build_verbatim_response()`` / ``_run_manual_dispatch()`` — response-assembly and
     400-error manual-dispatch logic
   - ``SYSTEM_PROMPT`` — implements the tiered knowledge-source policy (tools-only for dataset
     facts, tools-first-with-disclaimer for domain Q&A/workflows, pre-trained knowledge allowed
     for foundational concepts) and the tool-routing rules the ReAct agent follows

.. dropdown:: src/assistant/tools.py — Tool Interface
   :icon: file-directory-fill

   - ``search_datasets`` / ``get_dataset_details`` — dataset discovery (semantic + structured)
   - ``get_dataset_profile`` — single-dataset deep-dive profile, sub-node/``INPUT_FOR`` pipeline
     structure, file-format/data-location and reuse-suitability reasoning (backed by
     ``src/prompts/dataset_profile.yaml``); called once per dataset for comparisons
   - ``reason_about_dataset_content`` — relationship/content questions no literal field can
     answer ("paired tomographic and segmented images"). Ranks precomputed ``Dataset.factSheet``
     summaries, then runs one cited reasoning pass (``src/prompts/corpus_reasoning.yaml``) behind
     a fixed honesty framing — see :doc:`../user_guide/content_reasoning`. Reached both by agent
     routing and by the deterministic ``_needs_content_reasoning()`` gate that
     ``get_dataset_details``/``search_datasets`` run before committing to a Cypher answer
   - ``get_workflow_guidance`` / ``get_educational_context`` — domain Q&A and workflow guidance,
     backed by ``data/domain_workflows.yaml`` and ``data/tutorials.yaml``
   - ``search_portal_docs`` — DPM Portal documentation search
   - ``search_literature`` — Semantic Scholar search
   - ``expand_query`` — LLM-based query expansion + inferred metadata filters (not a LangChain
     tool itself — called internally by ``search_datasets``)
   - ``build_langchain_tools()`` — registers all tools with the LangGraph agent

.. dropdown:: src/assistant/graph_store.py — Dataset Graph Search
   :icon: file-directory-fill

   - ``GraphStore`` class — two layers:

     - **High-level** (``search()``, ``hybrid_search()``, ``component_search()``,
       ``cypher_qa()``, ``get_dataset_profile()``, ``rank_fact_sheets()``,
       ``fetch_fact_sheets()``) via ``langchain-neo4j``/the raw driver — used by ``tools.py``.
       ``get_dataset_profile()`` resolves a title/DOI/dataset-number reference to one
       ``Dataset`` node and fetches its full ``PART_OF``/``INPUT_FOR`` sub-node graph with one
       small query per node/edge type — see :doc:`../user_guide/dataset_profiles`.
       ``rank_fact_sheets()`` reuses the same vector+BM25 Reciprocal Rank Fusion as
       ``hybrid_search()``, pointed at the fact-sheet indexes — see
       :doc:`../user_guide/content_reasoning`
     - **Low-level** (``semantic_search()``, ``filter_by_metadata()``, ``search_datasets()``,
       ``execute_cypher()``) via the raw ``neo4j`` driver, for hybrid structured/semantic queries
   - Accepts a ``filters: dict`` (not hardcoded field names), per the Croissant extensibility
     constraint in ``CLAUDE.md``
   - Degrades gracefully: all search methods return empty results immediately if
     ``USE_NEO4J=false``, without importing the Neo4j driver

.. dropdown:: src/assistant/literature_search.py — Literature Search
   :icon: file-directory-fill

   - ``LiteratureSearch`` class — wraps the Semantic Scholar API
   - Works with or without ``SEMANTIC_SCHOLAR_API_KEY`` (unauthenticated requests allowed,
     just rate-limited)

.. dropdown:: src/assistant/portal_docs_retrieval.py + portal_docs_tree.py — Portal Doc Search
   :icon: file-directory-fill

   - PageIndex-style heading-tree retrieval over the DPM Portal's user documentation
     (``data/portal_docs/``), replacing an earlier FAISS/chunk-based approach
   - Returns results labeled ``[portal docs]``

.. dropdown:: src/prompts/ — Assistant Prompts
   :icon: file-directory-fill

   - ``assistant.yaml`` — a standalone 6-intent classifier; used for testing/offline analysis
     only, **not** called by ``ConversationManager`` at runtime (routing there is implicit — see
     the Request Lifecycle diagram above)
   - ``query_expander.yaml`` — renders ``expand_query()``'s semantic expansion + filter
     inference (see :doc:`../user_guide/dataset_discovery`)
   - ``educational.yaml`` — shared synthesis prompt for both ``get_educational_context`` and
     ``get_workflow_guidance`` (see :doc:`../user_guide/domain_qa`,
     :doc:`../user_guide/workflow_guidance`)
   - ``dataset_profile.yaml`` — synthesis prompt for ``get_dataset_profile``: tiered
     knowledge policy, concise-overview-vs-specific-field framing, organizational-structure
     rendering (see :doc:`../user_guide/dataset_profiles`)
   - ``portal_docs.yaml`` — synthesis prompt for ``search_portal_docs``
     (see :doc:`../user_guide/portal_docs`)

.. dropdown:: src/assistant/assistant_ui.py — Streamlit Tab
   :icon: file-directory-fill

   - ``render_assistant_tab()`` — chat interface, added as the ``"General Assistant"`` page in
     ``rocco_ui.py``
   - Renders colored source-label badges, linkifies DOIs/URLs, and normalizes LaTeX delimiters
     for KaTeX
   - Session state keys prefixed ``assistant_`` to avoid collisions with the curator tab

**Configuration**

- ``USE_NEO4J`` — set to ``false`` to disable dataset graph search
- ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` — Neo4j connection details
- ``SEMANTIC_SCHOLAR_API_KEY`` — optional, raises the Semantic Scholar rate limit

See :doc:`../user_guide/configuration` for the full reference.

Maintenance
-----------

.. dropdown:: Adding or updating datasets in the Neo4j graph
   :icon: sync

   Three steps, run from the repo root with ``NEO4J_URI``/``NEO4J_USER``/``NEO4J_PASSWORD`` set:

   1. **Fetch/refresh source metadata** — ``python scripts/scrape_metadata.py`` downloads DRP
      metadata JSONs from TACC Corral into ``data/metadata/`` (gitignored).
   2. **Load into Neo4j**:

      .. code-block:: bash

         # Incremental — merges new/changed datasets, preserves embeddings on
         # unchanged nodes. Use this for adding a handful of new datasets.
         python scripts/load_graph.py --mode upsert

         # Full rebuild — clears and reloads everything. Use after a schema change.
         python scripts/load_graph.py --mode rebuild

      ``load_graph.py`` does **not** generate embeddings or LLM keywords — see step 3.
   3. **Re-embed**:

      .. code-block:: bash

         # Re-embed everything (needed after `--mode rebuild`, or after changing
         # the embedding model / text-assembly logic — see CLAUDE.md's Index
         # Rebuild section for when this applies)
         python scripts/build_dataset_vector_index.py

         # Or patch a single dataset added via `--mode upsert`
         python scripts/reembed_single_dataset.py --doi 10.xxxx/xxxx

   Verify with ``python scripts/audit_schema.py --neo4j --verify`` (node/property counts,
   embedding coverage). See :doc:`../neo4j_schema` for the full schema reference and
   CLAUDE.md's "Index Rebuild" section for `CREATE ... IF NOT EXISTS` / re-run safety notes.

.. dropdown:: Pulling updates from dpm_docs (portal documentation)
   :icon: sync

   ``search_portal_docs`` reads from ``data/portal_docs/docs/`` — a synced copy of the
   `dpm_docs <https://github.com/digital-porous-media/dpm_docs>`_ repo, not a live fetch per
   query. There is **no separate index/build step**: ``portal_docs_tree.py`` parses these
   markdown files into a heading tree at query/import time, so re-syncing and restarting the
   app is all that's needed.

   .. code-block:: bash

      # Check whether the local copy is behind dpm_docs' current HEAD, without fetching
      python scripts/sync_dpm_docs.py --check

      # Fetch and overwrite data/portal_docs/docs/ with the latest dpm_docs content
      python scripts/sync_dpm_docs.py

   dpm_docs updates roughly every 3–6 months upstream; ``--check`` compares against
   ``data/portal_docs/_sync_meta.json``'s last-synced commit SHA. Requires network access to
   ``api.github.com`` and ``raw.githubusercontent.com``.

See Also
--------

- :doc:`../user_guide/streamlit_app` — Description Curator user-facing workflow
- :doc:`../user_guide/assistant` — General Assistant user-facing workflow
- :doc:`../developer_guide/contributing` — Development guidelines
- ``CLAUDE.md`` — Detailed implementation patterns for both modules
