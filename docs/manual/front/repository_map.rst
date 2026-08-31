Repository map
==============

This chapter maps the source repository. Every path is relative to the
repository root. Directories marked *not in version control* are created
locally at runtime or hold data too large or too sensitive to commit.

Top level
---------

.. code-block:: text

   dpm_rocco_curator/
   ├── rocco_ui.py              Streamlit entry point. Runs both modules as tabs.
   ├── assistant_demo.py        Minimal command-line demo of the General Assistant.
   ├── pyproject.toml           Package metadata, dependencies, optional extras.
   ├── pytest.ini               Test configuration. Excludes the `live` marker by default.
   ├── README.md                Project overview and quick start.
   ├── CONTRIBUTING.md          Contribution process and coding standards.
   ├── DEPLOYMENT.md            Server deployment runbook.
   ├── CHANGELOG.md             Release history.
   ├── CITATION.cff             Citation metadata.
   ├── codemeta.json            Software metadata for indexing services.
   ├── LICENSE                  MIT license.
   ├── .env.example             Template for local configuration. Copy to `.env`.
   ├── assets/                  Chat avatars used by the web interface.
   ├── benchmarks/              Evaluation study: notebook, data, and figures.
   ├── data/                    Runtime data and domain knowledge files.
   ├── docs/                    Documentation sources, including this manual.
   ├── scripts/                 Maintenance and index-building scripts.
   ├── src/                     Application code.
   └── tests/                   Test suite.

Application code
----------------

.. code-block:: text

   src/
   ├── llm/
   │   ├── client.py            RoccoClient. The single LLM interface for both modules.
   │   ├── embeddings.py        Embedding-provider factory.
   │   ├── content_screener.py  Validates user feedback before it reaches the editor.
   │   └── schemas.py           Structured output schemas for evaluation and editing.
   ├── evaluator/
   │   ├── evaluator.py         Scores a description against the rubric.
   │   ├── rubric.json          The ten evaluation criteria.
   │   └── examples_v3.json     Few-shot examples that calibrate the scorer.
   ├── editor/
   │   └── editor.py            Rewrites a description and produces citations.
   ├── ingestor/
   │   ├── document_ingestor.py Loads and chunks uploaded PDF and DOCX files.
   │   └── embedder.py          Turns chunks into vectors.
   ├── retriever/
   │   └── retriever.py         FAISS vector store for uploaded documents.
   ├── prompts/
   │   ├── loader.py            Loads and renders the versioned prompt files.
   │   └── *.yaml               Nine prompt definitions. See Appendix C.
   ├── assistant/
   │   ├── conversation_manager.py  Orchestrator: routing gates, agent, response assembly.
   │   ├── tools.py             The eight callable tools and their routing gates.
   │   ├── graph_store.py       Neo4j access: hybrid search, Cypher generation, profiles.
   │   ├── literature_search.py Semantic Scholar API client.
   │   ├── portal_docs_retrieval.py  Retrieval over the portal documentation tree.
   │   ├── portal_docs_tree.py  Parses portal documentation into a heading tree.
   │   ├── llm.py               Shared client and embedding singletons.
   │   ├── assistant_ui.py      The General Assistant page of the web interface.
   │   └── assistant.py         Backwards-compatible re-export of ConversationManager.
   └── common/
       └── utils.py             Shared helpers.

Scripts
-------

Run these from the repository root with the environment activated. Every script
accepts ``--help``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Purpose
   * - ``scripts/scrape_metadata.py``
     - Downloads dataset metadata from TACC Corral into ``data/metadata/``.
   * - ``scripts/load_graph.py``
     - Loads metadata into Neo4j as nodes and relationships. Creates no indexes.
   * - ``scripts/build_dataset_vector_index.py``
     - Builds the embeddings, fact sheets, and all five search indexes.
   * - ``scripts/reembed_single_dataset.py``
     - Re-embeds one dataset after an incremental load.
   * - ``scripts/audit_schema.py``
     - Audits property coverage and generates Appendix B.
   * - ``scripts/sync_dpm_docs.py``
     - Pulls the latest portal documentation into ``data/portal_docs/``.
   * - ``scripts/check_embedding_health.py``
     - Probes the embedding endpoint.
   * - ``scripts/check_neo4j_vector_support.py``
     - Verifies the Neo4j server supports vector indexes.

Data
----

.. code-block:: text

   data/
   ├── domain_workflows.yaml    Fifteen digital rock physics workflows.
   ├── tutorials.yaml           User goals mapped to verified portal tutorial links.
   ├── vector_store/            FAISS index for uploaded documents.
   ├── metadata/                Scraped dataset metadata.       (not in version control)
   └── portal_docs/             Synced portal documentation.    (not in version control)

Documentation
-------------

.. code-block:: text

   docs/
   ├── conf.py                  Sphinx configuration for the documentation website.
   ├── index.rst                Website landing page.
   ├── user_guide/              Parts I through IV of this manual.
   ├── developer_guide/         Part V of this manual.
   ├── neo4j_schema.md          Appendix B. Generated by `scripts/audit_schema.py`.
   ├── architecture_diagram.svg System diagram used in the project README.
   ├── assistant_diagram.svg    Assistant diagram used in the project README.
   ├── _static/                 Stylesheets and screenshots.
   └── manual/                  Build configuration for this manual.
       ├── conf.py              Print-specific Sphinx overrides.
       ├── index.rst            The manual's master document.
       ├── print.css            Page geometry, running heads, and table of contents styling.
       ├── front/               Preface and this chapter.
       ├── parts/               Part title pages.
       ├── appendix/            Appendix source files.
       └── overrides/           Copy-edited chapter text used only by the manual.

Tests
-----

The suite has two tiers. The default run excludes tests marked ``live``, which
call external services:

.. code-block:: console

   $ pytest tests/ -v          # default tier
   $ pytest tests/ -m live -v  # tests that call live endpoints

.. code-block:: text

   tests/
   ├── conftest.py
   ├── test_curator_integration.py   Evaluator, editor, and screener against RoccoClient.
   ├── test_llm_client.py            Provider routing and configuration.
   ├── test_vector_store.py          Embedding batches and document alignment.
   └── assistant/
       ├── conftest.py               Mock Neo4j driver and graph store fixtures.
       ├── test_conversation_manager.py  Routing gates, response assembly, cross-turn state.
       ├── test_tools.py             The eight tools and their gates.
       ├── test_graph_store.py       Search methods and Cypher safety.
       ├── test_search_integration.py    The acceptance suite. See Appendix D.
       ├── test_fact_sheet_builder.py
       ├── test_portal_docs_retrieval.py
       ├── test_literature_search.py
       ├── test_intent_classifier.py
       ├── test_prompts.py
       └── test_assistant_ui.py
