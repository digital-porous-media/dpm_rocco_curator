Appendix A. Python API reference
================================

This appendix documents every public class and function in the Rocco codebase.
It is generated from the source docstrings, so it always matches the version of
the code this manual was built from.

Entry points such as ``rocco_ui.py`` execute Streamlit calls at import time and
are described in prose at the end of this appendix rather than generated.

Language model layer
--------------------

``RoccoClient`` is the single interface to the language model for both the
Description Curator and the General Assistant. It inherits from LangChain's
``BaseChatModel``, so it works directly with LangGraph agents.

.. automodule:: src.llm.client
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.llm.embeddings
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.llm.schemas
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.llm.content_screener
   :members:
   :undoc-members:
   :show-inheritance:

Description Curator
-------------------

.. automodule:: src.evaluator.evaluator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.editor.editor
   :members:
   :undoc-members:
   :show-inheritance:

Document ingestion and retrieval
--------------------------------

.. automodule:: src.ingestor.document_ingestor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.ingestor.embedder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.retriever.retriever
   :members:
   :undoc-members:
   :show-inheritance:

Prompt loading
--------------

.. automodule:: src.prompts.loader
   :members:
   :undoc-members:
   :show-inheritance:

General Assistant
-----------------

.. automodule:: src.assistant.conversation_manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.assistant.tools
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.assistant.graph_store
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.assistant.literature_search
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.assistant.portal_docs_retrieval
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.assistant.portal_docs_tree
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.assistant.llm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.assistant.assistant_ui
   :members:
   :undoc-members:
   :show-inheritance:

Shared utilities
----------------

.. automodule:: src.common.utils
   :members:
   :undoc-members:
   :show-inheritance:

Maintenance scripts
-------------------

These modules are command-line tools. Each one runs standalone and accepts
``--help``. See :doc:`/front/repository_map` for what each script is for.

.. automodule:: scrape_metadata
   :members:
   :undoc-members:

.. automodule:: load_graph
   :members:
   :undoc-members:

.. automodule:: build_dataset_vector_index
   :members:
   :undoc-members:

.. automodule:: reembed_single_dataset
   :members:
   :undoc-members:

.. automodule:: audit_schema
   :members:
   :undoc-members:

.. automodule:: sync_dpm_docs
   :members:
   :undoc-members:

.. automodule:: check_embedding_health
   :members:
   :undoc-members:

.. automodule:: check_neo4j_vector_support
   :members:
   :undoc-members:

Entry points
------------

Two modules run as scripts rather than libraries. Both call Streamlit or
configure logging at import time, so they are listed here instead of generated
above.

``rocco_ui.py``
   The Streamlit application. It renders a page selector that switches between
   the Description Curator and the General Assistant, owns all curator session
   state, and delegates the assistant page to
   :func:`src.assistant.assistant_ui.render_assistant_page`. Start it with
   ``streamlit run rocco_ui.py``.

``assistant_demo.py``
   A minimal command-line loop that constructs a
   :class:`~src.assistant.conversation_manager.ConversationManager` and prints
   responses. Useful for checking assistant behavior without the web interface.

``src/assistant/assistant.py``
   A one-line re-export of ``ConversationManager``, kept so older imports keep
   working. New code should import from
   :mod:`src.assistant.conversation_manager`.
