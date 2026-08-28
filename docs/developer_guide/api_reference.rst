API Reference
=============

Auto-generated documentation for all public classes and functions.
Click any section header to expand it.

.. dropdown:: LLM Client
   :open:

   .. automodule:: src.llm.client
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: Evaluator

   .. automodule:: src.evaluator.evaluator
      :members:
      :undoc-members:
      :show-inheritance:


.. dropdown:: Editor

   .. automodule:: src.editor.editor
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: Document Ingestor

   .. automodule:: src.ingestor.document_ingestor
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: Document Embedder

   .. automodule:: src.ingestor.embedder
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: Vector Store Manager

   .. automodule:: src.retriever.retriever
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: Content Screener

   .. automodule:: src.llm.content_screener
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: Prompt Loader

   .. automodule:: src.prompts.loader
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: Output Schemas

   .. automodule:: src.llm.schemas
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: General Assistant — Conversation Manager

   .. automodule:: src.assistant.conversation_manager
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: General Assistant — Tools

   .. automodule:: src.assistant.tools
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: General Assistant — Graph Store

   .. automodule:: src.assistant.graph_store
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: General Assistant — Literature Search

   .. automodule:: src.assistant.literature_search
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: General Assistant — Portal Doc Retrieval

   .. automodule:: src.assistant.portal_docs_retrieval
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: General Assistant — Portal Doc Heading Tree

   .. automodule:: src.assistant.portal_docs_tree
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: Configuration

   Environment variables (set in ``.env``):

   - ``LLM_PROVIDER`` — Provider shortcut (``openai``, ``anthropic``, ``ollama``, etc.)
   - ``LLM_API_KEY`` — API key (required)
   - ``LLM_BASE_URL`` — Custom endpoint URL (optional)
   - ``LLM_MODEL`` — Model name (defaults to ``gpt-4o-mini``)
   - ``LLM_TIMEOUT`` — Request timeout in seconds (defaults to ``120``)
   - ``EMBEDDING_URL`` / ``EMBEDDING_MODEL`` / ``EMBEDDING_API_KEY`` — Override the embedding
     endpoint used for RAG and dataset-graph search (optional; auto-selected from
     ``LLM_PROVIDER`` when unset)
   - ``USE_NEO4J`` / ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` — General Assistant
     dataset graph search (optional; degrades gracefully if unset)
   - ``SEMANTIC_SCHOLAR_API_KEY`` — General Assistant literature search (optional)

   See :doc:`../user_guide/configuration` for all providers and setup.

----

See Also
--------

- :doc:`architecture` — System design and data flow
- :doc:`contributing` — Development guidelines
- :doc:`prompts` — Prompt YAML reference and editing guide
- ``CLAUDE.md`` — Implementation details and patterns
