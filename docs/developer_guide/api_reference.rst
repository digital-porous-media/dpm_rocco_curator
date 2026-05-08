API Reference
=============

Auto-generated documentation for all public classes and functions.
Click any section header to expand it.

.. dropdown:: Evaluator
   :open:

   .. automodule:: src.evaluator.evaluator
      :members:
      :undoc-members:
      :show-inheritance:

.. dropdown:: LLM Client

   .. automodule:: src.llm.client
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

.. dropdown:: Configuration

   Environment variables (set in ``.env``):

   - ``LLM_PROVIDER`` — Provider shortcut (``openai``, ``anthropic``, ``ollama``, etc.)
   - ``LLM_API_KEY`` — API key (required)
   - ``LLM_BASE_URL`` — Custom endpoint URL (optional)
   - ``LLM_MODEL`` — Model name (defaults to ``gpt-4o-mini``)

   See :doc:`../user_guide/configuration` for all providers and setup.

----

See Also
--------

- :doc:`architecture` — System design and data flow
- :doc:`contributing` — Development guidelines
- :doc:`prompts` — Prompt YAML reference and editing guide
- ``CLAUDE.md`` — Implementation details and patterns
