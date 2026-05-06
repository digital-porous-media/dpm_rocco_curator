API Reference
==============

Core Classes and Functions

Evaluator
---------

.. automodule:: src.evaluator.evaluator
   :members:
   :undoc-members:
   :show-inheritance:

LLM Client
----------

.. automodule:: src.llm.client
   :members:
   :undoc-members:
   :show-inheritance:

Editor
------

.. automodule:: src.editor.editor
   :members:
   :undoc-members:
   :show-inheritance:

Document Ingestor
-----------------

.. automodule:: src.ingestor.document_ingestor
   :members:
   :undoc-members:
   :show-inheritance:

Document Embedder
-----------------

.. automodule:: src.ingestor.embedder
   :members:
   :undoc-members:
   :show-inheritance:

Vector Store Manager
--------------------

.. automodule:: src.retriever.retriever
   :members:
   :undoc-members:
   :show-inheritance:

Content Screener
----------------

.. automodule:: src.llm.content_screener
   :members:
   :undoc-members:
   :show-inheritance:

Prompt Loader
-------------

.. automodule:: src.prompts.loader
   :members:
   :undoc-members:
   :show-inheritance:

Output Schemas
--------------

.. automodule:: src.llm.schemas
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

Environment variables (set in `.env`):

- ``LLM_PROVIDER`` — Provider shortcut (openai, anthropic, ollama, etc.)
- ``LLM_API_KEY`` — API key (required)
- ``LLM_BASE_URL`` — Custom endpoint URL (optional)
- ``LLM_MODEL`` — Model name (defaults to gpt-4o-mini)

See :doc:`../user_guide/configuration` for all providers and setup.

Common Usage Patterns
~~~~~~~~~~~~~~~~~~~~~

**Evaluate a Description**

.. code-block:: python

   from src.evaluator.evaluator import DescriptionEvaluator
   from src.llm.client import RoccoClient
   import json

   # Load resources
   with open("src/evaluator/rubric.json") as f:
       rubric = json.load(f)
   with open("src/evaluator/examples_v3.json") as f:
       examples = json.load(f)

   # Create evaluator
   client = RoccoClient()  # reads from .env
   evaluator = DescriptionEvaluator(client, rubric, examples)

   # Evaluate
   description = "Micro-CT images of Berea sandstone at 2µm resolution..."
   result = evaluator.evaluate(description)
   print(f"Score: {result.total_score}/10")

**Enhance a Description**

.. code-block:: python

   from src.editor.editor import DescriptionEditor
   from src.llm.client import RoccoClient
   from src.retriever.retriever import VectorStoreManager

   # Create editor with optional RAG context
   client = RoccoClient()
   vector_store = VectorStoreManager()
   editor = DescriptionEditor(client, rubric, vector_store)

   # Enhance
   original = "Sandstone micro-CT images"
   feedback = "Add voxel resolution and facility information"
   context = vector_store.similarity_search(original, k=5)

   result = editor.enhance(original, context, feedback)
   print(f"Enhanced: {result.updated_description}")
   print(f"Citations: {result.citations}")

**Ingest and Search Documents**

.. code-block:: python

   from src.ingestor.document_ingestor import DocumentIngestor
   from src.ingestor.embedder import DocumentEmbedder
   from src.retriever.retriever import VectorStoreManager

   # Prepare components
   ingestor = DocumentIngestor(chunk_size=500, chunk_overlap=100)
   embedder = DocumentEmbedder()
   vector_store = VectorStoreManager()

   # Ingest documents
   documents = ingestor.ingest_from_files(["paper1.pdf", "paper2.pdf"])
   embeddings = embedder.embed_documents(documents)
   vector_store.add_documents(embeddings)

   # Search
   query = "micro-CT imaging porosity"
   results = vector_store.similarity_search_with_score(query, k=5)
   for doc, score in results:
       print(f"{doc.metadata['doc_title']}: {score:.2f}")

**Validate Feedback**

.. code-block:: python

   from src.llm.content_screener import ContentScreener
   from src.llm.client import RoccoClient

   client = RoccoClient()
   screener = ContentScreener(client)

   feedback = "Add details about sample preparation"
   result = screener.screen(feedback)

   print(f"Recommendation: {result.recommendation}")  # "accept", "reject", "flag"
   print(f"Confidence: {result.confidence:.2f}")

Error Handling
~~~~~~~~~~~~~~

Rocco raises standard exceptions:

- ``ValueError`` — Invalid input (e.g., empty description)
- ``FileNotFoundError`` — Missing file (rubric.json, examples.json)
- ``RuntimeError`` — LLM API error (no key, quota exceeded, model unavailable)

Example:

.. code-block:: python

   try:
       result = evaluator.evaluate(description)
   except ValueError as e:
       print(f"Invalid input: {e}")
   except RuntimeError as e:
       print(f"LLM error (check API key and quota): {e}")

Best Practices
~~~~~~~~~~~~~~

1. **Cache LLM clients** — Create once per session, reuse for multiple calls
2. **Use context managers** — FAISS indices should be saved periodically
3. **Validate input** — Empty or extremely long descriptions may fail
4. **Handle timeouts** — LLM calls can timeout; implement retries if needed
5. **Monitor costs** — Different models have different pricing; choose wisely for your use case

Performance Tips
~~~~~~~~~~~~~~~~

- **Evaluation**: ~2–5 seconds (single LLM call)
- **Enhancement**: ~5–10 seconds (RAG retrieval + LLM call)
- **Embedding**: ~0.1–0.5 seconds per chunk (depends on embedder)
- **Vector search**: <100ms for FAISS on CPU

For large-scale deployments:
- Use GPU-accelerated embedders (FAISS-GPU)
- Consider managed vector stores (Pinecone, Weaviate)
- Implement caching for frequent queries

See Also
--------

- :doc:`architecture` — System design and data flow
- :doc:`contributing` — Development guidelines
- ``CLAUDE.md`` — Implementation details and patterns
- ``README.md`` — Overview and quick start
