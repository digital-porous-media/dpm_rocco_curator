Document RAG
============

Retrieval-Augmented Generation (RAG) lets Rocco ground its enhancements in *your* uploaded research
papers rather than in the LLM's general training knowledge. This section explains how the pipeline
works and how to use it programmatically.

What RAG Means Here
-------------------

When you upload documents, Rocco:

1. **Ingests** them — splits each file into overlapping text chunks and enriches each chunk with source metadata
2. **Embeds** the chunks — converts each chunk to a dense vector using a sentence-transformer model
3. **Stores** the vectors in a FAISS index held in memory
4. **Retrieves** the most relevant chunks at enhancement time — querying the index with the description text and each failing rubric criterion to find supporting context

The retrieved chunks are passed to the Writer, which is instructed to cite them explicitly when adding new information to the description.

Document Ingestion
------------------

**Class:** ``src.ingestor.document_ingestor.DocumentIngestor``

Supported formats:

- PDF (``.pdf``) — loaded with LangChain's ``PyPDFLoader``
- DOCX (``.docx``, ``.doc``) — loaded with LangChain's ``Docx2txtLoader``

Chunking uses LangChain's ``RecursiveCharacterTextSplitter``:

- **Chunk size:** 500 characters (default)
- **Chunk overlap:** 100 characters — ensures context is not lost at boundaries
- **Separators tried in order:** ``\n\n``, ``\n``, ``.``, `` ``, ``""``

Each chunk is enriched with metadata:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Value
   * - ``doc_title``
     - Filename without extension (e.g., ``"Smith_et_al_2015"``)
   * - ``page``
     - Page number (PDFs only; ``None`` for DOCX)
   * - ``chunk_index``
     - Sequential index within the document (0-based)
   * - ``source``
     - Full file path (set by LangChain)

Embedding
---------

**Class:** ``src.ingestor.embedder.DocumentEmbedder``

Chunks are embedded using `BAAI/bge-large-en-v1.5 <https://huggingface.co/BAAI/bge-large-en-v1.5>`_
via ``langchain-huggingface``. This model produces 1024-dimensional dense vectors normalised to unit
length, which makes cosine similarity equivalent to dot-product similarity.

The model runs on CPU by default. To use a GPU, pass ``model_kwargs={"device": "cuda"}`` to
``DocumentEmbedder``.

.. note::

   The embedding model is downloaded from HuggingFace on first use (~1.3 GB). Subsequent runs use
   the local cache.

Vector Store
------------

**Class:** ``src.retriever.retriever.VectorStoreManager``

Embeddings are stored in a FAISS flat-L2 index (via LangChain's FAISS wrapper). Key methods:

- ``add_documents(chunks)`` — embed chunks and add them to the index
- ``similarity_search(query, k)`` — return the top-k most similar ``Document`` objects
- ``similarity_search_with_score(query, k)`` — same, but also returns the distance score
- ``save(path)`` / ``load(path)`` — persist and reload the index to/from disk

Query Behavior
--------------

**How Rocco Uses RAG**

During enhancement, Rocco generates targeted search queries based on the evaluation rubric to retrieve relevant context. Each of the 10 evaluation criteria has associated keywords that guide the vector store search:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Rubric Criterion
     - Query Keywords
   * - **Self-Contained Description**
     - data description, data summary, data overview
   * - **Context of Creation**
     - research goals, objectives, study purpose, motivation
   * - **Porous Media Type**
     - sample material, porous media type, lithology
   * - **Research Problem**
     - research problem, research question, hypothesis
   * - **Reuse and Beneficiaries**
     - applications, reuse, reproducibility, validation, machine learning, simulation
   * - **Methodology**
     - methodology, experimental setup, x-ray imaging technique, data collection, image acquisition, scanning, image processing
   * - **Contents and Organization**
     - dataset structure, organization, file, data, format, contents
   * - **Quality Control**
     - quality control, validation, verification, calibration, inspection
   * - **Clarity and Accessibility**
     - *(No automatic query)*
   * - **Keywords**
     - keywords, relevant concepts, domain-specific terminology, nomenclature

By default, Rocco queries all criteria (``query_all=True``). This retrieves relevant context across the full rubric, which generally produces stronger enhancements. Optionally, queries can be limited to only criteria that scored ≤0.5, focusing context retrieval on areas of weakness.

Running Without the UI
----------------------

.. code-block:: python

   from src.ingestor.document_ingestor import DocumentIngestor
   from src.ingestor.embedder import DocumentEmbedder
   from src.retriever.retriever import VectorStoreManager

   # Build the pipeline
   ingestor = DocumentIngestor(chunk_size=500, chunk_overlap=100)
   embedder = DocumentEmbedder()
   vsm = VectorStoreManager(embedder)

   # Ingest documents
   chunks = ingestor.ingest(["paper.pdf", "protocol.docx"])
   vsm.add_documents(chunks)

   # Save the index to reuse later
   vsm.save("my_faiss_index/")

   # Search
   results = vsm.similarity_search_with_score("Berea sandstone segmentation method", k=5)
   for doc, score in results:
       title = doc.metadata.get("doc_title", "unknown")
       page  = doc.metadata.get("page", "?")
       print(f"{title} p.{page} (score {score:.3f})")
       print(doc.page_content[:200])
       print()

Loading a saved index:

.. code-block:: python

   vsm = VectorStoreManager(embedder)
   vsm.load("my_faiss_index/")

   results = vsm.similarity_search("porosity measurement", k=3)

Tips
----

**What documents work best**

- Research publications by the data contributors that provide background context, analysis, applications, and methodology.
- Method papers describing the imaging or analysis technique used to produce the dataset
- Technical protocols or instrument specifications

**Multiple documents**

Rocco blends results from all uploaded documents — more documents generally produce stronger context.
However, contradictory information across documents may appear in the output; always review citations.

**Large document sets**

Ingestion time scales with total file size (chunking + embedding). For very large sets,
consider pre-building the index offline and loading it at runtime with ``vsm.load()``.

See Also
--------

- :doc:`writer` — How retrieved chunks are used in description enhancement
- :doc:`streamlit_app` — Uploading documents through the web UI
- :doc:`../developer_guide/api_reference` — Full class documentation
