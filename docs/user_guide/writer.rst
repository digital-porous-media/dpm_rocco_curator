Writer
======

The Writer takes a dataset description, retrieved context chunks, and your feedback, then produces
an improved description with every change traced to a verifiable source. It is composed of two
components: a **Content Screener** that validates your feedback before it reaches the editor, and
a **Description Editor** that performs the enhancement.

Content Screening
-----------------

**Class:** ``src.llm.content_screener.ContentScreener``

Before the editor sees your feedback, the screener checks it against four criteria:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Check
     - What it asks
   * - **Relevant?**
     - Does the feedback pertain to the dataset being described?
   * - **Accurate?**
     - Is the feedback consistent with what is already known about the dataset?
   * - **Respectful?**
     - Is the tone professional?
   * - **Coherent?**
     - Does the feedback make sense as a dataset description instruction?

The screener returns one of three recommendations:

- ``accept`` — feedback is passed to the editor unchanged
- ``reject`` — feedback is discarded; you will be asked to rewrite it
- ``flag_for_review`` — feedback is shown in the UI with a warning for your review before proceeding

The screener prompt is stored in ``src/prompts/content_screener.yaml``.

How the Enhancement Prompt Works
---------------------------------

The enhancement prompt (``src/prompts/editor.yaml``) receives four inputs:

1. **Original description** — the text you want to improve
2. **Evaluation feedback** — the rubric breakdown from the Evaluator, so the LLM knows which criteria are failing
3. **Retrieved context chunks** — up to ``top_k_context`` (default 5) excerpts from your uploaded documents, formatted with source metadata headers
4. **Conversation history** — all previous rounds of feedback and enhanced descriptions, so each pass builds on the last

The LLM is instructed to:

- Improve the description while staying faithful to verifiable facts
- Add information *only* when it can be traced to a context chunk from an uploaded document, the original description, or your explicit feedback
- Return a structured JSON object with the improved text, a rationale, and a citation list

Citations
---------

Every statement added or substantially changed in the enhanced description gets a citation. An example citation might look like:

.. code-block:: json

   {
     "statement": "The raw images were segmented into solid and pore phases using Otsu's method.",
     "source": "uploaded_document",
     "quote": "...segmentation was performed using Otsu's method...",
     "doc_title": "Smith_et_al_2015",
     "page": 3,
     "chunk_index": 5
   }

**Source types:**

- ``original_description`` — the claim was already present in your original text
- ``uploaded_document`` — the claim is supported by an uploaded document; ``doc_title``, ``page``, and ``chunk_index`` point to the exact passage
- ``user_feedback`` — the claim came directly from your written feedback

Always verify citations before publishing: the LLM may occasionally link a statement to a passage
that only loosely supports it.

Running Without the UI
----------------------

.. code-block:: python

   import json
   from src.llm.client import RoccoClient
   from src.evaluator.evaluator import DescriptionEvaluator
   from src.editor.editor import DescriptionEditor
   from src.ingestor.embedder import DocumentEmbedder
   from src.retriever.retriever import VectorStoreManager

   with open("src/evaluator/rubric.json") as f:
       rubric = json.load(f)
   with open("src/evaluator/examples_v3.json") as f:
       examples = json.load(f)

   client   = RoccoClient()
   embedder = DocumentEmbedder()
   vsm = VectorStoreManager(embedder)
   # vsm.load("my_faiss_index/")  # optional: load a pre-built index

   draft_text = "This dataset contains micro-CT images of Berea sandstone ..."

   # enhance() requires a real EvaluatorOutput — draft_evaluation is not optional
   evaluation = DescriptionEvaluator(client, rubric, examples).evaluate(draft_text)

   editor = DescriptionEditor(client, rubric, vsm)

   result = editor.enhance(
       draft_text=draft_text,
       draft_evaluation=evaluation,
       user_feedback="The samples were imaged at 2 µm voxel resolution.",
   )

   print(result.suggested_text)
   for c in result.citation:
       print(f"  [{c.source}] {c.statement[:80]}...")

To run multiple rounds, call ``enhance()`` again on the improved text (re-evaluating it first,
since ``draft_evaluation`` must reflect the text being passed in):

.. code-block:: python

   evaluation2 = DescriptionEvaluator(client, rubric, examples).evaluate(result.suggested_text)
   result2 = editor.enhance(
       draft_text=result.suggested_text,
       draft_evaluation=evaluation2,
       user_feedback="Clarify the file naming convention.",
   )

The editor automatically carries conversation history across calls. To start fresh:

.. code-block:: python

   editor.reset_conversation_history()

Session Files
-------------

Session state — original description, current description, and full conversation history — is
*intended* to be saved to disk and reloaded later via ``editor.save_session(path)`` /
``editor.load_session(path)``.

.. warning::

   ``save_session()`` currently raises ``pydantic.ValidationError`` on every call:
   ``EditingSession.metadata`` (``src/llm/schemas.py``) is a required field, but
   ``save_session()`` never supplies it. This is a code bug, not a documentation gap — don't rely
   on session persistence until it's fixed.

.. note::

   Session file persistence is not yet wired into the Streamlit UI. The web app uses in-memory
   state only; refreshing the browser resets the session.

Multi-Turn Refinement
---------------------

Each call to ``enhance()`` appends to ``editor.conversation_history``. This list is injected into
the next prompt, so the LLM understands what feedback has already been incorporated and can focus
on what's still missing — each round narrows in on a specific gap (e.g. missing QA/QC details,
then wording) rather than repeating the same broad feedback.

Output Schema
-------------

``enhance()`` returns an ``EditorOutput`` Pydantic model:

.. code-block:: python

   class EditorOutput(BaseModel):
       original_text:  str          # the description passed in
       suggested_text: str          # the improved description
       rationale:      str          # summary of changes made
       citation:       List[Citation]

   class Citation(BaseModel):
       statement:   str
       source:      str             # "original_description" | "uploaded_document" | "user_feedback"
       quote:       str
       doc_title:   Optional[str]   # for uploaded_document sources
       page:        Optional[int]
       chunk_index: Optional[int]

See Also
--------

- :doc:`evaluator` — Generating the ``EvaluatorOutput`` passed to the Writer
- :doc:`rag` — Building the vector store that the Writer queries
- :doc:`streamlit_app` — Running the full workflow in the web UI
- :doc:`../developer_guide/api_reference` — Full class documentation
