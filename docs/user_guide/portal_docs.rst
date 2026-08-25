Portal Documentation Search
==============================

Ask how-to questions about the DPM Portal itself — uploading data, navigating the interface,
metadata field definitions and schema — and get an answer synthesized from the portal's own
documentation. This page covers the ``search_portal_docs`` tool.

When to Use This
------------------

Route here for **portal actions and navigation** ("how do I upload/download/copy/cite a
dataset", "how do I add collaborators") and for **metadata schema questions** — anything asking
about the definition, purpose, or difference between the portal's own entity types (``Dataset``,
``Sample``, ``Digital Dataset``, ``Analysis Dataset``). These are portal-specific terms with real
documented definitions, not general science concepts — never :doc:`domain_qa` or
:doc:`workflow_guidance` for these, since answering from general knowledge produces wrong,
made-up definitions.

What It Does
------------

``search_portal_docs(question)`` (``src/assistant/tools.py``) delegates to
``search_portal_docs_v2()`` in ``src/assistant/portal_docs_retrieval.py`` — a hand-rolled,
`PageIndex-style <https://github.com/VectifyAI/PageIndex>`_ retrieval approach with no chunking
and no embeddings:

1. **Index** (``portal_docs_tree.py``) — builds a hierarchical tree directly from each synced
   ``dpm_docs`` markdown page's ``#``/``##``/.../``######`` heading structure. Built at
   query/import time from ``data/portal_docs/docs/`` — there is no separate rebuild step or
   staleness problem; re-syncing the markdown (see
   :doc:`../developer_guide/architecture`'s "Maintenance" section) and restarting the app is all
   that's needed.
2. **Retrieval** (``portal_docs_retrieval.py``) — a single LLM call
   (``select_nodes_for_query``) reasons over the flattened list of section ids, page titles,
   section titles, and a short text/field-name snippet per section, and returns up to 4 node
   ids most relevant to the question. Unlike embedding similarity over prose, an LLM reasoning
   over short, clearly-differentiated titles isn't biased toward whichever section's text most
   densely repeats a shared word — and it's explicitly instructed to return **one node per named
   entity** for a comparison question ("difference between X and Y"), rather than just the
   single closest match. Falls back to a keyword/title-substring match if the LLM call fails to
   parse.
3. **Synthesis** — the selected sections' text is assembled into context (with a
   dataset-container disambiguation block prepended when relevant) and passed to an LLM call
   using ``src/prompts/portal_docs.yaml``, which is instructed to:

   - Answer only from the given excerpts, never general knowledge of the portal
   - Never blend ``Dataset``, ``DigitalDataset``, and ``AnalysisDataset`` definitions together
     even though excerpts about them all share the word "Dataset"
   - Cite sources in a ``Sources:`` line ([portal docs] entries with their doc URL)
   - Mention (never fabricate) a screenshot only when a retrieved excerpt contains a literal
     ``[Figure: ...]`` placeholder

Because this answer is already synthesized and cited, ``search_portal_docs`` is a
**self-contained tool** in the conversation manager (see :doc:`assistant`) — its answer is
returned directly, never re-synthesized by the outer agent. (Unlike the verbatim tools, its raw
retrieval is prose excerpts that only answer the question once something actually reads and
synthesizes them — pasting them unmodified produced disconnected, sometimes off-topic dumps.)

Data Source
------------

The underlying markdown lives in ``data/portal_docs/docs/``, synced from the public
`dpm_docs <https://github.com/digital-porous-media/dpm_docs>`_ repository via
``scripts/sync_dpm_docs.py``. See :doc:`../developer_guide/architecture`'s "Maintenance" section
for how to check for and pull updates.

Example Queries
----------------

.. code-block:: text

   How do I upload a dataset to the portal?
   How do I add collaborators to a dataset?
   What is the difference between a Digital Dataset and an Analysis Dataset?
   What fields does a Sample need?
   How do I cite a dataset from the DPM Portal?

See Also
--------

- :doc:`assistant` — Overview of how all capabilities fit together
- :doc:`domain_qa` — Porous media science Q&A (not portal-specific)
- :doc:`../developer_guide/architecture` — Portal doc sync/maintenance, module reference
- ``src/prompts/portal_docs.yaml`` — The synthesis prompt
