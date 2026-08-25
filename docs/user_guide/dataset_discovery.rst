Dataset Discovery
===================

Find datasets by describing what you're looking for in plain language — a rock type, an
imaging method, or an intended purpose ("suitable for lattice Boltzmann simulation"). This
page covers the ``search_datasets`` tool.

When to Use This vs. Structured Queries
------------------------------------------

The assistant routes between two different dataset tools depending on how your question is
phrased:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - This page (``search_datasets``)
     - :doc:`structured_queries` (``get_dataset_details``)
   * - "Sandstone datasets suitable for pore-scale simulation"
     - "Datasets with porosity above 0.3"
   * - "Something good for a teaching demo"
     - "Segmented datasets by Jane Doe"
   * - Open-ended, one value per field at most, no numeric comparisons
     - Numeric thresholds/ranges, exact values, named authors, multiple combined constraints

If you name a concrete, checkable property (a number, a specific value, a person), the
assistant should route to ``get_dataset_details`` instead — see that page for the full list of
queryable properties. ``search_datasets`` still attempts a structured Cypher lookup internally
as a safety net (see below), but routing directly to the right tool is more reliable.

What It Does
------------

Given a query, ``search_datasets`` (``src/assistant/tools.py``):

1. **Expands the query** via ``expand_query()`` — an LLM call using ``src/prompts/query_expander.yaml``.
   This rewrites the query with precise domain terminology (e.g. "tight rock" →
   "tight/ultra-tight formation, micro-porosity, sub-micron pore throats, nano-Darcy
   permeability") and infers any clearly-implied metadata filters
   (``porousMediaType``, ``source``, ``segmented``, ``voxelDimensions``). For purpose/suitability
   queries, it also reasons about what dataset characteristics would serve that purpose and
   explains why in a ``rationale`` field, which the assistant paraphrases (never quotes
   verbatim) back to you.
2. **Safety-net structured check** — if the query looks property-shaped (inferred filters were
   set, or it matches a known rock-type/imaging-method keyword, or names a person via a
   ``by``/``from``/``authored by`` pattern), it tries ``GraphStore.cypher_qa()`` first and
   returns that answer directly if it found something. This exists in case routing missed a
   property-shaped query.
3. **Hybrid search** — otherwise, calls ``GraphStore.hybrid_search()``: vector similarity search
   over the ``datasetEmbedding`` index, combined with Neo4j BM25 full-text search via
   Reciprocal Rank Fusion (RRF). This catches vocabulary mismatch (query says "FNO velocity
   field prediction", dataset description says "machine learning transport simulation
   benchmark") that pure vector similarity alone can miss. Results are labeled
   ``[hybrid match]``.
4. **Component-level second pass** — also calls ``GraphStore.component_search()`` over the
   ``componentEmbedding`` index (individual ``Sample``/``DigitalDataset``/``AnalysisDataset``
   sub-nodes, each embedded with parent-dataset context). This catches datasets whose parent
   description has weak signal but whose sub-nodes score better — e.g. one specific scan out of
   several under a multi-sample dataset. Results are labeled ``[component match]`` and include
   which sub-node matched.
5. **Result summarization** — one short LLM-generated sentence per result describing what it is
   and how it relates to the query (titles/DOIs stay verbatim from metadata; only this prose
   summary is LLM-authored).
6. **Weak-match detection** — if none of the results actually mention the query's concrete topic
   terms, the output is tagged ``[weak match: ...]`` so the assistant tells you plainly that
   nothing directly matched rather than presenting a loose result as if it were relevant.

Because ``search_datasets`` returns real DOIs and descriptions, its output is treated as a
**verbatim tool** by the conversation manager — see :doc:`assistant`'s "Response Assembly"
section. Only a short lead-in sentence is LLM-generated; the dataset list itself is spliced in
unmodified.

Filter Fields
--------------

This is the **complete, exhaustive list** of filters ``expand_query`` will infer — it's not an
oversight that there are only four. These are exactly the fields available as post-retrieval
metadata on a vector/BM25 search hit (``GraphStore.hybrid_search()``'s fulltext query only
projects these four onto each result); anything else requires a real Cypher query, which is
what :doc:`structured_queries` is for. The prompt (``src/prompts/query_expander.yaml``) is
explicit about not inventing fields outside this list:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Allowed values
   * - ``porousMediaType``
     - ``beads``, ``carbonate``, ``coal``, ``fibrous_media``, ``granite``, ``other``,
       ``sandstone``, ``soil``
   * - ``source``
     - ``artificial``, ``natural`` (lab-made vs. field-collected)
   * - ``segmented``
     - ``yes``, ``no``
   * - ``voxelDimensions``
     - ``micrometer``, ``millimeter``, ``nanometer``, ``other`` (a coarse bucket — cannot
       express a numeric cutoff like "< 2 microns"; that needs :doc:`structured_queries`)

For the much larger set of properties queryable via Cypher (porosity, grain size, imaging
equipment, authors, and more), see :doc:`structured_queries`'s "Queryable Properties" section
and the `DPM data model diagram <https://digital-porous-media.github.io/dpm_docs/images/data_model_v2.png>`_.

Example Queries
----------------

.. code-block:: text

   Sandstone datasets suitable for pore-scale simulation
   Something good for a teaching demo
   Datasets for training a Fourier Neural Operator on velocity fields
   Microbial transport data
   High-resolution carbonate scan in nanometers

See Also
--------

- :doc:`structured_queries` — Numeric/exact-value/named-author dataset queries
- :doc:`dataset_profiles` — Follow-up detail questions on a dataset you've already found
- :doc:`assistant` — Overview of how all capabilities fit together
- :doc:`../developer_guide/architecture` — Maintenance: adding datasets to the graph
- ``src/prompts/query_expander.yaml`` — The query expansion prompt
