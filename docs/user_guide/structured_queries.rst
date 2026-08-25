Structured Dataset Queries
============================

Ask about exact, checkable dataset properties — a numeric threshold, a specific value, or a
named author — and get an answer generated from a real Cypher query against the dataset graph,
not a similarity match. This page covers the ``get_dataset_details`` tool.

When to Use This vs. Dataset Discovery
-----------------------------------------

See :doc:`dataset_discovery`'s comparison table. In short: if your question names a concrete,
checkable property — a number, a threshold, an exact value, or a person — it belongs here, even
if it also mentions a rock type or imaging method (e.g. "sandstone datasets with porosity above
0.3" routes here, not to dataset discovery, because of the ``> 0.3`` constraint).

What It Does
------------

``get_dataset_details(question)`` (``src/assistant/tools.py``) passes your question directly to
``GraphStore.cypher_qa()`` (``src/assistant/graph_store.py``), which wraps LangChain's
``GraphCypherQAChain``:

1. The chain's LLM generates a Cypher query against a **hardcoded schema** fed via
   ``MANUAL_SCHEMA`` (not introspected live — ``refresh_schema=False`` skips the
   ``apoc.meta.data()`` call so this works without APOC installed). The Cypher-generation
   prompt explicitly forbids ``apoc.*`` calls, keeping queries portable across local Neo4j,
   AuraDB, and the TACC VM.
2. The query runs against the live graph with ``return_intermediate_steps=True``, so the code
   can distinguish "the query ran and found zero rows" (an honest, complete answer) from "the
   chain's own default failure string" (something actually went wrong) — both cases would
   otherwise return the same generic "I don't know the answer."
3. For result rows that look like a dataset listing (every row has a ``title``), the bullet
   list is built directly from the raw rows in Python (``_format_dataset_rows``), rather than
   trusting the chain's own answer-synthesis LLM call to reproduce titles/DOIs correctly — that
   call is not reliably steerable by prompt wording alone. Other result shapes (counts,
   aggregates) still use the chain's own prose answer.

Because this can return real dataset titles/DOIs, ``get_dataset_details`` is also a
**verbatim tool** in the conversation manager's response assembly (see :doc:`assistant`) — its
output reaches you unmodified, aside from a short LLM-generated lead-in sentence.

Queryable Properties
----------------------

The tool's own description (what the routing LLM sees) is **generated dynamically** from
``GraphStore.get_queryable_field_names()``, which parses the same ``MANUAL_SCHEMA`` string that
drives Cypher generation — so a field added to the schema is automatically reflected in routing
without a second place to edit. Broadly, queryable properties span:

- **Sample**: ``porousMediaType``, ``porosity``, ``grainSizeAvg``/``Min``/``Max``,
  ``collectionMethod``, ``source`` (artificial/natural), ``geographicOrigin``, and more
- **DigitalDataset**: ``voxelDimensions``, ``imagingEquipmentAndModel``, ``imageFormat``,
  ``segmented``, ``dimensionality``, and more
- **AnalysisDataset**: ``segmented``, ``type``, ``referencedDigitalDataset``, and more
- **Dataset**: ``title``, ``doi``, ``authors`` (via ``RelatedPublication``), ``datasetNumber``

For the visual entity-relationship diagram behind this schema, see the DPM Portal's
`data model reference <https://digital-porous-media.github.io/dpm_docs/images/data_model_v2.png>`_.
For the full, canonical field-by-field reference with coverage percentages against the live
graph, see :doc:`../neo4j_schema` — some fields (e.g. imaging-center metadata) are 0% populated
in current data, and the assistant must not assume they exist.

A named person ("datasets by Jane Doe", "who has published data on sandstone permeability")
counts as a checkable property here too — it maps to the ``authors`` field. The routing prompt
distinguishes this from an incidental name mention ("Hi, I'm Bernie"), which is small talk, not
an author lookup.

Example Queries
----------------

.. code-block:: text

   Which datasets have porosity above 0.3?
   Show datasets with permeability < 1 millidarcy and rock type = sandstone
   Micro-CT images with resolution better than 5 microns
   What datasets did Jane Doe publish?
   How many sandstone datasets are segmented?

See Also
--------

- :doc:`dataset_discovery` — Open-ended/suitability dataset search
- :doc:`dataset_profiles` — Follow-up detail questions on a dataset you've already found
- :doc:`assistant` — Overview of how all capabilities fit together
- :doc:`../neo4j_schema` — Full graph schema reference and coverage stats
- `DPM data model diagram <https://digital-porous-media.github.io/dpm_docs/images/data_model_v2.png>`_ — visual entity-relationship reference
- :doc:`../developer_guide/architecture` — Maintenance: adding datasets to the graph
