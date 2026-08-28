Dataset Profiles and Follow-Up Questions
===========================================

Ask a deeper follow-up question about a dataset you've already found — "tell me more about
this one," a specific property, how its samples/scans/analyses relate, how to read its files,
or whether it suits a particular use case — or compare two or more datasets against each other.
This page covers the ``get_dataset_profile`` tool.

When to Use This vs. Dataset Discovery / Structured Queries
----------------------------------------------------------------

.. list-table::
   :widths: 34 33 33
   :header-rows: 1

   * - Question shape
     - Tool
     - Why
   * - "Find datasets suitable for X" (many candidates, none yet identified)
     - ``search_datasets``
     - Open-ended discovery — see :doc:`dataset_discovery`
   * - "Which datasets have porosity above 0.3?" (a checkable property, across the catalog)
     - ``get_dataset_details``
     - Structured Cypher query — see :doc:`structured_queries`
   * - "Tell me more about this one" / "is this suitable for X" / "how do I read this file?"
       (ONE dataset already identified, from a prior turn or named directly)
     - ``get_dataset_profile``
     - This page
   * - "Compare dataset A and dataset B for X"
     - ``get_dataset_profile`` (called once per dataset)
     - This page

The distinguishing signal is whether a *specific* dataset is already identified. If the user
hasn't named or previously seen a specific dataset, route to discovery or structured queries
instead — ``get_dataset_profile`` has no way to resolve "this dataset" on its own.

What It Does
------------

``get_dataset_profile(dataset_reference, question)`` (``src/assistant/tools.py``) resolves
``dataset_reference`` against the graph in three tiers, stopping at the first with a match:

1. Dataset number, exact (e.g. ``"42"``, ``"DRP-42"``)
2. DOI, exact and case-insensitive (with or without the ``https://doi.org/`` prefix)
3. Title, case-insensitive substring match

It then fetches the full ``Dataset`` node plus every ``PART_OF`` sub-node (``Sample``,
``DigitalDataset``, ``AnalysisDataset``, ``RelatedPublication``, and — where the live schema
confirms it — ``RelatedSoftware``/``RelatedDataset``), along with the ``INPUT_FOR`` edges that
describe which Sample fed which DigitalDataset fed which AnalysisDataset
(``GraphStore.get_dataset_profile()`` in ``src/assistant/graph_store.py``).

- **Zero matches** → an honest "no dataset was found" message, no LLM call.
- **Multiple matches** (an ambiguous title fragment) → a disambiguation list of candidates, no
  LLM call — resolved from the tier-matching query alone, with no second round-trip to Neo4j.
- **Exactly one match** → the fetched data (with every empty/``None``/``[]`` field dropped
  before it ever reaches the model — sparse fields are never surfaced as clutter) is rendered
  into a structured context and passed to ``src/prompts/dataset_profile.yaml`` for one grounded
  LLM synthesis call.

Because that synthesis call must reason over the data (not just repeat it — see "File-Format
and Reuse-Suitability Reasoning" below), ``get_dataset_profile`` is a **self-contained tool**
in the conversation manager's response assembly (see :doc:`assistant`), like
``get_workflow_guidance``/``get_educational_context``. Its answer is returned directly, with a
code-generated ``[dataset profile] <title> (DOI: ...)`` header that is never retyped by an LLM.

General "tell me more" questions get a concise, relevant, high-level overview — the
description, the organizational structure at a summary level, and the most salient recorded
characteristics — not an exhaustive dump of every field returned. Full field-by-field detail is
reserved for when you ask about a specific field.

Comparing Multiple Datasets
----------------------------

There is no separate comparison tool. For "compare A and B," the agent calls
``get_dataset_profile`` once per dataset — each with its own resolved reference and the
comparison question — and synthesizes the comparison itself from both results. This reuses the
same multi-tool-call mechanism already used for cross-intent queries (see :doc:`assistant`):
when more than one tool call happens in a turn, the outer agent's own final-message synthesis
runs instead of the single-call short-circuit.

Organizational Structure (Sample → Digital Dataset → Analysis Dataset)
--------------------------------------------------------------------------

A dataset's samples, scans, and analyses are connected by ``INPUT_FOR`` edges — which physical
sample was scanned to produce which digital dataset, and which digital dataset was analyzed to
produce which analysis dataset. ``get_dataset_profile`` renders these as explicit chains (e.g.
``Core 1 -> Scan 1 -> PNM 1``) rather than three unrelated flat lists, and calls out any digital
dataset with no recorded sample/analysis link so the structure doesn't look silently complete
when it isn't.

.. warning::

   ``INPUT_FOR`` points **child → parent** ("was derived from") — the same direction as
   ``PART_OF``, despite the name. The correct pattern is
   ``(dd:DigitalDataset)-[:INPUT_FOR]->(s:Sample)``; writing it the intuitive way round matches
   zero rows and fails silently. See :doc:`../neo4j_schema` for the verified edge counts.

File-Format, Data Location, and Reuse-Suitability Reasoning
----------------------------------------------------------------

No *file-level* path or filename is ever recorded in the graph — the assistant can never tell
you the exact file to open. It can, however, give you a real, non-fabricated **dataset-level**
download location: the DPM Portal mirrors every published dataset on TACC Corral at
``https://web.corral.tacc.utexas.edu/digitalporousmedia/archive/DRP-{datasetNumber}/``, derived
directly from the dataset's real ``datasetNumber``. The archive keeps every published version
as its own subdirectory (``DRP-{n}``, ``DRP-{n}v2``, ...), and the graph doesn't record which
version is current — so this link may not always point at the latest version; the assistant
will say so and point you to the dataset's portal page to confirm.

For "how do I read this in Python" and reuse-suitability questions ("is this suitable for
two-phase flow simulation"), the assistant reasons from the dataset's actual recorded file
types/formats and properties, combined with general programming/domain knowledge where the
graph itself has no answer — always explicitly flagged as general knowledge, not a
portal-verified claim, per the tiered Knowledge Source Policy in :doc:`assistant`. An absence
of a recorded property is never treated as evidence against suitability — only as an honest
gap.

Example Queries
----------------

.. code-block:: text

   Tell me more about this dataset.
   What file types does the first one use, and how would I read them in Python?
   Where can I download this dataset?
   Is this dataset suitable for two-phase flow simulation?
   How are the samples in this dataset related to its scans and analyses?
   Compare Dataset A and Dataset B for pore network extraction.

See Also
--------

- :doc:`dataset_discovery` — Open-ended/suitability dataset search across many candidates
- :doc:`structured_queries` — Exact/numeric property queries across the catalog
- :doc:`content_reasoning` — Relationship/content questions no single field can answer
- :doc:`multi_turn` — How "this dataset" / "the second one" is resolved before this tool runs
- :doc:`assistant` — Overview of how all capabilities fit together, including response assembly
- :doc:`../neo4j_schema` — Full graph schema reference and coverage stats
- :doc:`../developer_guide/architecture` — Request lifecycle and tool registry
