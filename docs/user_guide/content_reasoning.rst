Relationship and Content Questions
====================================

Some questions about the catalog cannot be answered by looking up a field, no matter how good
the query is — not because the information is missing, but because it is either *unstructured*
(mentioned in a description rather than stored in a field) or *relational* (it depends on how a
dataset's parts relate to each other). This page covers the ``reason_about_dataset_content``
tool, which answers those honestly instead of quietly answering a narrower question.

The Problem It Solves
---------------------

Take *"are there paired tomographic and segmented images?"*

There is no usable imaging-modality field in the DPM schema — ``imagingCenter`` and
``imagingEquipmentAndModel`` exist but are populated on roughly 4% of nodes, which is worse than
useless as a filter: it answers for 4% of the catalog and silently drops the rest (see
:doc:`../neo4j_schema`). And "paired" is not a property at all: it is a statement about two scans of the *same sample*
standing in a particular relationship. A structured query can only pick out the one literal
clause it recognises, ``segmented = 'yes'``, and answer that.

That answer is not a partial answer — it is a **wrong** answer presented confidently. A generic
"has some segmented data" list looks exactly like a verified list of paired datasets, and
nothing in the response tells the reader that "paired" was silently dropped.

The fix is not to keep adding query patterns for each new phrasing. It is one mechanism that
recognises the whole class of question and answers it as what it is: reasoning, not a lookup.

When It Fires
-------------

.. list-table::
   :widths: 45 25 30
   :header-rows: 1

   * - Question
     - Tool
     - Why
   * - "sandstone datasets with porosity above 0.3"
     - ``get_dataset_details``
     - A plain conjunction of independent literal fields — each clause narrows the catalog on
       its own, and Cypher expresses both exactly.
   * - "datasets suitable for LBM simulation"
     - ``search_datasets``
     - Open-ended suitability with no checkable property named — see :doc:`dataset_discovery`.
   * - "paired tomographic and segmented images"
     - ``reason_about_dataset_content``
     - Relational. ``segmented`` is real, but it is a clause *inside* a relational claim, and
       is not independently displayable as an answer.
   * - "the same sample imaged at different resolutions"
     - ``reason_about_dataset_content``
     - Requires comparing one dataset's scans against each other, not matching a field.
   * - "datasets imaged on an Xradia scanner"
     - ``reason_about_dataset_content``
     - Real information, but it lives in a sample's free-text description, not a field.

The dividing line is **not** "the field exists" vs. "it doesn't". It is *"is every property in
the question a plain, literal, structured field?"* If yes, it stays on Cypher. If anything in it
is relational or free-text, the **whole** question routes here — it is never split, because a
literal clause pulled out of a relational claim is exactly the misleading partial answer this
tool exists to eliminate.

Gated in Code, Not by Routing
------------------------------

The split is enforced by a deterministic check, ``_needs_content_reasoning()``
(``src/assistant/tools.py``), which both ``get_dataset_details`` and ``search_datasets`` run
*before* committing to a Cypher answer. If it fires, they hand the whole question to this tool
instead and return its result.

This is deliberate. The two sides of the line are worded almost identically — "segmented and
porosity above 0.3" is plain, "segmented and imaged the same way" is relational — and this
project has repeatedly found that prompt-level routing is not reliable for nuanced binary calls
like that (see ``HANDOFF.md``). Gating in code makes the split correct regardless of which tool
the agent happened to pick. ``search_datasets`` already sets this precedent with
``_is_plain_property_query()``.

Borderline cases — a relational phrase alongside a genuinely literal property — still route
here, but are logged so the heuristic can be reviewed against real usage over time.

How It Works
------------

Everything expensive happens offline, at index-build time. At query time:

1. **Narrow.** Rank the precomputed fact sheets (below) with the same vector + BM25 Reciprocal
   Rank Fusion that ``hybrid_search`` already runs for dataset discovery — just pointed at the
   fact-sheet indexes instead of the description ones (``GraphStore.rank_fact_sheets``). No LLM
   call in this step. One general mechanism serves every relational phrasing, including ones
   nobody anticipated; there is no per-relationship Cypher pattern to author and re-verify.
2. **Fetch.** A plain, ID-based read of ``Dataset.factSheet`` for the shortlist — nothing
   computed, nothing pattern-specific.
3. **Reason.** One LLM call (``src/prompts/corpus_reasoning.yaml``) judges each candidate and
   must cite the specific recorded fact or quoted sentence it relied on.
4. **Compose.** A fixed honesty framing, then the cited shortlist.

For exhaustive questions that ranking cannot legitimately narrow ("list *every* dataset
where…"), it falls back to a batched map-reduce pass: the corpus is screened in parallel
batches by a cheap first-pass call, and the survivors go through the same careful reasoning
step. This bounds cost per call and scales as the catalog grows.

Fact Sheets
-----------

A **fact sheet** is a precomputed, edge-preserving summary of one dataset, stored on its
``Dataset`` node as ``factSheet`` (JSON) and ``factSheetText`` (rendered prose), and built by
``scripts/build_dataset_vector_index.py`` alongside the embeddings.

It holds each node's title and description (this is where instrument mentions and processing
narration actually live), the key structured properties (``porousMediaType``,
``voxelDimensions``, ``segmented``, ``type``), any related publication abstracts, and —
critically — **which** ``DigitalDataset`` belongs to **which** ``Sample``. That last part is why
this is not simply reusing the embedding text: the embedding builder deliberately flattens
sub-node properties into aggregated lines, which is right for similarity search and makes "does
this sample have scans at two different resolutions?" unanswerable.

Fact sheets cache **raw material only, never a verdict.** Whether a dataset satisfies the
relationship a given question describes is inherently query-dependent, so that judgment always
happens live. Precomputing only removes the cost of re-deriving each summary from the graph on
every call.

Grounding
---------

Two guarantees are enforced in code, not left to the prompt:

- **No citation, no candidate.** A candidate returned without a supporting quote or recorded
  fact is dropped before the user sees it.
- **No dataset the model wasn't shown.** A candidate whose title isn't in the shortlist that was
  actually sent is dropped as a likely fabrication.

Titles and DOIs in the response come from the graph records, never retyped by the model — the
same pattern as the ``[dataset profile]`` header.

So while the *ranking* is approximate, what the answer *asserts* is not: the fact sheet carries
the literal recorded properties, so every cited claim traces back to real data.

What the User Sees
------------------

Every answer from this tool is framed the same way, regardless of which underlying case
produced it::

    [content reasoning] I can't confirm this from a database field — here's what reasoning
    over the available facts and descriptions suggests. Verify before relying on it.

    - **Bentheimer Sandstone for Analyzing Wetting Phenomena** (DOI: 10.17612/P8FQ-6Y93)
      The dataset contains both original tomographic images and segmented images.
      *Basis:* Original tomographic image — voxelDimensions: 4.95 x 4.95 x 4.95 micrometers;
      segmented: no; Segmented image — 4.95 x 4.95 x 4.95 micrometers; segmented: yes

The ``Basis`` line is cleaned up for readability before display — whitespace collapsed onto one
line, the fact sheet's own section header dropped, and the verbose stored voxel-dimension
phrasing (``X, Y, Z units (in micrometers): 4.54, 4.54, 4.54``) compacted. Only presentation
changes: the recorded values are never rounded, reordered, or dropped, since they are the
grounding. Collapsing newlines is also load-bearing rather than cosmetic — a newline inside a
citation ends the markdown list item, so the remainder renders as a loose paragraph that reads
as the previous dataset's rationale having escaped its bullet.

If nothing plausibly matches, it says so plainly rather than padding the list. The tool is
registered as self-contained (``_SELF_CONTAINED_TOOLS`` in ``conversation_manager.py``), so this
output reaches the user as written — the framing sentence and the citations are never handed
back to a model to re-word.

Limitations
-----------

- **Ranking recall is not guaranteed.** A hand-written Cypher condition for one specific
  relationship would have zero recall risk *for that one relationship*. Hybrid vector + BM25
  ranking narrows that gap considerably but does not close it. This is a deliberate trade:
  accepting some residual recall risk in exchange for never needing new code for the next
  relational phrasing someone asks about.
- **It reasons over text and recorded properties only.** It never opens or classifies image
  files.
- **It depends on the fact sheets being built.** If they haven't been (see
  :doc:`../neo4j_schema`), the tool says so rather than falling back to a partial field lookup.
- **Answers are candidates, not verified results** — which is exactly what the framing says.

Rebuilding
----------

Fact sheets and their indexes are rebuilt with the rest of the index build:

.. code-block:: bash

   python scripts/build_dataset_vector_index.py                  # everything
   python scripts/build_dataset_vector_index.py --only fact-sheets   # fact sheets only

Re-running is safe (``SET`` upserts, ``CREATE INDEX ... IF NOT EXISTS``). Rebuild when datasets
are added or when the fact-sheet assembly logic changes.

See Also
--------

- :doc:`dataset_discovery` — open-ended semantic search
- :doc:`structured_queries` — literal field lookups via generated Cypher
- :doc:`dataset_profiles` — deep dive on ONE already-identified dataset
- :doc:`multi_turn` — how these results are tracked for follow-up questions
- :doc:`../neo4j_schema` — schema, indexes, and the ``INPUT_FOR`` direction gotcha
