Follow-Ups and Multi-Turn Refinement
=======================================

The assistant keeps track of the datasets it just showed you, so you can narrow a result set
("of these, which are coal?"), point at one entry ("tell me more about the second one"), or ask
a bare follow-up constraint ("how about any below 0.25?") without restating the whole question.

This page covers the conversational state in ``src/assistant/conversation_manager.py``. It is
mechanism shared by every dataset tool, not a tool of its own — see :doc:`assistant` for how a
single turn is routed and assembled.

What Gets Remembered
---------------------

Three pieces of per-session state, all held in memory by the ``ConversationManager`` instance
and lost on restart:

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - State
     - What it holds
   * - ``_last_dataset_mentions``
     - The ordered ``{title, doi}`` list from the most recent dataset *listing* — what "these"
       and "the second one" refer to.
   * - ``_cumulative_filter_text``
     - The filter chain built up so far ("sandstone datasets" → "…that are segmented"), used to
       decide whether a new question continues that chain or starts over.
   * - ``_last_profiled_dataset``
     - The single dataset from the most recent :doc:`dataset_profiles` call — what a bare "that
       dataset" resolves to in a later comparison.

A tool counts as a dataset listing based on the **shape of its output**, not which relay path it
took: ``search_datasets``, ``get_dataset_details``, and ``reason_about_dataset_content`` all
render ``Title (DOI: ...)`` entries, so the same parser handles all three
(``_extract_dataset_mentions``). **Any future tool that lists datasets must be added to
``_DATASET_LISTING_TOOLS``** — a listing tool that isn't registered leaves the *previous* turn's
results in place looking current, and a later "of these" then silently refines a set the user has
already moved on from.

Every one of ``chat()``'s return paths — the single-tool-call short-circuit, the deterministic
refinement and comparison dispatches, and the normal end-of-stream path — funnels through
``_track_dataset_listing()`` for the same reason: if any path skipped it, reference resolution
would work or fail depending on which path a given turn happened to take.

Resolving "the second one" / "that dataset"
---------------------------------------------

A follow-up naming no dataset explicitly is resolved against ``_last_dataset_mentions`` by
``_resolve_reference()``, which handles two forms:

- **Ordinals** — "the first one", "the 3rd", "the last result". Recognised up to *fifth*, plus
  *last*; a higher or out-of-range ordinal is not resolved.
- **Titles** — a listed title, or a distinguishing leading portion of one. Titles are usually
  longer than how people refer to them ("Leman Sandstone" for *Leman Sandstone SEM Images*), so
  the first two words act as an anchor — long enough that a single generic word like "sandstone"
  won't match everything.

Both are deliberately conservative: an out-of-range ordinal, or a title fragment matching more
than one prior entry, returns nothing rather than guessing. A missed resolution just falls back
to the LLM reading the replayed conversation history; a wrong guess would silently point you at
the wrong dataset.

The resolved title/DOI is what gets passed to ``get_dataset_profile``, so the deep dive is
always about a dataset that actually appeared in the transcript rather than one the model
recalled from memory.

(Literal DOIs typed into a message, and bare anaphora like "that dataset", are handled on the
comparison path below rather than here.)

Narrowing a Result Set
-----------------------

Two different mechanisms handle narrowing, because two different things can happen to a
constraint. Both end with ``get_dataset_details`` receiving ``restrict_to_titles`` — the exact
titles from the prior listing — so the answer is bounded by that set instead of re-running the
new filter over the whole catalog.

**1. Additive narrowing** — "of these, which are segmented?"

Matched by ``_REFINEMENT_RE`` (phrasings that explicitly name the prior set: "of these", "which
ones", "among those", "narrow it down"). The new constraint is ANDed onto the accumulated filter
chain and dispatched deterministically.

**2. Superseding constraints** — "how about any below 0.25?"

Matched by ``_ELLIPTICAL_REFINEMENT_RE`` — a bare constraint carrying no subject of its own.
These deliberately do **not** take the AND-composition path, because the new constraint
*replaces* an earlier one on the same property: "porosity above 0.3" ANDed with "any below 0.25"
is a contradiction that returns nothing. Instead the agent composes the question itself (it
supersedes correctly), and ``_with_result_set_restriction()`` injects the scope guarantee on top.

That second path also fires on a phrasing-independent signal, ``_continues_filter_chain()``:
if the agent's composed question still carries every subject-bearing term of the chain so far,
it is a refinement regardless of how the user worded it. This is the primary signal, and it
exists because recognising refinement from the user's phrasing kept failing — "of these",
"which ones", "any below 0.25", "are there any with porosity > 0.3", and "how about with
porosity > 0.2" are one intent worded five ways, and each new transcript brought a phrasing the
pattern list didn't have.

Both signals are guarded: the restriction is only ever added to ``get_dataset_details`` (the
only dataset tool that accepts the parameter), only when a prior listing actually exists, and
never over the top of a restriction the caller already set.

.. warning::

   ``cypher_qa`` treats ``restrict_to_titles=[]`` as *no restriction at all*, so an empty list
   silently widens the query back to the entire catalog while the logs still say "restricted
   search". Both a filter chain **and** a non-empty title list are required before the
   refinement dispatch fires; otherwise the turn falls through to normal routing.

When a Chain Resets
--------------------

A **fresh** listing turn that names no datasets (a genuine "no results", or an output shape the
parser can't read) clears the remembered mentions. Keeping them would pair the previous turn's
results with this turn's brand-new filter text, so a later "of these" would narrow a set
unrelated to the chain it was being ANDed onto.

A **refinement** turn that comes back empty keeps them — "of these, which are coal?" returning
nothing doesn't change what "these" means, so you can still narrow the same set a different way.

Topic changes are meant to fall out of this rather than be detected: "What about carbonate
datasets?" names a new subject, so it fails ``_continues_filter_chain()`` and searches the whole
catalog. This is also why "how about"/"what about" alone doesn't match the elliptical pattern —
it has to be followed by a comparison word.

Comparisons Across Turns
-------------------------

"How does that dataset compare with the Bentheimer one?" resolves *both* sides:
``_detect_comparison_references()`` matches the named dataset against the listing and the bare
anaphor against ``_last_profiled_dataset``. ``get_dataset_profile`` is then called once per
dataset, and because that is more than one tool call in the turn, the single-call short-circuit
never fires and the outer agent synthesizes the comparison from both profiles (see
:doc:`dataset_profiles`).

Limitations
------------

- **In-process only.** State lives on the ``ConversationManager`` instance; restarting the app
  clears it. Nothing is persisted to disk.
- **Titles, not IDs.** Restriction is by exact title string, because that is what the listing
  tools emit as plain text. A dataset whose title didn't parse cleanly can't be part of a
  restricted set.
- **One result set at a time.** There is no way to refer back to a listing from several turns
  ago once a newer listing has replaced it.

See Also
--------

- :doc:`assistant` — How a single turn is routed and assembled
- :doc:`dataset_discovery`, :doc:`structured_queries` — The tools that produce listings
- :doc:`dataset_profiles` — Deep dive on one resolved dataset, and comparisons
- :doc:`content_reasoning` — Also produces listings, and is tracked the same way
- :doc:`../developer_guide/architecture` — Request lifecycle and module reference
