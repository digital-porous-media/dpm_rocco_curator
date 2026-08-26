Workflow Guidance
===================

Get step-by-step guidance for a digital rock physics workflow — with real portal tutorial
notebook links where one exists — rather than a purely conceptual explanation. This page covers
the ``get_workflow_guidance`` tool.

When to Use This vs. Domain Q&A
-----------------------------------

See :doc:`domain_qa`'s comparison table. In short: practical "how do I do/set up/prepare X"
questions and suitability-of-a-specific-item questions route here; "what is X" / "how do I
compute/derive X" questions route to :doc:`domain_qa`. Portal *actions* (upload, download, cite)
and portal *entity-schema* questions never route here — those go to :doc:`portal_docs` instead.

What It Does
------------

``get_workflow_guidance(goal)`` (``src/assistant/tools.py``) shares the same context-assembly
and synthesis machinery as :doc:`domain_qa`'s ``get_educational_context``, using the same
``src/prompts/educational.yaml`` prompt:

1. **Matched workflows** — up to 3 entries from ``data/domain_workflows.yaml`` most relevant to
   the goal (LLM-selected over the workflow index, keyword fallback on parse failure). Each
   workflow entry contributes its description, ordered method-level steps, common software
   (portal-featured tools listed first), best practices, and 2–4 example DRP dataset IDs that
   have been verified to exist.
2. **Matched tutorials** — portal Jupyter notebooks from ``data/tutorials.yaml`` whose keywords
   overlap the goal. Tutorials live in the `dpm_teach <https://github.com/digital-porous-media/dpm_teach>`_
   repo and are mirrored to JupyterHub Community Data; ``access_instructions`` in the same YAML
   file describes the navigation path shown to users.
3. **Literature fallback** — if no tutorial matched, a Semantic Scholar search result set is
   included as an alternative starting point (see :doc:`literature_search`).

The response then goes through two deterministic post-processing guards specific to this tool:

- ``_strip_fabricated_tutorial_reference`` — removes any ``**Goal:** ... **Notebook:**
  \`...ipynb\``` block the model invented that doesn't correspond to an actually-matched
  tutorial. Tutorial paths are treated with the same strictness as dataset DOIs — never
  fabricated.
- ``_ensure_all_tutorials_mentioned`` — if multiple tutorials matched but the model's answer
  only mentioned some of them, appends the missing ones explicitly, so a real matched tutorial
  is never silently dropped from the response.

Because the answer is already synthesized and cited, ``get_workflow_guidance`` is a
**self-contained tool** in the conversation manager (see :doc:`assistant`) — returned directly,
never re-synthesized by the outer agent.

Workflow Data Schema
----------------------

Each entry in ``data/domain_workflows.yaml`` follows a fixed schema (see the file's own header
comment for the authoritative version): ``id``, ``name``, ``keywords``, ``description``,
``prerequisites``, ``inputs``, ``outputs``, ``steps``, ``software``, ``example_datasets``,
``best_practices``, ``references``, and optional ``decision_points`` for branching workflows.
Workflows are written to be **method-focused, not tool-specific** — steps describe the method so
users can substitute equivalent software, and portal-featured tools are listed before
third-party commercial ones (Avizo, Dragonfly, GeoDict).

Example Queries
----------------

.. code-block:: text

   How do I compute absolute permeability from a segmented image?
   How do I set up a lattice Boltzmann simulation for this image?
   What's the best way to organize core flooding data?
   Does this image have enough resolution for a simulation?

("How should I cite a dataset?" is a portal *action*, not a workflow — that routes to
:doc:`portal_docs`.)

See Also
--------

- :doc:`domain_qa` — Conceptual/theoretical DRP questions (same underlying data)
- :doc:`portal_docs` — Portal actions and entity-schema questions (never this tool)
- :doc:`literature_search` — Used as a fallback when no tutorial matches
- :doc:`assistant` — Overview of how all capabilities fit together
- ``data/domain_workflows.yaml`` — The curated workflow data and its schema
- ``data/tutorials.yaml`` — Portal tutorial notebook mapping
