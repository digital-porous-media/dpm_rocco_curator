Domain Q&A
============

Ask conceptual questions about porous media science, digital rock physics (DRP), and best
practices, and get an answer grounded in a curated workflow library — falling back to general
domain knowledge, with a disclaimer, when nothing specific is on file. This page covers the
``get_educational_context`` tool.

When to Use This vs. Workflow Guidance
------------------------------------------

Both tools share the same synthesis prompt (``src/prompts/educational.yaml``) and the same
backing data (``data/domain_workflows.yaml``), but answer different question shapes:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - This page (``get_educational_context``)
     - :doc:`workflow_guidance` (``get_workflow_guidance``)
   * - "What is relative permeability?"
     - "How do I compute relative permeability?"
   * - "How does capillary pressure relate to pore size?"
     - "How do I set up a lattice Boltzmann simulation for this image?"
   * - Theory, concepts, relationships, scientific terminology
     - Practical steps, tool usage, portal workflows, suitability of a specific item

The rule of thumb: a *"what is X"* / *"why does X happen"* / *"how does X relate to Y"* question
about theory or terminology comes here. Anything asking **how to actually do it** — including
*"how do I compute/derive/calculate X"* — goes to :doc:`workflow_guidance`, because a method
question is answered by the workflow steps for that method. Portal *actions* (upload, download,
cite) and portal *entity-schema* questions go to :doc:`portal_docs` instead, never here.

.. note::

   Earlier versions of this page split ``"how do I compute X"`` (method/math) from ``"how do I
   set up X"`` (practical steps), sending the first here. That rule came from
   ``src/prompts/assistant.yaml``, the standalone intent classifier that is **not called at
   runtime**. Live routing is governed by ``SYSTEM_PROMPT`` in ``conversation_manager.py`` and by
   each tool's own description, both of which route a scientific/analysis method to
   ``get_workflow_guidance``.

What It Does
------------

``get_educational_context(question)`` (``src/assistant/tools.py``) assembles context from three
sources, then makes one synthesis LLM call:

1. **Matched workflows** — up to 3 entries from ``data/domain_workflows.yaml`` selected by an
   LLM call over the workflow index (id + name + description), with a keyword-overlap fallback
   if that call fails to parse.
2. **Global best practices** — deterministic keyword-triggered sections from
   ``domain_workflows.yaml``'s ``global_best_practices`` block (representativeness, resolution,
   segmentation uncertainty, boundary conditions, reproducibility, connectivity), included when
   the question's wording matches their trigger terms.
3. **Matched tutorials** — portal tutorial notebooks from ``data/tutorials.yaml`` whose keywords
   overlap the question, included the same way as in :doc:`workflow_guidance`.
4. **Literature fallback** — if no tutorials matched, a Semantic Scholar search
   (:doc:`literature_search`) is run and its results included as additional context, so a
   question with no curated tutorial still gets a grounded starting point.

All of this is rendered into ``src/prompts/educational.yaml``'s ``{{ context }}`` template
variable, and the LLM is instructed to:

- Prefer the provided context; if it doesn't cover the topic, supplement with general domain
  expertise, prefaced with *"I don't have portal-specific data on this, but generally…"*
- Answer foundational concepts directly and completely, no disclaimer needed
- Never fabricate a tutorial notebook path — treat notebook paths with the same strictness as
  dataset titles/DOIs (a deterministic guard, ``_strip_fabricated_tutorial_reference``, also
  strips any notebook-shaped path the model invents that wasn't actually in the matched list)
- Use LaTeX delimiters for all math (``$...$`` inline, ``$$...$$`` block)

Because this answer is already synthesized and grounded, ``get_educational_context`` is a
**self-contained tool** in the conversation manager (see :doc:`assistant`) — returned directly,
never re-synthesized by the outer agent.

Example Queries
----------------

.. code-block:: text

   What is relative permeability?
   How does capillary pressure relate to pore size?
   What is interfacial tension and why does it matter?
   Explain pore-scale modeling
   What's the difference between porosity and permeability?

See Also
--------

- :doc:`workflow_guidance` — Practical, step-by-step DRP workflows (same underlying data)
- :doc:`literature_search` — Used as a fallback when no tutorial matches
- :doc:`assistant` — Overview of how all capabilities fit together
- ``src/prompts/educational.yaml`` — The synthesis prompt
- ``data/domain_workflows.yaml`` — The curated workflow/best-practices data
