General Assistant
==================

Rocco's **General Assistant** is a conversational tab in the same Streamlit app as the
Description Curator. It helps you find datasets, understand digital rock physics concepts,
get step-by-step workflow guidance, and search the literature — all through natural language.

This page explains how a message actually flows through the assistant end to end. For how any
one capability works in depth (its prompt, its backing data, its edge cases), see the
capability pages linked from the table below.

Launching the App
------------------

.. code-block:: bash

   streamlit run rocco_ui.py

The app opens at ``http://localhost:8501``. Select **"General Assistant"** from the page
selector to switch to the assistant tab (the Description Curator lives on its own tab in the
same app — see :doc:`streamlit_app`).

.. note::

   Dataset discovery requires a running Neo4j instance. If ``USE_NEO4J=false`` (or Neo4j is
   unreachable), the assistant degrades gracefully: dataset search is disabled, but domain Q&A,
   workflow guidance, portal documentation search, and literature search all keep working. See
   :doc:`quickstart_assistant` for setup.

How a Message Is Handled
--------------------------

There is no hardcoded intent dispatcher. Every message passes through a small chain of cheap,
tools-unbound LLM "gate" calls before it ever reaches the tool-bound agent, which is itself a
LangGraph ReAct agent (``langgraph.prebuilt.create_react_agent``) that picks a tool by reading
its description against the system prompt — not a lookup table. This is implemented in
``src/assistant/conversation_manager.py``.

1. **Off-domain gate** — a dedicated classifier call decides if the message has *any* plausible
   connection to datasets, DRP science, the portal, or domain-related coding help. If not, a
   fixed steer-back message is returned immediately — no further LLM calls.
2. **Tool-need gate** — a second classifier decides whether the message needs a lookup at all.
   Greetings, small talk, brainstorming, and self-contained foundational-science questions
   ("What is porosity?") are answered directly, with no tools bound to that call (so the model
   can't hallucinate a tool call it doesn't have).
3. **ReAct agent** — for everything else, the agent is given all seven tools (see the table
   below) and the system prompt's routing rules, and picks one or more based on the query.
   Cross-intent queries ("explain relative permeability and find datasets that measure it"), and
   multi-dataset comparisons ("compare dataset A and dataset B"), trigger multiple sequential
   tool calls.
4. **Response assembly** — how the final answer is built depends on which tool(s) ran:

   - **Verbatim tools** (``search_datasets``, ``get_dataset_details``) return real dataset
     titles/DOIs/descriptions that must reach the user unmodified. The agent's own retelling of
     these is discarded; only a short LLM-generated lead-in sentence is added, then the tool's
     output is spliced in byte-for-byte, followed by a fixed verification disclaimer.
   - **Self-contained tools** (``get_workflow_guidance``, ``get_educational_context``,
     ``search_portal_docs``, ``get_dataset_profile``, ``reason_about_dataset_content``)
     already return a complete, synthesized,
     cited answer from their own internal LLM call. That answer is returned directly — it is
     never re-synthesized by the outer agent, which has no grounding in the underlying data.
   - **Cross-intent / multi-tool turns** are synthesized by the outer agent into one coherent
     response, with source labels preserved from each tool's raw output.
   - A **manual-dispatch fallback** exists for a known tool-call-format issue with one supported
     model (Llama-4-Maverick via SambaNova): if the backend rejects the model's native tool-call
     syntax, the intended call is parsed out of the error and dispatched directly, following the
     same verbatim/self-contained rules above.

What the Assistant Can Do
--------------------------

.. list-table::
   :widths: 22 30 24 24
   :header-rows: 1

   * - Capability
     - Example query
     - Primary tool
     - Details
   * - Dataset discovery
     - "Find sandstone datasets suitable for pore-scale simulation"
     - ``search_datasets``
     - :doc:`dataset_discovery`
   * - Structured dataset queries
     - "Which datasets have porosity above 0.3?"
     - ``get_dataset_details``
     - :doc:`structured_queries`
   * - Dataset detail follow-up / comparison
     - "Tell me more about this dataset" / "Compare A and B for two-phase flow simulation"
     - ``get_dataset_profile``
     - :doc:`dataset_profiles`
   * - Relationship / content questions
     - "Are there paired tomographic and segmented images?"
     - ``reason_about_dataset_content``
     - :doc:`content_reasoning`
   * - Portal how-to / schema
     - "How do I upload a dataset to the portal?"
     - ``search_portal_docs``
     - :doc:`portal_docs`
   * - Domain Q&A
     - "What is relative permeability?"
     - ``get_educational_context``
     - :doc:`domain_qa`
   * - Workflow guidance
     - "How do I compute absolute permeability from a segmented image?"
     - ``get_workflow_guidance``
     - :doc:`workflow_guidance`
   * - Literature search
     - "Find papers on micro-CT imaging of carbonate rocks"
     - ``search_literature``
     - :doc:`literature_search`

Reading Source Badges
----------------------

Every assistant response is annotated with colored badges showing where each piece of
information came from:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Badge
     - Meaning
   * - ``graph match``
     - Pure vector-similarity match against the Neo4j dataset embedding index
   * - ``hybrid match``
     - Combined vector + BM25 full-text match (Reciprocal Rank Fusion) — the default for
       ``search_datasets``
   * - ``component match``
     - Match against a dataset sub-node (e.g. a specific sample or scan), not the parent dataset
   * - ``cypher match``
     - Structured Cypher query result — exact property values, numeric comparisons, named authors
   * - ``dataset profile``
     - Deep-dive answer about one already-identified dataset, or one dataset within a
       multi-dataset comparison
   * - ``semantic scholar``
     - Result from the Semantic Scholar literature API
   * - ``portal docs``
     - Result from the DPM Portal's user documentation

Knowledge Source Policy
------------------------

The assistant follows a tiered policy for how strictly it relies on tool results versus its own
pre-trained knowledge:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Question type
     - Policy
   * - Dataset facts / portal content
     - **Tools only.** If nothing is found, the assistant says so rather than guessing —
       hallucinated dataset properties would erode trust in the catalog.
   * - Domain Q&A / workflows
     - **Tools first** (curated workflow library, portal tutorials, Semantic Scholar). Falls back
       to general knowledge with an explicit disclaimer, e.g. *"I don't have portal-specific data
       on this, but generally…"*
   * - Foundational concepts
     - Pre-trained knowledge is used directly — these are stable, well-established facts (e.g.
       "What is porosity?").

If the assistant says it doesn't have portal-specific data on something, that's this policy
working as intended, not a bug.

Session State
-------------

Like the curator tab, the assistant stores conversation history in Streamlit's
``st.session_state`` (keys prefixed ``assistant_``) — it is **not persisted to disk**. Refreshing
the browser or restarting the app clears the conversation.

Troubleshooting
----------------

**"Graph search is disabled (USE_NEO4J=false)"**
   Dataset discovery requires Neo4j. Either set up a Neo4j instance and set ``USE_NEO4J=true``
   with valid connection details, or continue using the assistant for domain Q&A, workflow
   guidance, and literature/portal doc search, which don't require it.

**"No datasets found matching that query"**
   Try broadening your search — remove specific filters, use more general terminology, or search
   by rock type or imaging method instead of a precise numeric threshold.

**"No papers found on Semantic Scholar for that query"**
   Try a broader or differently-phrased query. A ``SEMANTIC_SCHOLAR_API_KEY`` is optional but
   gives a higher rate limit — see :doc:`configuration`.

**"That's outside what I can help with..."**
   The off-domain gate decided your message has no plausible connection to datasets, DRP
   science, the portal, or domain-related coding help. Rephrase to connect it to one of those.

See Also
--------

- :doc:`quickstart_assistant` — Setup and example queries for first-time use
- :doc:`configuration` — Neo4j and Semantic Scholar environment variables
- :doc:`streamlit_app` — The Description Curator tab
- :doc:`dataset_discovery`, :doc:`structured_queries`, :doc:`dataset_profiles`,
  :doc:`content_reasoning`, :doc:`portal_docs`, :doc:`domain_qa`, :doc:`workflow_guidance`,
  :doc:`literature_search` —
  How each capability is implemented
- :doc:`../developer_guide/architecture` — Assistant module reference, including how to add
  datasets to the graph and sync portal documentation updates ("Maintenance" section)
- ``CLAUDE.md`` — Full assistant architecture and design constraints
