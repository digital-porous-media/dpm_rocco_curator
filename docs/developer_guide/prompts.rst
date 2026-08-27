Prompt Reference
================

Rocco's behavior is defined by versioned YAML prompt files.
This page documents what each prompt does, its template variables, and how to modify or create prompts.

Overview
--------

**How it works:**

1. Prompt files live in ``src/prompts/<name>.yaml``
2. Loaded at runtime with :func:`src.prompts.loader.load_prompt`
3. Template variables are rendered with Jinja2 via :func:`src.prompts.loader.render`

.. code-block:: python

   from src.prompts.loader import load_prompt, render

   prompt = load_prompt("evaluator")   # returns dict: version, description, user
   text   = render(prompt["user"], rubric=rubric_json, description=desc)

Curator prompts carry only a ``user`` field; every assistant prompt also carries a ``system``
field, and ``corpus_reasoning.yaml`` additionally carries a second ``batch_screen_system`` /
``batch_screen_user`` pair for its map-reduce fallback.

**The prompts at a glance:**

.. list-table::
   :widths: 26 12 20 42
   :header-rows: 1

   * - File
     - Version
     - Module
     - Role
   * - ``evaluator.yaml``
     - 1.0.0
     - Curator
     - Rubric scoring
   * - ``editor.yaml``
     - 1.1.0
     - Curator
     - Description enhancement + citations
   * - ``content_screener.yaml``
     - 1.0.0
     - Curator
     - User-feedback validation
   * - ``query_expander.yaml``
     - 0.3.0
     - Assistant
     - Query expansion + filter inference
   * - ``educational.yaml``
     - 0.1.10
     - Assistant
     - Domain Q&A + workflow synthesis
   * - ``dataset_profile.yaml``
     - 0.1.1
     - Assistant
     - Single-dataset deep-dive profile
   * - ``corpus_reasoning.yaml``
     - 1.0.0
     - Assistant
     - Relationship/content reasoning over fact sheets
   * - ``portal_docs.yaml``
     - 0.1.3
     - Assistant
     - Portal documentation answer synthesis
   * - ``assistant.yaml``
     - 0.2.1
     - Assistant
     - 6-intent classifier — **not called at runtime** (see below)

**Versioning** (``major.minor.patch`` in each YAML):

- ``major`` — breaking change to the output format (callers must be updated)
- ``minor`` — new template variable added
- ``patch`` — wording or clarity tweak, no structural change

Git history is the authoritative changelog for prompt changes.

----

Evaluator Prompt
----------------

**File:** ``src/prompts/evaluator.yaml`` · **Version:** 1.0.0

**Role:** Scores a dataset description against the 10-criterion rubric. Used by
:class:`src.evaluator.evaluator.DescriptionEvaluator`.

Template Variables
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Variable
     - Required
     - Description
   * - ``{{ rubric }}``
     - Yes
     - The rubric JSON serialised to a string (loaded from ``src/evaluator/rubric.json``)
   * - ``{{ examples }}``
     - Yes
     - Few-shot examples JSON serialised to a string (from ``src/evaluator/examples_v3.json``)
   * - ``{{ description }}``
     - Yes
     - The plain-text dataset description to evaluate

Output Format
~~~~~~~~~~~~~

The LLM returns a JSON object. The caller parses it into :class:`src.llm.schemas.EvaluatorOutput`.

.. code-block:: json

   {
     "rubric_breakdown": [
       {"criterion": "Self-Contained Description", "score": 1, "explanation": "..."},
       {"criterion": "Context of Creation",         "score": 0, "explanation": "..."}
     ]
   }

Full Prompt Text
~~~~~~~~~~~~~~~~

.. code-block:: text

   ## ROLE
   You are an expert data curator for the Digital Porous Media Portal.
   You are provided 10 guidelines, each of which is worth one point.
   Descriptions only get the point if the guideline is addressed explicitly.
   You are to evaluate the description for each guideline. Follow the examples provided.
   Only evaluate the 10 guidelines, do not try to sum everything at the end.
   Return your evaluation as a JSON object with the following format:
   {
     "rubric_breakdown": [
       {"criterion": "Self-Contained Description", "score": 1, "explanation": "..."},
       {"criterion": "Context of Creation", "score": 0.5, "explanation": "..."},
       ...
     ]
   }
   Do not provide any additional text outside the JSON.

   Rubric:
   {{ rubric }}

   Examples:
   {{ examples }}

   Now follows the description you must rate. Do not round.

   Description: {{ description }}

   Explanation:

----

Editor Prompt
-------------

**File:** ``src/prompts/editor.yaml`` · **Version:** 1.1.0

**Role:** Rewrites or refines a dataset description, integrating rubric feedback and RAG context
from uploaded research papers. Used by :class:`src.editor.editor.DescriptionEditor`.

The prompt operates in two modes controlled by the ``{{ mode }}`` variable:

- **new** — Start fresh: maximise rubric compliance from the original description + paper context.
- **refinement** — Iterative pass: integrate the user's latest feedback throughout the existing text.

Template Variables
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Variable
     - Required
     - Description
   * - ``{{ mode }}``
     - Yes
     - ``"new"`` or ``"refinement"``
   * - ``{{ rubric_str }}``
     - Yes
     - Rubric JSON serialised to string
   * - ``{{ original_description }}``
     - Yes
     - The unmodified description text provided by the researcher
   * - ``{{ evaluation_feedback }}``
     - Yes
     - Structured feedback from the Evaluator (per-criterion scores + explanations)
   * - ``{{ context_str }}``
     - No
     - Top-k RAG chunks from uploaded papers, each prefixed with ``Source: <doc>, Page <n>, Chunk <n>``
   * - ``{{ history }}``
     - No
     - Serialised conversation history for multi-turn refinement
   * - ``{{ user_feedback }}``
     - No
     - Free-text feedback entered by the user in the current turn

Critical Rules (enforced in prompt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Only use information **explicitly stated** in the original description or paper context
- No speculative language: *potentially, possibly, likely, may include, probably*, etc.
- If information is missing, omit it — do not acknowledge the gap
- Every new or more-specific statement **must** carry a citation

Output Format
~~~~~~~~~~~~~

.. code-block:: json

   {
     "updated_description": [
       {
         "updated_description": "Improved description text...",
         "rationale": "Brief summary of key changes",
         "citations": [
           {
             "statement": "Exact statement from the improved description",
             "source": "uploaded_document",
             "quote": "Exact supporting quote from source",
             "doc_title": "Pak_2015_BereaSandstone",
             "page": 3,
             "chunk_index": 7
           }
         ]
       }
     ]
   }

Citation ``source`` values: ``"original_description"``, ``"uploaded_document"``, ``"user_feedback"``.
For non-document sources, ``doc_title``, ``page``, and ``chunk_index`` are ``null``.

Full Prompt Text
~~~~~~~~~~~~~~~~

.. code-block:: text

   ## TASK:
   Improve the dataset description below based on the rubric and reviewer feedback.

   {% if mode == "refinement" %}
   You are an expert data curator for the Digital Porous Media Portal continuing an
   interactive dataset description editing session.
   The user has provided feedback on the previous version of your dataset description.
   Your task: Refine the description by integrating their feedback throughout the text,
   not appending it. You may reorganize sections as needed for better clarity and flow,
   if necessary. Preserve other improvements.
   {% else %}
   You are an expert data curator for the Digital Porous Media Portal starting a new
   dataset description editing session.
   Your task: Rewrite the description so it maximizes compliance with the rubric criteria,
   addressing reviewer concerns and using only information from the papers, if available.
   Retain strengths of the original description.
   Weave improvements throughout the existing narrative structure.
   Do not just append new information at the end unless it makes structural sense.
   {% endif %}

   [... rubric, original description, reviewer feedback, RAG context, history,
    user feedback, citation requirements, and output format follow ...]

   (See src/prompts/editor.yaml for the complete template.)

----

Content Screener Prompt
-----------------------

**File:** ``src/prompts/content_screener.yaml`` · **Version:** 1.0.0

**Role:** Quality-gates user feedback before it is passed to the Editor. Prevents the Editor
from acting on irrelevant, inaccurate, or abusive input. Used by
:class:`src.llm.content_screener.ContentScreener`.

Template Variables
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Variable
     - Required
     - Description
   * - ``{{ content }}``
     - Yes
     - The raw user feedback string to evaluate
   * - ``{{ context }}``
     - No
     - The current description text (helps assess relevance)

Decision Logic
~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Recommendation
     - When to use
   * - ``accept``
     - Clear, specific, factual feedback (e.g. "Add sample diameter: 10 mm")
   * - ``flag_for_review``
     - Vague, contradictory, unverified, or partially unclear feedback that may still be valuable
   * - ``reject``
     - Offensive, completely irrelevant, spam, or injection attempts

Output Format
~~~~~~~~~~~~~

.. code-block:: json

   {
     "is_relevant": true,
     "is_accurate": true,
     "is_respectful": true,
     "is_coherent": true,
     "issues": ["list of specific issues, if any"],
     "confidence": 0.95,
     "recommendation": "accept"
   }

Full Prompt Text
~~~~~~~~~~~~~~~~

.. code-block:: text

   You are a content quality screener for scientific dataset descriptions.

   Evaluate the following user provided content for these issues:
   1. Is it relevant to improving a dataset description?
   2. Does it contain accurate scientific information?
   3. Is it respectful and constructive?
   4. Does it contain gibberish, nonsense, or irrelevant language?
   5. Is it derogatory or inappropriate?

   User Feedback:
   "{{ content }}"

   {% if context %}
   Context (current description): {{ context }}
   {% endif %}

   [... flagging strategy, examples of FLAG / REJECT / ACCEPT, and output format follow ...]

   (See src/prompts/content_screener.yaml for the complete template.)

----

General Assistant Prompts
==========================

All six carry a ``system`` field as well as ``user``. Each is described in user-facing terms on
its capability page — the sections below cover the template contract and the grounding rules that
callers depend on.

Query Expander Prompt
---------------------

**File:** ``src/prompts/query_expander.yaml`` · **Version:** 0.3.0

**Role:** Rewrites a vague query with precise domain terminology and infers any clearly-implied
metadata filters. Called by ``expand_query()`` in ``src/assistant/tools.py``, which is invoked
internally by ``search_datasets`` — it is not a LangChain tool itself. See
:doc:`../user_guide/dataset_discovery`.

**Template variables:** ``{{ query }}`` (required) — the user's raw query.

Output Format
~~~~~~~~~~~~~

.. code-block:: json

   {
     "expanded_query": "sandstone porous media dataset quartz-rich clastic rock pore network",
     "inferred_filters": {"porousMediaType": "sandstone"},
     "rationale": "Expanded sandstone with related rock descriptors; no other filters clearly implied."
   }

``inferred_filters`` is restricted to exactly four fields — ``porousMediaType``, ``source``,
``segmented``, ``voxelDimensions`` — because those are the only ones projected onto a
vector/BM25 search hit. The prompt is explicit about not inventing fields outside that list;
anything else needs real Cypher (:doc:`../user_guide/structured_queries`). ``rationale`` is
paraphrased back to the user, never quoted verbatim.

----

Educational Prompt
------------------

**File:** ``src/prompts/educational.yaml`` · **Version:** 0.1.10

**Role:** Shared synthesis prompt for **both** ``get_educational_context`` (conceptual Q&A) and
``get_workflow_guidance`` (practical steps). See :doc:`../user_guide/domain_qa` and
:doc:`../user_guide/workflow_guidance`.

**Template variables:** ``{{ question }}`` and ``{{ context }}`` (both required) — ``context``
is the assembled block of matched workflows, global best practices, matched tutorials, and any
literature fallback.

Critical Rules (enforced in prompt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Implements the tiered knowledge-source policy: prefer the provided context; supplement with
  general domain expertise only when prefaced *"I don't have portal-specific data on this, but
  generally…"*; answer foundational concepts directly with no disclaimer
- **Never fabricate a tutorial notebook path** — treated with the same strictness as a dataset
  DOI. Backed in code by ``_strip_fabricated_tutorial_reference`` and
  ``_ensure_all_tutorials_mentioned``
- All math in LaTeX delimiters (``$...$`` inline, ``$$...$$`` block), never plain-text notation.
  ``data/domain_workflows.yaml`` stores equations in plain Unicode on purpose; the conversion to
  LaTeX happens here, at the prompt layer, and is rendered by ``st.markdown()``/KaTeX

----

Dataset Profile Prompt
----------------------

**File:** ``src/prompts/dataset_profile.yaml`` · **Version:** 0.1.1

**Role:** Synthesizes a deep-dive answer about one already-identified dataset. Called by
``get_dataset_profile``. See :doc:`../user_guide/dataset_profiles`.

**Template variables:** ``{{ question }}`` and ``{{ context }}`` (both required) — ``context``
is the rendered ``Dataset`` node plus its sub-nodes and ``INPUT_FOR`` pipeline structure, with
every empty/``None``/``[]`` field already dropped before rendering.

Critical Rules (enforced in prompt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Portal facts come from ``context`` only; file-format and reuse-suitability reasoning may draw
  on general knowledge but must say so explicitly
- A general "tell me more" gets a concise high-level overview, not a field-by-field dump; full
  detail is reserved for a question about a specific field
- Reasons internally but shows only the useful conclusion — no chain-of-thought scaffold in the
  output (stripped in code by ``_strip_reasoning_scaffold`` if it leaks)
- Absence of a recorded property is an honest gap, never evidence against suitability

The ``[dataset profile] <title> (DOI: ...)`` header is generated in code from graph records and
is never produced by this prompt.

----

Corpus Reasoning Prompt
-----------------------

**File:** ``src/prompts/corpus_reasoning.yaml`` · **Version:** 1.0.0

**Role:** Judges whether a described relationship plausibly holds for each of a shortlist of
datasets, reasoning over their precomputed fact sheets. Called by
``reason_about_dataset_content``. See :doc:`../user_guide/content_reasoning`.

**Template variables:** ``{{ question }}`` and ``{{ context }}`` (both required) — ``context``
is the shortlisted fact sheets.

Output Format
~~~~~~~~~~~~~

.. code-block:: json

   {
     "candidates": [
       {
         "title": "Bentheimer Sandstone for Analyzing Wetting Phenomena",
         "reason": "The dataset contains both original tomographic and segmented images.",
         "citation": "Original tomographic image — segmented: no; Segmented image — segmented: yes"
       }
     ],
     "caveat": "Optional: one sentence on what could not be determined."
   }

Grounding (enforced in code, not just prompt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two guards run after parsing, so a prompt violation degrades to a dropped candidate rather than
an ungrounded claim:

- **No citation, no candidate** — an entry without a supporting quote or recorded fact is dropped
- **No dataset the model wasn't shown** — an entry whose title isn't in the shortlist actually
  sent is dropped as a likely fabrication

Titles and DOIs in the rendered answer come from graph records, never retyped by the model. The
prompt also asks for short citations for a concrete reason: a long one crowds out later
candidates and can truncate the list.

Map-Reduce Fallback
~~~~~~~~~~~~~~~~~~~

This file carries a second prompt pair, ``batch_screen_system`` / ``batch_screen_user``, used
only for exhaustive questions that ranking cannot legitimately narrow ("list *every* dataset
where…"). The corpus is batched by character budget, each batch gets one cheap screening call
returning a JSON array of titles to keep, and survivors go through the full reasoning pass above.
The screen is deliberately generous — it excludes only clear non-matches, since the careful pass
still demands a citation.

----

Portal Docs Prompt
------------------

**File:** ``src/prompts/portal_docs.yaml`` · **Version:** 0.1.3

**Role:** Answers a portal how-to or data-model question from retrieved ``dpm_docs`` sections.
Called by ``search_portal_docs``. See :doc:`../user_guide/portal_docs`.

**Template variables:** ``{{ question }}`` and ``{{ context }}`` (both required) — ``context``
is the selected heading-tree sections' text, with a dataset-container disambiguation block
prepended when relevant.

Critical Rules (enforced in prompt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Answer only from the given excerpts, never from general knowledge of the portal
- **Never blend** ``Dataset``, ``DigitalDataset``, and ``AnalysisDataset`` definitions together,
  even though excerpts about all three share the word "Dataset"
- Cite sources in a ``Sources:`` line with each entry's doc URL
- Mention a screenshot only when an excerpt contains a literal ``[Figure: ...]`` placeholder —
  never invent one

----

Intent Classifier Prompt
------------------------

**File:** ``src/prompts/assistant.yaml`` · **Version:** 0.2.1

**Role:** Classifies a query into one of six intents (``semantic_search``, ``metadata_filter``,
``domain_qa``, ``workflow_guidance``, ``query_expansion``, ``literature_search``).

.. warning::

   **This prompt is not called at runtime.** ``ConversationManager`` has no intent-dispatch
   step — routing is implicit, done by the ReAct agent matching tool descriptions against the
   system prompt (see :doc:`architecture`'s Request Lifecycle). This file is kept for offline
   analysis and for ``tests/assistant/test_intent_classifier.py``. Editing it changes no runtime
   behavior; to change routing, edit ``SYSTEM_PROMPT`` in
   ``src/assistant/conversation_manager.py`` or the tool docstrings in ``src/assistant/tools.py``.

**Template variables:** ``{{ query }}`` (required).

----

Prompts That Live in Code, Not YAML
====================================

Not every LLM call in the assistant is backed by a YAML file. The gate and assembly prompts are
module constants in ``src/assistant/conversation_manager.py``, because they are tightly coupled
to control flow there rather than being independently versionable behavior:

- ``SYSTEM_PROMPT`` — the tiered knowledge-source policy and the tool-routing rules the ReAct
  agent follows. **This is where routing changes go.**
- ``_OFF_DOMAIN_GATE_SYSTEM_PROMPT`` / ``_GATE_SYSTEM_PROMPT`` /
  ``_FOLLOWUP_TOOL_GATE_SYSTEM_PROMPT`` — the three cheap, tools-unbound classifier calls
- ``_COMPARISON_SYNTHESIS_SYSTEM_PROMPT`` — multi-dataset comparison synthesis
- ``_WRAPPER_SYSTEM_PROMPT`` — the lead-in sentence for a verbatim splice

In ``src/assistant/graph_store.py``, for the same reason:

- ``MANUAL_SCHEMA`` — the hardcoded schema fed to ``GraphCypherQAChain`` (no live
  introspection, so no APOC dependency). It is also parsed by
  ``get_queryable_field_names()`` to build ``get_dataset_details``'s routing description, so a
  field added here is reflected in routing automatically.
- ``CYPHER_GENERATION_TEMPLATE`` — the Cypher-generation prompt. Forbids ``apoc.*`` calls, and
  carries the worked example for the ``IS NULL OR`` trap: with an ``OPTIONAL MATCH``, a clause
  like ``WHERE s IS NULL OR toLower(s.porousMediaType) = 'sandstone'`` is true for every dataset
  that has *no* ``Sample`` at all, so the filter looks present while admitting almost
  everything. When a question filters on a property, require the node with a plain ``MATCH``.
- ``QA_GENERATION_TEMPLATE`` — the chain's answer-synthesis prompt. Note that dataset *listings*
  bypass it: those bullets are built from the raw rows in Python, because this call could not be
  steered to reproduce titles/DOIs reliably.

----

Editing an Existing Prompt
--------------------------

1. **Open** the YAML file in ``src/prompts/``.

2. **Edit** the ``user`` field. It is a Jinja2 template — use ``{{ variable_name }}`` for
   injected values and ``{% if ... %}`` blocks for conditional sections.

3. **Bump the version** according to the semantic rules:

   - Wording/clarity change with no variable changes → increment **patch** (``1.0.0`` → ``1.0.1``)
   - New ``{{ variable }}`` added → increment **minor** (``1.0.0`` → ``1.1.0``); update the caller to pass the new variable
   - Output format change (different JSON keys, removed fields) → increment **major** (``1.0.0`` → ``2.0.0``); update the caller's parsing logic and output schema in ``src/llm/schemas.py``

4. **Update the caller** if you added or removed template variables. Callers live in:

   - ``src/evaluator/evaluator.py`` — calls ``render(..., rubric=..., examples=..., description=...)``
   - ``src/editor/editor.py`` — calls ``render(..., mode=..., rubric_str=..., ...)``
   - ``src/llm/content_screener.py`` — calls ``render(..., content=..., context=...)``
   - ``src/assistant/tools.py`` — every assistant prompt except ``portal_docs.yaml``
   - ``src/assistant/portal_docs_retrieval.py`` — ``portal_docs.yaml``

5. **Run tests** to verify nothing broke:

   .. code-block:: bash

      pytest tests/
      pytest tests/assistant/test_prompts.py -v   # every prompt YAML loads and renders

----

Creating a New Prompt
---------------------

1. **Create** ``src/prompts/<name>.yaml`` with the required fields:

   .. code-block:: yaml

      version: "1.0.0"
      description: "One-line description of what this prompt does"

      user: |
        You are a ...

        {{ my_variable }}

        Respond as JSON: { "result": "..." }

2. **Load and render** it in your component:

   .. code-block:: python

      from src.prompts.loader import load_prompt, render

      prompt = load_prompt("my_prompt")           # loads src/prompts/my_prompt.yaml
      text   = render(prompt["user"], my_variable="value")

3. **Pass the rendered text** to :class:`src.llm.client.RoccoClient`:

   .. code-block:: python

      from src.llm.client import RoccoClient

      client   = RoccoClient()
      response = client.send_prompt(prompt=text, context="You are a helpful assistant.")

4. **Define an output schema** in ``src/llm/schemas.py`` if the prompt returns structured JSON,
   then parse the LLM response into it.

5. **Add tests** under ``tests/`` that mock the LLM call and assert the parsed output schema.

----

See Also
--------

- :doc:`../user_guide/evaluator` — Rubric criteria and scoring details
- :doc:`../user_guide/writer` — How the Editor uses RAG context and citations
- :doc:`api_reference` — Auto-generated class documentation
- :doc:`architecture` — System data flow, including the assistant request lifecycle
- :doc:`../user_guide/dataset_discovery`, :doc:`../user_guide/dataset_profiles`,
  :doc:`../user_guide/content_reasoning`, :doc:`../user_guide/portal_docs`,
  :doc:`../user_guide/domain_qa`, :doc:`../user_guide/workflow_guidance` — What each assistant
  prompt does from the user's side
