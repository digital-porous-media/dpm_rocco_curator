Prompt Reference
================

Rocco's AI behavior is entirely defined by three versioned YAML prompt files in ``src/prompts/``.
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
             "source": "context_chunk",
             "quote": "Exact supporting quote from source",
             "doc_title": "Pak_2015_BereaSandstone",
             "page": 3,
             "chunk_index": 7
           }
         ]
       }
     ]
   }

Citation ``source`` values: ``"original_description"``, ``"context_chunk"``, ``"user_feedback"``.
For non-chunk sources, ``doc_title``, ``page``, and ``chunk_index`` are ``null``.

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

Editing an Existing Prompt
--------------------------

1. **Open** the YAML file in ``src/prompts/`` (``evaluator.yaml``, ``editor.yaml``, or ``content_screener.yaml``).

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

5. **Run tests** to verify nothing broke:

   .. code-block:: bash

      pytest tests/

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
      response = client.call(system="You are a helpful assistant.", user=text)

4. **Define an output schema** in ``src/llm/schemas.py`` if the prompt returns structured JSON,
   then parse the LLM response into it.

5. **Add tests** under ``tests/`` that mock the LLM call and assert the parsed output schema.

----

See Also
--------

- :doc:`../user_guide/evaluator` — Rubric criteria and scoring details
- :doc:`../user_guide/writer` — How the Editor uses RAG context and citations
- :doc:`api_reference` — Auto-generated class documentation
- :doc:`architecture` — System data flow
