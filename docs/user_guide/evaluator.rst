Evaluator
=========

The Evaluator scores a dataset description against a 10-criterion rubric. The rubric is currently designed for porous media research datasets, but is easily adaptable to other domains (see :doc:`/developer_guide/prompts`).
Each criterion is worth 1 point, giving a total of 10 possible points.

What It Does
------------

Given a plain-text description, the Evaluator:

1. Loads the rubric from ``src/evaluator/rubric.json`` and three few-shot examples from ``src/evaluator/examples_v3.json``
2. Builds a system prompt containing the rubric and the few-shot examples
3. Sends the description to the configured LLM and asks for a structured breakdown
4. Returns an ``EvaluatorOutput`` object containing the per-criterion scores and an overall total

The Evaluation Rubric
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Criterion
     - What it checks
   * - **Self-Contained Description**
     - Can the description be understood independently, without reading a related paper?
   * - **Context of Creation**
     - Does the description explain why the dataset was created (study goals, motivation)?
   * - **Porous Media Type**
     - Is the porous material clearly identified (rock type, cement, soil, etc.)?
   * - **Research Problem**
     - Does the description state the high-level research question the data addresses?
   * - **Reuse and Beneficiaries**
     - Does it explain how others could reuse the data and who would benefit (e.g., ML training, flow simulation, water resource management)?
   * - **Methodology**
     - Is the data collection method described (imaging technique, experimental setup, simulation approach)?
   * - **Contents and Organization**
     - Does it describe file types, folder structure, and what each file contains?
   * - **Quality Control**
     - Were QA/QC steps performed and documented (artifact correction, alignment, calibration)?
   * - **Clarity and Accessibility**
     - Is the language clear for both domain experts and general audiences? Are acronyms spelled out?
   * - **Keywords**
     - Are porous media type, imaging method, and research methodology mentioned in a way that aids search?

Score Interpretation
--------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Score
     - Interpretation
   * - **8–10**
     - Excellent. Clear, complete, ready for archival or citation.
   * - **6–7**
     - Good. Minor gaps; a single round of enhancement typically brings it to 8+.
   * - **4–5**
     - Fair. Significant gaps to address during the enhancement process.
   * - **0–3**
     - Poor. Description is too vague or incomplete.


How the Prompt Works
--------------------

The evaluator prompt is stored in ``src/prompts/evaluator.yaml`` and rendered with Jinja2. At a high level, it:

1. **Presents the rubric** — each criterion's name and description are serialised to JSON and injected into the system message, so the LLM understands exactly what to look for.
2. **Provides few-shot examples** — three (description, score, explanation) pairs from ``src/evaluator/examples_v3.json`` demonstrate the expected output format and calibrate the LLM's scoring style.
3. **Asks for structured output** — the LLM is instructed to return a JSON object with a ``rubric_breakdown`` list (one entry per criterion) and a ``total_score``.

The few-shot examples are crucial for consistency: without them the LLM's interpretation of marginal
cases (e.g., a criterion that is half-met) varies significantly across calls.

Running Without the UI
----------------------

.. code-block:: python

   import json
   from src.llm.client import RoccoClient
   from src.evaluator.evaluator import DescriptionEvaluator

   with open("src/evaluator/rubric.json") as f:
       rubric = json.load(f)
   with open("src/evaluator/examples_v3.json") as f:
       examples = json.load(f)

   client = RoccoClient()           # reads LLM_* from .env
   evaluator = DescriptionEvaluator(client, rubric, examples)

   description = "This dataset contains micro-CT images of Berea sandstone ... "
   result = evaluator.evaluate(description)

   print(f"Score: {result.total_score}/10")
   for item in result.rubric_breakdown:
       status = "✓" if item.score >= 1 else "✗"
       print(f"  {status} {item.criterion}: {item.explanation}")

You can also print a formatted summary:

.. code-block:: python

   evaluator.print_evaluation_result(result)

Output Schema
-------------

``evaluate()`` returns an ``EvaluatorOutput`` dataclass:

.. code-block:: python

   @dataclass
   class RubricItem:
       criterion: str    # criterion name
       score: float      # 0 or 1
       explanation: str  # why the criterion passed or failed

   @dataclass
   class EvaluatorOutput:
       total_score: float           # sum of all criterion scores
       rubric_breakdown: List[RubricItem]
       comments: Optional[str]      # overall LLM commentary, if any

Modifying the Rubric
--------------------

If you need to add, remove, or reword criteria:

1. Edit ``src/evaluator/rubric.json``
2. Update ``src/evaluator/examples_v3.json`` so the few-shot examples reflect the new criteria
3. Bump the version in ``src/prompts/evaluator.yaml`` (major version if the score scale changes)

See Also
--------

- :doc:`writer` — Using evaluation results to drive description enhancement
- :doc:`streamlit_app` — Running the Evaluator through the web UI
- :doc:`../developer_guide/api_reference` — Full class documentation
