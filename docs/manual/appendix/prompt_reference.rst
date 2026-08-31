Appendix C. Prompt and rubric reference
=======================================

Rocco keeps every language model prompt in a versioned YAML file under
``src/prompts/``. Editing these files changes model behavior without touching
Python code, which makes them the primary tuning surface.

This appendix reproduces all nine prompt files in full, along with the
evaluation rubric and the few-shot examples that calibrate the scorer. For an
explanation of what each prompt does and which variables it accepts, see the
prompt chapter in Part V.

Each file carries a semantic version. Increment the major version for a breaking
change to the output format, the minor version when you add a template variable,
and the patch version for wording changes.

Evaluation prompts
------------------

``src/prompts/evaluator.yaml``
   Scores a description against the ten-criterion rubric.

.. literalinclude:: /_prompts/evaluator.yaml
   :language: yaml

``src/prompts/content_screener.yaml``
   Validates user feedback for relevance, accuracy, and coherence before the
   editor acts on it.

.. literalinclude:: /_prompts/content_screener.yaml
   :language: yaml

Enhancement prompts
-------------------

``src/prompts/editor.yaml``
   Rewrites a description using retrieved context and requires a citation for
   every added statement.

.. literalinclude:: /_prompts/editor.yaml
   :language: yaml

Assistant prompts
-----------------

``src/prompts/query_expander.yaml``
   Expands a user query into search terms and inferred metadata filters.

.. literalinclude:: /_prompts/query_expander.yaml
   :language: yaml

``src/prompts/educational.yaml``
   Answers domain questions and synthesizes workflow guidance.

.. literalinclude:: /_prompts/educational.yaml
   :language: yaml

``src/prompts/corpus_reasoning.yaml``
   Reasons over ranked dataset fact sheets to answer relationship and content
   questions, with a mandatory citation for each candidate. Also carries the
   batch-screening prompt used by the exhaustive map-reduce fallback.

.. literalinclude:: /_prompts/corpus_reasoning.yaml
   :language: yaml

``src/prompts/dataset_profile.yaml``
   Synthesizes a single-dataset deep-dive profile.

.. literalinclude:: /_prompts/dataset_profile.yaml
   :language: yaml

``src/prompts/portal_docs.yaml``
   Answers portal how-to and data-model questions from the documentation tree.

.. literalinclude:: /_prompts/portal_docs.yaml
   :language: yaml

``src/prompts/assistant.yaml``
   A six-intent classifier. This prompt is **not called at runtime**; it is kept
   for offline evaluation and tests.

.. literalinclude:: /_prompts/assistant.yaml
   :language: yaml

Evaluation rubric
-----------------

``src/evaluator/rubric.json`` defines the ten criteria, each worth one point.
The rubric is domain-specific: it currently targets porous media datasets. To
adapt Rocco to another field, replace this file and the few-shot examples that
follow it.

.. literalinclude:: /_prompts/rubric.json
   :language: json

Few-shot examples
-----------------

``src/evaluator/examples_v3.json`` calibrates the scorer. These examples
directly shape output quality, so test any change to them against a known set of
descriptions.

.. literalinclude:: /_prompts/examples_v3.json
   :language: json
