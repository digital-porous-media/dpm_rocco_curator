Contributing
=============

Thank you for your interest in contributing to Rocco! We welcome bug reports, feature requests, and pull requests.

Getting Started
---------------

See the `CONTRIBUTING.md <https://github.com/digital-porous-media/dpm_rocco_curator/blob/main/CONTRIBUTING.md>`_ file for the full contribution guide, including:

- Setting up a development environment
- Code style and testing requirements
- PR submission workflow
- Adding support for new LLM providers

Quick Contribution Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork** the repo
2. **Branch** from main: ``git checkout -b feature/your-feature``
3. **Code** with style: ``black . && isort .``
4. **Test**: ``pytest tests/``
5. **Commit** with clear messages
6. **Push** and create a pull request

Development Setup
-----------------

Clone the full repository (not a shallow clone) so you have the complete git history:

.. code-block:: bash

   git clone https://github.com/digital-porous-media/dpm_rocco_curator.git
   cd dpm_rocco_curator
   pip install -e ".[dev]"

**Using Rocco as a library in another project** (without cloning):

.. code-block:: bash

   pip install git+https://github.com/digital-porous-media/dpm_rocco_curator.git@v1.0.0

Development Commands
--------------------

.. code-block:: bash

   # Install with dev dependencies (editable mode)
   pip install -e ".[dev]"

   # Format code
   black . --line-length 100
   isort .

   # Run tests
   pytest tests/
   pytest -v tests/test_file.py

   # Run the app locally
   streamlit run rocco_ui.py

Key Contribution Areas
----------------------

**Bug Reports**

File an issue on GitHub with:
- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- Full error traceback

**Feature Requests**

Describe:
- The problem you're solving
- Your proposed solution
- Alternative approaches considered
- Use cases

**Code Contributions**

Focus areas:

1. **LLM Provider Support** — Add a new provider to ``PROVIDER_URLS`` in ``src/llm/client.py`` and document it in `.env.example`
2. **Evaluation Rubric** — Enhance or refine criteria in ``src/evaluator/rubric.json`` and examples
3. **Prompts** — Improve or localize prompts in ``src/prompts/``
4. **UI/UX** — Enhance the Streamlit interface in ``rocco_ui.py``
5. **Documentation** — Improve guides in ``docs/``
6. **Tests** — Add unit or integration tests in ``tests/``

Code Quality Standards
----------------------

**Python Style**

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Use black and isort for formatting

**Docstrings**

- One-line for simple functions
- Multi-line for complex logic (Google style)
- Include examples for public APIs

.. code-block:: python

   def enhance_description(self, description: str, context: List[str]) -> str:
       """Enhance a description using RAG context.

       Args:
           description: The original description text
           context: List of context chunks from the vector store

       Returns:
           The enhanced description with citations
       """

**Comments**

- Explain **why**, not what (code should be self-documenting)
- Link to related issues or design docs
- Flag known limitations or TODOs

.. code-block:: python

   # Known limitation: FAISS similarity search is O(n) on CPU
   # Consider upgrading to GPU for large corpus (>100k chunks)
   results = self.vector_store.similarity_search(query)

**Tests**

- Aim for >80% coverage
- Test happy paths and edge cases
- Use descriptive test names

.. code-block:: python

   def test_evaluator_scores_complete_description():
       """Evaluator should score a complete description highly."""
       description = "Micro-CT images of sandstone at 2µm voxel resolution..."
       result = evaluator.evaluate(description)
       assert result.total_score >= 7

   def test_evaluator_scores_sparse_description_low():
       """Evaluator should score sparse descriptions low."""
       description = "Images"
       result = evaluator.evaluate(description)
       assert result.total_score <= 3

Review Process
--------------

PRs are reviewed for:

1. **Functionality** — Does it work as intended?
2. **Code quality** — Follows style, well-tested, well-documented
3. **Performance** — No regressions, efficient algorithms
4. **Security** — No credential leaks, safe API calls, input validation
5. **Compatibility** — Works across Python 3.9+, supported LLM providers

Reviewer feedback will be constructive and focused on improvement.

Documentation
~~~~~~~~~~~~~

Pull requests should include:
- Docstrings for new public functions/classes
- Updates to relevant docs (README, CLAUDE.md, user guides)
- Changelog entry (if applicable)

Testing
~~~~~~~

All pull requests must pass:
- Unit tests: ``pytest tests/``
- Code style: ``black . --line-length 100 && isort .``
- Type checking (if added: ``mypy src/``)

Release Process
---------------

Rocco uses semantic versioning (major.minor.patch):

- **Major** — Breaking API changes, major new features
- **Minor** — New features, backwards-compatible
- **Patch** — Bug fixes, documentation

To trigger a release:

1. Update ``version`` in ``pyproject.toml``
2. Update ``CHANGELOG.md`` (if present)
3. Create a GitHub Release with tag ``v{version}``
4. Zenodo will auto-publish and assign DOI

Getting Help
~~~~~~~~~~~~

- **Questions?** Open a GitHub Discussion or issue
- **Design advice?** Comment on an issue or draft PR
- **Stuck?** Ask in the issue thread — maintainers are here to help

Community Guidelines
~~~~~~~~~~~~~~~~~~~~

We're committed to a welcoming, inclusive environment:

- Be respectful and constructive
- Assume good intent
- Give credit and acknowledge contributions
- Report concerning behavior to maintainers

See also: `Code of Conduct <https://github.com/digital-porous-media/dpm_rocco_curator/blob/main/CONTRIBUTING.md#code-of-conduct>`_

Next Steps
----------

- Pick a `good first issue <https://github.com/digital-porous-media/dpm_rocco_curator/labels/good%20first%20issue>`_ to start
- Read :doc:`architecture` to understand the codebase
- Join discussions on `GitHub Issues <https://github.com/digital-porous-media/dpm_rocco_curator/issues>`_

Thanks for contributing to Rocco! 🙏
