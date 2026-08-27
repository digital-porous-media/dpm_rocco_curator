Contributing
=============

Thank you for your interest in contributing to Rocco! The canonical contribution guide —
development environment setup (including Neo4j for assistant work), branch naming, code style
rules, the test suite and its ``live`` marker, commit message format, PR process, and the
assistant-specific constraints reviewers check for — lives in
`CONTRIBUTING.md <https://github.com/digital-porous-media/dpm_rocco_curator/blob/main/CONTRIBUTING.md>`_
at the repository root. Read it before opening a PR; this page only covers what that file doesn't.

Quick Start
-----------

.. code-block:: bash

   git clone https://github.com/digital-porous-media/dpm_rocco_curator.git
   cd dpm_rocco_curator
   pip install -e ".[dev]"            # curator only
   pip install -e ".[dev,graph]"      # add this if your work touches src/assistant/

See CONTRIBUTING.md's "Setting Up Your Development Environment" section for the Neo4j setup that
assistant work also needs.

**Using Rocco as a library** (without cloning):

.. code-block:: bash

   pip install git+https://github.com/digital-porous-media/dpm_rocco_curator.git@v1.0.0

Release Process
----------------

Rocco uses semantic versioning (major.minor.patch — breaking changes / new backwards-compatible
features / fixes and docs). To cut a release: bump ``version`` in ``pyproject.toml``, update
``CHANGELOG.md``, and create a GitHub Release tagged ``v{version}`` — Zenodo auto-publishes and
assigns a version DOI (see the DOI badges in `README.md`).

See Also
--------

- `CONTRIBUTING.md <https://github.com/digital-porous-media/dpm_rocco_curator/blob/main/CONTRIBUTING.md>`_ — the full contribution guide
- :doc:`architecture` — codebase structure and extension points
- `Good first issues <https://github.com/digital-porous-media/dpm_rocco_curator/labels/good%20first%20issue>`_
