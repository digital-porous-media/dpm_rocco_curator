Rocco: AI Curator for Dataset Descriptions
===========================================

Welcome to Rocco! An AI-powered description curator and evaluator for the Digital Porous Media (DPM) Portal.

Rocco helps researchers improve dataset descriptions using:
- **Rubric-based evaluation** (10 criteria, 0-10 scale)
- **RAG-powered enhancement** (semantic search over uploaded papers)
- **Interactive refinement** (user feedback with full citation tracking)
- **Multi-LLM support** (OpenAI, Anthropic, Ollama, DeepSeek, Gemini, and more)

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/configuration
   user_guide/usage

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/architecture
   developer_guide/contributing
   developer_guide/api_reference

Quick Links
-----------

- **GitHub Repository**: `dpm-rocco-curator <https://github.com/USER/dpm-rocco-curator>`_
- **Issue Tracker**: `GitHub Issues <https://github.com/USER/dpm-rocco-curator/issues>`_
- **License**: MIT

Getting Started
---------------

To get started with Rocco in 5 minutes:

1. **Install**: ``pip install -e .``
2. **Configure**: ``cp .env.example .env`` and set your LLM provider
3. **Run**: ``streamlit run rocco_ui.py``

For detailed instructions, see :doc:`user_guide/quickstart`.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
