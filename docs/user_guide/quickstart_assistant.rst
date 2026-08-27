Quick Start: General Assistant
==================================

Get up and running with the General Assistant!

.. note::

   Looking for the Description Curator instead? See :doc:`quickstart_curator`.

The General Assistant lives in the same Streamlit app, on a separate tab. It doesn't require
uploading a description — you just ask it questions.

Step 1: Install & Configure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Clone and install the repository:**

.. code-block:: bash

   git clone https://github.com/digital-porous-media/dpm_rocco_curator.git
   cd dpm_rocco_curator
   pip install .

**Set up your LLM provider:**

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your chosen provider
   # See Configuration guide for detailed provider setup: docs/user_guide/configuration

**Quick provider choices:**

- **Gemini (free tier)**: Get a free key at https://aistudio.google.com/app/apikey
- **Ollama (free, local)**: Follow the WSL2 setup instructions in the Configuration guide (no API key needed)
- **Anthropic, OpenAI, DeepSeek, etc.**: All supported — see Configuration for full list

Step 2: (Optional) Configure Neo4j and Semantic Scholar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset discovery is backed by `Neo4j <https://neo4j.com/>`_. If you don't have a Neo4j
instance available, set ``USE_NEO4J=false`` in ``.env`` — the assistant will still answer
domain Q&A, workflow guidance, literature, and portal documentation questions.

If you do want dataset discovery, install the ``graph`` extra first (it pulls in the Neo4j
driver and ``langchain-neo4j``; Step 1's plain install doesn't include it):

.. code-block:: bash

   pip install -e ".[graph]"

To set up Neo4j:

- **Local**: install `Neo4j Desktop <https://neo4j.com/download/>`_ or run the
  `official Docker image <https://hub.docker.com/_/neo4j>`_.
- **Hosted (free tier)**: create a database on `Neo4j AuraDB <https://neo4j.com/cloud/aura-free/>`_
  — no local install needed.

.. code-block:: ini

   # In .env
   USE_NEO4J=true
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-neo4j-password-here

   # Optional — raises the Semantic Scholar rate limit; unauthenticated requests work too.
   # Get a key at https://www.semanticscholar.org/product/api
   SEMANTIC_SCHOLAR_API_KEY=

See :doc:`configuration` for full details.

Step 3: Start the App
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   streamlit run rocco_ui.py

Select **"General Assistant"** from the page selector.

Step 4: Ask a Question
~~~~~~~~~~~~~~~~~~~~~~~~

Try one example from each capability:

.. code-block:: text

   Find sandstone datasets suitable for pore-scale simulation
   What is relative permeability?
   How do I compute absolute permeability from a segmented image?
   Find papers on micro-CT imaging of carbonate rocks
   How do I upload a dataset to the portal?

Each response is tagged with a colored source badge (e.g. ``[graph match]``,
``[cypher match]``, ``[semantic scholar]``, ``[portal docs]``) showing where the information
came from. See :doc:`assistant` for the full guide, including the badge reference and the
assistant's tiered knowledge-source policy.

What's Next?
-------------

- **Full configuration guide**: :doc:`configuration` — all LLM providers, models, and options
- **General Assistant guide**: :doc:`assistant` — how the pieces fit together, badge reference
- **Capability pages** — how each capability is implemented:

  - :doc:`dataset_discovery` — semantic dataset search
  - :doc:`structured_queries` — exact/numeric dataset property queries
  - :doc:`dataset_profiles` — follow-up detail questions and dataset comparisons
  - :doc:`content_reasoning` — relationship questions no single metadata field can answer
  - :doc:`multi_turn` — narrowing a result set and referring back to earlier results
  - :doc:`portal_docs` — portal how-to and schema questions
  - :doc:`domain_qa` — porous media science Q&A
  - :doc:`workflow_guidance` — step-by-step DRP workflows
  - :doc:`literature_search` — Semantic Scholar search

- **Under the hood**: :doc:`../developer_guide/architecture` — how Rocco works, extending it
- **Description Curator quick start**: :doc:`quickstart_curator`
- **Contributing**: :doc:`../developer_guide/contributing` — report issues, contribute improvements
