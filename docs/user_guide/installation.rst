Installation
=============

Prerequisites
-------------

**Requirements:**

   - **Python 3.9 or higher**
   - **An LLM API key OR local Ollama** (see :doc:`configuration`)
   - **OS**: Linux, macOS, or Windows (WSL2 strongly recommended for Windows users)

----

Run the App (Researchers)
--------------------------

Follow these steps to install Rocco and launch the Streamlit web interface.

Step 1: Download the Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the latest tagged release with a shallow clone:

.. code-block:: bash

   git clone --branch v1.0.0 --depth 1 https://github.com/digital-porous-media/dpm_rocco_curator.git
   cd dpm_rocco_curator

.. tip::

   Replace ``v1.0.0`` with a newer tag if one is available. See the
   `Releases page <https://github.com/digital-porous-media/dpm_rocco_curator/releases>`_ for the latest version.

.. tip::

   **Optional: Use a virtual environment** — It's good practice to isolate Python dependencies:

   .. code-block:: bash

      # venv
      python3 -m venv venv
      source venv/bin/activate

      # conda
      conda create -n rocco_env python=3.12
      conda activate rocco_env


Step 2: Install the Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install .


Step 3: Configure Your LLM Provider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy the environment template and fill in your credentials:

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your LLM provider credentials
   LLM_PROVIDER=
   LLM_API_KEY=
   LLM_MODEL=

See :doc:`configuration` for detailed provider setup.

Step 4: Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Verify imports
   python -c "import src; print('✓ Rocco imported successfully')"

   # Run the Streamlit app
   streamlit run rocco_ui.py

The app should open at ``http://localhost:8501``.

----

Optional: Enable the General Assistant
----------------------------------------

The Streamlit app always includes a **"General Assistant"** tab. Domain Q&A, workflow guidance,
portal documentation search, and literature search work out of the box. **Dataset discovery**
additionally requires the ``graph`` extra, which pulls in the Neo4j driver and
``langchain-neo4j``/``langchain-openai``:

.. code-block:: bash

   pip install -e ".[graph]"

If you skip this, or don't have a Neo4j instance available, set ``USE_NEO4J=false`` in ``.env``
— the assistant tab still loads and the rest of its capabilities work normally. See
:doc:`configuration` for Neo4j and Semantic Scholar setup, and :doc:`assistant` for a full
walkthrough.

----

Troubleshooting
---------------

**ImportError: No module named 'src'**
   Make sure you're in the repository root directory and ran ``pip install .``

**ModuleNotFoundError: No module named 'streamlit'**
   The installation may have failed. Try: ``pip install --upgrade pip`` then ``pip install .``

**streamlit: command not found**
   Ensure your virtual environment is activated and Streamlit was installed.

**FAISS library issues**
   Some systems require additional build tools. 
   Try upgrading FAISS: ``pip install --upgrade faiss-cpu``

----

Next Steps
----------

- Read the :doc:`quickstart_curator` guide for your first evaluation
- See :doc:`configuration` for all LLM provider options
- See :doc:`assistant` for the General Assistant's capabilities
