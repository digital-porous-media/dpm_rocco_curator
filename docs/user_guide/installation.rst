Installation
=============

Prerequisites
-------------

**Required software:**
- **Python 3.9 or higher**
- **pip** and **git**
- **An LLM API key OR local Ollama** (see :doc:`configuration`)

**System requirements:**

- **RAM**: 4GB minimum (8GB recommended for document processing)
- **Disk**: 500MB for package + dependencies; additional space for vector stores and uploaded documents
- **OS**: Linux, macOS, or Windows (WSL2 strongly recommended for Windows users)

Step 1: Clone the Repository
-----------------------------

.. code-block:: bash

   git clone https://github.com/USER/dpm-rocco-curator.git
   cd dpm-rocco-curator

Step 2: Create a Virtual Environment
-------------------------------------

It's recommended to use a Python virtual environment to isolate dependencies:

.. code-block:: bash

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate

Step 3: Install the Package
---------------------------

Install Rocco in editable mode with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

If you plan to use documentation features:

.. code-block:: bash

   pip install -e ".[docs,dev]"

For Neo4j graph support (for future expansion):

.. code-block:: bash

   pip install -e ".[graph,dev]"

Step 4: Configure Your LLM Provider
------------------------------------

Copy the environment template and configure it:

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your LLM provider credentials

See :doc:`configuration` for detailed provider setup.

Step 5: Verify Installation
---------------------------

Test that everything is working:

.. code-block:: bash

   # Verify imports
   python -c "import src; print('✓ Rocco imported successfully')"

   # Run the Streamlit app
   streamlit run rocco_ui.py

The app should open at ``http://localhost:8501``.

Troubleshooting
---------------

**ImportError: No module named 'src'**
   Make sure you're in the repository root directory and ran ``pip install -e .``

**ModuleNotFoundError: No module named 'streamlit'**
   The installation may have failed. Try: ``pip install --upgrade pip`` then ``pip install -e "."``.

**streamlit: command not found**
   Ensure your virtual environment is activated and Streamlit was installed.

**FAISS library issues**
   Some systems require additional build tools. On macOS, try: ``pip install --upgrade faiss-cpu``. On Ubuntu, install: ``apt-get install python3-dev``.

Next Steps
----------

- Read the :doc:`quickstart` guide for your first evaluation
- See :doc:`configuration` for all LLM provider options
- Check :doc:`usage` for the full workflow
