Quick Start
===========

Get up and running with Rocco!

Step 1: Install & Configure
----------------------------

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

- **Gemini (free tier)**: Get a free key at https://studio.google.dev/gemini
- **Ollama (free, local)**: Follow the WSL2 setup instructions in the Configuration guide (no API key needed)
- **Anthropic, OpenAI, DeepSeek, etc.**: All supported — see Configuration for full list

Step 2: Start the App
---------------------

.. code-block:: bash

   streamlit run rocco_ui.py

Your browser will automatically open to ``http://localhost:8501``.

Step 3: Enter a Dataset Description
-------------------------------------

Paste any dataset description into the text area. Example:

.. code-block:: text

   This dataset contains high-resolution micro-CT images of sandstone samples.

Step 4: Evaluate
-----------------

Click **"Evaluate Description"** to score your description against **10 research-backed criteria**:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Criterion
     - What it checks
   * - **Self-Contained Description**
     - Does the description stand alone without external context?
   * - **Context of Creation**
     - Are the goals of the study clearly described?
   * - **Porous Media Type**
     - Is the type of porous media specified?
   * - **Research Question**
     - Does the description state the research question the data is helping to solve?
   * - **Reuse and Beneficiaries**
     - Does the description explain who would benefit from reusing this data and how?
   * - **Methodology**
     - Are the methods used to create the data described?
   * - **Contents and Organization**
     - Is there an overview of the dataset's contents and organization?
   * - **Quality Control**
     - Are quality assurance/quality control procedures documented?
   * - **Clarity and Accessibility**
     - Is the description clear and accessible to broad audiences?
   * - **Keywords**
     - Are relevant keywords included to aid discoverability?

**Rocco returns a score out of 10** — each criterion is worth 1 point.

Step 5: Enhance with RAG
-------------------------

Rocco suggests improvements to your description using **Retrieval-Augmented Generation (RAG)** and your feedback.

.. tabs::

   .. tab:: Without context documents

      Simply write feedback in the "Your Feedback" text area:

      .. code-block:: text

         Add information about sample preparation.
         Explain any image processing techniques used.
         Include any derived metrics (porosity, permeability, etc.).

      Then, click **"Enhance with Rocco"**

   .. tab:: With context documents

      Click **"Upload Files"** and add one or more PDF or DOCX file with relevant background. These could be:

      - Research papers and manuscripts
      - Technical protocols or standards
      - Dataset documentation
      - Related methodology papers

      Then, click **"Enhance with Rocco"**

Rocco will:

- Retrieve relevant excerpts from your uploaded documents
- Integrate them into the description with proper citations
- Address the feedback you provided
- Show you exactly where each claim came from (with quotes)

.. important:: Rocco will **NOT**:

   - Add information that isn't supported by your feedback or documents
   - Store or share any of your uploaded documents or feedback. Everything is processed in-memory and discarded after the session.


Step 6: Review & Iterate
-------------------------

When Rocco displays the suggested enhancements, you can:

- **✓ Accept** the improvement and save the new version
- **✗ Reject** and try again with different feedback or documents
- **✏️ Edit** manually and refine further
- **↻ Iterate** — try multiple rounds of feedback to perfect your description

Rocco shows citations for **all** changes and updates. Citations will show:

- The exact statement added
- Where it came from (your feedback, an uploaded paper, or the original description)
- The source document and page number (if applicable)
- The exact quote

**That's it!** You've just improved a dataset description using AI and research documents.


Tips & Best Practices
----------------------

**Quality Input and Feedback**
   Clear, complete descriptions lead to better evaluations. Include as much detail as possible in your feedback, Rocco will summarize and organize it.

**Upload Context Documents**
   Rocco works best with background materials. Upload:

   - Method papers (how the imaging/analysis was done)
   - Technical documentation (instrument specs, protocols)
   - Related datasets (to cite similar work)

**Iterate Strategically**
   After the first enhancement:

   1. Review the citations for accuracy
   2. Provide targeted feedback on what's still missing
   3. Upload more focused documents if needed
   4. Enhance again

**Try Different Models**
   Different LLM models have different strengths. Try changing the model in your ``.env`` file.



What's Next?
-------------

- **Full configuration guide**: :doc:`configuration` — all LLM providers, models, and options
- **Under the hood**: :doc:`../developer_guide/architecture` — how Rocco works, extending it
- **Contributing**: :doc:`../developer_guide/contributing` — report issues, contribute improvements
