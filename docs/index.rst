Rocco — AI Curator for Dataset Descriptions
============================================

.. raw:: html

   <p class="rocco-tagline">
     AI-powered dataset description evaluation &amp; enhancement framework for the
     <strong>Digital Porous Media Portal</strong>.
   </p>

.. grid:: 3
   :gutter: 3
   :class-container: rocco-hero-grid

   .. grid-item-card:: :octicon:`book;1em` Rubric-Based Evaluation
      :link: user_guide/evaluator
      :link-type: doc
      :text-align: center

      Score any dataset description against **10 domain-specific criteria**.

   .. grid-item-card:: :octicon:`search;1em` RAG-Powered Enhancement
      :link: user_guide/writer
      :link-type: doc
      :text-align: center

      Improve descriptions with relevant excerpts drawn from your papers.

   .. grid-item-card:: :octicon:`sync;1em` Iterate
      :link: user_guide/streamlit_app
      :link-type: doc
      :text-align: center

      Refine with **interactive feedback** across multiple rounds until your description is publication-ready.

----

Getting Started Locally
-----------------------

.. tab-set::

   .. tab-item:: Quick Install

      .. code-block:: bash

         git clone --branch v1.0.0 --depth 1 https://github.com/digital-porous-media/dpm_rocco_curator.git
         cd dpm_rocco_curator
         pip install .

   .. tab-item:: Configure LLM Endpoints

      .. code-block:: bash

         cp .env.example .env

         # Edit .env to set LLM_PROVIDER, LLM_API_KEY, LLM_MODEL.
         # Rocco supports OpenAI-compatible APIs (OpenAI, Anthropic, Gemini, HuggingFace, Ollama, and more!)


   .. tab-item:: Run Rocco UI

      .. code-block:: bash

         streamlit run rocco_ui.py

         # Opens at http://localhost:8501

For a full walkthrough, see :doc:`user_guide/quickstart`.

----

.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   user_guide/installation
   user_guide/quickstart

.. toctree::
   :caption: Description Curator
   :maxdepth: 1

   user_guide/streamlit_app
   user_guide/evaluator
   user_guide/rag
   user_guide/writer

.. toctree::
   :caption: AI Providers
   :maxdepth: 1

   user_guide/configuration

.. toctree::
   :caption: Developer Guide
   :maxdepth: 1

   developer_guide/architecture
   developer_guide/contributing
   developer_guide/api_reference
   developer_guide/prompts
