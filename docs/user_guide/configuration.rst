Configuration
==============

Rocco supports multiple LLM providers. Configure your preferred provider in the ``.env`` file.

Setting Up .env
---------------

Copy the template:

.. code-block:: bash

   cp .env.example .env

Then edit ``.env`` with your chosen provider.

Supported LLM Providers
-----------------------

OpenAI (Default)
~~~~~~~~~~~~~~~~

.. code-block:: ini

   LLM_PROVIDER=openai
   LLM_API_KEY=sk-proj-your-key-here
   LLM_MODEL=gpt-4o-mini

Get an API key: https://platform.openai.com/api-keys

**Model options:**

Check https://developers.openai.com/api/docs/models for the latest available models and pricing.

SambaNova
~~~~~~~~~~~~~~~~

.. code-block:: ini

   LLM_PROVIDER=sambanova
   LLM_API_KEY=sk-your-key-here
   LLM_MODEL=Llama-4-Maverick-17B-128E-Instruct

**SambaNova at TACC endpoint:** https://ai.tejas.tacc.utexas.edu/v1

Anthropic (Claude)
~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   LLM_PROVIDER=anthropic
   LLM_API_KEY=sk-ant-your-key-here
   LLM_MODEL=claude-opus-4-7

Get an API key: https://console.anthropic.com/

**Model options:**
Check https://platform.claude.com/docs/en/about-claude/models/overview for the latest available models and pricing.

Google Gemini
~~~~~~~~~~~~~~

.. code-block:: ini

   LLM_PROVIDER=gemini
   LLM_API_KEY=AIza-your-key-here
   LLM_MODEL=gemini-2.0-flash

Get an API key: https://aistudio.google.com/app/apikey

**Model options:**
Check https://ai.google.dev/gemini-api/docs/models for the latest available models and pricing.


DeepSeek
~~~~~~~~~

.. code-block:: ini

   LLM_PROVIDER=deepseek
   LLM_API_KEY=sk-your-key-here
   LLM_MODEL=deepseek-chat

Get an API key: https://platform.deepseek.com/

**Model options:**
Check https://api-docs.deepseek.com/quick_start/pricing for the latest available models and pricing.


HuggingFace (Serverless Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   LLM_PROVIDER=huggingface
   LLM_API_KEY=hf_your-token-here
   LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

Get an API key: https://huggingface.co/settings/tokens

**Model options:** Use only an *instruction-tuned* model ID from the
`HuggingFace Hub <https://huggingface.co/models?sort=trending&search=instruction>`_.

.. warning:: **Important caveats:**

   - **Only instruction-tuned models work** (not base/pretrained models)
   - **Free tier is heavily rate-limited** — HF Pro subscription or Inference Endpoints recommended
   - **Not all models support OpenAI-compatible API** — check model docs for compatibility

Ollama (Local)
~~~~~~~~~~~~~~

Ollama runs locally on your machine.

**Installing Ollama**
If you haven't installed Ollama yet, you can do so from the command line.

For more information, see the `Ollama installation guide <https://ollama.com/download>`_.

.. note::
   We **recommend installing Ollama on WSL2 (Windows Subsystem for Linux)** if you're on Windows, rather than the Windows Desktop app, as it provides better integration with development tools.

   If you don't have WSL2 installed, see `Windows Subsystem for Linux Installation <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

1. Install Ollama from the command line:

   .. code-block:: bash

      curl -fsSL https://ollama.com/install.sh | sh

2. Start Ollama:

   .. code-block:: bash

      ollama serve

3. In another terminal, pull a model (e.g., llama2):

   .. code-block:: bash

      ollama pull llama2

4. Verify Ollama is running:

   .. code-block:: bash

      curl http://localhost:11434/api/tags


**Configure Rocco for Ollama**

Edit your ``.env`` file:

.. code-block:: ini

   LLM_PROVIDER=ollama
   LLM_BASE_URL=http://localhost:11434/v1
   # LLM_API_KEY is auto-set to "ollama" (no real key needed)
   LLM_MODEL=llama2

**Available models:** ``llama2``, ``llama3``, ``mistral``, ``phi3``, ``neural-chat``, etc. (any model you've pulled)



Custom OpenAI-Compatible Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For any OpenAI-compatible API — custom servers, proxies, or open-source models deployed with OpenAI compatibility:

.. code-block:: ini

   LLM_PROVIDER=openai_compatible
   LLM_API_KEY=your-key-here
   LLM_BASE_URL=https://your-custom-endpoint.com/v1
   LLM_MODEL=your-model-name

**What "OpenAI-compatible" means:**
The API must expose a ``/v1/chat/completions`` endpoint with the same request/response format as OpenAI. Examples:

- Text Generation Inference (TGI) with ``--openai-compatible-api`` flag
- vLLM with ``--served-model-name``
- Any proxy or router that wraps an LLM with OpenAI API compatibility

**If your endpoint is NOT OpenAI-compatible** (e.g., native HuggingFace API, custom format), you'll need to:

1. Wrap your API with an OpenAI-compatible adapter, **OR**
2. Fork Rocco and modify ``src/llm/client.py`` to support your specific API format


Environment Variable Reference
-------------------------------

.. list-table::
   :widths: 25 12 63
   :header-rows: 1

   * - Variable
     - Required?
     - Description
   * - ``LLM_PROVIDER``
     - No
     - Shortcut to pre-fill ``LLM_BASE_URL``. Options: ``openai``, ``anthropic``, ``gemini``,
       ``deepseek``, ``huggingface``, ``ollama``, ``sambanova``, ``openai_compatible`` (custom).
   * - ``LLM_API_KEY``
     - **Yes**
     - Your API key. For Ollama or local services, can be a dummy value (e.g., ``ollama`` or
       ``unused``).
   * - ``LLM_BASE_URL``
     - No
     - Custom endpoint URL. Required for custom or proprietary providers. If not set, provider
       mapping is used (or defaults to OpenAI).
   * - ``LLM_MODEL``
     - No
     - Model name (e.g., ``gpt-4o-mini``). Defaults to ``gpt-4o-mini``.
   * - ``LLM_TIMEOUT``
     - No
     - Request timeout in seconds. Defaults to ``120``.
   * - ``USE_NEO4J``
     - No
     - General Assistant only. Set to ``false`` to disable Neo4j-backed dataset search; the
       assistant still answers domain Q&A, workflow, literature, and portal doc questions.
       Defaults to ``true``.
   * - ``NEO4J_URI``
     - No\*
     - Neo4j connection URI, e.g. ``bolt://localhost:7687`` (local) or
       ``neo4j+s://<id>.databases.neo4j.io`` (AuraDB). \*Required if ``USE_NEO4J=true``.
   * - ``NEO4J_USER``
     - No\*
     - Neo4j username (typically ``neo4j``). \*Required if ``USE_NEO4J=true``.
   * - ``NEO4J_PASSWORD``
     - No\*
     - Neo4j password. \*Required if ``USE_NEO4J=true``.
   * - ``EMBEDDING_URL`` / ``EMBEDDING_MODEL`` / ``EMBEDDING_API_KEY``
     - No
     - Override the embedding model/endpoint used for RAG and dataset-graph search. Auto-selected
       from ``LLM_PROVIDER`` when unset — see "Embedding Model Overrides" below.
   * - ``SEMANTIC_SCHOLAR_API_KEY``
     - No
     - General Assistant only. Optional API key for the Semantic Scholar literature search tool
       — unauthenticated requests work but with a lower rate limit.

Embedding Model Overrides
--------------------------

Rocco auto-selects an embedding model/endpoint based on ``LLM_PROVIDER`` if
``EMBEDDING_URL``/``EMBEDDING_MODEL``/``EMBEDDING_API_KEY`` are left unset:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - ``LLM_PROVIDER``
     - Default embedding model
   * - ``openai``
     - ``text-embedding-3-small`` (API)
   * - ``sambanova``
     - ``E5-Mistral-7B-Instruct`` (API, TACC endpoint, 4096-dim)
   * - ``gemini``
     - ``gemini-embedding-2`` (API)
   * - ``ollama`` / ``huggingface`` / ``anthropic`` / ``deepseek`` / other
     - ``BAAI/bge-large-en-v1.5`` (local — these providers have no embedding API)

Set any of ``EMBEDDING_URL``, ``EMBEDDING_MODEL``, ``EMBEDDING_API_KEY`` to override — e.g. to
use OpenAI embeddings alongside an Anthropic chat model. Local embedding models require the
``local-embeddings`` extra (``pip install -e ".[local-embeddings]"``, pulls in ``torch``).

Changing Providers
------------------

To switch providers at runtime, just edit ``.env`` and restart the Streamlit app.

General Assistant: Neo4j and Semantic Scholar
-----------------------------------------------

The General Assistant tab uses the same ``.env`` file. Two additional integrations are
configured here — see :doc:`assistant` for what each one enables.

Neo4j (Dataset Discovery)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   USE_NEO4J=true
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-neo4j-password-here

Set ``USE_NEO4J=false`` to disable this — the assistant falls back gracefully and keeps
domain Q&A, workflow guidance, literature search, and portal documentation search working.
Requires the ``graph`` extra (see :doc:`installation`).

Semantic Scholar (Literature Search)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   SEMANTIC_SCHOLAR_API_KEY=

Optional — unauthenticated requests work without a key, but a key gives a dedicated rate limit.
Get one at https://www.semanticscholar.org/product/api.


Troubleshooting
---------------

**"API Key not found"**
   Check that ``.env`` exists in the repo root and ``LLM_API_KEY`` is set.

**"Invalid API key"**
   Verify your key is correct (no extra spaces, matches the provider format).

**"Connection timeout"**
   For Ollama, ensure the server is running (``ollama serve``). For cloud providers, check your internet connection.

**"Model not found"**
   Verify the model name is correct for the provider (e.g., ``gpt-4o-mini`` for OpenAI, not ``gpt-4``).

Next Steps
==========

- Ready to use Rocco? See :doc:`quickstart_curator` or :doc:`quickstart_assistant`
- Want to try the General Assistant? See :doc:`assistant`
- Want to understand the architecture? See :doc:`../developer_guide/architecture`
- Need help? Check :doc:`../developer_guide/contributing`
