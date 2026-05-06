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

**Best for:** General-purpose, reliable, good performance.

.. code-block:: ini

   LLM_PROVIDER=openai
   LLM_API_KEY=sk-proj-your-key-here
   LLM_MODEL=gpt-4o-mini

Get an API key: https://platform.openai.com/api-keys

**Model options:**
- ``gpt-4o`` (most capable)
- ``gpt-4o-mini`` (fast, cost-effective) — recommended for most use cases
- ``gpt-4`` (older, still capable)

Anthropic (Claude)
~~~~~~~~~~~~~~~~~~~

**Best for:** Extended reasoning, good at nuanced analysis.

.. code-block:: ini

   LLM_PROVIDER=anthropic
   LLM_API_KEY=sk-ant-your-key-here
   LLM_MODEL=claude-opus-4-7

Get an API key: https://console.anthropic.com/

**Model options:**
- ``claude-opus-4-7`` (most capable)
- ``claude-sonnet-4-6`` (balanced)
- ``claude-haiku-4`` (fast, cost-effective)

Google Gemini
~~~~~~~~~~~~~~

**Best for:** Competitive pricing, fast inference.

.. code-block:: ini

   LLM_PROVIDER=gemini
   LLM_API_KEY=AIza-your-key-here
   LLM_MODEL=gemini-2.0-flash

Get an API key: https://aistudio.google.com/app/apikey

**Model options:**
- ``gemini-2.0-flash`` (latest, fastest)
- ``gemini-1.5-pro`` (most capable)

DeepSeek
~~~~~~~~~

**Best for:** Cost-effective, good reasoning.

.. code-block:: ini

   LLM_PROVIDER=deepseek
   LLM_API_KEY=sk-your-key-here
   LLM_MODEL=deepseek-chat

Get an API key: https://platform.deepseek.com/

**Model options:**
- ``deepseek-chat`` (standard)
- ``deepseek-reasoner`` (extended reasoning)

HuggingFace (Serverless Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for:** Easy access to open-source instruction-tuned models without infrastructure setup.

.. code-block:: ini

   LLM_PROVIDER=huggingface
   LLM_API_KEY=hf_your-token-here
   LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

Get an API key: https://huggingface.co/settings/tokens

**Model options:** Any instruction-tuned model ID from HuggingFace Hub, e.g.:
- ``meta-llama/Llama-3.1-8B-Instruct``
- ``meta-llama/Llama-3.1-70B-Instruct``
- ``mistralai/Mistral-7B-Instruct-v0.2``
- ``mistralai/Mixtral-8x7B-Instruct-v0.1``

**Important caveats:**
- **Only instruction-tuned models work** (not base/pretrained models)
- **Free tier is heavily rate-limited** — HF Pro subscription or Inference Endpoints recommended for production use
- **Gated models** (e.g., Llama models) require you to: (1) accept the model's license on HF Hub, (2) use a token with model access
- **Not all OpenAI features supported** — advanced fields like ``tool_calls``, ``logprobs``, and structured outputs may not work depending on model/TGI version

HuggingFace Text Generation Inference (TGI) — Self-Hosted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for:** Custom fine-tuned models or if you need full control over compute resources.

Deploy Text Generation Inference locally or on your own infrastructure with the OpenAI-compatible API:

.. code-block:: bash

   docker run -p 8080:80 \
     --shm-size 1g \
     -e MODEL_ID=meta-llama/Llama-2-7b-chat \
     ghcr.io/huggingface/text-generation-inference:latest \
     --openai-compatible-api

Then configure Rocco:

.. code-block:: ini

   LLM_PROVIDER=openai_compatible
   LLM_API_KEY=unused-for-local-tgi
   LLM_BASE_URL=http://localhost:8080/v1
   LLM_MODEL=meta-llama/Llama-2-7b-chat

**Model options:** Any model ID from HuggingFace Hub (local or via Inference Endpoints)

Ollama (Local)
~~~~~~~~~~~~~~

**Best for:** Privacy, no API costs, complete control, reproducible local development.

Ollama runs locally on your machine. We **recommend installing Ollama on WSL2 (Windows Subsystem for Linux)** if you're on Windows, rather than the Windows Desktop app, as it provides better integration with development tools.

**Setup on Windows (WSL2)**

If you don't have WSL2 installed, see `Windows Subsystem for Linux Installation <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

1. Install Ollama in WSL2:

   .. code-block:: bash

      # Install Ollama in WSL2
      curl -fsSL https://ollama.ai/install.sh | sh

2. Start Ollama in a separate terminal (keep it running):

   .. code-block:: bash

      ollama serve

3. In another terminal, pull a model:

   .. code-block:: bash

      ollama pull llama2      # ~4GB, fast, good balance
      # or
      ollama pull llama3      # ~7GB, higher quality
      # or
      ollama pull mistral     # ~4GB, very capable

4. Verify Ollama is running:

   .. code-block:: bash

      curl http://localhost:11434/api/tags

**Setup on macOS/Linux**

1. Download and install from https://ollama.ai

2. Start Ollama (runs in background):

   .. code-block:: bash

      ollama serve

3. Pull a model:

   .. code-block:: bash

      ollama pull llama2      # Recommended for most use cases
      ollama pull llama3      # Larger, higher quality
      ollama pull mistral     # Fast and capable

**Configure Rocco for Ollama**

Edit your ``.env`` file:

.. code-block:: ini

   LLM_PROVIDER=ollama
   LLM_BASE_URL=http://localhost:11434/v1
   # LLM_API_KEY is auto-set to "ollama" (no real key needed)
   LLM_MODEL=llama2

**Available models:** ``llama2``, ``llama3``, ``mistral``, ``phi3``, ``neural-chat``, etc. (any model you've pulled)

**Performance notes:**
- Local Ollama requires 4–8GB RAM depending on model size
- Recommend ``llama2`` (7B, 4GB) or ``mistral`` (7B, 4GB) for general use
- Larger models like ``llama3`` (7B, 7GB) or ``mistral`` (8x7B, 8GB) offer better quality but require more resources
- First run of a model may take longer (loading into memory)
- **Windows WSL2 tip:** Keep the ``ollama serve`` terminal open in the background while using Rocco

SambaNova (TACC)
~~~~~~~~~~~~~~~~

**Best for:** High-performance institutional setups.

.. code-block:: ini

   LLM_PROVIDER=sambanova
   LLM_API_KEY=sk-your-key-here
   LLM_MODEL=Llama-4-Maverick-17B-128E-Instruct

**SambaNova endpoint:** https://ai.tejas.tacc.utexas.edu/v1

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
1. Wrap your API with an OpenAI-compatible adapter, OR
2. Fork Rocco and modify ``src/llm/client.py`` to support your specific API format

Environment Variable Reference
-------------------------------

==================  =========  ==============================================
Variable            Required?  Description
==================  =========  ==============================================
``LLM_PROVIDER``    No         Shortcut to pre-fill ``LLM_BASE_URL``.
                               Options: ``openai``, ``anthropic``, ``gemini``,
                               ``deepseek``, ``huggingface``, ``ollama``,
                               ``sambanova``, ``openai_compatible`` (custom).


``LLM_API_KEY``     **Yes**    Your API key. For Ollama or local services, can
                               be dummy value (e.g., ``ollama`` or ``unused``).

``LLM_BASE_URL``    No         Custom endpoint URL. Required for custom or
                               proprietary providers. If not set, provider
                               mapping is used (or defaults to OpenAI).

``LLM_MODEL``       No         Model name (e.g., ``gpt-4o-mini``). Defaults to
                               ``gpt-4o-mini``.
==================  =========  ==============================================

Changing Providers
------------------

To switch providers at runtime, just edit ``.env`` and restart the Streamlit app (press **R** in the browser).


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

Performance Tuning
------------------

- **Fast inference**: Use ``gpt-4o-mini``, ``gemini-2.0-flash``, or local Ollama
- **Best quality**: Use ``gpt-4o``, ``claude-opus-4-7``, or ``deepseek-reasoner``
- **Cost-effective**: Use ``gemini-2.0-flash`` or local Ollama

Next Steps
==========

- Ready to use Rocco? See :doc:`quickstart`
- Want to understand the architecture? See :doc:`../developer_guide/architecture`
- Need help? Check :doc:`../developer_guide/contributing`
