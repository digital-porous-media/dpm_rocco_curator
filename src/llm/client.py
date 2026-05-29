import logging
import os
from typing import Optional, Dict, Any, List, ClassVar
import openai

from pydantic import Field
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _strip_json_fences(text: str) -> str:
	"""Strip markdown code fences from LLM responses that ignore 'return JSON only' instructions."""
	if "```json" in text:
		text = text.split("```json", 1)[1].split("```", 1)[0]
	elif "```" in text:
		text = text.split("```", 1)[1].split("```", 1)[0]
	return text.strip()


class LLMClient:
    """
    Provider-agnostic LLM client supporting OpenAI, Anthropic, Gemini, DeepSeek, HuggingFace, Ollama, SambaNova, and any OpenAI-compatible API.
    Pure utility class (does not inherit from Pydantic models).
    """

    # Provider-to-base-URL mapping for convenience
    PROVIDER_URLS = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "deepseek": "https://api.deepseek.com/v1",
        "huggingface": "https://router.huggingface.co/v1",
        "ollama": "http://localhost:11434/v1",
        "sambanova": "https://ai.tejas.tacc.utexas.edu/v1",
    }

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        model: str = None,
        provider: str = None,
        timeout: int = 60
    ):
        """
        Initialize LLMClient with provider-agnostic configuration.

        Args:
            api_url: Base URL for the LLM API endpoint. Overrides ``LLM_BASE_URL`` env var.
            api_key: API key. Overrides ``LLM_API_KEY`` env var. Defaults to ``"ollama"`` for local Ollama.
            model: Model name. Overrides ``LLM_MODEL`` env var. Defaults to ``"gpt-4o-mini"``.
            provider: Shortcut alias (``openai``, ``anthropic``, ``gemini``, ``deepseek``,
                ``huggingface``, ``ollama``, ``sambanova``). Overrides ``LLM_PROVIDER`` env var.
            timeout: Request timeout in seconds. Defaults to 60.
        """
        # Load from environment with fallback order:
        # 1. Direct parameter
        # 2. Environment variable
        # 3. Infer from LLM_BASE_URL if set
        # 4. Default to openai

        self.provider = provider or os.getenv("LLM_PROVIDER", "").lower()
        if not self.provider:
            # If custom base URL is set without explicit provider, mark as custom
            if api_url or os.getenv("LLM_BASE_URL"):
                self.provider = "custom"
            else:
                self.provider = "openai"

        # Determine API key
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if self.provider == "ollama" and not self.api_key:
            self.api_key = "ollama"

        # Determine API URL
        if api_url:
            self.api_url = api_url
        elif os.getenv("LLM_BASE_URL"):
            self.api_url = os.getenv("LLM_BASE_URL")
        else:
            # Use provider mapping if available
            self.api_url = self.PROVIDER_URLS.get(self.provider, "https://api.openai.com/v1")

        # Determine model
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
        )
        self.timeout = timeout

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using LLM provider: {self.provider}, model: {self.model}, endpoint: {self.api_url}")

    def list_models(self) -> List[str]:
        """Return a list of model IDs available from the configured provider endpoint."""
        try:
            self.logger.info("Listing available models...")
            response = self.client.models.list()
            models = [m.id for m in response.data]
            self.logger.info(f"Available models: {models}")
            return models
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []

    def send_prompt(self, prompt: str, context: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a prompt to the LLM and return the response text.
        """
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})

        call_params = {"model": self.model,
                       "messages": messages,
                       "timeout": self.timeout}
        if params:
            call_params.update(params)

        try:
            self.logger.info(f"Sending prompt to model {self.model}...")

            response = self.client.chat.completions.create(
                **call_params
            )
            result = response.choices[0].message.content
            result = _strip_json_fences(result)
            self.logger.info("Received response from model.")
            return result
        except Exception as e:
            self.logger.error(f"Error sending prompt: {str(e)}")
            raise RuntimeError(f"LLM API error ({self.provider}): {str(e)}")


class RoccoClient(BaseChatModel):
    """
    RoccoClient implements LangChain's BaseChatModel interface with provider-agnostic LLM support.

    Uses LLMClient internally for provider-agnostic API calls. Works with OpenAI, Anthropic,
    Gemini, DeepSeek, HuggingFace, Ollama, SambaNova, and any OpenAI-compatible API.
    """

    # Pydantic fields for LangChain integration
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.)")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    api_url: str = Field(default="https://api.openai.com/v1", description="Base URL for the LLM API")
    model: str = Field(default="gpt-4o-mini", description="Model name")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    llm_client: Any = Field(default=None, exclude=True, description="Wrapped LLMClient instance")

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        model: str = None,
        provider: str = None,
        timeout: int = 60,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize RoccoClient with BaseChatModel support.

        Args:
            api_url: Base URL for the LLM API endpoint.
            api_key: API key for authentication.
            model: Model name to use.
            provider: Provider shortcut alias (openai, anthropic, etc.).
            timeout: Request timeout in seconds.
            temperature: Temperature for LLM generation (0.0-1.0).
            **kwargs: Additional arguments for BaseChatModel.
        """
        # Determine provider and api_url using same logic as LLMClient
        resolved_provider = provider or os.getenv("LLM_PROVIDER", "").lower()
        if not resolved_provider:
            if api_url or os.getenv("LLM_BASE_URL"):
                resolved_provider = "custom"
            else:
                resolved_provider = "openai"

        # Determine API URL using provider mapping
        if api_url:
            resolved_api_url = api_url
        elif os.getenv("LLM_BASE_URL"):
            resolved_api_url = os.getenv("LLM_BASE_URL")
        else:
            resolved_api_url = LLMClient.PROVIDER_URLS.get(resolved_provider, "https://api.openai.com/v1")

        # Initialize BaseChatModel (Pydantic) with configuration
        super().__init__(
            provider=resolved_provider,
            api_key=api_key or os.getenv("LLM_API_KEY"),
            api_url=resolved_api_url,
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            timeout=timeout,
            temperature=temperature,
            **kwargs
        )

        # Create wrapped LLMClient instance (handles the actual API calls)
        self.llm_client = LLMClient(
            api_url=self.api_url,
            api_key=self.api_key,
            model=self.model,
            provider=self.provider,
            timeout=self.timeout
        )

    @property
    def _llm_type(self) -> str:
        """Return the LangChain LLM type identifier."""
        return f"rocco_client_{self.provider}"

    def send_prompt(self, prompt: str, context: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Backward-compatible method that delegates to the wrapped LLMClient.
        """
        return self.llm_client.send_prompt(prompt, context, params)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from the LLM.

        Converts LangChain messages to LLMClient format and back.
        """
        system_content = None
        user_content = None

        for msg in messages:
            if msg.type == "system":
                system_content = msg.content
            elif msg.type in ("human", "user"):
                user_content = msg.content

        response_text = self.llm_client.send_prompt(
            prompt=user_content or "",
            context=system_content,
            params={"temperature": self.temperature},
        )
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message, text=response_text)
        return ChatResult(generations=[generation])
