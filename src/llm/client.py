import json
import logging
import os
from typing import Optional, Dict, Any, List
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
        timeout: int = None
    ):
        """
        Initialize LLMClient with provider-agnostic configuration.

        Args:
            api_url: Base URL for the LLM API endpoint. Overrides ``LLM_BASE_URL`` env var.
            api_key: API key. Overrides ``LLM_API_KEY`` env var. Defaults to ``"ollama"`` for local Ollama.
            model: Model name. Overrides ``LLM_MODEL`` env var. Defaults to ``"gpt-4o-mini"``.
            provider: Shortcut alias (``openai``, ``anthropic``, ``gemini``, ``deepseek``,
                ``huggingface``, ``ollama``, ``sambanova``). Overrides ``LLM_PROVIDER`` env var.
            timeout: Request timeout in seconds. Overrides ``LLM_TIMEOUT`` env var. Defaults to 120
                (some providers, e.g. SambaNova/TACC, routinely take well over 60s to respond).
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
        self.timeout = timeout if timeout is not None else int(os.getenv("LLM_TIMEOUT", "120"))

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using LLM provider: {self.provider}, model: {self.model}, endpoint: {self.api_url}")

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
    timeout: int = Field(default=120, description="Request timeout in seconds")
    temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    llm_client: Any = Field(default=None, exclude=True, description="Wrapped LLMClient instance")

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        model: str = None,
        provider: str = None,
        timeout: int = None,
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
            timeout: Request timeout in seconds. Overrides ``LLM_TIMEOUT`` env var. Defaults
                to 120 (some providers, e.g. SambaNova/TACC, routinely take well over 60s).
            temperature: Temperature for LLM generation (0.0-1.0).
            **kwargs: Additional arguments for BaseChatModel.
        """
        # LLMClient owns the provider/URL/key/model/timeout resolution rules (see its
        # __init__); build it from the raw arguments and read the resolved values back
        # off it, rather than reimplementing the same env-fallback chain here and having
        # to keep two copies in step.
        llm_client = LLMClient(
            api_url=api_url,
            api_key=api_key,
            model=model,
            provider=provider,
            timeout=timeout,
        )

        # Initialize BaseChatModel (Pydantic) with the resolved configuration.
        super().__init__(
            provider=llm_client.provider,
            api_key=llm_client.api_key,
            api_url=llm_client.api_url,
            model=llm_client.model,
            timeout=llm_client.timeout,
            temperature=temperature,
            **kwargs
        )

        # The wrapped instance handles the actual API calls.
        self.llm_client = llm_client

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

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Enable tool use for LangGraph ReAct agents."""
        from langchain_core.utils.function_calling import convert_to_openai_tool
        formatted = [convert_to_openai_tool(t) for t in tools]
        extra: Dict[str, Any] = {"tools": formatted}
        if tool_choice is not None:
            extra["tool_choice"] = tool_choice
        extra.update(kwargs)
        return self.bind(**extra)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response, supporting tool calls for LangGraph agents."""
        oai_messages = []
        for msg in messages:
            if msg.type == "system":
                oai_messages.append({"role": "system", "content": msg.content})
            elif msg.type in ("human", "user"):
                oai_messages.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                oai_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    oai_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                oai_messages.append(oai_msg)
            elif msg.type == "tool":
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,  # type: ignore[attr-defined]
                    "content": msg.content,
                })

        call_params: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if stop:
            call_params["stop"] = stop
        call_params.update(kwargs)

        response = self.llm_client.client.chat.completions.create(**call_params)
        choice = response.choices[0].message

        if choice.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                    "type": "tool_call",
                }
                for tc in choice.tool_calls
            ]
            message = AIMessage(content=choice.content or "", tool_calls=tool_calls)
        else:
            message = AIMessage(content=choice.content or "")

        return ChatResult(generations=[ChatGeneration(message=message, text=message.content)])
