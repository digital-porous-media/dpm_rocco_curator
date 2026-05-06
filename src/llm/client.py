import logging
import os
from typing import Optional, Dict, Any, List
import openai

class LLMClient:
    """
    Provider-agnostic LLM client supporting OpenAI, Anthropic, Gemini, DeepSeek, HuggingFace, Ollama, SambaNova, and any OpenAI-compatible API.
    """

    # Provider-to-base-URL mapping for convenience
    # Note: Only providers that expose OpenAI-compatible /v1 endpoints are listed here.
    # For any custom endpoint, use LLM_BASE_URL with a custom wrapper or compatible API.
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

        # Initialize OpenAI-compatible client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
        self.timeout = timeout

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using LLM provider: {self.provider}, model: {self.model}, endpoint: {self.api_url}")
        
    
    def list_models(self) -> List[str]:
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
            self.logger.info("Received response from model.")
            return result
        except Exception as e:
            self.logger.error(f"Error sending prompt: {str(e)}")
            raise RuntimeError(f"LLM API error ({self.provider}): {str(e)}")

class RoccoClient(LLMClient):
    """
    RoccoClient extends LLMClient for specific Rocco interactions.
    """
    def __init__(self, api_url: str = None, api_key: str = None, model: str = None, provider: str = None, timeout: int = 60):
        super().__init__(api_url, api_key, model, provider, timeout)
        
    def evaluate_description(self, draft_text: str, rubric: Dict[str, Any], examples: List[Dict[str, Any]], context: Optional[List[str]] = None) -> str:
        """
        Evaluate a dataset description using the provided rubric and examples.
        """
        pass
    
    def improve_description(self, draft_text: str, context: Optional[List[str]] = None) -> str:
        """
        Improve a dataset description based on the provided context.
        """
        pass