import logging
import os
from typing import Optional, Dict, Any, List
import openai

class LLMClient:
    """
    Base LLM client for interacting with LLM API (e.g., SambaNova)
    """
        
    def __init__(self, api_url: str = None, api_key: str = None, model: str = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("SAMBANOVA_API_KEY")
        self.api_url = api_url or os.getenv("SAMBANOVA_API_URL", "https://ai.tejas.tacc.utexas.edu/v1")
        self.model = model or os.getenv("SAMBANOVA_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
        self.timeout = timeout
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using model: {self.model}")
        
    
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
            return

class RoccoClient(LLMClient):
    """
    RoccoClient extends LLMClient for specific Rocco interactions.
    """
    def __init__(self, api_url: str = None, api_key: str = None, model: str = None, timeout: int = 60):
        super().__init__(api_url, api_key, model, timeout)
        
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