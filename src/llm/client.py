import os

class RoccoClient:
    """Wraps LLM API calls"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SAMBANOVA_API_KEY")
        
    def call(self, prompt: str, max_tokens: int = 1000) -> str:
        pass
    
    