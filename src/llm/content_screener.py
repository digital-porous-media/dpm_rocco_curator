from src.llm.client import RoccoClient
from src.prompts.loader import load_prompt, render
import json

class ContentScreener:
    """Screen contents for usefulness"""
    def __init__(self, model: RoccoClient):
        """
        Args:
            model: Configured :class:`~src.llm.client.RoccoClient` used to call the LLM.
        """
        self.model = model
    
    def screen_user_content(self, content: str, context: str = None) -> dict:
        """
        Screen user provided content

        Returns: {
            'is_valid': bool,
            'issues': list of issues found
            'confidence': float (0-1)
            'recommendation': str
        }
        """

        # Load prompt template and render
        prompt_data = load_prompt("content_screener")
        screening_prompt = render(
            prompt_data["user"],
            content=content,
            context=context or ""
        )

        response = self.model.send_prompt(prompt=screening_prompt)
        # print(response)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            return {
                "is_relevant": False,
                "is_accurate": False,
                "is_respectful": False,
                "is_coherent": False,
                "issues": ["Unable to parse feedback"],
                "confidence": 0.5,
                "recommendation": "flag_for_review"
            }
        
        