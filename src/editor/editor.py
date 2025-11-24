from llm.client import RoccoClient
from llm.schemas import EditorOutput

class DescriptionEnhancer:
    """Improves dataset descriptions based on evaluation and context"""

    def __init__(self, model: RoccoClient, rubric: dict):
        self.model = model
        self.rubric = rubric

    def build_prompt(self, draft_text: str, evaluator_output: dict, context: list[str]) -> str:
        """Prepare prompt for improving the draft"""
        return "Editor prompt placeholder"

    def enhance(self, draft_text: str, evaluator_output: dict, context: list[str]) -> EditorOutput:
        prompt = self.build_prompt(draft_text, evaluator_output, context)
        raw_resp = self.model.call(prompt, max_tokens=1200)
        return EditorOutput(raw_resp)
