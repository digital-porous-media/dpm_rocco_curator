# src/rocco/evaluator/evaluator.py

from model.client import ModelClient
from model.schemas import EvaluatorOutput

class DescriptionEvaluator:
    """Evaluates dataset descriptions against a rubric"""

    def __init__(self, model: ModelClient, rubric: dict, examples: list):
        self.model = model
        self.rubric = rubric
        self.examples = examples

    def build_prompt(self, draft_text: str, context: list[str]) -> str:
        """Combine rubric, examples, context, and draft into prompt"""
        return "Evaluator prompt placeholder"

    def evaluate(self, draft_text: str, context: list[str]) -> EvaluatorOutput:
        """Call the LLM and return structured evaluation"""
        prompt = self.build_prompt(draft_text, context)
        raw_resp = self.model.call(prompt, max_tokens=800)
        return EvaluatorOutput(raw_resp)
