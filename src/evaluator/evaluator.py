from src.llm.client import RoccoClient
from src.llm.schemas import EvaluatorOutput, RubricItem
from src.common.utils import _build_rubric, _build_few_shot_examples
import json
import re
from typing import List, Dict, Any, Optional


class DescriptionEvaluator:
    """Evaluates dataset descriptions against a rubric"""

    def __init__(self, model: RoccoClient, rubric: List[Dict[str, Any]], examples: List[Dict[str, Any]]):
        self.model = model
        self.rubric = rubric
        self.examples = examples

    def build_prompt(self, draft_text: str, context: Optional[list[str]]=None) -> str:
        """Combine rubric, examples, context, and draft into prompt"""
        rubric_str = _build_rubric(self.rubric)
        examples_str = _build_few_shot_examples(self.examples)
        
        # Use your system instructions and preface
        system_instructions = (
            "You are an expert data curator for the Digital Porous Media Portal. "
            "You are provided 10 guidelines, each of which is worth one point. "
            "Descriptions only get the point if the guideline is addressed explicitly. "
            "You are to evaluate the description for each guideline. Follow the examples provided. "
            "Only evaluate the 10 guidelines, do not try to sum everything at the end.\n"
            "Return your evaluation as a JSON object with the following format:\n"
            "{\n"
            '  "rubric_breakdown": [\n'
            '    {"criterion": "Self-Contained Description", "score": 1, "explanation": "..."},\n'
            '    {"criterion": "Context of Creation", "score": 0.5, "explanation": "..."},\n'
            '...\n'
            "  ]\n"
            "}\n"
            "Do not provide any additional text outside the JSON.\n"
        )
        preface = "\nNow follows the description you must rate. Do not round.\n"
        context_str = ""
        if context:
            context_str = "Additional context:\n" + "\n".join(context) + "\n"
            
        prompt = (
            f"{system_instructions}\n"
            f"Rubric:\n{rubric_str}\n\n"
            f"Examples:\n{examples_str}\n"
            f"{context_str}"
            f"{preface}"
            f"Description: {draft_text}\n"
            "Explanation:"
        )
        return prompt

    def evaluate(self, draft_text: str) -> EvaluatorOutput: #, context: Optional[list[str]]=[""]) -> EvaluatorOutput:
        """Call the LLM and return structured evaluation"""
        prompt = self.build_prompt(draft_text)
        raw_resp = self.model.send_prompt(prompt) 
        # print(raw_resp)       
        # Try to parse as JSON first (if your prompt requests JSON output)
        try:
            data = json.loads(raw_resp.strip())
            rubric_breakdown = [
                RubricItem(
                    criterion=item["criterion"],
                    score=item["score"],
                    explanation=item.get("explanation", "")
                )
                for item in data["rubric_breakdown"]
            ]
            total_score = sum(item.score for item in rubric_breakdown)
            comments = data.get("comments", None)
            return EvaluatorOutput(
                total_score=total_score,
                rubric_breakdown=rubric_breakdown,
                comments=comments
            )
        except Exception:
            # print("Failed to parse JSON from response.")
            
            # Fallback: parse numbered list
            rubric_breakdown = []
            total_score = 0
            # Example regex for "1. 0.5 points for..."
            item_regex = re.compile(
                r'\{"criterion":\s*"([^"]+)",\s*"score":\s*([\d.]+),\s*"explanation":\s*"([^"]*)"\s*\}',
                re.MULTILINE
            )
            matches = []
            for match in item_regex.finditer(raw_resp):
                criterion = match.group(1)
                score = float(match.group(2))
                explanation = match.group(3)
                matches.append((criterion, score, explanation))
                
                
            rubric_breakdown = []
            total_score = 0
            for criterion, score, explanation in matches:
                score = float(score)
                rubric_breakdown.append(RubricItem(criterion, score, explanation))
                total_score += score
            return EvaluatorOutput(
                total_score=total_score,
                rubric_breakdown=rubric_breakdown,
                comments=None
            )
        return EvaluatorOutput(raw_resp)

    def print_evaluation_result(self, evaluation_output: EvaluatorOutput) -> None:
        """Utility to print evaluation results"""
        print(f"Total Score: {evaluation_output.total_score}")
        print("Justifications:\n")
        print("-" * 80)
        for item in evaluation_output.rubric_breakdown:
            print(f"Criterion: {item.criterion} \t Score: {item.score}")
            print(f"Explanation: {item.explanation}\n")
    
