from src.llm.client import RoccoClient
from src.llm.schemas import EvaluatorOutput, RubricItem
from src.common.utils import _build_rubric, _build_few_shot_examples
from src.prompts.loader import load_prompt, render
import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _to_evaluator_output(data: Dict[str, Any]) -> EvaluatorOutput:
    """Build an EvaluatorOutput from a parsed rubric payload.

    Shared by all three JSON-extraction paths in ``evaluate`` (direct parse, JSON block
    embedded in prose, JSON inside a markdown fence) — they differ only in how they get
    ``data`` out of the raw response, not in how the result is assembled.
    """
    rubric_breakdown = [
        RubricItem(
            criterion=item["criterion"],
            score=item["score"],
            explanation=item.get("explanation", "")
        )
        for item in data["rubric_breakdown"]
    ]
    return EvaluatorOutput(
        total_score=sum(item.score for item in rubric_breakdown),
        rubric_breakdown=rubric_breakdown,
        comments=data.get("comments", None)
    )


class DescriptionEvaluator:
    """Evaluates dataset descriptions against a rubric"""

    def __init__(self, model: RoccoClient, rubric: List[Dict[str, Any]], examples: List[Dict[str, Any]]):
        self.model = model
        self.rubric = rubric
        self.examples = examples

    def build_prompt(self, draft_text: str) -> str:
        """Combine rubric, examples, and draft into prompt"""
        rubric_str = _build_rubric(self.rubric)
        examples_str = _build_few_shot_examples(self.examples)

        # Load prompt template and render
        prompt_data = load_prompt("evaluator")
        prompt = render(
            prompt_data["user"],
            rubric=rubric_str,
            examples=examples_str,
            description=draft_text
        )
        return prompt

    def evaluate(self, draft_text: str) -> EvaluatorOutput: #, context: Optional[list[str]]=[""]) -> EvaluatorOutput:
        """Call the LLM and return structured evaluation"""
        prompt = self.build_prompt(draft_text)
        try:
            raw_resp = self.model.send_prompt(prompt, params={"response_format": {"type": "json_object"}})
        except Exception:
            raw_resp = self.model.send_prompt(prompt)

        if raw_resp is None:
            raise RuntimeError("LLM returned no response. Check your API key and network connection.")

        # Try to parse as JSON first (if your prompt requests JSON output)
        try:
            return _to_evaluator_output(json.loads(raw_resp.strip()))
        except Exception as json_err:
            logger.warning(f"JSON parsing failed: {str(json_err)}")
            logger.debug(f"Raw response (first 500 chars): {raw_resp[:500]}")

            # Fallback 1.5: try to extract JSON block from mixed text (e.g., "Here is my evaluation:\n{...}")
            json_block_match = re.search(r'(\{[\s\S]*\})', raw_resp)
            if json_block_match:
                try:
                    output = _to_evaluator_output(json.loads(json_block_match.group(1)))
                    logger.info("Successfully parsed JSON from mixed-text response")
                    return output
                except Exception as block_err:
                    logger.warning(f"JSON block extraction failed: {str(block_err)}")

            # Fallback: try to extract JSON from markdown code blocks (Gemini often wraps in ```json)
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_resp)
            if json_match:
                try:
                    output = _to_evaluator_output(json.loads(json_match.group(1)))
                    logger.info("Successfully parsed JSON from markdown code block")
                    return output
                except Exception as markdown_err:
                    logger.warning(f"Markdown JSON extraction failed: {str(markdown_err)}")

            # Final fallback: parse regex
            rubric_breakdown = []
            total_score = 0
            item_regex = re.compile(
                r'\{"criterion":\s*"([^"]+)",\s*"score":\s*([\d.]+),\s*"explanation":\s*"([^"]*)"\s*\}',
                re.MULTILINE
            )
            matches = list(item_regex.finditer(raw_resp))

            if not matches:
                logger.error(f"Failed to parse response from LLM. Response: {raw_resp[:1000]}")
                raise ValueError(
                    f"Could not parse LLM response. Expected JSON format. "
                    f"Got (first 200 chars): {raw_resp[:200]}"
                )

            for match in matches:
                criterion = match.group(1)
                score = float(match.group(2))
                explanation = match.group(3)
                rubric_breakdown.append(RubricItem(criterion, score, explanation))
                total_score += score

            return EvaluatorOutput(
                total_score=total_score,
                rubric_breakdown=rubric_breakdown,
                comments=None
            )

    def print_evaluation_result(self, evaluation_output: EvaluatorOutput) -> None:
        """Utility to print evaluation results"""
        print(f"Total Score: {evaluation_output.total_score}")
        print("Justifications:\n")
        print("-" * 80)
        for item in evaluation_output.rubric_breakdown:
            print(f"Criterion: {item.criterion} \t Score: {item.score}")
            print(f"Explanation: {item.explanation}\n")

