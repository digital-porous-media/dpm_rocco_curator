from typing import List, Dict, Any
from src.llm.schemas import EvaluatorOutput

def _build_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    """
    Converts a list of example dicts into a formatted string for few-shot prompting.
    Each example includes a description and a rubric breakdown with scores and explanations.
    """
    prompt_parts = []
    for idx, ex in enumerate(examples, 1):
        part = [f"EXAMPLE {idx}"]
        part.append(f"Description: {ex['description']}")
        part.append("Explanation:")
        for i, item in enumerate(ex['rubric_breakdown'], 1):
            score = item.get('score', '')
            explanation = item.get('explanation', '')
            part.append(f"{i}. {score} points for {explanation}")
        prompt_parts.append("\n".join(part))
    return "\n\n".join(prompt_parts)

def _build_rubric(rubric: List[Dict[str, Any]]) -> str:
    """
    Converts the rubric from JSON into a formatted string for prompting.
    Each rubric item includes a criterion and its maximum score.
    """
    
    rubric_lines = []
    for idx, item in enumerate(rubric, 1):
        rubric_lines.append(f"{idx}. {item['criterion']}: {item['description']} (Max {item['max_score']})")
    return "\n".join(rubric_lines)

def _build_evaluation_text(evaluation: EvaluatorOutput) -> str:
    evaluation_text = "\n".join([
            f"- Criterion {item.criterion}: {item.score}/1 - {item.explanation}"
            for item in evaluation.rubric_breakdown
        ])
    return evaluation_text