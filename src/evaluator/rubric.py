import json

def load_rubric(rubric_path: str = "src/evaluator/rubric.json") -> dict:
    with open(rubric_path, "r") as f:
        rubric = json.load(f)
    return rubric
    
    