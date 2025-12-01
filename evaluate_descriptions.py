import json
import os
from dotenv import load_dotenv
import logging    
from src.llm.client import RoccoClient
from src.llm.schemas import EvaluatorOutput
from src.evaluator.evaluator import DescriptionEvaluator
from typing import List, Dict, Any
import numpy as np
import httpx


if __name__ == "__main__":
    
    with open("src/evaluator/examples_v3.json", "r") as f:
        examples = json.load(f)
    with open("src/evaluator/rubric.json", "r") as f:
        rubric = json.load(f)  
        
    load_dotenv()
    api_key = os.getenv("SAMBANOVA_API_KEY")
    api_url = os.getenv("SAMBANOVA_API_URL")
    
    # import httpx, time

    # API_KEY = api_key

    # models = [
    #     "Meta-Llama-3.1-8B-Instruct",
    #     "Qwen3-32B",
    #     "Llama-4-Maverick-17B-128E-Instruct",
    #     "Meta-Llama-3.3-70B-Instruct",
    # ]

    # with httpx.Client(timeout=120) as client:
    #     for m in models:
    #         print(f"Testing {m}...")
    #         try:
    #             r = client.post(
    #                 "https://ai.tejas.tacc.utexas.edu/v1/chat/completions",
    #                 headers={"Authorization": f"Bearer {API_KEY}"},
    #                 json={"model": m, "messages": [{"role": "user", "content": "ping"}]},
    #             )
    #             print("Status:", r.status_code)
    #             print("Response:", r.text)
    #         except Exception as e:
    #             print("Failed:", e)

    #         print("-" * 40)
    #         time.sleep(1)


        
        
    logging.basicConfig(level=logging.INFO)

    client = RoccoClient(api_url=api_url, api_key=api_key)# , model="Meta-Llama-3.3-70B-Instruct")
    grader = DescriptionEvaluator(model=client, rubric=rubric, examples=examples)
    
    # Load the draft description
    with open('description.txt', "r", encoding="utf-8") as f:
        description_string = f.read()

    # description_string = "The workflow that produces the STL files is provided in a Jupyter notebook (.ipynb file). The workflow includes generating a mesh with specific dimensions from a 2D image of the porous media. Before STL generation, the image is eroded to ensure that the metal ball can pass through. The workflow also includes a tool that analyzes whether a ball of a specific size can pass - an example with a 2.5 mm ball is shown - you can use the bike ball bearings for this purpose. This mesh is then merged into a game base modeled in Blender."
    # Rocco Evaluation
    evaluation = grader.evaluate(draft_text=description_string)
    
    # Print the results
    print(f"Final Score: {evaluation.total_score}")
    print(f"Justifications:")
    for item in evaluation.rubric_breakdown:
        print(f"Criterion: {item.criterion} \t\t Score: {item.score}")
        print(f"Explanation: {item.explanation}\n")
        
