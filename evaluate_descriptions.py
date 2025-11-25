import json
import os
from dotenv import load_dotenv
import logging    
from src.llm.client import RoccoClient
from src.llm.schemas import EvaluatorOutput
from src.evaluator.evaluator import DescriptionEvaluator
from typing import List, Dict, Any

if __name__ == "__main__":
   
    with open("src/evaluator/examples_v3.json", "r") as f:
        examples = json.load(f)
    with open("src/evaluator/rubric.json", "r") as f:
        rubric = json.load(f)  
        
        
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    api_key = os.getenv("SAMBANOVA_API_KEY")
    api_url = os.getenv("SAMBANOVA_API_URL")
    client = RoccoClient(api_url=api_url, api_key=api_key)
    grader = DescriptionEvaluator(model=client, rubric=rubric, examples=examples)
    
    description_string = "A North Sea sandstone with 23% porosity and 640 mD permeability was cleaned with solvents, dried at 60°C, and imaged using micro-CT at a 2.3 μm resolution over 22 hours. It was then cut, resin-impregnated, polished, and carbon-coated for 2D mineral mapping using SEM-EDS with QEMSCAN at a 2.0 μm resolution. The 2D mineral map was registered to the 3D micro-CT image, and minerals were segmented into seven groups based on X-ray intensity, though some minerals with similar attenuation could not be fully distinguished. The data is uploaded as 4 netCDF blocks that combine into a full 3D data set of 1,000^3. Full details are provided in the netCDF header files. The segmented domain is associated with LBPM (https://github.com/OPM) simulations results available at https://zenodo.org/records/13836047"
    evaluation = grader.evaluate(draft_text=description_string)
    print(f"Final Score: {evaluation.total_score}")
    print(f"Justifications: {evaluation.rubric_breakdown}")
    