import json
import os
from pathlib import Path
from dotenv import load_dotenv
import logging    
from src.llm.client import RoccoClient
from src.llm.schemas import EvaluatorOutput
from src.evaluator.evaluator import DescriptionEvaluator
from src.ingestor.document_ingestor import DocumentIngestor
from src.ingestor.embedder import DocumentEmbedder
from src.retriever.retriever import VectorStoreManager
from src.editor.editor import DescriptionEditor
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
        
    logging.basicConfig(level=logging.INFO)

    client = RoccoClient(api_url=api_url, api_key=api_key)# , model="Meta-Llama-3.3-70B-Instruct")
    # print(client.list_models())
    grader = DescriptionEvaluator(model=client, rubric=rubric, examples=examples)
    
    # Load the draft description
    with open('DPMP-461_description.txt', "r", encoding="utf-8") as f:
        description_string = f.read()

    # Rocco Evaluation
    evaluation = grader.evaluate(draft_text=description_string)
    grader.print_evaluation_result(evaluation)
    
    # Enter paper
    vector_store_path = "data/vector_store"
    embedder = DocumentEmbedder(model_name="BAAI/bge-large-en-v1.5",
                                model_kwargs={'device': 'cpu'},
                                encode_kwargs={'normalize_embeddings': True})
    vector_store_manager = VectorStoreManager(embedder)
    
    if Path(vector_store_path).exists():
        vector_store_manager.load(vector_store_path)
    else:
        # TODO: Add logic to ingest paper and generate embeddings
        vector_store_manager = None

    # Description Enhancement
    editor = DescriptionEditor(model=client, rubric=rubric, vector_store_manager=vector_store_manager, use_rag=True, top_k_context=5)
    enhanced_description = editor.enhance(draft_text=description_string, draft_evaluation=evaluation, retrieve_context=True)
    
    print(f"Original Description:\n{description_string}\n")
    print(f"Enhanced Description:\n{enhanced_description.suggested_text}\n")
    print(f"Justifications:\n {enhanced_description.rationale}")
    print(f"Citations:\n {enhanced_description.citation}")
    # Rerun the evaluation with the enhanced description
    reevaluation = grader.evaluate(draft_text=enhanced_description.suggested_text)
    grader.print_evaluation_result(reevaluation)