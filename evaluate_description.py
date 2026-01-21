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
from src.llm.content_screener import ContentScreener
from typing import List, Dict, Any


if __name__ == "__main__":
    
    with open("src/evaluator/examples_v3.json", "r") as f:
        examples = json.load(f)
    with open("src/evaluator/rubric.json", "r") as f:
        rubric = json.load(f)  
        
    load_dotenv()
    api_key = os.getenv("SAMBANOVA_API_KEY")
    api_url = os.getenv("SAMBANOVA_API_BASE_URL")
        
    logging.basicConfig(level=logging.INFO)

    client = RoccoClient(api_url=api_url, api_key=api_key)
    print(client.list_models())
    grader = DescriptionEvaluator(model=client, rubric=rubric, examples=examples)
    
    # Load the draft description
    with open('DPMP-523_description.txt', "r", encoding="utf-8") as f:
        description_string = f.read()

    # Rocco Evaluation
    evaluation = grader.evaluate(draft_text=description_string)
    grader.print_evaluation_result(evaluation)
    
    # # Enter paper
    # pdf_path = "data/DPMP-523.pdf"
    # pdf_filename = Path(pdf_path).stem
    # vector_store_path = f"data/vector_store/{pdf_filename}"
    # embedder = DocumentEmbedder(model_name="BAAI/bge-large-en-v1.5",
    #                             model_kwargs={'device': 'cpu'},
    #                             encode_kwargs={'normalize_embeddings': True})
    # vector_store_manager = VectorStoreManager(embedder)
    
    # if Path(vector_store_path).exists():
    #     logging.info(f"Loading existing vector store from: {vector_store_path}")
    #     vector_store_manager.load(vector_store_path)
    # else:
    #     logging.info(f"Vector store not found at: {vector_store_path}, creating a new one.")
        
    #     ingestor = DocumentIngestor(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])
    #     chunks = ingestor.ingest(pdf_path)
    #     logging.info(f"Ingested {len(chunks)} chunks from {pdf_path}")

    #     # Create the vector store via VectorStoreManager and save to disk
    #     vector_store_manager.create_from_documents(chunks)
    #     logging.info("Created vector store from document chunks.")
    #     vector_store_manager.save(vector_store_path)
    #     # print(f"Vector store created and saved to: {vector_store_path}")
    #     logging.info(f"Vector store saved to {vector_store_path}")
    # # Description Enhancement
    # editor = DescriptionEditor(model=client, rubric=rubric, vector_store_manager=vector_store_manager, use_rag=True, top_k_context=5)
    # enhanced_description = editor.enhance(draft_text=description_string, draft_evaluation=evaluation, retrieve_context=True)
    
    # editor.print_enhancement_result(enhanced_description)
    
    # # Rerun the evaluation with the enhanced description
    # reevaluation = grader.evaluate(draft_text=enhanced_description.suggested_text)
    # grader.print_evaluation_result(reevaluation)
    
    # Provide some user feedback
    # user_feedback = "The dataset can be reused for simulation and machine learning training." # "This dataset is organized into grayscale and segmented Samples. Each sample contains an image of the dry scan and images at various fractional flow levels."
    # screener = ContentScreener(model=client)
    # screen_result = screener.screen_user_content(user_feedback, description_string)
    # print(screen_result)
    # user_refinement1 = editor.enhance(
    #     draft_text = description_string, # enhanced_description.suggested_text,
    #     draft_evaluation= evaluation, # reevaluation,
    #     user_feedback=user_feedback
    # )
    
    # editor.print_enhancement_result(user_refinement1)
    # # Rerun the evaluation with the enhanced description
    # reevaluation = grader.evaluate(draft_text=user_refinement1.suggested_text)
    # grader.print_evaluation_result(reevaluation)
    
    
    # # Provide some user feedback
    # user_feedback = "This dataset was created to demonstrate the dynamic fluid interface behavior from a real fractional flow experiment on an imaged sample thats size is closer to a representative elementary volume and can provide greater certainty in informing upscaling multiphase flow behavior from pore-scale to continuum-scale."
    
    # user_refinement2 = editor.enhance(
    #     draft_text = user_refinement1.suggested_text,
    #     draft_evaluation=reevaluation,
    #     user_feedback=user_feedback
    # )
    
    # editor.print_enhancement_result(user_refinement2)
    
    # # Rerun the evaluation with the enhanced description
    # reevaluation = grader.evaluate(draft_text=user_refinement2.suggested_text)
    # grader.print_evaluation_result(reevaluation)
    