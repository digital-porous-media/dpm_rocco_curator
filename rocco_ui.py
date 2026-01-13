import streamlit as st
from dotenv import load_dotenv
import os
import json
from pathlib import Path
import uuid
from datetime import datetime

from src.llm.client import RoccoClient
from src.evaluator.evaluator import DescriptionEvaluator
from src.editor.editor import DescriptionEditor
from src.ingestor.embedder import DocumentEmbedder
from src.ingestor.document_ingestor import DocumentIngestor
from src.retriever.retriever import VectorStoreManager

# --- Constants and Page Config ---
ROCCO_AVATAR = "assets/rocco_avatar.jpg"
USER_AVATAR = "assets/user_avatar.jpg"

st.set_page_config(page_title="Rocco - DPM Curator", layout="wide")
st.title("Rocco - Your Digital Porous Media AI Curator")

# --- Helper Functions ---
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY")
api_url = os.getenv("SAMBANOVA_API_BASE_URL")

# --- Session State Initialization ---
if "description_text" not in st.session_state:
    st.session_state.description_text = ""
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = None
if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = ""
if "original_description" not in st.session_state:
    st.session_state.original_description = None
if "enhanced_description" not in st.session_state:
    st.session_state.enhanced_description = None

# --- Initialization ---
@st.cache_resource
def load_resources():
    """Load rubric, examples, and initialize clients."""
    with open("src/evaluator/rubric.json", "r") as f:
        rubric = json.load(f)
    with open("src/evaluator/examples_v3.json", "r") as f:
        examples = json.load(f)
    
    client = RoccoClient(api_url=api_url, api_key=api_key)
    grader = DescriptionEvaluator(model=client, rubric=rubric, examples=examples)
    editor = DescriptionEditor(model=client, rubric=rubric, vector_store_manager=st.session_state.vector_store_manager)
    embedder = DocumentEmbedder(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    ingestor = DocumentIngestor(chunk_size=500, chunk_overlap=100)
    return rubric, examples, client, grader, editor, embedder, ingestor

if api_key and api_url:
    rubric, examples, client, grader, editor, embedder, ingestor = load_resources()
    if st.session_state.vector_store_manager:
        editor.vector_store_manager = st.session_state.vector_store_manager
else:
    st.error("API Key or URL not found. Please set SAMBANOVA_API_KEY and SAMBANOVA_API_URL.")
    st.stop()

# --- Main App Layout ---
st.header("Dataset Description")
st.info("Enter your dataset description below. You can then use the tools to evaluate and enhance it.")
description_text = st.text_area("Enter your dataset description here:", value=st.session_state.description_text, height=250, label_visibility="collapsed")
st.session_state.description_text = description_text

evaluate_button = st.button("Evaluate Description", use_container_width=True)

# --- Workflow Logic ---
if evaluate_button and description_text:
    with st.spinner("Rocco is evaluating your description..."):
        evaluation = grader.evaluate(draft_text=description_text)
        st.session_state.evaluation = evaluation
        st.rerun()

st.divider()

# Display comparison view if an enhanced description is available
if st.session_state.enhanced_description:
    st.header("Compare and Finalize Description")
    col1, col2 = st.columns(2)
    with col1:
        st.info("Your Original Description")
        st.text_area("Original", value=st.session_state.original_description, height=300, key="original_desc_readonly", disabled=True)
    with col2:
        st.info("Rocco's Suggested Enhancement")
        st.text_area("Enhanced", value=st.session_state.enhanced_description, height=300, key="enhanced_desc_readonly", disabled=True)

    # --- Action buttons for the new description ---
    accept_col, reject_col = st.columns(2)
    with accept_col:
        if st.button("✅ Adopt Rocco's Version", use_container_width=True):
            st.session_state.description_text = st.session_state.enhanced_description
            st.session_state.original_description = None
            st.session_state.enhanced_description = None
            st.session_state.evaluation = None
            st.rerun()
    with reject_col:
        if st.button("❌ Keep Original Version", use_container_width=True):
            st.session_state.description_text = st.session_state.original_description
            st.session_state.original_description = None
            st.session_state.enhanced_description = None
            st.session_state.evaluation = None
            st.rerun()
    st.divider()

# Display evaluation results and enhancement tools if an evaluation is present
if st.session_state.evaluation:
    eval_col, enhance_col = st.columns(2)

    with eval_col:
        with st.container(border=True):
            st.header("Evaluation Results")
            st.write("Here is Rocco's evaluation of your current description:")
            st.json(st.session_state.evaluation.model_dump())

    with enhance_col:
        with st.container(border=True):
            st.header("Provide Context to Enhance")
            st.write("Upload documents, add written feedback, or both.")

            # --- Document Upload ---
            st.subheader("Add Context from Documents")
            uploaded_files = st.file_uploader("Upload relevant papers or manuscripts (PDF).", type="pdf", accept_multiple_files=True)

            if uploaded_files:
                uploaded_file_names = sorted([file.name for file in uploaded_files])
                
                if st.session_state.processed_files != uploaded_file_names:
                    with st.spinner(f"Processing {len(uploaded_file_names)} file(s)..."):
                        session_id = get_session_id()
                        temp_dir = Path(f"temp_{session_id}")
                        temp_dir.mkdir(exist_ok=True)
                        
                        pdf_paths = []
                        for uploaded_file in uploaded_files:
                            pdf_path = temp_dir / uploaded_file.name
                            with open(pdf_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            pdf_paths.append(pdf_path)

                        vector_store_manager = VectorStoreManager(embedder)
                        chunks = ingestor.ingest(str(pdf_paths))
                        vector_store_manager.create_from_documents(chunks)
                        
                        st.session_state.vector_store_manager = vector_store_manager
                        editor.vector_store_manager = vector_store_manager
                        st.session_state.processed_files = uploaded_file_names
                        st.success(f"File(s) processed and ready.")
                else:
                    st.info("File(s) already processed and ready.")

            # --- User Feedback ---
            st.subheader("Add Written Feedback")
            user_feedback = st.text_area("Provide feedback or supplemental info based on the evaluation.", key="user_feedback_input")
            st.session_state.user_feedback = user_feedback

            # --- Enhance Button and Logic ---
            disable_enhance = not st.session_state.evaluation or (not st.session_state.vector_store_manager and not st.session_state.user_feedback)
            
            if st.button("✨ Enhance with Rocco", use_container_width=True, disabled=disable_enhance):
                with st.spinner("Rocco is enhancing your description..."):
                    use_rag = st.session_state.vector_store_manager is not None
                    
                    enhanced_description_obj = editor.enhance(
                        draft_text=st.session_state.description_text,
                        draft_evaluation=st.session_state.evaluation,
                        user_feedback=st.session_state.user_feedback if st.session_state.user_feedback else None,
                        retrieve_context=use_rag,
                    )
                    st.session_state.original_description = st.session_state.description_text
                    st.session_state.enhanced_description = enhanced_description_obj.suggested_text
                    st.session_state.user_feedback = "" # Clear feedback
                    st.rerun()

            if disable_enhance:
                if not st.session_state.vector_store_manager and not st.session_state.user_feedback:
                    st.warning("Provide context to enable enhancement.")

elif not st.session_state.enhanced_description:
    st.info("Click 'Evaluate Description' to get started.")