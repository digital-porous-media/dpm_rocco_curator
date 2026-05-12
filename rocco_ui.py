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
from src.llm.content_screener import ContentScreener

# Suppress transformers library warnings about optional vision model imports
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Constants and Page Config ---
ROCCO_AVATAR = "assets/rocco_avatar.jpg"
USER_AVATAR = "assets/user_avatar.jpg"

st.set_page_config(page_title="Rocco - DPM Curator", layout="wide")
st.title("Rocco - Your Digital Porous Media AI Curator")

# --- Display LLM Configuration ---
load_dotenv()
api_key = os.getenv("LLM_API_KEY")
api_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL", "gpt-4o-mini")
provider = os.getenv("LLM_PROVIDER", "").lower()

st.write(f"Using **{provider.upper() if provider else 'Custom'}** — {model}")


# --- Helper Functions ---
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


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
if "enhanced_description_obj" not in st.session_state:
    st.session_state.enhanced_description_obj = None
if "edited_enhanced_description" not in st.session_state:
    st.session_state.edited_enhanced_description = None
if "skip_screening" not in st.session_state:
    st.session_state.skip_screening = False
if "pending_enhancement" not in st.session_state:
    st.session_state.pending_enhancement = False
if "screening_result" not in st.session_state:
    st.session_state.screening_result = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "context_manager_edits" not in st.session_state:
    st.session_state.context_manager_edits = {}


# --- Initialization ---
@st.cache_resource
def load_resources():
    """Load rubric, examples, and initialize clients."""
    with open("src/evaluator/rubric.json", "r") as f:
        rubric = json.load(f)
    with open("src/evaluator/examples_v3.json", "r") as f:
        examples = json.load(f)

    client = RoccoClient(api_url=api_url, api_key=api_key, model=model)
    grader = DescriptionEvaluator(model=client, rubric=rubric, examples=examples)
    editor = DescriptionEditor(
        model=client,
        rubric=rubric,
        vector_store_manager=st.session_state.vector_store_manager,
        top_k_context=5,
    )
    embedder = DocumentEmbedder(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ingestor = DocumentIngestor(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""]
    )
    screener = ContentScreener(model=client)
    return rubric, examples, client, grader, editor, embedder, ingestor, screener


if api_key or provider == "ollama":
    rubric, examples, client, grader, editor, embedder, ingestor, screener = (
        load_resources()
    )
    if st.session_state.vector_store_manager:
        editor.vector_store_manager = st.session_state.vector_store_manager
else:
    st.error(
        "LLM API Key not found. Please set LLM_API_KEY in your environment or .env file. "
        "See .env.example for configuration details and supported LLM providers."
    )
    st.stop()

# --- Main App Layout ---
st.header("Dataset Description")
st.info(
    "Enter your dataset description below. You can then use the tools to evaluate, edit, and refine it."
)
description_text = st.text_area(
    "Enter your dataset description here:",
    value=st.session_state.description_text,
    height=250,
    label_visibility="collapsed",
)
st.session_state.description_text = description_text

evaluate_button = st.button("Evaluate Description", use_container_width=True)

# --- Workflow Logic ---
if evaluate_button and description_text:

    # Clear previous enhancement state when starting a new evaluation
    st.session_state.original_description = None
    st.session_state.enhanced_description = None
    st.session_state.enhanced_description_obj = None
    st.session_state.edited_enhanced_description = None
    st.session_state.user_feedback = ""
    st.session_state.processed_files = None
    st.session_state.vector_store_manager = None

    with st.spinner("Rocco is evaluating your description..."):
        evaluation = grader.evaluate(draft_text=description_text)
        st.session_state.evaluation = evaluation
        st.rerun()

st.divider()

# Display comparison view if an enhanced description is available
if st.session_state.enhanced_description:
    st.header("Compare and Update Description")

    col1, col2 = st.columns(2)
    with col1:
        st.info("Your Original Description")
        st.text_area(
            "Original",
            value=st.session_state.original_description,
            height=300,
            key="original_desc_readonly",
            disabled=True,
        )
    with col2:
        st.info("Rocco's Suggested Description")
        edited_text = st.text_area(
            "Enhanced",
            value=st.session_state.edited_enhanced_description
            or st.session_state.enhanced_description,
            height=300,
            key="enhanced_desc_editable",
            disabled=False,
            help="You can edit Rocco's suggested description before finalizing.",
            on_change=lambda: st.session_state.update(
                {"edited_enhanced_description": st.session_state.enhanced_desc_editable}
            ),
        )
        st.session_state.edited_enhanced_description = edited_text
        # st.text_area("Enhanced", value=st.session_state.enhanced_description, height=300, key="enhanced_desc_readonly", disabled=True)

    # --- Action buttons for the new description ---
    accept_col, reject_col = st.columns(2)
    with accept_col:
        if st.button("✅ Adopt Rocco's Version", use_container_width=True):
            st.session_state.description_text = (
                st.session_state.edited_enhanced_description
                or st.session_state.enhanced_description
            )
            st.session_state.original_description = None
            st.session_state.enhanced_description = None
            st.session_state.enhanced_description_obj = None
            st.session_state.edited_enhanced_description = None
            st.session_state.evaluation = None
            st.session_state.user_feedback = ""
            st.rerun()
    with reject_col:
        if st.button("❌ Keep Original Version", use_container_width=True):
            st.session_state.description_text = st.session_state.original_description
            st.session_state.original_description = None
            st.session_state.enhanced_description = None
            st.session_state.enhanced_description_obj = None
            st.session_state.edited_enhanced_description = None
            st.session_state.evaluation = None
            st.session_state.user_feedback = ""

            st.rerun()
    # --- Display Rationale and Citations ---
    with st.expander("View Enhancement Details", expanded=True):
        col_rationale, col_citations = st.columns(2)

        with col_rationale:
            st.subheader("Rationale for Changes")
            if (
                hasattr(st.session_state, "enhanced_description_obj")
                and st.session_state.enhanced_description_obj
            ):
                st.write(st.session_state.enhanced_description_obj.rationale)
            else:
                st.info("Rationale not available")

        with col_citations:
            st.subheader("Citations")
            if (
                hasattr(st.session_state, "enhanced_description_obj")
                and st.session_state.enhanced_description_obj
            ):
                citations = st.session_state.enhanced_description_obj.citation
                if citations:
                    for i, citation in enumerate(citations, 1):
                        with st.expander(f"Citation {i}: {citation.statement[:50]}..."):
                            st.write(f"**Statement:** {citation.statement}")
                            st.write(f"**Source:** {citation.source}")
                            if citation.doc_title:
                                loc_parts = [f"*{citation.doc_title}*"]
                                if citation.page is not None:
                                    loc_parts.append(f"p. {citation.page + 1}")
                                if citation.chunk_index is not None:
                                    loc_parts.append(f"chunk {citation.chunk_index}")
                                st.write(f"**Document:** {', '.join(loc_parts)}")
                            st.write(f"**Quote:** _{citation.quote}_")
                else:
                    st.info("No citations available")
            else:
                st.info("Citations not available")

    st.divider()

# Display evaluation results and enhancement tools if an evaluation is present
if st.session_state.evaluation:
    selected_history = []

    eval_col, enhance_col = st.columns(2)

    with eval_col:
        with st.container(border=True):
            st.header("Evaluation Results")
            st.write("Here is Rocco's evaluation of your current description:")
            # Display evaluation results formatted for Streamlit
            st.metric("Total Score", f"{st.session_state.evaluation.total_score}/10")
            st.subheader("Rubric Breakdown")
            for item in st.session_state.evaluation.rubric_breakdown:
                if item.score < 1.0:
                    with st.expander(
                        f"⚠️ **{item.criterion}** - Score: {item.score}/1.0"
                    ):
                        st.write(item.explanation)
                else:
                    with st.expander(
                        f"✅ **{item.criterion}** - Score: {item.score}/1.0"
                    ):
                        st.write(item.explanation)
                # with st.expander(f"**{item.criterion}** - Score: {item.score}/1.0"):
                #     st.write(item.explanation)
            # grader.print_evaluation_result(st.session_state.evaluation)
            # st.json(st.session_state.evaluation.model_dump())

    with enhance_col:
        with st.container(border=True):
            st.header("Provide Context to Enhance")
            st.write(
                "Upload documents and/or provide Rocco with supplemental information or specific feedback."
            )

            # --- Document Upload ---
            st.subheader("Add Context from Documents")
            uploaded_files = st.file_uploader(
                "Upload relevant papers or manuscripts (PDF).",
                type="pdf",
                accept_multiple_files=True,
            )

            if uploaded_files:
                uploaded_file_names = sorted([file.name for file in uploaded_files])
                if st.session_state.processed_files != uploaded_file_names:
                    with st.spinner(
                        f"Processing {len(uploaded_file_names)} file(s)... This could take a few minutes."
                    ):
                        session_id = get_session_id()
                        temp_dir = Path(f"temp_{session_id}")
                        temp_dir.mkdir(exist_ok=True)

                        pdf_paths = []
                        for uploaded_file in uploaded_files:
                            pdf_path = temp_dir / uploaded_file.name
                            with open(pdf_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            pdf_paths.append(str(pdf_path))

                        vector_store_manager = VectorStoreManager(embedder)
                        chunks = ingestor.ingest(pdf_paths)
                        vector_store_manager.create_from_documents(chunks)

                        st.session_state.vector_store_manager = vector_store_manager
                        editor.vector_store_manager = vector_store_manager
                        st.session_state.processed_files = uploaded_file_names
                        st.success(f"File(s) processed and ready.")
                else:
                    st.info("File(s) already processed and ready.")

            # --- User Feedback ---
            st.subheader("Add Written Feedback")
            user_feedback = st.text_area(
                "Provide feedback or supplemental info based on the evaluation.",
                key="user_feedback_input",
                on_change=lambda: st.session_state.update(
                    {"user_feedback": st.session_state.user_feedback_input}
                ),
            )

            st.session_state.user_feedback = user_feedback

            # --- Enhance Button and Logic ---
            disable_enhance = not st.session_state.evaluation or (
                not st.session_state.vector_store_manager
                and not st.session_state.user_feedback
            )

            if st.button(
                "✨ Enhance with Rocco",
                use_container_width=True,
                disabled=disable_enhance,
            ):
                if st.session_state.user_feedback:
                    with st.spinner("Screening your feedback..."):
                        screening_result = screener.screen_user_content(
                            st.session_state.user_feedback,
                            context=st.session_state.description_text,
                        )

                    if screening_result["recommendation"] == "reject":
                        st.error(
                            f"❌ Feedback rejected. Issues found: {', '.join(screening_result['issues'])}"
                        )
                        st.stop()
                    elif screening_result["recommendation"] == "flag_for_review":
                        st.session_state.screening_result = screening_result
                        st.session_state.pending_enhancement = True
                        st.rerun()
                    elif screening_result["recommendation"] == "accept":
                        st.session_state.skip_screening = True
                        st.session_state.pending_enhancement = False
                        st.rerun()
                else:
                    # No user feedback, proceed directly with enhancement
                    st.session_state.skip_screening = True
                    st.session_state.pending_enhancement = False
                    st.rerun()

            # Handle flagged feedback review
            if (
                st.session_state.get("pending_enhancement")
                and st.session_state.get("screening_result", {}).get("recommendation")
                == "flag_for_review"
            ):
                with st.expander("⚠️ Feedback flagged for review", expanded=True):
                    st.write(
                        f"**Issues:** {', '.join(st.session_state.screening_result['issues'])}"
                    )
                    col_continue, col_cancel = st.columns(2)
                    with col_continue:
                        if st.button(
                            "Continue anyway",
                            key="continue_flagged",
                            use_container_width=True,
                        ):
                            st.session_state.skip_screening = True
                            st.session_state.pending_enhancement = False
                            st.rerun()
                    with col_cancel:
                        if st.button(
                            "Cancel", key="cancel_flagged", use_container_width=True
                        ):
                            st.session_state.pending_enhancement = False
                            st.session_state.screening_result = None
                            st.rerun()

            # Proceed with enhancement if screening passed or was skipped
            if st.session_state.skip_screening:
                if (
                    st.session_state.user_feedback
                    or st.session_state.vector_store_manager
                ):
                    with st.spinner("Rocco is refining your description..."):
                        use_rag = st.session_state.vector_store_manager is not None
                        editor.use_rag = use_rag
                        draft_text = (
                            st.session_state.edited_enhanced_description
                            or st.session_state.enhanced_description
                            or st.session_state.description_text
                        )
                        enhanced_description_obj = editor.enhance(
                            draft_text=draft_text,
                            draft_evaluation=st.session_state.evaluation,
                            user_feedback=(
                                st.session_state.user_feedback
                                if st.session_state.user_feedback
                                else None
                            ),
                            retrieve_context=use_rag,
                            history_override=selected_history,
                        )
                        st.session_state.original_description = (
                            st.session_state.description_text
                        )
                        st.session_state.enhanced_description = (
                            enhanced_description_obj.suggested_text
                        )
                        st.session_state.enhanced_description_obj = (
                            enhanced_description_obj
                        )
                        st.session_state.edited_enhanced_description = (
                            None  # Reset edits
                        )
                        # Append the new turn to conversation history
                        if st.session_state.user_feedback:
                            st.session_state.conversation_history.append({
                                "role": "user",
                                "content": st.session_state.user_feedback,
                            })
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": enhanced_description_obj.suggested_text,
                            "rationale": enhanced_description_obj.rationale,
                            "context_used": enhanced_description_obj.context_used,
                        })
                        st.session_state.user_feedback = ""  # Clear feedback
                        st.session_state.skip_screening = False  # Reset flagging
                        st.session_state.pending_enhancement = False
                        st.session_state.screening_result = None
                        st.rerun()

            if disable_enhance:
                if (
                    not st.session_state.vector_store_manager
                    and not st.session_state.user_feedback
                ):
                    st.warning("Provide context to enable enhancement.")

    # --- Full-width Context Manager (outside columns, below eval+enhance) ---
    if st.session_state.conversation_history and not st.session_state.enhanced_description:
        st.divider()
        with st.expander("📋 Manage Context (Prior Turns)", expanded=False):
            st.caption("Select which prior turns to include in the next enhancement. Uncheck to exclude, edit feedback inline.")
            history = st.session_state.conversation_history
            # Group into (user_turn, assistant_turn) pairs
            pairs = []
            i = 0
            while i < len(history):
                if history[i]["role"] == "user" and i + 1 < len(history):
                    pairs.append((history[i], history[i + 1]))
                    i += 2
                else:
                    pairs.append((None, history[i]))
                    i += 1

            # Clear history button
            col_clear, col_space = st.columns([1, 4])
            with col_clear:
                if st.button("Clear history", key="ctx_clear", type="secondary"):
                    st.session_state.conversation_history = []
                    st.session_state.context_manager_edits = {}
                    st.rerun()

            for turn_idx, (user_turn, asst_turn) in enumerate(pairs):
                col_check, col_card = st.columns([0.05, 0.95])
                with col_check:
                    include = st.checkbox("", value=True, key=f"ctx_include_{turn_idx}",
                                          label_visibility="collapsed")
                with col_card:
                    label = f"Turn {turn_idx + 1}"
                    if user_turn:
                        label += f': "{user_turn["content"][:60]}..."'
                    with st.expander(label, expanded=False):
                        if user_turn:
                            st.markdown("**Feedback given:**")
                            edited = st.text_area(
                                "Edit feedback",
                                value=st.session_state.context_manager_edits.get(turn_idx, user_turn["content"]),
                                key=f"ctx_edit_{turn_idx}",
                                height=80,
                                label_visibility="collapsed",
                            )
                            st.session_state.context_manager_edits[turn_idx] = edited
                        if asst_turn:
                            # Show context chunks used
                            chunks = asst_turn.get("context_used", [])
                            if chunks:
                                st.markdown("**Documents retrieved:**")
                                for chunk in chunks:
                                    title = chunk.get("doc_title", "unknown")
                                    page = chunk.get("page")
                                    loc = f"*{title}*" + (f", p. {page + 1}" if page is not None else "")
                                    st.caption(f"↳ {loc} — {chunk.get('snippet', '')[:80]}...")
                            # Show result preview
                            snippet = asst_turn["content"][:200] + ("..." if len(asst_turn["content"]) > 200 else "")
                            st.markdown("**Result preview:**")
                            st.text(snippet)
                if include:
                    if user_turn:
                        selected_history.append({
                            "role": "user",
                            "content": st.session_state.context_manager_edits.get(turn_idx, user_turn["content"]),
                        })
                    if asst_turn:
                        selected_history.append({
                            "role": "assistant",
                            "content": asst_turn["content"],
                            "rationale": asst_turn.get("rationale", ""),
                        })

elif not st.session_state.enhanced_description:
    st.info("Click 'Evaluate Description' to get started.")
