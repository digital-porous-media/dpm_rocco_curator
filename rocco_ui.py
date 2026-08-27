import streamlit as st
from dotenv import load_dotenv
import os
import json
from pathlib import Path
import uuid

from src.llm.client import RoccoClient
from src.evaluator.evaluator import DescriptionEvaluator
from src.editor.editor import DescriptionEditor
from src.ingestor.embedder import DocumentEmbedder
from src.ingestor.document_ingestor import DocumentIngestor
from src.retriever.retriever import VectorStoreManager
from src.llm.content_screener import ContentScreener
from src.llm.embeddings import get_embeddings
from src.assistant.assistant_ui import render_assistant_tab

# Suppress transformers library warnings about optional vision model imports
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Page Config ---
st.set_page_config(page_title="Rocco - DPM Research Assistant", layout="wide")
st.title("Rocco - Your Digital Porous Media AI Assistant")

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


def _history_pairs(history):
    """Group a flat conversation history into (user_turn, assistant_turn) pairs.

    An assistant turn with no preceding user turn (the initial enhancement, which runs
    on uploaded documents alone) pairs with None.
    """
    pairs = []
    i = 0
    while i < len(history):
        if history[i]["role"] == "user" and i + 1 < len(history):
            pairs.append((history[i], history[i + 1]))
            i += 2
        else:
            pairs.append((None, history[i]))
            i += 1
    return pairs


def _selected_prior_turns():
    """The prior turns the user has left checked in the "Manage Context" panel, as the
    history_override list for the next enhance() call.

    Read from session_state rather than accumulated while the panel renders: the panel
    is laid out full-width *below* the enhance controls, so on the rerun that performs
    an enhancement its widgets have not executed yet. Building the list during that
    later render meant enhance() always received an empty override — and since an empty
    list is not None, build_prompt() read it as "an explicit empty history" and dropped
    the editor's own conversation_history too, silently discarding all multi-turn
    context. Streamlit retains each widget's value in session_state under its key across
    reruns, so the last-rendered selection is available here.

    An empty return is meaningful and must stay distinct from None: it means the user
    unchecked every turn, and build_prompt() is expected to suppress history in that case.
    """
    selected = []
    for turn_idx, (user_turn, asst_turn) in enumerate(
        _history_pairs(st.session_state.conversation_history)
    ):
        if not st.session_state.get(f"ctx_include_{turn_idx}", True):
            continue
        if user_turn:
            selected.append({
                "role": "user",
                "content": st.session_state.context_manager_edits.get(
                    turn_idx, user_turn["content"]
                ),
            })
        if asst_turn:
            selected.append({
                "role": "assistant",
                "content": asst_turn["content"],
                "rationale": asst_turn.get("rationale", ""),
            })
    return selected


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
    embedder = DocumentEmbedder(embeddings=get_embeddings())
    ingestor = DocumentIngestor(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""]
    )
    screener = ContentScreener(model=client)
    return grader, editor, embedder, ingestor, screener


if api_key or provider == "ollama":
    grader, editor, embedder, ingestor, screener = load_resources()
    if st.session_state.vector_store_manager:
        editor.vector_store_manager = st.session_state.vector_store_manager
else:
    st.error(
        "LLM API Key not found. Please set LLM_API_KEY in your environment or .env file. "
        "See .env.example for configuration details and supported LLM providers."
    )
    st.stop()


def render_curator_tab():
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
                    {
                        "edited_enhanced_description": st.session_state.enhanced_desc_editable
                    }
                ),
            )
            st.session_state.edited_enhanced_description = edited_text

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
                st.session_state.description_text = (
                    st.session_state.original_description
                )
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
                            with st.expander(
                                f"Citation {i}: {citation.statement[:50]}..."
                            ):
                                st.write(f"**Statement:** {citation.statement}")
                                st.write(f"**Source:** {citation.source}")
                                if citation.doc_title:
                                    loc_parts = [f"*{citation.doc_title}*"]
                                    if citation.page is not None:
                                        loc_parts.append(f"p. {citation.page + 1}")
                                    if citation.chunk_index is not None:
                                        loc_parts.append(
                                            f"chunk {citation.chunk_index}"
                                        )
                                    st.write(f"**Document:** {', '.join(loc_parts)}")
                                st.write(f"**Quote:** _{citation.quote}_")
                    else:
                        st.info("No citations available")
                else:
                    st.info("Citations not available")

        st.divider()

    # Display evaluation results and enhancement tools if an evaluation is present
    if st.session_state.evaluation:
        selected_history = _selected_prior_turns()

        eval_col, enhance_col = st.columns(2)

        with eval_col:
            with st.container(border=True):
                st.header("Evaluation Results")
                st.write("Here is Rocco's evaluation of your current description:")
                # Display evaluation results formatted for Streamlit
                st.metric(
                    "Total Score", f"{st.session_state.evaluation.total_score}/10"
                )
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
                    and st.session_state.get("screening_result", {}).get(
                        "recommendation"
                    )
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
                                st.session_state.conversation_history.append(
                                    {
                                        "role": "user",
                                        "content": st.session_state.user_feedback,
                                    }
                                )
                            st.session_state.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": enhanced_description_obj.suggested_text,
                                    "rationale": enhanced_description_obj.rationale,
                                    "context_used": enhanced_description_obj.context_used,
                                }
                            )
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
        if (
            st.session_state.conversation_history
            and not st.session_state.enhanced_description
        ):
            st.divider()
            with st.expander("📋 Manage Context (Prior Turns)", expanded=False):
                st.caption(
                    "Select which prior turns to include in the next enhancement. Uncheck to exclude, edit feedback inline."
                )
                pairs = _history_pairs(st.session_state.conversation_history)

                # Clear history button
                col_clear, col_space = st.columns([1, 4])
                with col_clear:
                    if st.button("Clear history", key="ctx_clear", type="secondary"):
                        st.session_state.conversation_history = []
                        st.session_state.context_manager_edits = {}
                        # Drop the per-turn checkbox state too, or a turn unchecked
                        # before the clear would still read as excluded once history
                        # grows back into that index.
                        for key in [
                            k for k in st.session_state if k.startswith("ctx_include_")
                        ]:
                            del st.session_state[key]
                        st.rerun()

                # The checkbox values are read back in _selected_prior_turns() on the
                # next rerun, via their session_state keys — nothing is accumulated here.
                for turn_idx, (user_turn, asst_turn) in enumerate(pairs):
                    col_check, col_card = st.columns([0.05, 0.95])
                    with col_check:
                        st.checkbox(
                            "",
                            value=True,
                            key=f"ctx_include_{turn_idx}",
                            label_visibility="collapsed",
                        )
                    with col_card:
                        label = f"Turn {turn_idx + 1}"
                        if user_turn:
                            label += f': "{user_turn["content"][:60]}..."'
                        with st.expander(label, expanded=False):
                            if user_turn:
                                st.markdown("**Feedback given:**")
                                edited = st.text_area(
                                    "Edit feedback",
                                    value=st.session_state.context_manager_edits.get(
                                        turn_idx, user_turn["content"]
                                    ),
                                    key=f"ctx_edit_{turn_idx}",
                                    height=80,
                                    label_visibility="collapsed",
                                )
                                st.session_state.context_manager_edits[turn_idx] = (
                                    edited
                                )
                            if asst_turn:
                                # Show context chunks used
                                chunks = asst_turn.get("context_used", [])
                                if chunks:
                                    st.markdown("**Documents retrieved:**")
                                    for chunk in chunks:
                                        title = chunk.get("doc_title", "unknown")
                                        page = chunk.get("page")
                                        loc = f"*{title}*" + (
                                            f", p. {page + 1}"
                                            if page is not None
                                            else ""
                                        )
                                        st.caption(
                                            f"↳ {loc} — {chunk.get('snippet', '')[:80]}..."
                                        )
                                # Show result preview
                                snippet = asst_turn["content"][:200] + (
                                    "..." if len(asst_turn["content"]) > 200 else ""
                                )
                                st.markdown("**Result preview:**")
                                st.text(snippet)

    elif not st.session_state.enhanced_description:
        st.info("Click 'Evaluate Description' to get started.")


# --- Main App Layout ---
_PAGES = ["General Assistant", "Description Curator"]

if "page" not in st.session_state:
    st.session_state.page = _PAGES[0]

with st.sidebar:
    st.markdown("### Navigation")
    for _pg in _PAGES:
        if st.button(
            _pg,
            use_container_width=True,
            type="primary" if st.session_state.page == _pg else "secondary",
        ):
            st.session_state.page = _pg
            st.rerun()

if st.session_state.page == "General Assistant":
    render_assistant_tab()
else:
    render_curator_tab()
