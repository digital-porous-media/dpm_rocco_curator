"""
Streamlit UI for the General Assistant tab.

Integrated into rocco_ui.py as a new tab during Week 6 (Bernie owns that integration).
All session state keys are prefixed with "assistant_" to avoid collisions with the
curator tab keys (which Bernie prefixes with "curator_" in the same Week 6 pass).
"""

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.assistant.conversation_manager import ConversationManager


def _get_session_id() -> str:
    return get_script_run_ctx().session_id


def _write_message(role: str, content: str, save: bool = True) -> None:
    if save:
        st.session_state.assistant_messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)


def render_assistant_tab() -> None:
    """Render the General Assistant chat interface."""

    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I'm Rocco! I can help you find porous media datasets, "
                    "answer questions about digital rock physics, and guide you "
                    "through portal workflows. How can I help?"
                ),
            }
        ]
    if "assistant_manager" not in st.session_state:
        st.session_state.assistant_manager = ConversationManager()

    for message in st.session_state.assistant_messages:
        _write_message(message["role"], message["content"], save=False)

    if question := st.chat_input("Ask about datasets, workflows, or porous media..."):
        _write_message("user", question)
        with st.spinner("Thinking..."):
            response = st.session_state.assistant_manager.chat(
                question, session_id=_get_session_id()
            )
        _write_message("assistant", response)
