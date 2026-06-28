"""
Streamlit UI for the General Assistant tab.

Integrated into rocco_ui.py as a new tab during Week 6 (Bernie owns that integration).
All session state keys are prefixed with "assistant_" to avoid collisions with the
curator tab keys (which Bernie prefixes with "curator_" in the same Week 6 pass).
"""

import streamlit as st

from src.assistant.conversation_manager import ConversationManager

_WELCOME = (
    "Hi, I'm Rocco! I can help you find porous media datasets, "
    "answer questions about digital rock physics, and guide you "
    "through portal workflows. How can I help?"
)


def render_assistant_tab() -> None:
    """Render the General Assistant chat interface."""

    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [{"role": "assistant", "content": _WELCOME}]
    if "assistant_manager" not in st.session_state:
        st.session_state.assistant_manager = ConversationManager()

    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask about datasets, workflows, or porous media..."):
        # Render user message immediately before blocking on the LLM call.
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.assistant_messages.append({"role": "user", "content": question})

        # history = all prior completed turns, skipping the static welcome message.
        # Tool-call internals are never stored here, so the backend only sees clean
        # user/assistant pairs — avoids BadRequestError on replay.
        prior_history = st.session_state.assistant_messages[1:-1]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant_manager.chat(question, history=prior_history)
            st.markdown(response)

        st.session_state.assistant_messages.append({"role": "assistant", "content": response})
