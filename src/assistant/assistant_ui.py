"""
Streamlit UI for the General Assistant tab.

Integrated into rocco_ui.py as a new tab during Week 6 (Bernie owns that integration).
All session state keys are prefixed with "assistant_" to avoid collisions with the
curator tab keys (which Bernie prefixes with "curator_" in the same Week 6 pass).
"""

import re
import streamlit as st

from src.assistant.conversation_manager import ConversationManager

_DOI_RE = re.compile(r'DOI:\s*(10\.[^\s)]+)')

_SOURCE_LABEL_RE = re.compile(
    r'\[(graph match|hybrid match|semantic match|component match|paper match|semantic scholar|cypher match)\]',
    re.IGNORECASE,
)

_LABEL_COLORS = {
    "graph match":      "#1f77b4",
    "hybrid match":     "#1f77b4",
    "semantic match":   "#2ca02c",
    "component match":  "#17becf",
    "paper match":      "#9467bd",
    "semantic scholar": "#9467bd",
    "cypher match":     "#ff7f0e",
}


def _badge(label: str) -> str:
    """Render a source label as a small colored HTML badge."""
    color = _LABEL_COLORS.get(label.lower(), "#888888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:0.75em;font-weight:bold;'
        f'margin-right:4px">{label}</span>'
    )


def _labelify_sources(text: str) -> str:
    """Convert [source label] patterns in response text into colored HTML badges."""
    return _SOURCE_LABEL_RE.sub(lambda m: _badge(m.group(1)), text)


def _linkify_dois(text: str) -> str:
    """Convert 'DOI: 10.xxxx/yyyy' patterns in LLM responses to markdown links."""
    return _DOI_RE.sub(lambda m: f'[DOI: {m.group(1)}](https://doi.org/{m.group(1)})', text)


def _normalize_latex_delimiters(text: str) -> str:
    """Convert \\(...\\) / \\[...\\] delimiters to $...$ / $$...$$ so KaTeX renders them.

    Some models (e.g. Llama-4-Maverick via SambaNova/TACC) emit \\(...\\)-style
    delimiters instead of the $...$ convention Streamlit's KaTeX renderer expects.
    """
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    return text


def _render_response(text: str) -> str:
    """Apply all response formatting: source badges first, then DOI links, then LaTeX normalization."""
    return _normalize_latex_delimiters(_linkify_dois(_labelify_sources(text)))


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
            st.markdown(_render_response(message["content"]), unsafe_allow_html=True)

    if question := st.chat_input("Ask about datasets, workflows, or porous media..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.assistant_messages.append({"role": "user", "content": question})

        prior_history = st.session_state.assistant_messages[1:-1]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant_manager.chat(question, history=prior_history)
            st.markdown(_render_response(response), unsafe_allow_html=True)

        st.session_state.assistant_messages.append({"role": "assistant", "content": response})
