"""
Shared conversation manager for the General Assistant.

Top-level orchestrator for all intents from both interns. Uses a LangGraph
ReAct agent with in-memory checkpointing for per-session conversation history.

Intern A's tools (graph search, Cypher QA) and Intern B's tools (educational
context, workflow guidance, literature search) are registered via
tools.build_langchain_tools().
"""

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.assistant.tools import build_langchain_tools

SYSTEM_PROMPT = (
    "You are Rocco, an expert research assistant for the Digital Porous Media Portal. "
    "Answer the user's question directly. When you call a tool, relay its response to the user "
    "verbatim or with light formatting — do not summarize, evaluate, or comment on the tool's "
    "output. For portal-specific questions (datasets, statistics) use your tools and report "
    "only what they return. For general domain knowledge, answer from your expertise."
)


class ConversationManager:
    """
    Wraps a LangGraph ReAct agent with per-session memory.

    Usage:
        manager = ConversationManager()
        response = manager.chat("Find sandstone datasets with porosity > 0.2", session_id="abc123")
    """

    def __init__(self):
        from src.assistant.llm import get_chat_model

        self._agent = create_react_agent(
            get_chat_model(),
            build_langchain_tools(),
            prompt=SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
        )

    def chat(self, user_input: str, session_id: str) -> str:
        """
        Send a message and get a response.

        Args:
            user_input: The user's message.
            session_id: Unique session identifier (use Streamlit's session ID for per-tab isolation).

        Returns:
            The assistant's response as a string.
        """
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": session_id}},
        )
        return result["messages"][-1].content
