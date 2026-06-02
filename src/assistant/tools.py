"""
Shared tool interface for the General Assistant.

All callable tools are defined here — both interns code to this interface.

Intern A owns (Week 2-3): search_datasets, get_dataset_details
Intern B owns (Week 3):   get_educational_context, get_workflow_guidance,
                          expand_query, search_literature
"""

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_graph_store = None


def _get_graph_store():
    global _graph_store
    if _graph_store is None:
        from src.assistant.graph_store import GraphStore
        _graph_store = GraphStore()
    return _graph_store


@tool
def search_datasets(query: str) -> str:
    """Find datasets by semantic similarity to a natural language query."""
    results = _get_graph_store().search(query)
    if not results:
        return "No datasets found matching that query."
    lines = []
    for r in results:
        meta = r.get("metadata", {})
        title = meta.get("title", "Unknown")
        doi = meta.get("doi", "")
        lines.append(f"[graph match] {title} (DOI: {doi})\n{r['text'][:300]}")
    return "\n\n".join(lines)


@tool
def get_dataset_details(question: str) -> str:
    """Answer structured questions about dataset properties using Cypher. Source: [cypher match]"""
    return _get_graph_store().cypher_qa(question)


@tool
def general_chat(query: str) -> str:
    """
    General porous media discussion not covered by dataset search or Cypher.
    Placeholder until educational.yaml (Intern B, Week 3) is wired up.
    """
    from src.assistant.llm import get_chat_model
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert researcher in porous media. Answer the user's question "
                   "thoroughly using your knowledge. If the question is about specific datasets "
                   "or portal content you don't have access to, say so explicitly."),
        ("human", "{input}"),
    ])
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"input": query})


# --- Intern B stubs (Week 3) ---

def get_educational_context(question: str) -> str:
    """Answer domain Q&A using domain_workflows.yaml and educational.yaml prompt."""
    # TODO (Intern B, Week 3): load domain_workflows.yaml, retrieve relevant context,
    # render educational.yaml prompt, call LLM
    raise NotImplementedError("get_educational_context not yet implemented")


def get_workflow_guidance(goal: str) -> str:
    """Return step-by-step DRP workflow guidance for a user goal."""
    # TODO (Intern B, Week 3): match goal against domain_workflows.yaml,
    # also check tutorials.yaml for portal URLs
    raise NotImplementedError("get_workflow_guidance not yet implemented")


def expand_query(query: str) -> dict:
    """Expand a vague query into a richer search query with inferred filters."""
    # TODO (Intern B, Week 3): render query_expander.yaml prompt, call LLM, return dict
    raise NotImplementedError("expand_query not yet implemented")


def search_literature(query: str) -> str:
    """Search curated publication FAISS first; fall back to Semantic Scholar if confidence < 0.75."""
    # TODO (Intern B, Week 3): wire up PublicationCorpus + LiteratureSearch with routing logic
    raise NotImplementedError("search_literature not yet implemented")


def build_langchain_tools() -> list:
    """Return the list of LangChain Tool objects for the ConversationManager agent."""
    return [general_chat, search_datasets, get_dataset_details]
