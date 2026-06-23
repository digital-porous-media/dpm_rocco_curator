"""
Shared tool interface for the General Assistant.

All callable tools are defined here — both interns code to this interface.

Intern A owns (Week 2-3): search_datasets, get_dataset_details
Bernie owns:              get_educational_context, get_workflow_guidance,
                          expand_query, search_literature
"""

import json
import logging
import os
from pathlib import Path

import yaml
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

_graph_store = None
_workflows_data = None
_tutorials_data = None
_lit_search = None


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

def _get_graph_store():
    global _graph_store
    if _graph_store is None:
        from src.assistant.graph_store import GraphStore
        _graph_store = GraphStore()
    return _graph_store


def _get_lit_search():
    global _lit_search
    if _lit_search is None:
        from src.assistant.literature_search import LiteratureSearch
        _lit_search = LiteratureSearch()
    return _lit_search


def _load_workflows() -> dict:
    """Load and cache data/domain_workflows.yaml."""
    global _workflows_data
    if _workflows_data is None:
        path = Path(__file__).parents[2] / "data" / "domain_workflows.yaml"
        with open(path, "r") as f:
            _workflows_data = yaml.safe_load(f)
    return _workflows_data


def _load_tutorials() -> dict:
    """Load and cache data/tutorials.yaml."""
    global _tutorials_data
    if _tutorials_data is None:
        path = Path(__file__).parents[2] / "data" / "tutorials.yaml"
        with open(path, "r") as f:
            _tutorials_data = yaml.safe_load(f)
    return _tutorials_data


# ---------------------------------------------------------------------------
# Keyword matching helpers
# ---------------------------------------------------------------------------

def _match_workflows(query: str, max_results: int = 3) -> list[dict]:
    """Return up to max_results workflows whose keywords overlap with query."""
    data = _load_workflows()
    query_lower = query.lower()
    scored = []
    for wf in data.get("workflows", []):
        keywords = [str(k).lower() for k in wf.get("keywords", [])]
        hits = sum(1 for kw in keywords if kw in query_lower)
        if hits > 0:
            scored.append((hits, wf))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [wf for _, wf in scored[:max_results]]


def _match_tutorials(query: str) -> list[dict]:
    """Return tutorials whose keywords overlap with query."""
    data = _load_tutorials()
    query_lower = query.lower()
    matched = []
    for t in data.get("tutorials", []):
        keywords = [str(k).lower() for k in t.get("keywords", [])]
        if any(kw in query_lower for kw in keywords):
            matched.append(t)
    return matched


def _global_practices_context(query: str) -> str:
    """Return relevant global_best_practices sections as a string."""
    data = _load_workflows()
    gbp = data.get("global_best_practices", {})
    if not gbp:
        return ""

    # Map section names to trigger terms
    triggers = {
        "representativeness": ["rev", "representative elementary volume", "sub-volume"],
        "resolution": ["resolution", "voxel", "pixel"],
        "segmentation_uncertainty": ["segment", "threshold", "binarize"],
        "boundary_conditions": ["boundary", "inlet", "outlet", "buffer"],
        "reproducibility": ["reproducib", "publish", "provenance"],
        "connectivity": ["connectivity", "percolat", "connected pore", "effective porosity"],
    }

    query_lower = query.lower()
    lines = []
    for section, terms in triggers.items():
        if section in gbp and any(t in query_lower for t in terms):
            lines.append(f"Global best practices — {section}:")
            for item in gbp[section]:
                lines.append(f"  - {item}")
    return "\n".join(lines)


def _workflow_context_str(workflows: list[dict], tutorials: list[dict]) -> str:
    """Assemble a readable context block from matched workflows and tutorials."""
    parts = []

    for wf in workflows:
        section = [f"## Workflow: {wf.get('name', wf['id'])}"]
        section.append(wf.get("description", "").strip())

        steps = wf.get("steps", [])
        if steps:
            section.append("Steps:")
            for i, s in enumerate(steps, 1):
                section.append(f"  {i}. {s.strip()}")

        software = wf.get("software", [])
        if software:
            section.append("Software: " + ", ".join(software))

        best = wf.get("best_practices", [])
        if best:
            section.append("Best practices:")
            for b in best:
                section.append(f"  - {b.strip()}")

        examples = [str(e) for e in wf.get("example_datasets", []) if e]
        if examples:
            section.append("Example datasets: " + ", ".join(examples))

        parts.append("\n".join(section))

    if tutorials:
        access = _load_tutorials().get("access_instructions", {})
        steps_txt = " → ".join(access.get("steps", []))
        tut_lines = ["## Portal Tutorials", f"Access: {steps_txt}"]
        for t in tutorials:
            tut_lines.append(f"  - Goal: {t['goal']}")
            tut_lines.append(f"    Notebook: {t['notebook']}")
        parts.append("\n".join(tut_lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset search tools (Intern A)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Educational and workflow tools (Bernie)
# ---------------------------------------------------------------------------

def expand_query(query: str) -> dict:
    """Expand a vague query into a richer search query with inferred filters.

    Not a LangChain tool — called internally by the agent before search.
    Returns dict with keys: expanded_query, inferred_filters, rationale.
    """
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    prompt = load_prompt("query_expander")
    system = prompt["system"]
    user = render(prompt["user"], query=query)

    try:
        raw = get_chat_model().send_prompt(user, context=system, params={"temperature": 0.2, "max_tokens": 400})
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("expand_query: JSON parse failed (%s); returning passthrough", e)
        return {"expanded_query": query, "inferred_filters": {}, "rationale": "Parse error"}


@tool
def get_workflow_guidance(goal: str) -> str:
    """Return step-by-step DRP workflow guidance for a user goal, with tutorial links."""
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    workflows = _match_workflows(goal, max_results=3)
    tutorials = _match_tutorials(goal)
    context = _workflow_context_str(workflows, tutorials)

    prompt = load_prompt("educational")
    system = render(prompt["system"], context=context)
    user = render(prompt["user"], question=goal)

    return get_chat_model().send_prompt(user, context=system, params={"temperature": 0.3, "max_tokens": 1000})


@tool
def get_educational_context(question: str) -> str:
    """Answer domain Q&A using domain_workflows.yaml, global best practices, and tutorials."""
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    workflows = _match_workflows(question, max_results=3)
    tutorials = _match_tutorials(question)
    workflow_ctx = _workflow_context_str(workflows, tutorials)
    global_ctx = _global_practices_context(question)

    parts = [p for p in [workflow_ctx, global_ctx] if p]
    context = "\n\n".join(parts)

    prompt = load_prompt("educational")
    system = render(prompt["system"], context=context)
    user = render(prompt["user"], question=question)

    return get_chat_model().send_prompt(user, context=system, params={"temperature": 0.3, "max_tokens": 1000})


# ---------------------------------------------------------------------------
# Literature search tool (Bernie)
# ---------------------------------------------------------------------------

@tool
def search_literature(query: str) -> str:
    """Search Semantic Scholar for papers related to a query. Source: [semantic scholar]"""
    papers = _get_lit_search().search_external_literature(query, max_results=5)
    if not papers:
        return "No papers found on Semantic Scholar for that query."
    lines = []
    for p in papers:
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        doi_str = f" DOI: {p.doi}" if p.doi else ""
        lines.append(
            f"[semantic scholar] {p.title} ({p.year}) — {authors}{doi_str}\n"
            f"{p.abstract or 'No abstract.'}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def build_langchain_tools() -> list:
    """Return the list of LangChain Tool objects for the ConversationManager agent."""
    return [
        search_datasets,
        get_dataset_details,
        get_workflow_guidance,
        get_educational_context,
        search_literature,
    ]
