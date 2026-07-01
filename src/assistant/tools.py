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
import re
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
    """Return up to max_results workflows semantically relevant to query."""
    from src.assistant.llm import get_chat_model

    data = _load_workflows()
    all_workflows = data.get("workflows", [])

    index_lines = []
    for wf in all_workflows:
        desc = wf.get("description", "").replace("\n", " ").strip()[:120]
        index_lines.append(f"- {wf['id']}: {wf.get('name', wf['id'])} — {desc}")
    index_str = "\n".join(index_lines)

    system = (
        "You are a workflow retrieval system. Given a user query and a list of workflows, "
        "return a JSON array of the most relevant workflow IDs. Return at most "
        f"{max_results} IDs, ordered by relevance. Return ONLY valid JSON, no explanation. "
        "If nothing is relevant, return []."
    )
    user_msg = f"Query: {query}\n\nWorkflows:\n{index_str}"

    try:
        raw = get_chat_model().send_prompt(user_msg, context=system, params={"temperature": 0, "max_tokens": 100})
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        ids = json.loads(cleaned.strip())
        if not isinstance(ids, list):
            raise ValueError("not a list")
    except Exception as e:
        logger.warning("_match_workflows semantic call failed (%s); falling back to keyword match", e)
        return _match_workflows_keyword(query, max_results)

    id_to_wf = {wf["id"]: wf for wf in all_workflows}
    return [id_to_wf[wid] for wid in ids if wid in id_to_wf][:max_results]


def _match_workflows_keyword(query: str, max_results: int = 3) -> list[dict]:
    """Keyword fallback — used only when the LLM call fails."""
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


def _match_tutorials(query: str, max_results: int = 5) -> list[dict]:
    """Return up to max_results tutorials whose keywords overlap with query, ranked by hit count."""
    data = _load_tutorials()
    query_lower = query.lower()
    scored = []
    for t in data.get("tutorials", []):
        keywords = [str(k).lower() for k in t.get("keywords", [])]
        hits = sum(1 for kw in keywords if kw in query_lower)
        if hits > 0:
            scored.append((hits, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:max_results]]


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


def _literature_fallback_context(query: str, max_results: int = 3) -> str:
    """Return a literature context block for use when no tutorials match."""
    try:
        papers = _get_lit_search().search_external_literature(query, max_results=max_results)
    except Exception as e:
        logger.warning("Literature fallback search failed: %s", e)
        return ""
    if not papers:
        return ""
    lines = ["## Related Literature [semantic scholar]"]
    for p in papers:
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        doi_str = f" DOI: {p.doi}" if p.doi else ""
        lines.append(f"  - {p.title} ({p.year}) — {authors}{doi_str}")
        if p.abstract:
            lines.append(f"    {p.abstract[:200].rstrip()}…")
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
        tut_lines = [
            "## Portal Tutorials",
            f"How to access: {steps_txt}",
            "",
            "Relevant notebooks:",
        ]
        for t in tutorials:
            nb_path = t["notebook"]
            # Derive a readable name: strip chapter prefix and extension
            nb_filename = nb_path.split("/")[-1]
            # e.g. "5-2-1_lbm_d2q9_bgk.ipynb" → "lbm_d2q9_bgk"
            name_part = nb_filename.replace(".ipynb", "")
            # Strip leading chapter number like "5-2-1_"
            name_clean = re.sub(r"^\d[\d\-]*_", "", name_part).replace("_", " ")
            tut_lines.append(f'  - **{name_clean}** — {t["goal"]}')
            tut_lines.append(f'    Path in Community Data: `{nb_path}`')
        parts.append("\n".join(tut_lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset search tools (Intern A)
# ---------------------------------------------------------------------------

@tool
def search_datasets(query: str, top_k: int = 5) -> str:
    """Find datasets by semantic similarity to a natural language query. Use for dataset discovery, finding datasets by rock type or imaging method, and suitability queries like 'sandstone datasets suitable for LBM simulation'. Do NOT use for how-to or workflow questions (e.g. 'how to compute permeability') — those belong to get_workflow_guidance."""
    expansion = expand_query(query)
    expanded = expansion.get("expanded_query", query)
    filters = expansion.get("inferred_filters", {})
    rationale = expansion.get("rationale", "")

    graph_store = _get_graph_store()
    results = graph_store.hybrid_search(expanded, filters=filters, top_k=top_k)

    # Second-pass: component-level search to catch datasets whose parent description
    # has weak signal but whose sub-nodes (e.g. AnalysisDataset) score better.
    seen_dois = {r.get("metadata", {}).get("doi") for r in results if r.get("metadata", {}).get("doi")}
    comp_results = graph_store.component_search(expanded, top_k=top_k)
    extras = 0
    extra_limit = max(3, top_k - len(results))
    for cr in comp_results:
        if extras >= extra_limit:
            break
        cm = cr.get("metadata", {})
        doi = cm.get("doi", "")
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            results.append({
                "text": cr["text"],
                "metadata": {"title": cm.get("datasetTitle", "Unknown"), "doi": doi},
                "source_label": "[component match]",
            })
            extras += 1

    if not results:
        return "No datasets found matching that query."
    lines = []
    for r in results:
        meta = r.get("metadata", {})
        title = meta.get("title", "Unknown")
        raw_doi = meta.get("doi", "")
        # Normalize: strip any number of leading https://doi.org/ prefixes
        doi_id = raw_doi
        while doi_id.startswith("https://doi.org/"):
            doi_id = doi_id[len("https://doi.org/"):]
        doi_str = f"DOI: {doi_id}" if doi_id else ""
        label = r.get("source_label", "[graph match]")
        lines.append(f"{label} {title} ({doi_str})\n{r['text'][:300]}")

    output = "\n\n".join(lines)
    if rationale:
        output = f"[search reasoning: {rationale}]\n\n" + output
    return output


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
    lit_ctx = _literature_fallback_context(goal) if not tutorials else ""
    context = "\n\n".join(p for p in [_workflow_context_str(workflows, tutorials), lit_ctx] if p)

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
    lit_ctx = _literature_fallback_context(question) if not tutorials else ""
    workflow_ctx = _workflow_context_str(workflows, tutorials)
    global_ctx = _global_practices_context(question)

    parts = [p for p in [workflow_ctx, global_ctx, lit_ctx] if p]
    context = "\n\n".join(parts)

    prompt = load_prompt("educational")
    system = render(prompt["system"], context=context)
    user = render(prompt["user"], question=question)

    return get_chat_model().send_prompt(user, context=system, params={"temperature": 0.3, "max_tokens": 1000})


# ---------------------------------------------------------------------------
# Portal documentation search (stub — full pipeline pending)
# ---------------------------------------------------------------------------

@tool
def search_portal_docs(question: str) -> str:
    """Search the DPM Portal user documentation for how-to guides and metadata schema reference.

    Covers: dataset submission guidelines, portal navigation, metadata field definitions,
    and file format requirements sourced from https://github.com/digital-porous-media/dpm_docs.

    NOTE: Full implementation pending — the portal docs ingestion pipeline (fetch → chunk →
    vector index) has not been built yet. When implemented, this tool will query a FAISS or
    Neo4j vector index built from the dpm_docs markdown pages.
    """
    return (
        "Portal documentation search is not yet available. "
        "For step-by-step workflow guidance, try get_workflow_guidance(). "
        "For structured dataset property queries, try get_dataset_details()."
    )


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
        search_portal_docs,
        get_workflow_guidance,
        get_educational_context,
        search_literature,
    ]
