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
_portal_docs_store = None


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


def _get_portal_docs_store():
    """Lazily load the FAISS index built by scripts/build_portal_docs_index.py.

    Returns None (rather than raising) if the index hasn't been built yet, so
    search_portal_docs can produce an honest gap message instead of crashing.
    """
    global _portal_docs_store
    if _portal_docs_store is None:
        index_dir = Path(__file__).parents[2] / "data" / "portal_docs_index"
        if not index_dir.exists():
            return None
        from src.assistant.llm import get_embeddings_model
        from src.ingestor.embedder import DocumentEmbedder
        from src.retriever.retriever import VectorStoreManager

        # Resolve the real Embeddings instance (not the lazy proxy) — FAISS
        # does an isinstance(..., Embeddings) check that the proxy fails.
        manager = VectorStoreManager(DocumentEmbedder(embeddings=get_embeddings_model()))
        manager.load(index_dir)
        _portal_docs_store = manager
    return _portal_docs_store


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

# Known closed-vocabulary schema values (mirrors graph_store.py's MANUAL_SCHEMA) plus
# common imaging-method terms. Used to detect queries that already state a concrete
# schema property, so we can deterministically suppress the search-reasoning narration
# for those — rather than relying on the LLM to self-classify (unreliable, see HANDOFF.md).
_ROCK_TYPES = ("beads", "carbonate", "coal", "fibrous_media", "fibrous media",
               "granite", "sandstone", "soil")
_SOURCE_TERMS = ("artificial", "natural")
_SEGMENTED_TERMS = ("segmented", "unsegmented")
_IMAGING_KEYWORDS = ("micro-ct", "microct", "fib-sem", "fibsem", "x-ray", "xray",
                     "nano-ct", "nanoct", "sem", "mri")
_PLAIN_PROPERTY_TERMS = _ROCK_TYPES + _SOURCE_TERMS + _SEGMENTED_TERMS + _IMAGING_KEYWORDS

# A named person ("datasets by Jane Doe") names a concrete, checkable property — the
# authors field — just as much as "sandstone" or "segmented" does, but no fixed keyword
# list can enumerate every possible name. Detect the *pattern* instead: a capitalized
# multi-word proper-noun phrase following "by"/"from"/"authored by".
_AUTHOR_QUERY_RE = re.compile(
    r"\b(?:by|from|authored by)\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)+)"
)


def _mentions_named_person(query: str) -> bool:
    """True if the query plausibly names a specific person (maps to the authors field)."""
    return bool(_AUTHOR_QUERY_RE.search(query))


def _schema_field_names() -> list[str]:
    """Lazily fetch the derived list of Cypher-queryable schema fields (see
    graph_store.get_queryable_field_names) — single source of truth, not a
    hand-maintained duplicate list."""
    from src.assistant.graph_store import get_queryable_field_names
    return get_queryable_field_names()


def _summarize_dataset_results(query: str, results: list[dict]) -> list[str]:
    """One sentence per result describing what the dataset is and how it relates to
    the query. Batched into a single LLM call. Title/DOI stay verbatim from metadata —
    only this prose summary is LLM-authored, same pattern as the search lead-in."""
    from src.assistant.llm import get_chat_model

    texts = [re.sub(r"\s+", " ", r.get("text", "")).strip() for r in results]
    fallback = [t[:200].rstrip() + ("…" if len(t) > 200 else "") for t in texts]

    system = (
        "You summarize dataset search results for a research assistant. For each "
        "dataset below (title, DOI, and description), write ONE short sentence "
        "(at most 25 words) describing what the dataset is and how it relates to "
        "the user's query. Base each summary only on the given description — never "
        "invent details not present in it. Return ONLY a JSON array of strings, one "
        "per dataset, in the same order given, no markdown fences, no explanation."
    )
    numbered = [
        f"{i}. {r.get('metadata', {}).get('title', 'Unknown')}: {text}"
        for i, (r, text) in enumerate(zip(results, texts), 1)
    ]
    user_msg = f"Query: {query}\n\nDatasets:\n" + "\n".join(numbered)

    try:
        raw = get_chat_model().send_prompt(
            user_msg, context=system, params={"temperature": 0.2, "max_tokens": 60 * len(results)}
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        summaries = json.loads(cleaned.strip())
        if not isinstance(summaries, list) or len(summaries) != len(results):
            raise ValueError("summary count mismatch")
        return [str(s).strip() for s in summaries]
    except Exception as e:
        logger.warning("Result summarization failed (%s); falling back to raw snippet", e)
        return fallback


def _is_plain_property_query(query: str) -> bool:
    """True if the query already names a concrete schema property/keyword directly,
    rather than describing a task or purpose that requires inferring properties."""
    lowered = query.lower()
    return any(term in lowered for term in _PLAIN_PROPERTY_TERMS)


def _extract_query_topic_terms(query: str, inferred_filters: dict) -> list[str]:
    """Pull out the concrete topic term(s) a query is actually asking about, from the
    known schema/imaging vocabularies plus any filter values expand_query inferred.
    Used to deterministically detect weak/off-topic semantic search results — no
    embedding-score threshold required (score scale is model/index-dependent and an
    arbitrary numeric cutoff would be fragile)."""
    lowered = query.lower()
    terms = [term for term in _PLAIN_PROPERTY_TERMS if term in lowered]
    for value in inferred_filters.values():
        if isinstance(value, str) and value.strip():
            terms.append(value.strip().lower().replace("_", " "))
    return list(dict.fromkeys(terms))  # de-dupe, preserve order


def _results_mention_any(results: list[dict], terms: list[str]) -> bool:
    for r in results:
        haystack = f"{r.get('text', '')} {r.get('metadata', {}).get('title', '')}".lower()
        if any(term in haystack for term in terms):
            return True
    return False


@tool
def search_datasets(query: str, top_k: int = 5) -> str:
    """Find datasets by semantic similarity to a natural language query. Use for open-ended dataset discovery and suitability/purpose queries with no precise checkable property named, like 'sandstone datasets suitable for LBM simulation' or 'something good for a teaching demo'. Do NOT use this for queries that name a concrete, checkable property — a numeric threshold or range (e.g. 'porosity above 0.3', 'voxel size smaller than 2 microns', 'resolution finer than 5 micrometers'), a specific metadata value, a named person/author, or multiple values/fields (e.g. 'sandstone or carbonate', 'segmented and porosity above 0.3') — even if a rock type or imaging method is also mentioned; those belong to get_dataset_details, which generates real Cypher and can express comparisons and combinations this tool's filters cannot. This tool's own voxelDimensions filter is a coarse micrometer/millimeter/nanometer bucket only — it cannot express a numeric cutoff like "< 2 microns". Do NOT use for how-to or workflow questions (e.g. 'how to compute permeability') — those belong to get_workflow_guidance. Note: this tool also attempts a structured Cypher lookup first for property-shaped queries (including named authors) as a safety net in case routing missed it — but call get_dataset_details directly when you recognize the property, since it's the more reliable path."""
    expansion = expand_query(query)
    expanded = expansion.get("expanded_query", query)
    filters = expansion.get("inferred_filters", {})
    rationale = expansion.get("rationale", "")

    # Deterministic safety net: don't rely solely on the outer agent's routing having
    # correctly classified the query up front (see HANDOFF.md — routing-by-example is
    # brittle for property types no one thought to enumerate, e.g. author names). If the
    # query looks property-shaped, attempt the structured Cypher path first and use it
    # if it produced a real, grounded answer; only fall through to semantic/hybrid search
    # when the structured attempt genuinely found nothing (or the query doesn't look
    # property-shaped at all, e.g. pure suitability/discovery queries).
    _NO_STRUCTURED_ANSWER = (
        "the query ran successfully and found no matching",
        "no answer found",
        "graph search is disabled",
    )
    looks_structured = bool(filters) or _is_plain_property_query(query) or _mentions_named_person(query)
    if looks_structured:
        try:
            structured = _get_graph_store().cypher_qa(query)
        except Exception as e:
            logger.warning("Structured-first lookup failed (%s); falling back to semantic search", e)
            structured = ""
        if structured and not structured.strip().lower().startswith(_NO_STRUCTURED_ANSWER):
            return structured

    graph_store = _get_graph_store()
    results = graph_store.hybrid_search(expanded, filters=filters, top_k=top_k)

    # Second-pass: component-level search to catch datasets whose parent description
    # has weak signal but whose sub-nodes (e.g. AnalysisDataset) score better.
    seen_dois = {r.get("metadata", {}).get("doi") for r in results if r.get("metadata", {}).get("doi")}
    comp_results = graph_store.component_search(expanded, filters=filters, top_k=top_k)
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
                "metadata": {
                    "title": cm.get("datasetTitle", "Unknown"),
                    "doi": doi,
                    # Which specific sub-node matched (e.g. a single DigitalDataset out
                    # of several under this dataset) — surfaced to the user below so a
                    # multi-sample/multi-scan dataset isn't a mystery match.
                    "component_title": cm.get("componentTitle"),
                    "component_type": cm.get("componentType"),
                },
                "source_label": "[component match]",
            })
            extras += 1

    if not results:
        return "No datasets found matching that query. Try broadening your search — for example, remove specific filters, use more general terminology, or search by rock type (sandstone, carbonate, coal) or imaging method (micro-CT, FIB-SEM)."
    summaries = _summarize_dataset_results(query, results)
    lines = []
    for r, summary in zip(results, summaries):
        meta = r.get("metadata", {})
        title = meta.get("title", "Unknown")
        raw_doi = meta.get("doi", "")
        # Normalize: strip any number of leading https://doi.org/ prefixes
        doi_id = raw_doi
        while doi_id.startswith("https://doi.org/"):
            doi_id = doi_id[len("https://doi.org/"):]
        doi_str = f"DOI: {doi_id}" if doi_id else ""
        label = r.get("source_label", "[graph match]")
        component_title = meta.get("component_title")
        component_type = meta.get("component_type")
        matched_via = f' — matched via {component_type} "{component_title}"' if component_title else ""
        lines.append(f"{label} {title}{matched_via} ({doi_str})\n{summary}")

    output = "\n\n".join(lines)
    if rationale and not _is_plain_property_query(query):
        output = f"[search reasoning: {rationale}]\n\n" + output

    topic_terms = _extract_query_topic_terms(query, filters)
    if topic_terms and not _results_mention_any(results, topic_terms):
        shown = " / ".join(f'"{t}"' for t in topic_terms)
        output = (
            f"[weak match: none of the results below directly mention {shown}; "
            "showing the closest available results, which may not be relevant]\n\n"
        ) + output
    return output


@tool
def get_dataset_details(question: str) -> str:
    """Answer structured questions about dataset properties using Cypher. Source: [cypher match]"""
    return _get_graph_store().cypher_qa(question)


# The static docstring above is deliberately generic — the routing detail (which
# properties count as "structured") is appended dynamically below, derived from
# graph_store.get_queryable_field_names() rather than a hand-picked example list. This
# keeps the agent's routing signal in sync with the actual schema fed to
# GraphCypherQAChain: a field added to MANUAL_SCHEMA is automatically reflected here
# without a second edit. Wrapped in try/except so a parsing hiccup degrades to the
# static docstring rather than breaking tool registration.
try:
    get_dataset_details.description += (
        ". Covers any of these dataset/sample properties, including numeric "
        "comparisons, exact values, or a named person explicitly as the subject of a "
        "dataset/author search (e.g. 'datasets by Jane Doe' — maps to authors; a name "
        "mentioned incidentally, such as someone introducing themselves, is not this "
        "case): " + ", ".join(_schema_field_names()) + "."
    )
except Exception as _e:  # pragma: no cover - defensive only
    logger.warning("Could not derive schema field list for get_dataset_details description: %s", _e)


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


_HONEST_NO_TUTORIAL_MSG = (
    "We don't currently have a dedicated tutorial for this topic, but we welcome community "
    "contributions — if you'd like to see one added or are interested in contributing, please "
    "reach out to the DPM Portal team."
)

# Matches the "**Goal:** ... **Notebook:** `path.ipynb`" block the educational.yaml prompt
# instructs the model to use when a real tutorial was retrieved.
_NOTEBOOK_BLOCK_RE = re.compile(
    r"\*\*Goal:\*\*[^\n]*\n\s*\*\*Notebook:\*\*\s*`[^`]*\.ipynb`\n?",
    re.IGNORECASE,
)


_NOTEBOOK_PATH_RE = re.compile(r"[^\s`\"']*\.ipynb")


def _strip_fabricated_tutorial_reference(response: str, tutorials: list[dict]) -> str:
    """Deterministic guard against tutorial-path hallucination.

    Some models (observed with Llama-4-Maverick) fabricate a plausible-looking
    notebook path even when no real tutorial was retrieved — or, worse, fabricate
    *additional* paths alongside genuinely retrieved ones — despite the prompt
    instructing it to only echo paths verbatim from context. Prompt-only fixes
    reduced but did not eliminate this, so any notebook path not present in the
    actual `tutorials` match list is stripped, regardless of whether other real
    paths are also present in the response.
    """
    if ".ipynb" not in response.lower():
        return response

    valid_paths = {t["notebook"] for t in tutorials}

    if not valid_paths:
        cleaned = _NOTEBOOK_BLOCK_RE.sub("", response).strip()
        if ".ipynb" in cleaned.lower():
            # Didn't match the expected block format but still references a notebook —
            # can't safely excise just the offending part, so replace wholesale.
            return _HONEST_NO_TUTORIAL_MSG
        if _HONEST_NO_TUTORIAL_MSG[:20].lower() not in cleaned.lower():
            cleaned = (cleaned + "\n\n" + _HONEST_NO_TUTORIAL_MSG).strip()
        return cleaned

    def _drop_invalid_block(match: re.Match) -> str:
        block = match.group(0)
        path_match = re.search(r"`([^`]*\.ipynb)`", block)
        path = path_match.group(1) if path_match else None
        return block if path in valid_paths else ""

    cleaned = _NOTEBOOK_BLOCK_RE.sub(_drop_invalid_block, response).strip()

    # Catch fabricated paths mentioned outside the expected Goal/Notebook block
    # format (e.g. inline prose). Excise just the path text — a line may also
    # contain a genuine path, so dropping the whole line would lose real content.
    stray_paths = {p for p in _NOTEBOOK_PATH_RE.findall(cleaned) if p not in valid_paths}
    for p in stray_paths:
        cleaned = cleaned.replace(p, "")
    if stray_paths:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n[ \t]+", "\n", cleaned).strip()

    return cleaned


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

    response = get_chat_model().send_prompt(user, context=system, params={"temperature": 0.3, "max_tokens": 1000})
    return _strip_fabricated_tutorial_reference(response, tutorials)


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

    response = get_chat_model().send_prompt(user, context=system, params={"temperature": 0.3, "max_tokens": 1000})
    return _strip_fabricated_tutorial_reference(response, tutorials)


# ---------------------------------------------------------------------------
# Portal documentation search
# ---------------------------------------------------------------------------

# FAISS.similarity_search always returns the top-k nearest chunks regardless of
# relevance — there's no free "no match" signal. similarity_search_with_score
# returns L2 distance (lower = closer); empirically, on-topic queries against
# the current index score ~0.4-0.6 while clearly unrelated queries score
# ~0.9-1.3 (tested against the built dpm_docs index during development).
# 0.9 sits in the gap between those clusters. This is index/embedding-model
# specific — recalibrate if the embedding model or corpus changes materially.
_NO_MATCH_SCORE_THRESHOLD = 0.9

# A long section (e.g. "1. Dataset") gets split into multiple overlapping chunks by
# the 500-char/100-overlap splitter (build_portal_docs_index._MD_SPLITTER). Because
# they're near-duplicates of the same passage, plain top-k similarity search often
# ranks two or three of them consecutively — wasting several of the final slots on
# redundant text instead of covering different sections (e.g. both halves of
# "1. Dataset" ranking above "2. Sample"/"3. Digital Dataset" entirely, even though
# all four sections are on the same page and equally relevant to a "what's the
# difference" question). _PORTAL_DOCS_CANDIDATE_K casts a wider net; _dedupe_by_section
# then keeps only the best-scoring chunk per (page_title, section) before truncating
# to _PORTAL_DOCS_FINAL_K, trading duplicate coverage of one section for breadth
# across sections.
_PORTAL_DOCS_CANDIDATE_K = 12
_PORTAL_DOCS_FINAL_K = 5

# Named schema entities (the same node labels documented in graph_store.py's
# MANUAL_SCHEMA: Dataset, Sample, DigitalDataset, AnalysisDataset) each have a
# definitional chunk on upload_data.md's numbered "Curate Your Dataset" reference
# section. Raw similarity ranking systematically under-ranks the more specific
# entities (DigitalDataset, AnalysisDataset) because "1. Dataset"'s long, generic
# section mentions the word "Dataset" densely and wins top-k for almost any query
# containing it, regardless of how far _PORTAL_DOCS_CANDIDATE_K is raised (verified
# up to k=25 — still crowded out by other, less-relevant sections). Widening k alone
# doesn't fix this without also flooding results with noise, so instead: if a query
# names one of these entities, deterministically guarantee its definitional chunk is
# included rather than leaving it purely to embedding rank.
_PORTAL_DOCS_PAGE = "How to Upload Data: A Step-by-Step Guide"
_SCHEMA_ENTITY_ANCHORS = [
    (re.compile(r"\bdigital\s*dataset\b", re.IGNORECASE), (_PORTAL_DOCS_PAGE, "3. Digital Dataset")),
    (re.compile(r"\banalysis\s*dataset\b", re.IGNORECASE), (_PORTAL_DOCS_PAGE, "4. Analysis Dataset")),
    (re.compile(r"\bsample\b", re.IGNORECASE), (_PORTAL_DOCS_PAGE, "2. Sample")),
    (re.compile(r"\bdataset\b", re.IGNORECASE), (_PORTAL_DOCS_PAGE, "1. Dataset")),
]

# Fetch wide enough to cover the whole portal_docs corpus (247 chunks as of the last
# rebuild) so the metadata filter below can find the target section regardless of its
# raw similarity rank.
_ANCHOR_FETCH_K = 300


def _anchor_chunks_for_query(store, question: str) -> list:
    """Return definitional chunks for any schema entity named in `question`, fetched
    directly by (page_title, section) via FAISS's metadata `filter` — a targeted
    lookup, not a ranked search, so it's immune to the vocabulary-dominance ranking
    bias described above. Returns [] if the store doesn't support this or nothing
    matches; never raises."""
    vector_store = getattr(store, "vector_store", None)
    if vector_store is None:
        return []

    matched_sections = []
    seen = set()
    for pattern, section_key in _SCHEMA_ENTITY_ANCHORS:
        if section_key in seen or not pattern.search(question):
            continue
        seen.add(section_key)
        matched_sections.append(section_key)

    anchors = []
    for page_title, section in matched_sections:
        try:
            hits = vector_store.similarity_search_with_score(
                question, k=1, filter={"page_title": page_title, "section": section}, fetch_k=_ANCHOR_FETCH_K
            )
        except Exception as e:
            logger.warning("Anchor lookup failed for section %r (%s)", section, e)
            continue
        anchors.extend(doc for doc, _score in hits)
    return anchors


def _dedupe_by_section(scored: list[tuple]) -> list:
    """Keep only the best-scoring chunk per (page_title, section), preserving score
    order. `scored` is already score-ascending (best first), so the first chunk seen
    for a given section is its best-scoring one; later chunks from the same section
    are near-duplicate overlap splits and are dropped rather than the reverse."""
    seen: set[tuple] = set()
    deduped = []
    for doc, score in scored:
        key = (doc.metadata.get("page_title"), doc.metadata.get("section"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, score))
    return deduped


def _expand_portal_query(question: str) -> str:
    """Restate a possibly truncated/keyword-style tool argument as a full natural-
    language question before embedding it (src/prompts/portal_query_expander.yaml).

    The routing prompt instructs the agent to forward the user's question verbatim,
    but tool-calling LLMs routinely compress a question into a short keyword phrase
    anyway (e.g. "How do I upload a dataset?" -> "upload a dataset") — this loses
    context that materially changes which chunks rank highest. Falls back to the
    original question on any failure; this is a best-effort enrichment step, not a
    hard dependency of search_portal_docs."""
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    try:
        prompt = load_prompt("portal_query_expander")
        user = render(prompt["user"], query=question)
        expanded = get_chat_model().send_prompt(
            user, context=prompt["system"], params={"temperature": 0.0, "max_tokens": 150}
        ).strip()
        return expanded or question
    except Exception as e:
        logger.warning("Portal query expansion failed (%s); using original question", e)
        return question


def _strip_fabricated_figure_reference(response: str, has_figure: bool) -> str:
    """Deterministic guard against figure-mention hallucination.

    portal_docs.yaml instructs the model to mention that a screenshot exists ONLY when
    an excerpt it actually used contains a "[Figure: ...]" placeholder. Observed
    (Llama-4-Maverick via SambaNova/TACC) over-generalizing this to "mention a
    screenshot whenever discussing a UI step," fabricating the mention even when none
    of the retrieved excerpts contained a placeholder at all. Prompt wording alone
    wasn't reliable for the analogous tutorial-path hallucination (see
    _strip_fabricated_tutorial_reference) — same fix here: strip any sentence
    mentioning a screenshot rather than trusting the prompt to gate it."""
    if has_figure or "screenshot" not in response.lower():
        return response

    kept_paragraphs = []
    for para in response.split("\n"):
        if not para.strip():
            kept_paragraphs.append(para)
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para)
        kept_sentences = [s for s in sentences if "screenshot" not in s.lower()]
        if kept_sentences:
            kept_paragraphs.append(" ".join(kept_sentences))
    cleaned = "\n".join(kept_paragraphs)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()


_FIGURE_APPEND_NOTE = "See the screenshot on the linked page for this step."


def _ensure_figure_reference(response: str, has_figure: bool) -> str:
    """Complement to _strip_fabricated_figure_reference: when a used excerpt genuinely
    contains a "[Figure: ...]" placeholder, the disclosure that a screenshot exists is
    useful enough (and easy enough to get right deterministically) that it shouldn't
    depend on the model choosing to mention it — observed inconsistently omitting it
    even with an explicit prompt instruction to include it. If has_figure is True and
    the response doesn't already mention a screenshot, append a fixed honest note."""
    if not has_figure or "screenshot" in response.lower():
        return response
    return response.rstrip() + "\n\n" + _FIGURE_APPEND_NOTE


def _format_portal_doc_chunk(doc) -> str:
    """Render one retrieved dpm_docs chunk as labeled context text for the
    synthesis LLM — same [portal docs] labeling shown to users.

    doc.page_content already starts with a "{page_title} — {section}" header baked in
    at embed time (see build_portal_docs_index.chunk_markdown_file) — only the
    "[portal docs]" source-type tag is added here, not a second copy of the title."""
    doc_url = doc.metadata.get("doc_url", "")
    return f"[portal docs] {doc.page_content}\nSource: {doc_url}"


@tool
def search_portal_docs(question: str) -> str:
    """Search the DPM Portal user documentation for how-to guides and metadata schema reference.

    Covers: dataset submission guidelines, portal navigation, metadata field definitions,
    and file format requirements sourced from https://github.com/digital-porous-media/dpm_docs.
    Source label: [portal docs].
    """
    store = _get_portal_docs_store()
    if store is None:
        return (
            "Portal documentation search is not yet available (index not built). "
            "Run scripts/build_portal_docs_index.py to enable it. "
            "For step-by-step workflow guidance, try get_workflow_guidance(). "
            "For structured dataset property queries, try get_dataset_details()."
        )

    expanded_question = _expand_portal_query(question)
    scored = store.similarity_search_with_score(expanded_question, k=_PORTAL_DOCS_CANDIDATE_K)
    scored = [(doc, score) for doc, score in scored if score <= _NO_MATCH_SCORE_THRESHOLD]
    deduped = _dedupe_by_section(scored)[:_PORTAL_DOCS_FINAL_K]
    results = [doc for doc, score in deduped]

    # Guarantee any named schema entity's definitional chunk is present — never
    # displaces a genuinely good top-k match, only fills gaps the ranking misses.
    existing_keys = {(d.metadata.get("page_title"), d.metadata.get("section")) for d in results}
    for anchor_doc in _anchor_chunks_for_query(store, expanded_question):
        key = (anchor_doc.metadata.get("page_title"), anchor_doc.metadata.get("section"))
        if key not in existing_keys:
            results.append(anchor_doc)
            existing_keys.add(key)

    if not results:
        return (
            "No portal documentation found matching that question. "
            "Try get_workflow_guidance() or get_educational_context()."
        )

    # Raw retrieved chunks are independent nearest-neighbor hits, not a coherent
    # answer — some are routinely off-topic even after the score filter above (e.g.
    # an image-caption fragment surfacing for an unrelated question). Hand them to an
    # LLM to synthesize an actual answer and cite only the chunks it actually used,
    # rather than returning the raw chunk dump directly (see search_portal_docs'
    # entry in conversation_manager._SELF_CONTAINED_TOOLS for why this must be a
    # self-contained answer rather than a verbatim tool).
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    has_figure = any("[Figure:" in doc.page_content for doc in results)
    context = "\n\n".join(_format_portal_doc_chunk(doc) for doc in results)
    prompt = load_prompt("portal_docs")
    system = render(prompt["system"], context=context)
    user = render(prompt["user"], question=question)
    response = get_chat_model().send_prompt(user, context=system, params={"temperature": 0.2, "max_tokens": 800})
    response = _strip_fabricated_figure_reference(response, has_figure)
    return _ensure_figure_reference(response, has_figure)


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
