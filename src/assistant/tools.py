"""
Shared tool interface for the General Assistant.

All callable tools are defined here — both interns code to this interface.

Intern A owns (Week 2-3): search_datasets, get_dataset_details

Bernie owns: get_educational_context, get_workflow_guidance, expand_query, search_literature
"""

import json
import logging
import re
from pathlib import Path

import yaml
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Pure helpers/dataclasses only — safe at module level (no neo4j/langchain_neo4j import,
# those stay deferred to GraphStore.__init__ so USE_NEO4J=false stays fast/dependency-free).
from src.assistant.graph_store import DatasetProfileAmbiguous, _strip_doi_prefix

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


# ---------------------------------------------------------------------------
# Content-reasoning gate
# ---------------------------------------------------------------------------
#
# The dividing line is NOT "missing field" vs. "hopeless" — it is "is every property in
# the question a plain, literal, structured field?" A conjunction of independent literal
# properties ("sandstone AND porosity > 0.3") stays on Cypher, because each clause
# genuinely narrows the space on its own. A question containing relational language
# ("paired", "corresponding", "same X", "derived from") must route ENTIRELY to
# reason_about_dataset_content and is never split: a literal sub-condition inside a
# relational claim ("segmented" inside "paired ... and segmented") is not an
# independently valid partial answer, and displaying it as one produces exactly the
# overclaim this tool exists to remove — a generic "has some segmented data" list
# presented as if it had verified "paired".
#
# This is a deterministic code-level gate, not a tool-description hint, for the reason
# this codebase keeps rediscovering (see HANDOFF.md, conversation_manager.py): prompting
# is not reliable for nuanced binary calls, and the two sides here are worded almost
# identically ("segmented and porosity above 0.3" is plain; "segmented and imaged the
# same way" is relational). search_datasets already sets the precedent with
# _is_plain_property_query — same idea, applied to the other half of the split.

_RELATIONAL_PATTERNS = (
    r"\bpair(?:ed|s|ing)?\b",
    r"\bcorrespond(?:s|ing|ence)?\b",
    r"\bcounterparts?\b",
    r"\bequivalents?\s+(?:of|to)\b",
    r"\bsame\s+\w+",
    r"\bderived\s+from\b",
    r"\bcomes?\s+from\s+the\s+\w+",
    r"\bversions?\s+of\b",
    r"\bbefore\s+and\s+after\b",
    r"\bmatched\s+(?:pairs?|sets?|scans?|images?)\b",
    r"\baccompan(?:y|ies|ied|ying)\b",
    r"\b(?:different|differing|varying|multiple|several|two)\s+"
    r"(?:resolutions?|scales?|voxel\s+sizes?|magnifications?)\b",
    r"\bimaged\s+(?:the\s+same|at\s+(?:different|multiple|two))\b",
    r"\bboth\s+\w+(?:\s+\w+)?\s+and\s+\w+(?:\s+\w+)?\s+"
    r"(?:images?|scans?|versions?|forms?|datasets?)\b",
)

_RELATIONAL_RE = re.compile("|".join(_RELATIONAL_PATTERNS), re.IGNORECASE)

# Questions that ask for an exhaustive sweep of the catalog rather than "find me some" —
# these are the ones ranking can't legitimately narrow, so they take the map-reduce path.
_EXHAUSTIVE_RE = re.compile(
    r"\b(?:all|every|each|complete list|exhaustive|how many|count of|"
    r"list (?:all|every)|are there any)\b",
    re.IGNORECASE,
)


def _content_reasoning_signal(question: str) -> str | None:
    """Return the relational phrase that makes this question un-answerable by a plain
    field lookup, or None. Split out from _needs_content_reasoning so the matched phrase
    can be logged — the same detector does double duty as a live gate and as the
    monitoring signal for tuning it as false positives/negatives turn up."""
    m = _RELATIONAL_RE.search(question or "")
    return m.group(0) if m else None


def _needs_content_reasoning(question: str) -> bool:
    """True if `question` describes a relationship, a comparison across a dataset's
    sub-nodes, or a pattern implied by methodology/content — rather than being a plain,
    literal, structured-field question (including a plain conjunction of several such
    fields, which stays on Cypher).

    Borderline cases — a relational phrase alongside a genuinely literal property, e.g.
    "paired tomographic and segmented images" — still route here (the whole question
    goes to content reasoning, never split), but are logged so the heuristic can be
    reviewed periodically against real usage."""
    signal = _content_reasoning_signal(question)
    if not signal:
        return False
    if _is_plain_property_query(question):
        logger.warning(
            "Content-reasoning gate fired on a question that ALSO names a literal property "
            "(borderline — logged for review): signal=%r question=%r", signal, question,
        )
    else:
        logger.warning("Content-reasoning gate fired: signal=%r question=%r", signal, question)
    return True


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
    # Content-reasoning gate, checked before anything else — including before the
    # structured-first attempt below. A relational question ("paired tomographic and
    # segmented images") trips that attempt's `looks_structured` check on its literal
    # sub-clause alone ("segmented"), and answering it with a bare segmented='yes' list
    # presents a generic "has some segmented data" result as if "paired" had been
    # verified, which it never was. Route the WHOLE question to content reasoning
    # instead — same gate get_dataset_details applies, so the split is correct no matter
    # which of the two tools the agent happened to call.
    if _needs_content_reasoning(query):
        return _reason_about_dataset_content(query)

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
        doi_id = _strip_doi_prefix(raw_doi)
        doi_str = f"DOI: {doi_id}" if doi_id else "DOI: not available"
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
def get_dataset_details(question: str, restrict_to_titles: list[str] | None = None) -> str:
    """Answer structured questions about dataset properties using Cypher. Source: [cypher match]

    restrict_to_titles: internal use only — leave unset for a normal, catalog-wide
    question. When a caller is refining an earlier dataset listing (e.g. "which of
    these are segmented?"), pass the exact titles from that earlier listing here so
    the answer is deterministically narrowed to that set instead of re-running the
    new filter over the entire graph."""
    # Deterministic gate BEFORE committing to a plain Cypher answer: if the question
    # isn't purely literal-field-shaped, Cypher can only answer part of it, and a
    # partial answer to a relational question is a misleading answer (see
    # _needs_content_reasoning). Hand the whole question to content reasoning instead.
    # Applied here rather than left to the agent's tool choice for the same reason
    # search_datasets checks _is_plain_property_query internally.
    if _needs_content_reasoning(question):
        return _reason_about_dataset_content(question, restrict_to_titles=restrict_to_titles)
    return _get_graph_store().cypher_qa(question, restrict_to_titles=restrict_to_titles)


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
# Single-dataset deep-dive profile (follow-up detail queries)
# ---------------------------------------------------------------------------

def _corral_archive_url(dataset_number) -> str | None:
    """
    Returns the DPM Portal's TACC Corral archive directory for a dataset, derived
    deterministically from its datasetNumber (same URL pattern scripts/scrape_metadata.py
    already uses) — or None if no dataset number is available.

    Caveat (surfaced to the LLM via the prompt, not repeated here): the archive keeps every
    published version as its own directory (DRP-{n}, DRP-{n}v2, ...) and the graph doesn't
    record which version is current, so this bare-number link may not always be the latest.
    """
    if dataset_number in (None, ""):
        return None
    return f"https://web.corral.tacc.utexas.edu/digitalporousmedia/archive/DRP-{dataset_number}/"


def _corral_repl_path(dataset_number) -> str | None:
    """
    Returns the TACC-internal filesystem path (usable directly from a TACC system, e.g. a
    Lonestar6/Stampede3 job or an interactive session — NOT a URL) where this dataset's
    published files live on Corral, derived deterministically from its datasetNumber the same
    way as `_corral_archive_url()`. Only relevant to users already working on a TACC system;
    everyone else should use the web.corral URL (or, preferably, the DOI/portal page) instead.

    Same versioning caveat as `_corral_archive_url()`: the graph doesn't record which published
    version (v1, v2, ...) is current, so the bare-number path may not be the latest.
    """
    if dataset_number in (None, ""):
        return None
    return f"/corral-repl/utexas/OTH21076/data_prod/published/DRP-{dataset_number}/"


# Hard cap on how many sub-nodes of one type get rendered into the profile context. Some
# datasets have far more Sample/DigitalDataset/AnalysisDataset sub-nodes than a typical one
# (e.g. large multi-scan collections) — without a cap, _build_profile_context's output can grow
# large enough to blow past the model's context window on a single call, with no prior
# conversation history needed to reproduce it. Truncation is never silent: a "not shown" note
# with the real omitted count is always appended so the LLM (and, downstream, the user) knows
# the profile is partial rather than assuming these are literally all the sub-nodes that exist.
_MAX_NODES_PER_TYPE = 25

# Known vector-embedding property names (Dataset.datasetEmbedding, and componentEmbedding on
# the DatasetComponent secondary label shared by Sample/DigitalDataset/AnalysisDataset — see
# graph_store.py's schema docstring). GraphStore.get_dataset_profile()'s Cypher already nulls
# these out via map projection so they never cross the wire, but this is a second, independent
# layer: it also catches ANY other property that happens to be a long list of numbers (the
# actual root cause hit in production — a real embedded dataset's Sample/DigitalDataset nodes
# carried a 4096-float vector straight into the LLM context, alone enough to exceed the model's
# context window on a single call with no prior conversation history).
_EMBEDDING_KEYS = {"datasetEmbedding", "componentEmbedding"}
_EMBEDDING_LIST_MIN_LEN = 16


def _is_embedding_like(key: str, value) -> bool:
    """True if `value` looks like a vector embedding rather than human-facing metadata —
    either a known embedding property name, or (as a backstop against renamed/unknown
    embedding fields) any list of _EMBEDDING_LIST_MIN_LEN+ numeric values."""
    if key in _EMBEDDING_KEYS:
        return True
    return (
        isinstance(value, list)
        and len(value) >= _EMBEDDING_LIST_MIN_LEN
        and all(isinstance(v, (int, float)) for v in value)
    )


def _render_node_list(nodes: list[dict], heading: str) -> str:
    """
    Renders a list of sub-node property dicts as a markdown block, one bullet per node,
    one `key: value` per populated property. Nodes/properties with no populated fields are
    skipped entirely — empty/None/[] values must never reach the LLM context as clutter
    (see HANDOFF.md / project memory: no empty metadata fields shown to the user).
    Embedding-vector properties are stripped regardless (see _is_embedding_like) — they are
    never human-facing metadata and are large enough on their own to blow the context window.

    Caps at _MAX_NODES_PER_TYPE nodes (see its docstring) to bound context size.
    """
    if not nodes:
        return ""
    total = len(nodes)
    truncated = nodes[:_MAX_NODES_PER_TYPE]
    lines = [f"### {heading}"]
    for node in truncated:
        populated = {
            k: v for k, v in node.items()
            if v not in (None, "", []) and not _is_embedding_like(k, v)
        }
        if not populated:
            continue
        label = populated.get("title") or populated.get("identifier") or "(untitled)"
        lines.append(f"- **{label}**")
        for key, value in populated.items():
            if key in ("title",):
                continue
            lines.append(f"  - {key}: {value}")
    if total > _MAX_NODES_PER_TYPE:
        lines.append(f"- ... and {total - _MAX_NODES_PER_TYPE} more {heading.lower()} not shown here.")
    return "\n".join(lines) if len(lines) > 1 else ""


def _render_pipeline_edges(profile) -> str:
    """Renders the Sample -> DigitalDataset -> AnalysisDataset INPUT_FOR chain as arrows,
    keyed by identifier against the already-rendered node lists. Nodes with no recorded
    edge are called out by name so the org structure doesn't look silently incomplete."""
    by_id = {}
    for node in profile.samples + profile.digital_datasets + profile.analysis_datasets:
        if node.get("identifier"):
            by_id[node["identifier"]] = node.get("title") or node["identifier"]

    chains = []
    digital_to_analysis = {e["digitalDataset"]: e["analysisDataset"] for e in profile.digital_to_analysis_edges}
    linked_digital_ids = set()
    for edge in profile.sample_to_digital_edges:
        sample_id, digital_id = edge["sample"], edge["digitalDataset"]
        linked_digital_ids.add(digital_id)
        chain = [by_id.get(sample_id, sample_id), by_id.get(digital_id, digital_id)]
        analysis_id = digital_to_analysis.get(digital_id)
        if analysis_id:
            chain.append(by_id.get(analysis_id, analysis_id))
        chains.append(" -> ".join(chain))

    orphaned = [
        by_id.get(dd.get("identifier"), dd.get("title"))
        for dd in profile.digital_datasets
        if dd.get("identifier") not in linked_digital_ids
    ]

    lines = []
    if chains:
        lines.append("### Organizational structure (Sample -> DigitalDataset -> AnalysisDataset)")
        shown_chains = chains[:_MAX_NODES_PER_TYPE]
        lines.extend(f"- {c}" for c in shown_chains)
        if len(chains) > _MAX_NODES_PER_TYPE:
            lines.append(f"- ... and {len(chains) - _MAX_NODES_PER_TYPE} more pipeline links not shown here.")
    if orphaned:
        lines.append("### Digital datasets with no recorded sample/analysis link")
        shown_orphaned = [n for n in orphaned if n][:_MAX_NODES_PER_TYPE]
        lines.extend(f"- {name}" for name in shown_orphaned)
        if len(orphaned) > _MAX_NODES_PER_TYPE:
            lines.append(f"- ... and {len(orphaned) - _MAX_NODES_PER_TYPE} more not shown here.")
    return "\n".join(lines)


def _build_profile_context(profile) -> str:
    """
    Renders a DatasetProfileMatch into the structured context string fed to the
    dataset_profile prompt. Every section omits properties/nodes with no populated value —
    the context must never show a blank/null field, since that clutters the eventual answer
    (see project memory on concise dataset-detail answers).
    """
    d = {
        k: v for k, v in profile.dataset.items()
        if v not in (None, "", []) and not _is_embedding_like(k, v)
    }
    sections = ["## Portal-verified facts", f"**{d.get('title', 'Untitled dataset')}**"]
    for key, value in d.items():
        if key != "title":
            sections.append(f"- {key}: {value}")

    archive_url = _corral_archive_url(d.get("datasetNumber"))
    repl_path = _corral_repl_path(d.get("datasetNumber"))
    doi_value = d.get("doi")
    if archive_url or repl_path or doi_value:
        lines = ["\n## Data location"]
        if doi_value:
            lines.append(f"- Portal page (preferred, general access): DOI {_strip_doi_prefix(doi_value)}")
        if archive_url:
            lines.append(f"- Direct/scripting download URL: {archive_url}")
        if repl_path:
            lines.append(f"- TACC filesystem path (only relevant if the user is working directly on a TACC system): {repl_path}")
        sections.append("\n".join(lines))

    for nodes, heading in (
        (profile.samples, "Samples"),
        (profile.digital_datasets, "Digital datasets"),
        (profile.analysis_datasets, "Analysis datasets"),
        (profile.related_publications, "Related publications"),
        (profile.related_software, "Related software"),
        (profile.related_datasets, "Related datasets"),
    ):
        rendered = _render_node_list(nodes, heading)
        if rendered:
            sections.append("\n" + rendered)

    pipeline = _render_pipeline_edges(profile)
    if pipeline:
        sections.append("\n" + pipeline)

    return "\n".join(sections)


@tool
def get_dataset_profile(dataset_reference: str, question: str) -> str:
    """Give a full profile / deep-dive answer about ONE already-identified dataset, using its
    real graph data (Sample -> DigitalDataset -> AnalysisDataset organizational structure, file
    types, imaging/segmentation metadata, related publications/software/datasets) plus
    reasoning grounded in that data. Use this for:
      - "tell me more about <dataset>" / general profile requests on a specific dataset
      - a specific-field follow-up about ONE already-identified dataset (e.g. "what's its
        porosity", "how many files does it have") when the dataset itself is already
        known/named/resolved — NOT for a fresh multi-dataset structured lookup (that's
        get_dataset_details) or fresh discovery across many datasets (that's search_datasets)
      - questions about a dataset's Sample -> Digital Dataset -> Analysis Dataset pipeline
        structure (which sample produced which scan, which scan produced which analysis)
      - file-type/format questions and "how do I read/open this data in Python" or "where can I
        download this" reasoning — this tool will reason from the dataset's actual recorded
        file types/formats (and its real TACC Corral archive location, derived from its
        dataset number) plus general file-format/programming knowledge, and will clearly flag
        which parts of the answer are general knowledge rather than portal-verified fact
      - reuse-suitability reasoning about ONE dataset ("is this suitable for two-phase flow
        simulation") grounded in its actual recorded properties

    dataset_reference must be a concrete title, DOI, or dataset number — resolve any pronoun or
    positional reference ("this dataset", "the first one", "the sandstone one") against the
    conversation history YOURSELF before calling; this tool has no memory of prior turns and
    will treat a bare pronoun as a literal (failing) search string.

    For a query comparing TWO OR MORE datasets, call this tool ONCE PER dataset (with each
    one's own resolved reference and the comparison question) — do not invent a separate
    comparison tool call; synthesize the comparison yourself from the multiple results.

    Do NOT use this to discover NEW datasets matching a description (use search_datasets) or to
    run a structured multi-dataset property query across the whole catalog (use
    get_dataset_details) — this tool answers about one dataset that is already identified.

    Source label: [dataset profile]
    """
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    profile = _get_graph_store().get_dataset_profile(dataset_reference)

    if profile is None:
        return (
            f'No dataset was found matching "{dataset_reference}". Try the exact title, '
            "DOI, or dataset number as shown in a prior search result."
        )
    if isinstance(profile, DatasetProfileAmbiguous):
        lines = [
            f'- **{c["title"]}** (DOI: {_strip_doi_prefix(c.get("doi")) or "not available"})'
            for c in profile.candidates
        ]
        return f'Multiple datasets match "{dataset_reference}" — which one did you mean?\n\n' + "\n".join(lines)

    context = _build_profile_context(profile)
    prompt = load_prompt("dataset_profile")
    system = render(prompt["system"], context=context)
    user = render(prompt["user"], question=question)

    response = get_chat_model().send_prompt(user, context=system, params={"temperature": 0.2, "max_tokens": 1200})
    title = profile.dataset.get("title") or dataset_reference
    doi = _strip_doi_prefix(profile.dataset.get("doi")) or "not available"
    header = f"[dataset profile] {title} (DOI: {doi})"
    return f"{header}\n\n{response}"


# ---------------------------------------------------------------------------
# Content/relationship reasoning over precomputed fact sheets
# ---------------------------------------------------------------------------

# Fixed, never-LLM-authored framing prepended to every answer this tool produces,
# regardless of which underlying case (structural inference, a fact buried in free text,
# a cross-sub-node comparison) produced it. The whole point of the tool is to turn "no
# path to an answer" into a cited, honestly-labelled shortlist — not to manufacture
# certainty — so the caveat is not optional and not the model's to phrase.
_CONTENT_REASONING_FRAMING = (
    "I can't confirm this from a database field — here's what reasoning over the available "
    "facts and descriptions suggests. Verify before relying on it."
)

# How many fact sheets the ranking step shortlists, and the hard ceiling on the context
# they assemble into. Recall matters more than precision here (the reasoning pass discards
# non-matches anyway), so this wants to be generous — but not unbounded, since large
# candidate sets measurably degrade reasoning quality ("lost in the middle") on top of the
# token cost.
#
# Sized against the real corpus rather than guessed: measured over all 184 live datasets,
# a rendered fact sheet is a median of ~4.5k characters (p90 ~11k, max ~21k). A 40-sheet
# shortlist — the band originally sketched for this feature, before fact-sheet sizes were
# known — would therefore be ~180k characters, roughly 45k tokens, a third of the model's
# whole context window on every relational question. 25 sheets lands at ~113k characters
# (~28k tokens), which typically fits inside the budget below in full, so the shortlist
# size is a real number rather than one that gets silently cut down to a third of itself
# on every call. Re-measure both if the fact-sheet content or the corpus size changes
# substantially.
_FACT_SHEET_SHORTLIST_K = 25
_FACT_SHEET_CONTEXT_CHAR_BUDGET = 120_000

# Map-reduce fallback sizing (exhaustive questions only — see _is_exhaustive_question).
# Batched by CHARACTER budget rather than item count: fact-sheet sizes vary ~30x across the
# corpus (measured median ~4.5k, max ~21k characters), so a fixed item count would make some
# batches overflow _build_fact_sheet_context's budget and drop sheets. On an exhaustive
# question ("list EVERY dataset where...") a silently dropped dataset is precisely the wrong
# failure — the budget here is small enough that a batch never overflows the context builder.
_MAP_REDUCE_BATCH_CHAR_BUDGET = 40_000
_MAP_REDUCE_MAX_WORKERS = 4
_MAP_REDUCE_MAX_SURVIVORS = 40


def _batch_records_by_chars(records: list[dict], budget: int) -> list[list[dict]]:
    """Pack fact-sheet records into batches whose combined text stays under `budget`.
    A single record larger than the budget gets a batch to itself rather than being
    dropped."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for record in records:
        size = len(_fact_sheet_text(record))
        if current and used + size > budget:
            batches.append(current)
            current, used = [], 0
        current.append(record)
        used += size
    if current:
        batches.append(current)
    return batches


def _is_exhaustive_question(question: str) -> bool:
    """True for a question demanding a complete sweep of the catalog ("list every dataset
    where...", "how many datasets have...") rather than "find me some". Only these take
    the map-reduce path: ranking legitimately narrows an ordinary discovery question, but
    silently ranking away 130 datasets when the user asked for *all* of them would be a
    wrong answer dressed as a shortlist."""
    return bool(_EXHAUSTIVE_RE.search(question or ""))


def _fact_sheet_text(record: dict) -> str:
    """The rendered fact-sheet text for one dataset. Prefers the precomputed
    `factSheetText` (the exact text the ranking indexes were built over, so the model
    reasons over what the ranker matched on); falls back to a plain dump of the JSON
    `factSheet` for a dataset embedded before factSheetText existed."""
    text = (record.get("factSheetText") or "").strip()
    if text:
        return text
    raw = record.get("factSheet")
    if not raw:
        return ""
    try:
        return json.dumps(json.loads(raw), indent=1, default=str)
    except (json.JSONDecodeError, TypeError):
        return str(raw)


def _build_fact_sheet_context(records: list[dict]) -> tuple[str, list[dict]]:
    """Assemble the shortlist's fact sheets into one context block, within the character
    budget. Returns (context, included_records) — the caller validates the LLM's cited
    titles against `included_records`, so a dataset dropped by the budget here can never
    be "found" downstream."""
    blocks = []
    included = []
    used = 0
    for record in records:
        text = _fact_sheet_text(record)
        if not text:
            continue
        block = f"--- Dataset fact sheet ---\n{text}"
        if used + len(block) > _FACT_SHEET_CONTEXT_CHAR_BUDGET and included:
            break
        blocks.append(block)
        included.append(record)
        used += len(block)

    omitted = len(records) - len(included)
    if omitted > 0:
        blocks.append(
            f"--- NOTE: {omitted} further shortlisted dataset(s) did not fit in this "
            "context and were NOT considered. ---"
        )
    return "\n\n".join(blocks), included


def _screen_fact_sheet_batches(question: str, records: list[dict]) -> list[dict]:
    """Map step of the exhaustive fallback: batch the corpus, ask one cheap
    "does this batch contain a plausible candidate?" screening call per batch in
    parallel, and return the survivors for the single careful reasoning pass.

    Bounds token cost per call and scales as the catalog grows, instead of hitting a hard
    context wall later. A batch whose screening call fails is kept whole rather than
    dropped — a false positive costs a little context in the reduce step, a false negative
    silently loses a real answer."""
    from concurrent.futures import ThreadPoolExecutor
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    prompt = load_prompt("corpus_reasoning")
    batches = _batch_records_by_chars(records, _MAP_REDUCE_BATCH_CHAR_BUDGET)

    def _screen(batch: list[dict]) -> list[dict]:
        context, included = _build_fact_sheet_context(batch)
        if not included:
            return []
        try:
            raw = get_chat_model().send_prompt(
                render(prompt["batch_screen_user"], question=question, context=context),
                context=prompt["batch_screen_system"],
                params={"temperature": 0, "max_tokens": 300},
            )
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            kept_titles = {str(t).strip().lower() for t in json.loads(cleaned.strip())}
        except Exception as e:
            logger.warning("Fact-sheet batch screening failed (%s); keeping whole batch", e)
            return included
        return [r for r in included if str(r.get("title", "")).strip().lower() in kept_titles]

    with ThreadPoolExecutor(max_workers=_MAP_REDUCE_MAX_WORKERS) as pool:
        survivors = [r for batch in pool.map(_screen, batches) for r in batch]

    if len(survivors) > _MAP_REDUCE_MAX_SURVIVORS:
        logger.warning(
            "Exhaustive screening kept %d datasets; capping the reasoning pass at %d.",
            len(survivors), _MAP_REDUCE_MAX_SURVIVORS,
        )
        survivors = survivors[:_MAP_REDUCE_MAX_SURVIVORS]
    return survivors


def _parse_reasoning_response(raw: str) -> dict | None:
    """Parse the reasoning pass's JSON, tolerating markdown fences and surrounding prose.

    Returns None — not {} — when nothing parseable came back. The caller must be able to
    tell "the model judged the shortlist and found nothing" from "we couldn't read the
    model's answer": reporting the second as the first states a negative finding that was
    never actually established, which is exactly the class of overclaim this whole tool
    exists to remove."""
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start: end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    return _salvage_truncated_candidates(cleaned)


def _salvage_truncated_candidates(text: str) -> dict | None:
    """Recover the complete candidate objects from a response cut off mid-array.

    A long shortlist with long citations can exhaust max_tokens partway through the JSON.
    The candidates already emitted are complete, well-formed, and cited — discarding all of
    them because a later one was truncated throws away a real answer. Each salvaged object
    still goes through the same citation and shortlist-membership checks downstream, so this
    loosens parsing, never grounding.

    Returns None if nothing complete can be recovered.
    """
    start = text.find('"candidates"')
    if start == -1:
        return None
    array_start = text.find("[", start)
    if array_start == -1:
        return None

    candidates = []
    depth = 0
    obj_start = None
    in_string = False
    escaped = False
    for i in range(array_start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(text[obj_start: i + 1])
                    if isinstance(obj, dict):
                        candidates.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == "]" and depth == 0:
            break

    if not candidates:
        return None
    logger.warning(
        "Content-reasoning response was truncated; salvaged %d complete candidate(s).",
        len(candidates),
    )
    return {"candidates": candidates, "_truncated": True}


_DOI_IN_TITLE_RE = re.compile(r"10\.\d{4,9}/[^\s)\]]+")
_TRAILING_DOI_RE = re.compile(r"\s*\((?:doi|DOI)[:\s][^)]*\)\s*$")


def _match_shortlisted_record(claimed_title: str, records: list[dict]) -> dict | None:
    """Resolve the title the model returned to one of the fact sheets actually sent, or
    None if it can't be resolved to exactly one.

    Matching has to tolerate formatting drift without ever loosening into "close enough":
    the model reliably echoes the fact sheet's own header line, which is
    `Dataset <n>: <title> (DOI: <doi>)`, so it commonly returns the title with the DOI
    appended. Matching that literally against the bare title dropped EVERY otherwise-valid,
    correctly-cited candidate — the guard was doing its job, the key was just wrong.

    Resolution order, each requiring an unambiguous hit:
      1. a DOI appearing anywhere in the claimed title (the strongest possible key),
      2. exact case-insensitive title match after stripping a trailing "(DOI: ...)",
      3. a unique containment match either direction (handles a truncated or prefixed title).
    Anything ambiguous returns None and the candidate is dropped, preserving the guarantee
    that the model cannot introduce a dataset it was never shown.
    """
    claimed = (claimed_title or "").strip()
    if not claimed:
        return None

    doi_match = _DOI_IN_TITLE_RE.search(claimed)
    if doi_match:
        wanted = _strip_doi_prefix(doi_match.group(0)).lower().rstrip(".,;")
        for record in records:
            if _strip_doi_prefix(record.get("doi") or "").lower() == wanted:
                return record

    bare = _TRAILING_DOI_RE.sub("", claimed).strip().lower()
    if not bare:
        return None
    exact = [r for r in records if str(r.get("title", "")).strip().lower() == bare]
    if len(exact) == 1:
        return exact[0]

    partial = [
        r for r in records
        if (t := str(r.get("title", "")).strip().lower()) and (t in bare or bare in t)
    ]
    return partial[0] if len(partial) == 1 else None


# A citation is copied out of the fact sheet, so it arrives carrying that sheet's rendering:
# the section header it sat under, the verbose stored voxelDimensions phrasing, and often real
# newlines. Newlines matter beyond looks — a bullet's continuation lines must be single lines or
# markdown ends the list item and renders the remainder as a separate paragraph, which reads as
# the last dataset's rationale having escaped its bullet.
_CITATION_SECTION_PREFIX_RE = re.compile(
    r"^(?:Samples|Digital datasets(?:\s*\(images/scans\))?|Analysis datasets|"
    r"Related publications|Structure[^:]*)\s*(?:\(\d+\))?\s*:\s*",
    re.IGNORECASE,
)

# "X, Y, Z units (in micrometers): 4.54, 4.54, 4.54" -> "4.54 x 4.54 x 4.54 micrometers".
# A faithful reformatting of the same recorded numbers and unit — not a paraphrase: the values
# are the grounding, so they are never rounded, reordered, or dropped.
# The unit is parenthesised in most sheets but absent in some ("X, Y, Z units: 1024.0, 943.0,
# None. Unit type not provided"), so it's optional here.
_VOXEL_DIMS_RE = re.compile(
    r"X,\s*Y,\s*Z\s*units?\s*(?:\(in\s+(\w+)\))?\s*:\s*"
    r"([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+|None))?",
    re.IGNORECASE,
)

_MAX_CITATION_CHARS = 400


def _one_line(text: str) -> str:
    """Collapse any run of whitespace (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tidy_citation(citation: str) -> str:
    """Make a fact-sheet-derived citation readable on one line, without changing what it
    asserts. Only presentation is touched: whitespace is collapsed, the fact sheet's own
    section header is dropped (it names a section, not evidence), the stored voxel-dimension
    phrasing is compacted, and stray wrapping quotes are removed. Values are never altered.
    Over-long citations are cut with an explicit ellipsis rather than silently."""
    text = _one_line(citation)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = _CITATION_SECTION_PREFIX_RE.sub("", text)

    def _compact_voxels(m: re.Match) -> str:
        unit = m.group(1)
        dims = [d for d in (m.group(2), m.group(3), m.group(4)) if d and d.lower() != "none"]
        compacted = " x ".join(dims)
        return f"{compacted} {unit}" if unit else compacted

    text = _VOXEL_DIMS_RE.sub(_compact_voxels, text)
    text = re.sub(r"\s+;", ";", text).strip()
    if len(text) > _MAX_CITATION_CHARS:
        text = text[:_MAX_CITATION_CHARS].rstrip() + " … (citation truncated)"
    return text


def _render_reasoning_answer(question: str, parsed: dict, records: list[dict]) -> str:
    """Compose the final answer: the fixed honest framing, then the cited shortlist.

    Two deterministic grounding guards run here rather than being left to the prompt
    (the project's standing lesson that a "always cite" instruction alone isn't reliable
    for this model):
      1. A candidate with no citation is dropped — no citation, no candidate.
      2. A candidate whose title isn't one of the fact sheets actually sent is dropped,
         so the model cannot introduce a dataset it was never shown.
    Titles and DOIs in the output come from the graph records, never retyped by the LLM.
    """
    lines = [f"[content reasoning] {_CONTENT_REASONING_FRAMING}", ""]

    kept = []
    seen_numbers = set()
    for candidate in parsed.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        title = str(candidate.get("title", "")).strip()
        citation = str(candidate.get("citation", "")).strip()
        if not citation:
            logger.warning("Dropping uncited content-reasoning candidate: %r", title)
            continue
        record = _match_shortlisted_record(title, records)
        if record is None:
            logger.warning(
                "Dropping content-reasoning candidate not present in the shortlist "
                "(possible fabrication): %r", title,
            )
            continue
        number = record.get("datasetNumber")
        if number in seen_numbers:
            continue  # same dataset named twice — show it once
        seen_numbers.add(number)
        kept.append((record, candidate, citation))

    if not kept:
        return (
            f"[content reasoning] {_CONTENT_REASONING_FRAMING}\n\n"
            "Reasoning over the available dataset facts and descriptions, I couldn't find a "
            "dataset that plausibly matches this — either no dataset does, or the evidence "
            "for it isn't recorded in the portal metadata. Try naming a concrete property "
            "(rock type, segmented status, voxel size, porosity) if one applies."
        )

    for record, candidate, citation in kept:
        doi = _strip_doi_prefix(record.get("doi")) or "not available"
        lines.append(f"- **{record.get('title')}** (DOI: {doi})")
        # Both continuation lines MUST be single lines — see _one_line/_tidy_citation.
        reason = _one_line(candidate.get("reason", ""))
        if reason:
            lines.append(f"  {reason}")
        lines.append(f"  *Basis:* {_tidy_citation(citation)}")

    caveat = _one_line(parsed.get("caveat", ""))
    if caveat:
        # Labelled and separated: an unlabelled trailing paragraph sitting directly under the
        # last bullet reads as that dataset's own rationale rather than a caveat on the whole list.
        lines.append("")
        lines.append(f"*Note: {caveat}*")
    if parsed.get("_truncated"):
        # Never let a truncated list read as a complete one.
        lines.append("")
        lines.append(
            "(This list was cut off before it finished — there may be further matching "
            "datasets not shown. Ask again to see more.)"
        )
    return "\n".join(lines)


def _reason_about_dataset_content(question: str, restrict_to_titles: list[str] | None = None) -> str:
    """Implementation behind the reason_about_dataset_content tool, callable directly by
    get_dataset_details/search_datasets' internal gates (a @tool-decorated function isn't
    ordinarily callable as a plain function).

    Query-time sequence:
      1. Narrow — rank the precomputed fact sheets with the same vector+BM25 RRF fusion
         hybrid_search already runs (GraphStore.rank_fact_sheets), or, for an exhaustive
         question, screen the whole corpus map-reduce style. No LLM call in the ranking
         step. If the caller already knows the exact set (a refinement of a previously
         listed result), skip narrowing entirely and reason over exactly those datasets.
      2. Fetch — a plain, generic, ID-based read of Dataset.factSheet for the shortlist.
      3. Reason — ONE LLM call over the shortlist's fact sheets, which must cite the
         specific fact it relied on per candidate.
      4. Compose — fixed honest framing, then the cited shortlist.
    """
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    store = _get_graph_store()

    try:
        if restrict_to_titles:
            # The caller already knows the exact set to consider (a refinement of a
            # previously listed result). Reason over exactly those — ranking could only
            # lose members of a set that is already correct and small.
            records = store.fetch_fact_sheets(titles=restrict_to_titles)
        elif _is_exhaustive_question(question):
            all_records = store.fetch_fact_sheets()
            logger.warning(
                "Exhaustive content question — map-reduce over %d fact sheets: %r",
                len(all_records), question,
            )
            records = _screen_fact_sheet_batches(question, all_records) if all_records else []
        else:
            ranked = store.rank_fact_sheets(question, top_k=_FACT_SHEET_SHORTLIST_K)
            records = store.fetch_fact_sheets(ranked)
    except Exception as e:
        logger.error("Fact-sheet retrieval failed: %s", e)
        return (
            "I couldn't reason about that right now — the dataset fact-sheet lookup failed. "
            "Please try again, or ask using a concrete property (rock type, segmented "
            "status, voxel size, porosity) so it can be answered by a direct query."
        )

    if not records:
        return (
            f"[content reasoning] {_CONTENT_REASONING_FRAMING}\n\n"
            "I don't have the precomputed dataset fact sheets needed to reason about this "
            "question — they may not have been built for this deployment yet (see "
            "scripts/build_dataset_vector_index.py). I'd rather tell you that than answer "
            "from a partial field lookup that wouldn't actually check what you asked."
        )

    context, included = _build_fact_sheet_context(records)
    prompt = load_prompt("corpus_reasoning")
    system = render(prompt["system"], context=context)
    user = render(prompt["user"], question=question)

    try:
        raw = get_chat_model().send_prompt(
            user, context=system, params={"temperature": 0.2, "max_tokens": 3000}
        )
    except Exception as e:
        logger.error("Content-reasoning LLM call failed: %s", e)
        return (
            "I couldn't complete that reasoning step due to an internal error. Could you "
            "try rephrasing the question?"
        )

    parsed = _parse_reasoning_response(raw)
    if parsed is None:
        logger.error("Could not parse content-reasoning response: %r", (raw or "")[:300])
        return (
            "I reasoned over the relevant datasets but couldn't read back a usable result — "
            "this is an internal formatting failure on my side, not a finding that nothing "
            "matches. Could you try asking again?"
        )
    return _render_reasoning_answer(question, parsed, included)


@tool
def reason_about_dataset_content(question: str) -> str:
    """Answer a question about datasets that CANNOT be settled by looking up a single literal
    field — use it for a relationship between datasets or between one dataset's parts, a
    comparison across a dataset's sub-nodes, or a pattern implied by methodology or content
    rather than by a stored property value.

    Use this when the question contains relational language — "paired", "corresponding", "the
    same sample/rock/resolution", "derived from", "before and after", "imaged at different
    resolutions", "both X and Y images" — or otherwise asks about something that would only
    show up in a description's free text (e.g. the instrument a scan was taken on, which the
    portal has no queryable field for). Pass the WHOLE question, including any literal
    property it also names: a literal clause inside a relational claim ("segmented" inside
    "paired tomographic and segmented images") is not a valid partial answer on its own, and
    must not be split off to get_dataset_details.

    Do NOT use this for a plain conjunction of independent literal properties ("sandstone
    datasets with porosity above 0.3", "segmented carbonate datasets") — each of those clauses
    narrows the catalog on its own and belongs to get_dataset_details, which generates real
    Cypher. Do NOT use it for open-ended discovery or suitability with no checkable property
    ("datasets suitable for LBM simulation") — that's search_datasets.

    The answer is always framed honestly as reasoning rather than a verified database result,
    and every dataset it names cites the specific recorded fact or quoted sentence it was
    inferred from. Source label: [content reasoning]
    """
    return _reason_about_dataset_content(question)


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


def _ensure_all_tutorials_mentioned(response: str, tutorials: list[dict]) -> str:
    """Complement to _strip_fabricated_tutorial_reference: educational.yaml already
    instructs the model to "list every matched tutorial explicitly... do not
    paraphrase or omit," but when more than one tutorial matches, this "always
    mention all of X" instruction was observed dropping one anyway (live, 2/4 runs
    for a query matching both the Minkowski Functionals and Connected Components
    tutorials — same query, same context, same prompt, only the model's own content
    selection varied). Per this project's own repeatedly-validated lesson, a prompt-
    only "always do X" instruction isn't reliable for this model — deterministically
    append any matched tutorial whose notebook path isn't already verbatim in the
    response, in the same Goal/Notebook format the prompt itself specifies."""
    missing = [t for t in tutorials if t["notebook"] not in response]
    if not missing:
        return response

    lines = []
    for t in missing:
        nb_path = t["notebook"]
        nb_filename = nb_path.split("/")[-1]
        name_part = nb_filename.replace(".ipynb", "")
        name_clean = re.sub(r"^\d[\d\-]*_", "", name_part).replace("_", " ")
        lines.append(f'  - **{name_clean}** — {t["goal"]}')
        lines.append(f'    Path in Community Data: `{nb_path}`')
    return response.rstrip() + "\n\n" + "\n".join(lines)


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
    response = _strip_fabricated_tutorial_reference(response, tutorials)
    return _ensure_all_tutorials_mentioned(response, tutorials)


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

@tool
def search_portal_docs(question: str) -> str:
    """Search the DPM Portal user documentation for how-to guides and metadata schema reference.

    Covers: dataset submission guidelines, portal navigation, metadata field definitions,
    and file format requirements sourced from https://github.com/digital-porous-media/dpm_docs.
    Source label: [portal docs].
    """
    from src.assistant.portal_docs_retrieval import search_portal_docs_v2
    return search_portal_docs_v2(question)


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
        get_dataset_profile,
        reason_about_dataset_content,
        search_portal_docs,
        get_workflow_guidance,
        get_educational_context,
        search_literature,
    ]
