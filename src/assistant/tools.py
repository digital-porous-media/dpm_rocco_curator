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
        search_portal_docs,
        get_workflow_guidance,
        get_educational_context,
        search_literature,
    ]
