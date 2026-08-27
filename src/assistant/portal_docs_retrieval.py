"""
LLM-reasoning-based retrieval over the portal-docs heading tree — the "retrieval"
half of a hand-rolled, PageIndex-style approach to search_portal_docs
(src/assistant/tools.py). See portal_docs_tree.py's module docstring for the
"index" half and background/rationale, and HANDOFF.md's PageIndex prototype
section for the full write-up, including the FAISS/chunk-based retrieval path
this replaced.

`search_portal_docs_v2` is `tools.search_portal_docs`'s entire implementation —
this module is not registered as its own LangChain tool; `tools.py` just
delegates to it directly.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_FIGURE_MARK = "[Figure:"

# How much of a node's own plain body text (not its descendants' — see node.text
# vs. node.full_text in portal_docs_tree.py) to surface to the selector alongside
# its title, for sections that are prose rather than a field list. Titles alone
# miss body-level field names ("Reference Sample") that never appear as their own
# heading — see HANDOFF.md's PageIndex prototype "Update 10" section for the
# concrete bug this fixes.
_SNIPPET_CHARS = 200

# dpm_docs consistently bolds field names in its "Curate Your Dataset" reference
# sections (e.g. "*   **Reference Sample***: ..."). A field name like "Reference
# Sample" can sit well past a fixed character-count prefix in a long bullet list
# (each bullet includes a "Best practice:" sub-explanation, so real field names
# routinely fall 500+ chars into a section) — pulling out every bolded term
# instead of truncating at a char count is what actually surfaces them regardless
# of list position.
_BOLD_TERM_RE = re.compile(r"\*\*([^*]+)\*\*")


def _index_line(node) -> str:
    """Serialize one node for the LLM selector: id, titles, then a body-text
    signal — every distinct bolded field name in the node's own text (if any),
    followed by a plain-text prefix (for prose sections with no bolded fields)."""
    base = f"{node.node_id}: {node.page_title} — {node.title} (H{node.level})"

    seen = set()
    fields = []
    for m in _BOLD_TERM_RE.finditer(node.text):
        term = m.group(1).strip().rstrip("*").strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            fields.append(term)

    snippet_parts = []
    if fields:
        snippet_parts.append("fields: " + ", ".join(fields))
    plain = " ".join(node.text.split())[:_SNIPPET_CHARS]
    if plain:
        snippet_parts.append(plain)
    snippet = " | ".join(snippet_parts)

    return f"{base} | {snippet}" if snippet else base


def select_nodes_for_query(question: str, nodes: list, max_nodes: int = 4) -> list[str]:
    """One LLM call: given the full flattened node list (id, page title, section
    title, heading level), return up to max_nodes node ids most relevant to the
    question. Mirrors tools._match_workflows' existing pattern (index_lines +
    JSON-array-of-ids response, same markdown-fence-stripping/parse-failure
    fallback) — reused verbatim rather than inventing a new response shape.

    Explicitly instructed to return one node per named entity for a comparison-style
    question ("difference between X and Y") — this directly targets the definition-
    conflation bug that motivated this prototype (HANDOFF.md Update 7 issue #1):
    unlike embedding similarity over prose (which was biased toward whichever
    section's text most densely repeated a shared word), an LLM reasoning over short,
    clearly-differentiated titles has no equivalent "long text wins" bias — but it
    still needs an explicit instruction to select *all* the entities asked about, not
    just the closest single match.

    Falls back to a keyword/title-substring match on JSON-parse failure — never
    raises, never returns an empty list just because this one LLM call hiccuped.
    """
    from src.assistant.llm import get_chat_model, strip_code_fences

    index_str = "\n".join(_index_line(n) for n in nodes)
    system = (
        "You are a documentation retrieval system. Given a user question and a list of "
        "documentation section ids (id: page title — section title (heading level)), "
        "return a JSON array of the ids most relevant to answering the question. "
        f"Return at most {max_nodes} ids, ordered by relevance. Only include an id if it "
        "is genuinely relevant to the question — it is fine and expected to return fewer "
        f"than {max_nodes} ids, or even just one, when only a small number of sections "
        "are actually relevant. Do not pad the list with weakly- or generically-related "
        "sections (e.g. an unrelated page that merely shares a common word like "
        "\"dataset\") just to reach the limit. "
        "If the question asks about a difference or comparison between two or more "
        "named entities/concepts, return one id per entity that has its own matching "
        "section — not just the single closest match. "
        "Return ONLY valid JSON, no explanation. If nothing is relevant, return []."
    )
    user_msg = f"Question: {question}\n\nSections:\n{index_str}"

    try:
        raw = get_chat_model().send_prompt(
            user_msg, context=system, params={"temperature": 0, "max_tokens": 150}
        )
        ids = json.loads(strip_code_fences(raw))
        if not isinstance(ids, list):
            raise ValueError("not a list")
    except Exception as e:
        logger.warning(
            "select_nodes_for_query LLM call failed (%s); falling back to keyword match", e
        )
        return _select_nodes_keyword(question, nodes, max_nodes)

    valid_ids = {n.node_id for n in nodes}
    seen = set()
    deduped_ids = []
    for nid in ids:
        if nid in valid_ids and nid not in seen:
            seen.add(nid)
            deduped_ids.append(nid)
    return deduped_ids[:max_nodes]


# Common words excluded from the keyword fallback's word-overlap scoring — without
# this, including full body text (see _select_nodes_keyword) would let generic words
# ("the", "in", "what") that appear in nearly every section's prose dilute the
# signal from actually-distinctive words, which titles (being short) didn't suffer
# from as much.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "is", "are",
    "what", "how", "do", "does", "i", "it", "this", "that", "with", "should",
    "put", "you", "your", "be", "can",
}


def _select_nodes_keyword(question: str, nodes: list, max_nodes: int) -> list[str]:
    """Keyword fallback — used only when the LLM call fails or returns unparseable
    JSON. Scores each node on two signals:

    - Plain word overlap between the question and the node's page title + section
      title + own body text (title words count double — a real title match should
      still generally win over an incidental body mention).
    - A verbatim bolded-field-name match (reusing _BOLD_TERM_RE, same field-name
      convention _index_line relies on) weighted heavily. Plain word overlap alone
      isn't discriminating enough when a query names a specific field ("Reference
      Sample") that shares generic words ("sample", "field") with an unrelated
      section ("Sample") — but "Reference Sample" as a whole bolded field name is
      distinctive and only actually defined in one section, so an exact phrase
      match there is a much stronger signal than word-level overlap.
    """
    q_words = set(re.findall(r"[a-z0-9]+", question.lower())) - _STOPWORDS
    q_lower = question.lower()
    scored = []
    for n in nodes:
        title_words = set(re.findall(r"[a-z0-9]+", f"{n.page_title} {n.title}".lower()))
        body_words = set(re.findall(r"[a-z0-9]+", n.text.lower()))
        hits = 2 * len(q_words & title_words) + len(q_words & body_words)

        field_terms = {
            m.group(1).strip().rstrip("*").strip().lower()
            for m in _BOLD_TERM_RE.finditer(n.text)
        }
        phrase_hits = sum(1 for term in field_terms if term and term in q_lower)
        hits += phrase_hits * 5

        if hits:
            scored.append((hits, n.node_id))
    scored.sort(key=lambda t: -t[0])
    return [nid for _, nid in scored[:max_nodes]]


# dpm_docs' field-reference sections always introduce a field as a bolded bullet,
# e.g. "*   **Name***: The name of the digital dataset." — this marks the boundary
# between a section's leading conceptual/prose overview and its field-by-field
# reference detail, regardless of whether that detail lives under a separate child
# heading (e.g. "3. Digital Dataset" -> "Core Digital Dataset Information") or
# inline in the same node with no sub-heading at all (e.g. "4. Analysis Dataset",
# which has no children — see HANDOFF.md's "Update 12" section: splitting only on
# node.children missed this case entirely, since the intro sentence and the entire
# field list are both part of that one node's own text). Splitting on the actual
# bullet boundary instead of the tree structure handles both shapes uniformly.
_FIRST_FIELD_BULLET_RE = re.compile(r"\n\*\s+\*\*")


def _split_overview_and_details(text: str) -> tuple[str, str]:
    """Split text at the first bolded field bullet into (leading prose, field-
    reference tail). Returns (text, "") if no bullet is found (pure prose, nothing
    to split) or ("", text) if the bullet list starts immediately (no separate
    overview to extract)."""
    m = _FIRST_FIELD_BULLET_RE.search(text)
    if not m:
        return text, ""
    return text[: m.start()].strip(), text[m.start():].lstrip("\n")


def _format_portal_doc_node(node) -> str:
    """Render one selected tree node as labeled context text for the synthesis LLM —
    same [portal docs] labeling shown to users. Uses node.full_text (own text + every
    descendant's text, concatenated) rather than a single chunk's truncated slice —
    this is what solves the truncated-procedure bugs (HANDOFF.md Update 7 issues #2/
    #3): a "### Step 2: Download the Dataset" node's full_text always contains the
    complete step, with no 500-char/100-overlap chunk boundary (the old FAISS path's
    chunk size — see HANDOFF.md) to cut it off mid-instruction.

    A section's leading conceptual overview (e.g. "A digital dataset is always
    linked to a specific Sample...") is typically a small fraction of full_text —
    the field-by-field reference detail that follows is often 10x+ longer (see
    HANDOFF.md's "Update 11"/"Update 12" sections). Handing over one undifferentiated
    blob biases synthesis toward the much larger field-list content regardless of
    prompt wording; explicitly labeling "Overview" vs. "Reference details" gives the
    model the same signal a human skimming the doc gets for free from font
    size/position."""
    header = f"{node.page_title} — {node.title}" if node.title != node.page_title else node.page_title
    full_text = node.full_text.strip()
    overview, detail = _split_overview_and_details(full_text)
    if overview and detail:
        body = (
            f"Overview: {overview}\n\n"
            f"Reference details (field-level specifics — use only if the question "
            f"asks about a specific field):\n{detail}"
        )
    else:
        body = full_text
    return f"[portal docs] {header}\n\n{body}\nSource: {node.doc_url}"


# upload_data.md's "## Curate Your Dataset" section has four sibling H3 children:
# "1. Dataset" (the overarching container), "2. Sample", "3. Digital Dataset", and
# "4. Analysis Dataset". "1. Dataset"'s own intro sentence ("This is the main
# container for your work, connecting the physical sample, digital data, analysis,
# and any related publications.") is what gives a definitional answer about one of
# the three sub-entities its relational framing — confirmed by direct comparison
# against the FAISS path, whose independent chunk-based retrieval happened to
# surface this sentence as a separate hit alongside the sub-entity's own chunk (see
# HANDOFF.md's "Update 11" section). The prototype's node selector has no equivalent
# mechanism, since these are siblings, not ancestor/descendant — surfacing it
# explicitly here is cheap (one short sentence) and scoped to this one known
# relationship rather than a general "always include siblings" rule.
_CURATE_DATASET_PREFIX = "curate-your-dataset/"
_DATASET_CONTAINER_NODE_SUFFIX = "curate-your-dataset/1-dataset"


def _dataset_container_context(results, id_to_node) -> str | None:
    """If any selected node is a Dataset sub-entity (Sample/DigitalDataset/
    AnalysisDataset) under upload_data.md's "Curate Your Dataset" section, and the
    "1. Dataset" container node itself wasn't already selected, return a lightweight
    rendering of just its own overview sentence — not its own field list, which
    isn't relevant here. Returns None when it doesn't apply, so callers can skip it
    cleanly."""
    if any(_DATASET_CONTAINER_NODE_SUFFIX in n.node_id for n in results):
        return None  # already selected on its own merits — don't duplicate
    if not any(_CURATE_DATASET_PREFIX in n.node_id for n in results):
        return None  # this page/section wasn't involved at all

    container = next(
        (n for nid, n in id_to_node.items() if nid.endswith(_DATASET_CONTAINER_NODE_SUFFIX)),
        None,
    )
    if container is None or not container.text.strip():
        return None

    header = f"{container.page_title} — {container.title}"
    return f"[portal docs] {header}\n\n{container.text.strip()}\nSource: {container.doc_url}"


# Keywords that mark a question as genuinely comparing two or more named entities
# — _restrict_to_best_title_match must not fire here, since select_nodes_for_query
# is deliberately instructed to return multiple ids for these (HANDOFF.md Update 7
# issue #1 / Update 8's whole rationale for this prototype).
_COMPARISON_KEYWORDS = ("difference", "compare", "comparison", "versus", " vs ", "distinguish")


def _is_comparison_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _COMPARISON_KEYWORDS)


def _normalize_title(title: str) -> str:
    """Strip leading list numbering ("3. ") and punctuation, lowercase — so
    "3. Digital Dataset" and "Digital Dataset" compare equal for substring
    matching against a normalized question."""
    stripped = re.sub(r"^\d+\.\s*", "", title)
    return re.sub(r"[^a-z0-9\s]", "", stripped.lower()).strip()


def _restrict_to_best_title_match(question: str, results: list) -> list:
    """select_nodes_for_query was observed padding its result list with weakly- or
    only-generically-related sections (e.g. "Community Data", a page-overview
    "Data Model" section) alongside the one node that's an exact match for the
    question's named entity — an explicit anti-padding prompt instruction reduced
    but did not reliably eliminate this (same lesson this project has repeatedly
    hit: prompt-only reliability fixes for this model are inconsistent). When
    exactly one selected node's title is a near-verbatim match for the query (and
    the query isn't a genuine multi-entity comparison, which legitimately needs
    several nodes), keep only that node — diluting a precise match with weaker
    padding measurably degraded synthesis quality (see HANDOFF.md's "Update 12"
    section: 8/8 live runs recited an exhaustive field list instead of a
    conceptual definition when padding nodes were present in context)."""
    if len(results) <= 1 or _is_comparison_query(question):
        return results
    q_norm = _normalize_title(question)
    strong_matches = [n for n in results if _normalize_title(n.title) and _normalize_title(n.title) in q_norm]
    if len(strong_matches) == 1:
        return strong_matches
    return results


def _ensure_source_urls_present(response: str, results) -> str:
    """portal_docs.yaml already instructs the model to list sources "with their
    doc_url," and every node's own rendered context includes a literal "Source:
    <url>" line — but the model was observed dropping the actual URL from its own
    "Sources:" line the large majority of the time (9/10 live runs for "What is a
    Digital Dataset?", same context, same prompt, only the model's own citation-line
    transcription varied). Same lesson as tools._ensure_all_tutorials_mentioned: a
    prompt-only "always include X" instruction isn't reliable for this model —
    deterministically append any used node's doc_url that isn't already verbatim in
    the response, rather than relying on the model to transcribe it every time."""
    seen = set()
    missing = []
    for n in results:
        if n.doc_url and n.doc_url not in seen:
            seen.add(n.doc_url)
            if n.doc_url not in response:
                missing.append(n.doc_url)
    if not missing:
        return response
    return response.rstrip() + "\n" + "\n".join(missing)


def _strip_fabricated_figure_reference(response: str, has_figure: bool) -> str:
    """Deterministic guard against figure-mention hallucination.

    portal_docs.yaml instructs the model to mention that a screenshot exists ONLY when
    an excerpt it actually used contains a "[Figure: ...]" placeholder. Observed
    (Llama-4-Maverick via SambaNova/TACC) over-generalizing this to "mention a
    screenshot whenever discussing a UI step," fabricating the mention even when none
    of the retrieved excerpts contained a placeholder at all. Prompt wording alone
    wasn't reliable for the analogous tutorial-path hallucination (see
    tools._strip_fabricated_tutorial_reference) — same fix here: strip any sentence
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


def search_portal_docs_v2(question: str) -> str:
    """`tools.search_portal_docs`'s entire implementation — see that function's
    docstring for the user-facing contract (conversation_manager._SELF_CONTAINED_TOOLS
    relies on this being a self-contained answer, not a verbatim tool). Selects
    whole documentation *sections* via LLM reasoning over the heading tree
    (select_nodes_for_query). See module docstring for background."""
    from src.assistant.portal_docs_tree import flatten, get_portal_docs_tree
    from src.prompts.loader import load_prompt, render
    from src.assistant.llm import get_chat_model

    forest = get_portal_docs_tree()
    if not forest:
        return (
            "Portal documentation search is not yet available (no documentation pages "
            "found). For step-by-step workflow guidance, try get_workflow_guidance(). "
            "For structured dataset property queries, try get_dataset_details()."
        )

    flat_nodes = flatten(forest)
    selected_ids = select_nodes_for_query(question, flat_nodes)
    id_to_node = {n.node_id: n for n in flat_nodes}
    results = [id_to_node[nid] for nid in selected_ids if nid in id_to_node]
    results = _restrict_to_best_title_match(question, results)

    if not results:
        return (
            "No portal documentation found matching that question. "
            "Try asking about a specific workflow or general topic instead, "
            "or rephrase your question."
        )

    has_figure = any(_FIGURE_MARK in n.full_text for n in results)
    context_blocks = [_format_portal_doc_node(n) for n in results]
    container_block = _dataset_container_context(results, id_to_node)
    if container_block:
        context_blocks.insert(0, container_block)
    context = "\n\n".join(context_blocks)
    prompt = load_prompt("portal_docs")
    system = render(prompt["system"], context=context)
    user = render(prompt["user"], question=question)
    response = get_chat_model().send_prompt(
        user, context=system, params={"temperature": 0.2, "max_tokens": 800}
    )
    # Only strip a fabricated screenshot mention — no longer proactively appends one
    # when a figure genuinely exists (see _strip_fabricated_figure_reference's
    # call site comment for the rationale).
    response = _strip_fabricated_figure_reference(response, has_figure)
    return _ensure_source_urls_present(response, results)
