from __future__ import annotations
"""
Shared conversation manager for the General Assistant.

Top-level orchestrator for the Rocco General Assistant. Wraps a LangGraph
ReAct agent with in-memory per-session checkpointing. There is no hardcoded
intent dispatcher — the LLM selects tools based on their descriptions and the
system prompt below.

Intent → Tool Routing
---------------------
The agent performs routing implicitly. The mapping is:

  Intent                  Primary tool(s)
  ----------------------  -------------------------------------------------------
  Dataset discovery       search_datasets        (semantic similarity, Neo4j vector index;
                                                  purpose/suitability queries with no
                                                  precise checkable property named)
  Structured queries      get_dataset_details    (Cypher QA; any query naming a concrete
                                                  property, numeric threshold/range, or
                                                  multiple values/fields — even combined
                                                  with a rock type or imaging method)
  Dataset follow-up /     get_dataset_profile    (full profile of ONE already-identified
  profile / comparison                           dataset: org structure, file types/data
                                                  location, reuse-suitability reasoning;
                                                  called once per dataset for comparisons)
  Relationship / content  reason_about_dataset_content
  questions                                      (anything not answerable by a literal
                                                  field: "paired ... images", "the same
                                                  sample at different resolutions",
                                                  instrument named only in free text —
                                                  ranked fact sheets + one cited
                                                  reasoning pass, honestly framed)
  Portal how-to / schema  search_portal_docs     (dpm_docs markdown parsed into a heading
                                                  tree at runtime, LLM-selected sections —
                                                  see src/assistant/portal_docs_tree.py)
  Domain Q&A              get_educational_context (workflows + global best practices)
  Workflow guidance       get_workflow_guidance   (step-by-step DRP workflows + tutorial links)
  Literature              search_literature       (Semantic Scholar API)
  Vague / ambiguous       expand_query (internal) (called before search; NOT a LangChain tool)

Cross-intent queries (e.g. "explain relative permeability and find me datasets that measure
it") trigger multiple tool calls in sequence. The agent synthesizes results into one response,
preserving the source labels ([graph match], [semantic scholar], etc.) returned by each tool.

Session isolation: each session_id maps to an independent MemorySaver thread. Memory resets
on process restart — there is no persistent storage.
"""

import json
import logging
import re

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent

from src.assistant.tools import build_langchain_tools

# Tools whose output must reach the user byte-for-byte: they already contain
# real DOIs/titles/descriptions assembled from tool code, not the model's own words.
# Letting the ReAct agent's final synthesis turn retype this content invites exactly
# the failure mode prompt instructions can't reliably prevent — dropped or
# hallucinated DOIs/descriptions — because the model is reproducing structured data
# from memory of what it just read rather than being handed it verbatim.
_VERBATIM_TOOLS = {"search_datasets", "get_dataset_details"}

# Tools that already return a complete, final, user-ready answer authored by their own
# grounded LLM/chain call (GraphCypherQAChain's QA step, educational.yaml's synthesis,
# etc.) — never re-run through the outer agent's own synthesis, which has no grounding
# in the underlying data and will invent identifiers/DOIs/citations it wasn't given
# (e.g. get_dataset_details' answer has no DOI to draw from unless its own Cypher
# fetched one, but the outer SYSTEM_PROMPT's "always include a DOI" instruction still
# pressures it to guess one if allowed to retype the answer).
# search_portal_docs belongs here rather than in _VERBATIM_TOOLS: unlike
# search_datasets/get_dataset_details, its raw retrieval is prose sections (dpm_docs
# excerpts) that answer the question only if something actually reads them and
# synthesizes — pasting them verbatim produced disconnected, sometimes off-topic
# dumps instead of an answer. Its own LLM call (portal_docs.yaml) does that
# synthesis and cites [portal docs] sources, so it's self-contained like
# get_educational_context/get_workflow_guidance.
# get_dataset_profile also belongs here, not in _VERBATIM_TOOLS: it must reason over its
# fetched graph data (file-format/"how do I read this" guidance, reuse-suitability
# judgments, a concise high-level synthesis for general "tell me more" questions) rather
# than reproduce a fixed data shape verbatim — its own LLM call (dataset_profile.yaml) is
# grounded in the real fetched profile and prepends a code-generated, never-retyped
# [dataset profile] title/DOI header, same as the other self-contained tools' own citations.
# reason_about_dataset_content belongs here for the same reason: it composes a fixed,
# non-negotiable honesty framing plus a citation-checked shortlist whose titles/DOIs come
# from graph records, never from the model. Letting the outer agent retype that would put
# the framing sentence and the citations back in the model's hands — exactly the two
# things the tool exists to guarantee in code.
_SELF_CONTAINED_TOOLS = {
    "get_workflow_guidance", "get_educational_context", "search_portal_docs", "get_dataset_profile",
    "reason_about_dataset_content",
}

# Tags search_datasets prepends for the narrator's benefit only — never meant to
# reach the user as literal bracketed text (see tools.py: rationale/weak-match tags).
_LEADING_TAG_RE = re.compile(r'^\[(?:search reasoning|weak match)[^\]]*\]\n\n', re.IGNORECASE)


def _strip_leading_tags(text: str) -> str:
    while True:
        m = _LEADING_TAG_RE.match(text)
        if not m:
            return text
        text = text[m.end():]


_WRAPPER_SYSTEM_PROMPT = """\
You write a short introduction sentence for a set of dataset search results. You do NOT \
see the actual list rendered — it will be spliced in verbatim right after your sentence.

Respond with a JSON object only, no markdown fences:
{"lead_in": "one short sentence introducing the results"}

Rules:
- Never write a DOI, title, or dataset description in your output — you have not seen \
the exact list, only the query and raw tool context, so anything specific you write \
about individual results risks being wrong.
- If the tool context includes a "[weak match: ...]" tag, the lead_in must plainly say \
no results directly matched the topic and the closest available results are shown \
instead — do not invent a reason they might be relevant.
- If the tool context includes a "[search reasoning: ...]" tag, the lead_in may \
paraphrase (never quote verbatim) why these properties suit the user's stated purpose.
- If the tool context says no datasets were found, say so plainly.
"""

# A closing synthesis sentence generated per-query ("these datasets share a focus on
# pore-scale processes...") was tried and dropped: with no access to the actual result
# data, the model could only produce vague, near-content-free observations. A fixed
# disclaimer is actually useful information, unlike a vague one; static text also runs
# no risk of quietly asserting something wrong about the results.
_VERIFICATION_DISCLAIMER = (
    "Please verify these datasets on the DPM Portal before use in your research — "
    "search results may not capture every detail of a dataset's metadata."
)


# Bare "nothing to show" tool outputs that short-circuit _build_verbatim_response
# before _generate_lead_in is ever called — kept at module level so both functions
# check against the same list instead of a second hand-copied one.
_BARE_MESSAGE_PREFIXES = (
    "no datasets found",
    "the query ran successfully and found no matching",
    "graph search is disabled",
    "no answer found",
)

# Catches a lead-in that asserts no results were found ("No datasets by X were found",
# "None found", "couldn't find any...") — used to detect when the lead-in LLM invents
# this framing on its own despite real results sitting right below it. The only
# legitimate case for this framing is a "[weak match: ...]" tag in tool_output (see
# _WRAPPER_SYSTEM_PROMPT); a flat no-results tool_output never reaches _generate_lead_in
# at all, since _build_verbatim_response short-circuits on _BARE_MESSAGE_PREFIXES first.
_NEGATION_LEADIN_RE = re.compile(
    r"\bno\b[^.]{0,40}\bfound\b|\bnone found\b|\bnot found\b|"
    r"\bdid(?:n't| not) find\b|\bweren't found\b|\bwasn't found\b",
    re.IGNORECASE,
)


_DEFAULT_LEAD_IN = "Here are the datasets matching your query:"


def _generate_lead_in(user_input: str, tool_output: str) -> str:
    """Ask the LLM for a one-sentence lead-in, with no access to reproducing the
    actual result data — the data itself is spliced in by code. Only called for tool
    output carrying a "[weak match]" or "[search reasoning]" tag (see
    _build_verbatim_response) — those are the only cases with anything nuanced for
    the LLM to add; a plain confident match uses _DEFAULT_LEAD_IN directly without an
    LLM call at all, since the model isn't reliably steerable away from inventing
    "no results" framing even when tool_output contains none (see history below)."""
    from src.assistant.llm import get_chat_model

    try:
        llm = get_chat_model()
        response = llm.invoke([
            {"role": "system", "content": _WRAPPER_SYSTEM_PROMPT},
            {"role": "user", "content": f"User query: {user_input}\n\nTool context:\n{tool_output}"},
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            raw = raw[len("json"):] if raw.startswith("json") else raw
        lead_in = (json.loads(raw.strip()).get("lead_in") or _DEFAULT_LEAD_IN).strip()
    except Exception as e:
        logger.warning("Lead-in generation failed (%s); using default.", e)
        return _DEFAULT_LEAD_IN

    # Deterministic backstop for the "[search reasoning]" case (a positive suitability
    # match, where a "no match" framing is never correct): the "[weak match]" case is
    # legitimately allowed to say results weren't found, so this only fires for the
    # other tagged path. Kept as a second line of defense, not the primary one — a
    # blocklist of negation phrasings is beatable by paraphrase (this model has been
    # observed rephrasing around an earlier version of this exact check, e.g. "did not
    # directly match... but here are some related datasets" instead of "not found").
    # The real fix is upstream in _build_verbatim_response: this function is no longer
    # called at all for the plain-match case that originally triggered the bug.
    if "[weak match" not in tool_output.lower() and _NEGATION_LEADIN_RE.search(lead_in):
        logger.warning(
            "Lead-in hallucinated a no-results framing despite real tool output; discarding: %r",
            lead_in,
        )
        return _DEFAULT_LEAD_IN

    return lead_in


def _build_verbatim_response(user_input: str, tool_output: str) -> str:
    """Assemble the final response with the tool's data untouched — only the
    lead-in sentence comes from the LLM; the closing note is fixed, static text."""
    display_block = _strip_leading_tags(tool_output).strip()
    lowered = display_block.lower()
    if lowered.startswith(_BARE_MESSAGE_PREFIXES):
        return display_block

    # An LLM-authored lead-in only has something case-specific to add when tool_output
    # carries a tag asking for nuance (paraphrase the suitability rationale, or honestly
    # flag a weak/off-topic match) -- the model is explicitly forbidden from naming
    # titles/DOIs either way, so for a plain confident match there is nothing for it to
    # contribute. Skip the LLM call entirely in that case rather than give a free-form
    # model an opportunity to invent framing that contradicts the real results sitting
    # right below it (the original "No datasets by X were found" bug, on a tool_output
    # with zero indication of that).
    output_lower = tool_output.lower()
    needs_nuanced_lead_in = "[weak match" in output_lower or "[search reasoning" in output_lower
    lead_in = _generate_lead_in(user_input, tool_output) if needs_nuanced_lead_in else _DEFAULT_LEAD_IN

    return "\n\n".join([lead_in, display_block, _VERIFICATION_DISCLAIMER])

logger = logging.getLogger(__name__)

# Llama-4-Maverick emits its native function-call tokens as plain text when
# LangGraph doesn't intercept them. Strip these so they never reach the UI.
_TOOL_CALL_RE = re.compile(r'<\|python_start\|>.*?<\|python_end\|>', re.DOTALL)

# Error substrings that indicate LiteLLM rejected the model's tool-call format.
_TOOL_FORMAT_ERRORS = ("JSONDecodeError", "dict_type", "Invalid function calling output")

# Maps each tool name to its required string parameter key(s), in call order.
# Used to reconstruct tool calls from the error's model_output field. Most tools take
# exactly one required param; get_dataset_profile takes two (dataset_reference, question) —
# both extraction functions below iterate this list rather than assuming a single key.
_TOOL_PARAM_KEYS: dict[str, list[str]] = {
    "get_educational_context": ["question"],
    "get_workflow_guidance": ["goal"],
    "search_datasets": ["query"],
    "get_dataset_details": ["question"],
    "get_dataset_profile": ["dataset_reference", "question"],
    "reason_about_dataset_content": ["question"],
    "search_literature": ["query"],
    "search_portal_docs": ["question"],
}

# Returned when a tool call was identified but produced nothing usable after a 400
# tool-format error — deliberately honest rather than falling back to an ungrounded
# direct LLM guess (mirrors tools.py's _HONEST_NO_TUTORIAL_MSG pattern).
_HONEST_TOOL_FAILURE_MSG = (
    "I wasn't able to complete that lookup due to an internal issue processing the "
    "request. Could you try rephrasing your question?"
)


def _non_empty(text: str | None, fallback: str = _HONEST_TOOL_FAILURE_MSG) -> str:
    """Guarantee a non-empty, non-whitespace-only response string.

    A leaked `<|python_start|>...<|python_end|>` block that is stripped down to nothing by
    _clean_response (or a synthesis LLM call that returns an empty completion) must never
    surface as a literal "" response: appending "" into the UI's session history poisons
    every later turn's replayed context (see HANDOFF.md — this was the actual mechanism
    behind "comparing two datasets silently returns nothing, and subsequent turns also
    return nothing until a different tool is used")."""
    return text if text and text.strip() else fallback


# Dataset-listing tools whose rendered output includes an ordered list of "Title (DOI: ...)"
# entries — used by _extract_dataset_mentions/_resolve_reference below to let a later
# ordinal/name-only follow-up ("the first one", "the Gildehauser sandstone sample") resolve
# deterministically instead of relying solely on the LLM re-deriving it from raw chat history.
#
# reason_about_dataset_content belongs here even though it is NOT a verbatim tool (it is
# self-contained — see _SELF_CONTAINED_TOOLS): what makes a tool trackable is the SHAPE of
# its output, not which relay path it takes. Its answer renders the same
# "- **Title** (DOI: ...)" bullets as _format_dataset_rows, from graph records rather than
# retyped by the model, so _extract_dataset_mentions parses it unchanged.
#
# Leaving it out was live-observed producing a wrong answer, not merely a missed one: a
# content-reasoning turn ("datasets with both raw and segmented images", 12 results) went
# completely untracked, so the very next turn's "What are the lithologies of these?" matched
# _REFINEMENT_RE and refined the STALE listing still held from several turns earlier
# ("datasets suitable for training a segmentation model"). "These" silently resolved to a
# result set the user had already moved on from, with nothing in the answer to reveal the
# substitution. An untracked dataset-listing turn doesn't just fail to update this state —
# it leaves the previous turn's state in place looking current.
_DATASET_LISTING_TOOLS = _VERBATIM_TOOLS | {"reason_about_dataset_content"}

_DATASET_MENTION_PATTERNS = [
    # search_datasets/hybrid_search: "[label] Title — matched via ... (DOI: xxx)"
    re.compile(r'^\[[\w\s]+\]\s+(?P<title>.+?)(?:\s+—\s+matched via[^(\n]*)?\s*\(DOI:\s*(?P<doi>[^)]*)\)', re.MULTILINE),
    # get_dataset_details / _format_dataset_rows: "- **Title** (DOI: xxx)"
    re.compile(r'^-\s+\*\*(?P<title>.+?)\*\*\s*\(DOI:\s*(?P<doi>[^)]*)\)', re.MULTILINE),
]


def _extract_dataset_mentions(tool_output: str) -> list[dict]:
    """Best-effort, ordered extraction of {"title", "doi"} pairs from a dataset-listing
    tool's rendered text. Not a structured API — these tools return plain strings by design
    (see _VERBATIM_TOOLS) — so this just parses the same "Title (DOI: xxx)" shapes a human
    reader would use to identify which dataset is "the first one"."""
    mentions: list[dict] = []
    seen_titles: set[str] = set()
    for pattern in _DATASET_MENTION_PATTERNS:
        for m in pattern.finditer(tool_output):
            title = m.group("title").strip()
            doi_raw = m.group("doi").strip()
            doi = doi_raw if doi_raw and doi_raw.lower() != "not available" else None
            if title and title not in seen_titles:
                seen_titles.add(title)
                mentions.append({"title": title, "doi": doi})
    return mentions


# get_dataset_profile's own code-generated, never-retyped header (tools.py:
# f"[dataset profile] {title} (DOI: {doi})") — parsed the same deterministic way as
# _DATASET_MENTION_PATTERNS above so a later anaphoric comparison ("how does that
# dataset compare with X") can resolve "that dataset" to whichever single dataset was
# profiled most recently, the same way _resolve_reference resolves against a listing.
_PROFILE_HEADER_RE = re.compile(r'^\[dataset profile\]\s+(?P<title>.+?)\s*\(DOI:\s*(?P<doi>[^)]*)\)', re.MULTILINE)


def _extract_profiled_dataset(tool_output: str) -> dict | None:
    """Parse the {"title", "doi"} of a single get_dataset_profile call from its header.
    Returns None if the text doesn't start with a recognizable profile header (e.g. an
    ambiguous-match or not-found response, which has no dataset to remember)."""
    m = _PROFILE_HEADER_RE.search(tool_output)
    if not m:
        return None
    title = m.group("title").strip()
    doi_raw = m.group("doi").strip()
    doi = doi_raw if doi_raw and doi_raw.lower() != "not available" else None
    return {"title": title, "doi": doi} if title else None


_ORDINAL_WORDS = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
    "last": -1,
}
_ORDINAL_RE = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in _ORDINAL_WORDS) + r')\b(?:\s+(?:one|dataset|result))?',
    re.IGNORECASE,
)


def _title_mentioned(title: str, lowered_message: str) -> bool:
    """True if `title` — or a distinguishing leading portion of it — appears in
    `lowered_message` (already lowercased). Titles are often longer/more formal than how a
    user refers to them conversationally (e.g. the user says "Leman Sandstone" for a mention
    titled "Leman Sandstone SEM Images"), so a full-title-only substring check misses real
    references. Falls back to the title's leading two words (or the whole title, if it's
    shorter than that) as an anchor — long enough to avoid a single generic word like
    "sandstone" alone matching everything, short enough to tolerate a truncated title."""
    t = title.lower()
    if t in lowered_message:
        return True
    words = t.split()
    anchor_len = min(2, len(words))
    if anchor_len == 0:
        return False
    return " ".join(words[:anchor_len]) in lowered_message


def _resolve_reference(user_input: str, mentions: list[dict]) -> dict | None:
    """Deterministically resolve an ordinal reference ("the first one", "the last result")
    or an unambiguous dataset-title mention (see _title_mentioned) in `user_input` against
    `mentions` (the most recent dataset-listing tool's parsed results this session). Returns
    the matched {"title", "doi"} dict, or None if there's no confident match — in which case
    chat() falls back to today's LLM-only resolution from replayed conversation history.

    Deliberately conservative: an ordinal out of range, or a title mention matching more
    than one prior mention, returns None rather than guessing — a missed deterministic
    resolution just falls back to the existing (imperfect) behavior; a wrong guess would
    silently point the user at the wrong dataset."""
    if not mentions:
        return None

    m = _ORDINAL_RE.search(user_input)
    if m:
        idx = _ORDINAL_WORDS[m.group(1).lower()]
        try:
            return mentions[idx]
        except IndexError:
            pass

    lowered = user_input.lower()
    name_matches = [mn for mn in mentions if mn["title"] and _title_mentioned(mn["title"], lowered)]
    if len(name_matches) == 1:
        return name_matches[0]

    return None


# Matches a literal DOI typed directly in a user message (e.g. "compare X (DOI: 10.17612/...)
# and Y (DOI: 10.17612/...)"). Stops before trailing sentence punctuation/closing paren so it
# doesn't swallow a ")" or "." immediately following the DOI.
_DOI_IN_TEXT_RE = re.compile(r'10\.\d{4,9}/[^\s,).]+', re.IGNORECASE)

# Bare anaphoric references to "the dataset I was just told about" rather than a named
# one — "that dataset", "this one", "the other dataset". Only meaningful alongside
# last_profiled below; on its own it can't be resolved to anything.
_DATASET_ANAPHORA_RE = re.compile(r'\b(that|this|the other) (dataset|one)\b', re.IGNORECASE)


# The argument each dataset-listing tool carries its actual search text in.
_FILTER_TEXT_ARG_KEYS = ("question", "query")


def _tool_filter_text(tool_args: dict | None, fallback: str) -> str:
    """The text that should become _cumulative_filter_text after a dataset-listing tool ran.

    Prefer the tool's OWN argument over the raw user message. The two diverge exactly when it
    matters: on a follow-up turn like "How about any below 0.25?", the raw message carries no
    trace of the constraints established earlier, while the agent (following SYSTEM_PROMPT's
    instruction to compose one self-contained question) actually calls the tool with
    "sandstone datasets with porosity below 0.25". Storing the raw message discarded the
    accumulated "sandstone" constraint, so the NEXT refinement composed its compound question
    from a filter chain that had silently forgotten two turns of context — observed live.

    Falls back to `fallback` (the user's message) when the call carries no usable text
    argument, which keeps behavior unchanged for the deterministic dispatch paths that pass
    their own composed text explicitly.
    """
    for key in _FILTER_TEXT_ARG_KEYS:
        value = (tool_args or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _tool_args_by_call_id(messages: list) -> dict:
    """Map tool_call_id -> args for every tool call requested in `messages`, so a ToolMessage
    result can be traced back to the arguments it was produced from."""
    args_by_id: dict = {}
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            call_id = call.get("id")
            if call_id:
                args_by_id[call_id] = call.get("args") or {}
    return args_by_id


def _detect_comparison_references(
    user_input: str, mentions: list[dict], last_profiled: dict | None = None
) -> list[str] | None:
    """Deterministically collect 2+ distinct dataset references named in ONE message —
    e.g. "compare X (DOI: ...) and Y (DOI: ...)", "difference between the first and second",
    "how do X and Y compare" (where X/Y are titles from a prior result list) — and return
    them as resolved reference strings (DOI when available, else title), or None if fewer
    than 2 are found.

    This exists because getting the ReAct agent to reliably call get_dataset_profile twice
    in one turn depends on three independent, individually-unreliable steps succeeding
    together (the followup-tool-gate classifier, the model's own choice to emit a second
    tool call, and — if a 400 forces manual recovery — both calls being parseable out of the
    error text). Live testing showed this chain drops the second dataset often enough, even
    when both DOIs are given explicitly, that it isn't viable as the sole mechanism. When
    this function finds 2+ references, chat() dispatches get_dataset_profile for all of them
    directly, bypassing the agent's own tool-selection for this turn entirely.

    last_profiled: the {"title", "doi"} of whichever single dataset get_dataset_profile most
    recently returned (see _extract_profiled_dataset), used to resolve a bare anaphoric
    reference — "how does THAT DATASET compare with the downscaling-based one" — to the
    dataset actually being talked about. Without this, "that dataset" matches none of the
    explicit-reference checks below (it's not a DOI, ordinal, or title substring), so a
    message naming one dataset explicitly plus one anaphoric reference was only ever
    resolving to a single ref and silently falling through to the unreliable agent path
    this function exists to bypass."""
    refs: list[str] = []
    seen: set[str] = set()

    for m in _DOI_IN_TEXT_RE.finditer(user_input):
        doi = m.group(0)
        if doi.lower() not in seen:
            seen.add(doi.lower())
            refs.append(doi)

    for om in _ORDINAL_RE.finditer(user_input):
        idx = _ORDINAL_WORDS[om.group(1).lower()]
        try:
            mn = mentions[idx]
        except IndexError:
            continue
        ref = mn.get("doi") or mn.get("title")
        if ref and ref.lower() not in seen:
            seen.add(ref.lower())
            refs.append(ref)

    lowered = user_input.lower()
    for mn in mentions:
        title = mn.get("title")
        if not title:
            continue
        doi = mn.get("doi")
        if doi and doi.lower() in seen:
            continue  # already captured via its literal DOI above
        if title.lower() in seen:
            continue
        if _title_mentioned(title, lowered):
            # Prefer the mention's known DOI over the bare title: get_dataset_profile
            # resolves a DOI exactly, but a bare title only CONTAINS-matches, which goes
            # ambiguous whenever another dataset's title contains this one as a substring
            # (e.g. "Belgian Fieldstone" vs "DRP Visualization Challenge: Belgian
            # Fieldstone") — exactly the ambiguity this deterministic path exists to avoid.
            seen.add(title.lower())
            if doi:
                seen.add(doi.lower())
            refs.append(doi or title)

    if len(refs) == 1 and last_profiled and _DATASET_ANAPHORA_RE.search(user_input):
        ref = last_profiled.get("doi") or last_profiled.get("title")
        if ref and ref.lower() not in seen:
            refs.append(ref)

    return refs if len(refs) >= 2 else None


# Matches a message that narrows/refines a previous dataset-listing result rather than
# starting a fresh, unrelated search — "of these", "which of these", "any of those",
# "now filter/narrow further", etc.
_REFINEMENT_RE = re.compile(
    r'\b(of (these|those)\b|which (of (these|those)|ones)\b|among (these|those)\b|'
    r'from (these|those)\b|out of (these|those)\b|any of (these|those)\b|'
    r'now (filter|narrow)|filter (further|again)|narrow (it |them )?(down|further))',
    re.IGNORECASE,
)


# Elliptical follow-ups: a bare constraint carrying no subject of its own ("how about any
# below 0.25?", "any with porosity under 0.2", "just the ones above 5 microns"). These refine
# the current result set just as much as _REFINEMENT_RE's phrasings do, but they deliberately
# do NOT take the same deterministic compound-question dispatch, because that path ANDs the
# new text onto the entire prior chain. That is right for a genuine narrowing ("of these,
# which are segmented") and wrong here, where the new constraint SUPERSEDES an earlier one on
# the same property: "porosity above 0.3" AND "any below 0.25" composes to a contradiction and
# returns nothing.
#
# The agent handles supersession correctly on its own — live-observed calling
# get_dataset_details with "sandstone datasets with porosity below 0.25" after exactly that
# exchange. What it does not do is keep the answer inside the previously listed set. So for
# these phrasings the agent composes the question and chat() injects restrict_to_titles into
# its call (see _with_result_set_restriction), combining the agent's phrasing with the
# deterministic scope guarantee.
#
# "How about"/"what about" alone is NOT enough to match: it must be followed by a comparison
# word. That is what keeps a topic change ("What about carbonate datasets?") — which names a
# new subject and should search the whole catalog — out of this path.
_ELLIPTICAL_REFINEMENT_RE = re.compile(
    r'\b(?:how|what)\s+about\b[^?]*\b(?:below|above|under|over|less\s+than|greater\s+than|'
    r'more\s+than|fewer\s+than|between|higher|lower|smaller|larger|finer|coarser)\b'
    r'|^\s*(?:and\s+|or\s+)?any\s+(?:with|below|above|under|over|less|greater|more|fewer|'
    r'smaller|larger|higher|lower|finer|coarser)\b'
    r'|\b(?:just|only)\s+(?:the\s+)?(?:ones|those)\b'
    r'|\bnarrow\s+(?:it|them|that|this)\b|\brestrict\b',
    re.IGNORECASE,
)


# Words that carry no subject information, so they can't distinguish "still talking about
# sandstone" from "now asking about carbonate". Comparison words and the generic
# dataset/data nouns are deliberately included: the comparison is exactly what CHANGES
# between refinement turns ("above 0.3" -> "below 0.25"), and every question in this domain
# says "dataset".
_CHAIN_STOPWORDS = frozenset("""
a an the and or of for in on to with within from by as at any all some each every both
are there is be been being was were do does did done has have had having
find show give list get me my i you your we can could would please
what which who whose that this these those it its them they how about when where why
dataset datasets data datum set sets
above below over under greater less than more fewer between higher lower smaller larger
finer coarser most least equal exactly around approximately only just also still other
containing contains include includes including using used based
""".split())


def _chain_terms(text: str) -> set[str]:
    """The subject-bearing words of a query, normalized for comparison.

    Numbers are dropped entirely (they are the part that changes between refinement turns),
    as are stopwords and simple plurals.
    """
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    terms = set()
    for w in words:
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w and w not in _CHAIN_STOPWORDS and len(w) > 2:
            terms.add(w)
    return terms


def _continues_filter_chain(new_question: str, prior_chain: str | None) -> bool:
    """True if a freshly composed tool question still carries every subject term of the
    filter chain built up so far — i.e. it narrows that chain rather than starting over.

    This replaces trying to recognise refinement from the USER's phrasing, which failed
    repeatedly: "of these", "which ones", "any below 0.25", "are there any with porosity >
    0.3", "how about with porosity > 0.2" are all the same intent worded five ways, and each
    new transcript brought a phrasing the pattern list didn't have — the growing-pattern-library
    problem this codebase keeps rediscovering.

    The agent's own composed question is a far better signal, and it is available on every
    turn: live logs show it reliably restating the accumulated constraints ("sandstone datasets
    with porosity > 0.2" two turns after the user last said "sandstone"). A genuine topic change
    drops them instead ("carbonate datasets"), which is exactly what this detects.

    Conservative by construction: it requires the prior chain's terms to SURVIVE, so a dropped
    subject means no restriction and today's catalog-wide behavior — never a wrongly narrowed
    answer.
    """
    prior_terms = _chain_terms(prior_chain)
    if not prior_terms:
        return False
    return prior_terms.issubset(_chain_terms(new_question))


# Trailing paragraphs that just restate bulleted dataset results in prose instead of
# adding new information. Matched only against a paragraph that begins with one of
# these openers, so a legitimate leading header ("Datasets:") before the bullets is
# never touched — this only strips paragraphs found AFTER the last bullet line.
_RECAP_OPENER_RE = re.compile(
    r'(?im)^(these (datasets|results) (are|is) related to|'
    r'these (datasets|results) may be|'
    r'(these datasets|the (datasets|results) (above|listed)) may (be|also be) relevant).*',
)


def _strip_recap_paragraph(text: str) -> str:
    """Strip a trailing recap paragraph that restates results already shown as bullets.
    Only removes paragraphs after the last bullet/list line, and only when they open
    with a known recap phrase — never touches a leading header before the bullets."""
    lines = text.split('\n')
    last_bullet_idx = -1
    for i, line in enumerate(lines):
        if re.match(r'^\s*([-*•]|\[[\w\s]+\])', line):
            last_bullet_idx = i
    if last_bullet_idx == -1:
        return text
    head = '\n'.join(lines[:last_bullet_idx + 1])
    tail = '\n'.join(lines[last_bullet_idx + 1:])
    tail = _RECAP_OPENER_RE.sub('', tail).strip()
    return (head + ('\n\n' + tail if tail else '')).strip()


# Chain-of-thought scaffolding this model sometimes emits as its user-facing answer:
# a numbered "Step 1: ... Step 8: ..." reasoning trace followed by a "The final answer
# is:" marker introducing the actual response. Both halves must be present to strip —
# see _strip_reasoning_scaffold for why that conjunction is the safety guard.
_STEP_SCAFFOLD_RE = re.compile(r'^\s*(?:\*\*|#{1,6}\s*)?Step\s+\d+\s*[:.]', re.MULTILINE | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r'^\s*(?:\*\*|#{1,6}\s*)?(?:The\s+)?final answer(?:\s+is)?\s*[:.]?\s*(?:\*\*)?\s*',
    re.MULTILINE | re.IGNORECASE,
)


def _strip_reasoning_scaffold(text: str) -> str:
    """Strip a leaked chain-of-thought scaffold, keeping only the model's own designated
    final answer. Llama-4-Maverick intermittently answers a synthesis prompt by emitting
    its reasoning verbatim ("Step 1: Identify... Step 8: ... The final answer is: <answer>")
    instead of just the answer — live-observed on the multi-dataset comparison path.

    Deliberately requires BOTH a "Step N:" scaffold AND a "final answer is:" marker before
    removing anything. A legitimate answer can absolutely contain numbered steps — a
    get_workflow_guidance response ("Step 1: Segment the image...") is exactly that shape —
    but such an answer never also announces "The final answer is:". Requiring the
    conjunction is what makes this safe to run over every response rather than only the
    comparison path, so the same leak is caught wherever it surfaces.

    This is a backstop, not the primary fix: the synthesis prompts also instruct the model
    not to emit a scaffold. Consistent with this project's repeated finding that prompt-only
    reliability fixes for this model are inconsistent, the guarantee lives in code."""
    if not _STEP_SCAFFOLD_RE.search(text):
        return text
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return text
    answer = text[matches[-1].end():].strip()
    if not answer:
        return text
    logger.warning("Stripped leaked chain-of-thought scaffold (%d chars -> %d chars)", len(text), len(answer))
    return answer


def _clean_response(text: str) -> str:
    return _strip_recap_paragraph(_strip_reasoning_scaffold(_TOOL_CALL_RE.sub('', text).strip()))


def _extract_tool_calls_from_text(text: str) -> list[dict]:
    """
    Extract Llama-style <|python_start|>fn(key="val")<|python_end|> calls from a text string.
    Used both for leaked tool-call text in the final agent response and as Strategy 1 in the
    400 error path.
    """
    calls: list[dict] = []
    seen: set[tuple] = set()
    python_call_re = re.compile(
        r'<\|python_start\|>\s*(\w+)\s*\(([^)]*)\)\s*<\|python_end\|>',
        re.DOTALL,
    )
    kwarg_re = re.compile(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"')
    for m in python_call_re.finditer(text):
        tool_name = m.group(1)
        if tool_name not in _TOOL_PARAM_KEYS:
            continue
        param_keys = _TOOL_PARAM_KEYS[tool_name]
        kwargs = dict(kwarg_re.findall(m.group(2)))
        args = {key: kwargs[key] for key in param_keys if key in kwargs}
        if not args and kwargs:
            # None of the expected key names matched (e.g. the model used different
            # names) — fall back to assigning whatever values were parsed, in order,
            # to the tool's expected param keys.
            args = dict(zip(param_keys, kwargs.values()))
        dedupe_key = (tool_name, tuple(sorted(args.items())))
        if args and dedupe_key not in seen:
            calls.append({"name": tool_name, "args": args})
            seen.add(dedupe_key)
    return calls


def _extract_tool_calls_from_error(err_str: str) -> list[dict]:
    """
    Extract the intended tool calls from a LiteLLM 400 format-mismatch error.

    Llama-4-Maverick uses a non-OpenAI tool-call format. LiteLLM rejects it with
    a 400 but echoes the raw model output in the error. We try three extraction
    strategies:

    1. Python-call format: <|python_start|>fn_name(key="value")<|python_end|>
    2. JSON "value" field: {"type":"string","value":"..."} (multiply-escaped)
    3. JSON parameter key field: {"param_key": "value"} near the tool name
    """
    calls: list[dict] = []
    seen: set[tuple] = set()

    # Strategy 1: Llama native <|python_start|>fn(key="val")<|python_end|> format
    calls = _extract_tool_calls_from_text(err_str)
    if calls:
        return calls

    # Strategies 2 & 3 operate on the normalized (unescaped) error string.
    # The actual err_str contains 4 literal backslashes before each quote (\\\\"),
    # e.g. \\\\"value\\\\" — stripping all double-backslash pairs normalizes it.
    normalized = err_str.replace('\\\\', '')

    for tool_name, param_keys in _TOOL_PARAM_KEYS.items():
        if tool_name not in normalized:
            continue
        # Scan for EVERY occurrence of tool_name, not just the first — a comparison
        # ("compare A and B") issues the same tool (get_dataset_profile) twice with
        # different args, and a single `.find()` here used to only ever recover the
        # first call, silently dropping the second dataset (see HANDOFF.md).
        search_start = 0
        while True:
            idx = normalized.find(tool_name, search_start)
            if idx == -1:
                break
            # Wide enough to span a multi-arg call's full JSON args blob (e.g.
            # get_dataset_profile's two keys), not just a single-arg one.
            snippet = normalized[idx: idx + 800]
            search_start = idx + len(tool_name)

            # Strategy 3: {"param_key": "value"} — direct JSON parameter key match, tried
            # for every expected key so multi-arg tools can recover all of them.
            args = {}
            for key in param_keys:
                mp = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', snippet)
                if mp:
                    args[key] = mp.group(1)

            # Strategy 2: {"type":"string","value":"..."} only ever recovers one positional
            # value, so it only applies as a fallback for single-param tools.
            if not args and len(param_keys) == 1:
                mv = re.search(r'"value":\s*"([^"]*)"', snippet)
                if mv:
                    args[param_keys[0]] = mv.group(1)

            dedupe_key = (tool_name, tuple(sorted(args.items())))
            if args and dedupe_key not in seen:
                calls.append({"name": tool_name, "args": args})
                seen.add(dedupe_key)

    return calls

_COMPARISON_SYNTHESIS_SYSTEM_PROMPT = """\
You are given 2 or more already-complete, grounded dataset write-ups below — each one was \
produced by its own dedicated lookup against the real portal graph, so every fact in them is \
already verified; you do not need to re-derive or re-verify anything, only organize/compare \
what's already there.

- Preserve every recorded property or fact mentioned in EACH write-up below. Never say a \
property, DOI, or detail "isn't provided," "isn't available," or "isn't mentioned" if it \
appears anywhere in the write-ups below — that would be a wrong under-report of data you \
were actually given; re-read both write-ups carefully before concluding something is missing.
- Organize by dataset, then cover similarities and differences relevant to the user's actual \
question — for a general "what's the difference" question, cover whatever properties both \
write-ups actually contain (rock type, porosity, imaging, organizational structure, etc.), \
not just the first difference you happen to notice.
- Preserve any [dataset profile] source labels, DOIs, and LaTeX math verbatim.
- If a property is genuinely absent from BOTH write-ups, it's fine to say so — but only after \
checking both carefully, not as a default hedge.
- Output ONLY the finished comparison, addressed to the researcher. Do not narrate your own \
reasoning process, do not emit numbered "Step 1: / Step 2:" analysis stages, and do not \
introduce your response with "The final answer is:" — the user sees your output verbatim, so \
any such scaffolding reads as a malfunction. Write the comparison directly, using headings \
and/or bullets to organize it.
"""


def _run_manual_dispatch(tool_calls: list[dict], user_input: str, prior: list[dict]) -> str | None:
    """
    Execute tool_calls directly (bypassing LangGraph), then synthesize a response.
    Returns the synthesized string, or None if dispatch produced no results.
    """
    from src.assistant.llm import get_chat_model
    from src.assistant.tools import build_langchain_tools

    tools_map = {t.name: t for t in build_langchain_tools()}
    results = []
    raw_results = []
    for call in tool_calls:
        fn = tools_map.get(call["name"])
        if fn:
            try:
                res = fn.invoke(call["args"])
                results.append(f"--- {call['name']} ---\n{res}")
                raw_results.append((call["name"], res))
                logger.warning("Manual dispatch: %s(%s)", call["name"], call["args"])
            except Exception as te:
                logger.warning("Tool %s failed in manual dispatch: %s", call["name"], te)

    if not results:
        return None

    # Self-contained tools already return a final, user-ready answer (see
    # _SELF_CONTAINED_TOOLS docstring above). Re-running them through a second LLM
    # synthesis pass governed by the generic SYSTEM_PROMPT (which has no verbatim-
    # citation rule) was silently dropping tutorial notebook references and inventing
    # DOIs — so for a single call to one of these tools, return its own output directly.
    if len(raw_results) == 1 and raw_results[0][0] in _SELF_CONTAINED_TOOLS:
        return _non_empty(_clean_response(raw_results[0][1]), fallback=_non_empty(raw_results[0][1]))

    # Same verbatim-passthrough rationale as the normal ReAct path in chat(): don't let
    # a second LLM call retype search_datasets' real DOIs/descriptions from memory.
    if len(raw_results) == 1 and raw_results[0][0] in _VERBATIM_TOOLS:
        return _non_empty(_build_verbatim_response(user_input, raw_results[0][1]))

    # Multiple calls to the SAME self-contained tool (the dataset-comparison case: N>=2
    # get_dataset_profile calls, one per dataset). Each raw result is already a complete,
    # grounded write-up (dataset_profile.yaml's own honesty/tiered-knowledge synthesis
    # already ran per dataset) — it does not need re-deriving, only presenting together.
    # Live testing showed running this through the GENERIC synthesis path below (full
    # SYSTEM_PROMPT + entire prior conversation history) actively hurt quality: it
    # under-reported real recorded facts that were plainly present in the per-dataset
    # write-ups just above it (e.g. claiming a property "isn't provided in the given
    # context" for one dataset while it plainly was, a few lines up) — the wider context
    # and full conversation history distracted the model from the two write-ups it was
    # actually supposed to compare. Use a narrow, comparison-only prompt with NO prior
    # history instead, whose only job is to organize/compare what's already there.
    tool_names_used = {name for name, _ in raw_results}
    if len(tool_names_used) == 1 and tool_names_used <= _SELF_CONTAINED_TOOLS:
        combined = "\n\n---\n\n".join(res for _, res in raw_results)
        try:
            llm = get_chat_model()
            compare_messages = [
                {"role": "system", "content": _COMPARISON_SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"{user_input}\n\n{combined}"},
            ]
            return _non_empty(_clean_response(llm.invoke(compare_messages).content), fallback=_non_empty(combined))
        except Exception as e:
            logger.error("Comparison synthesis failed: %s; returning raw profiles", e)
            return _non_empty(_clean_response(combined), fallback=_non_empty(combined))

    tool_output = "\n\n".join(results)
    try:
        llm = get_chat_model()
        synth_messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + prior
            + [{"role": "user", "content": (
                f"{user_input}\n\n"
                "[Knowledge base context — do not mention internal tool names "
                "(get_workflow_guidance, get_educational_context, etc.) in your response. "
                "Preserve any notebook paths, DOIs, and LaTeX math verbatim — do not "
                "paraphrase or drop them.]\n"
                f"{tool_output}"
            )}]
        )
        return _non_empty(_clean_response(llm.invoke(synth_messages).content), fallback=_non_empty(tool_output))
    except Exception as e:
        # The tool call(s) already succeeded and produced real, grounded data (results
        # is non-empty here) — a failure in this polish-only synthesis step must not
        # cause that real data to be thrown away and replaced by an ungrounded guess.
        # Fall back to the raw tool output directly rather than returning None.
        logger.error("Synthesis after manual dispatch failed: %s; returning raw tool output", e)
        return _non_empty(_clean_response(tool_output), fallback=_non_empty(tool_output))


SYSTEM_PROMPT = """\
You are Rocco, an expert research assistant for the Digital Porous Media (DPM) Portal. \
You help researchers discover datasets, understand porous media workflows, and find relevant literature.

## Knowledge tiers — follow these strictly

**Tier 0 — Conversation, brainstorming, and code assistance \
(e.g. "hi", "thanks", "Hi, I'm Bernie", "can you help me think through my sampling design?", \
"write a script to compute porosity from this CSV", "why is my segmentation pipeline crashing?")**
Respond directly — no tool call is required for this tier. This covers greetings and small \
talk (including self-introductions that happen to contain a name, e.g. "Hi, I'm Bernie" — a \
name mentioned this way is NOT an author-lookup request and must never trigger \
get_dataset_details), but also open-ended requests that don't map to a specific tool: \
brainstorming research ideas, writing or debugging code for porous-media-related data work, \
or talking through methodology. Keep greetings/small talk brief (a sentence or two); code and \
brainstorming responses can be as long as the task genuinely needs. If the conversation surfaces a need for \
an actual dataset lookup, workflow guide, or literature search, go ahead and call the relevant \
tool per Tiers 1-3 below rather than answering from memory. If the request has no connection \
to porous media, dataset/DRP research, or the kind of data/analysis work researchers do around \
Rocco's datasets (e.g. general programming help or personal topics unrelated to the domain), \
respond warmly but gently steer the conversation back — mention what you can help with (finding \
datasets, DRP workflows, literature, domain science, portal how-to guidance/documentation, or \
domain-related coding/analysis help) rather than refusing outright or acting as a general-purpose \
assistant. Acknowledge the mismatch in one short sentence, then stop — do not go on to answer \
the off-topic question(s), even partially, even to be helpful. This applies even if you know \
the answer. (A dedicated gate ahead of this prompt already catches most off-domain requests \
before they reach you — if you're seeing this instruction, treat it as the backstop, not the \
primary defense.)

**Tier 1 — Dataset and portal facts (e.g. "How many sandstone datasets have φ > 0.2?")**
Use tools only. Never assert a dataset property, count, or statistic that was not returned \
by a tool. If a tool returns no results, tell the user honestly and suggest rephrasing.

**Tier 2 — Domain Q&A, workflows, and portal how-to \
(e.g. "How do I compute relative permeability?", "How do I upload a dataset?")**
Call the appropriate tool first (get_educational_context, get_workflow_guidance, or \
search_portal_docs). If the tool context is sparse or missing, you may supplement with \
general domain expertise, but preface that supplement with: \
"I don't have portal-specific data on this, but generally…"

**Tier 3 — Foundational concepts (e.g. "What is porosity?", "Explain Darcy's law")**
Answer directly and completely. No tool calls needed, no disclaimers.

## Tool selection

The rules below apply only to Tiers 1-3. Tier 0 conversation/brainstorming/code-help \
input does not require any tool from this list, though a tool may still be called mid-turn \
if the conversation surfaces a genuine dataset/workflow/literature need.

- "How to X", "How do I X", "what are the steps to X" for a **scientific/analysis \
method** (compute relative permeability, segment an image, run a simulation \
conceptually/computationally, extract a pore network) → get_workflow_guidance (always \
first; call search_datasets afterward only if the user also wants datasets). Do NOT \
use get_workflow_guidance for a **portal action** (upload, download, copy, cite, \
manage collaborators, request publication) or for **operating a specific tool the \
portal documents** (e.g. running an LBPM simulation through the portal's LBPM \
interface, using the portal's Jupyter tools) — those route to search_portal_docs \
below, never to get_workflow_guidance, even when phrased as "how do I run/use X".
- **Any query that names a concrete, checkable dataset/sample property — a numeric \
threshold or range (porosity above/below/between X, grain size less than X), a specific \
metadata value or set of values (rock type, segmented status, voxel resolution), a named \
person explicitly as the subject of a dataset/author search (e.g. "datasets by Jane \
Doe", "who has published data on sandstone permeability" — maps to the authors field; a \
name mentioned incidentally, such as someone introducing themselves — "Hi, I'm Bernie" — \
is Tier 0, not this case), or a combination of these — → get_dataset_details, even if it \
also mentions a rock type or imaging method.** \
The full list of checkable properties is in get_dataset_details' own tool description \
(derived from the live schema — do not rely on this list of examples being exhaustive; \
if a query names ANY property in that tool's description, including one not called out \
here by name, route there). search_datasets can only match one value per field and \
cannot express numeric comparisons at all, so it silently drops these constraints; \
get_dataset_details generates real Cypher and handles any number/combination of them \
correctly.
- **A follow-up that narrows/refines a previous dataset-listing result** ("of these, are \
there any with X", "which of these also have Y", "now filter by Z") — get_dataset_details \
and search_datasets are both STATELESS per call: each call only ever sees the exact \
question/query string passed that call, with no memory of what was asked or filtered in an \
earlier turn. Passing just the new constraint in isolation ("are there any with porosity \
between 0.2 and 0.25") silently drops every constraint from the earlier turn(s) and searches \
the WHOLE catalog instead of narrowing the prior result set. Always compose ONE \
self-contained question that restates every constraint from this conversation so far \
(all earlier filters PLUS the new one) as the tool's argument — e.g. if the prior turn asked \
for "segmented sandstone datasets" and this turn adds "porosity between 0.2 and 0.25", call \
get_dataset_details with "segmented sandstone datasets with porosity between 0.2 and 0.25", \
not just the new clause alone.
- **A follow-up question about a dataset that is already identified** — from a prior \
search_datasets/get_dataset_details/get_dataset_profile result, or from the user directly \
naming/describing one dataset in this turn — including "tell me more about this/that/the \
first one", a specific property question about that one dataset, organizational-structure \
questions (which sample fed which scan fed which analysis), "how do I read this dataset's \
files in Python"/"where can I download this", or a reuse-suitability judgment about that one \
dataset ("is this suitable for X") → get_dataset_profile. Resolve the pronoun/positional \
reference ("this", "that", "the first one", "the sandstone one") to a concrete title, DOI, or \
dataset number from the conversation history BEFORE calling — the tool takes only a resolved \
reference string, never a bare pronoun. Once a dataset has been identified this way, do NOT \
re-call search_datasets or get_dataset_details for a further follow-up about that same \
dataset — keep using get_dataset_profile with the same resolved reference.
- **Comparing two or more already-identified datasets** ("compare dataset A and dataset B", \
"which of these two is better for X") → call get_dataset_profile once per dataset, each with \
its own resolved reference and the comparison question, then synthesize the comparison \
yourself from both results — do not look for or invent a separate comparison tool.
- **A question that no single literal field can settle — a RELATIONSHIP between \
datasets or between one dataset's own parts, a comparison across its sub-nodes, or a \
pattern implied by methodology/content** ("paired tomographic and segmented images", \
"the same sample imaged at different resolutions", "datasets with a segmented version \
of the same scan", "imaged on the same instrument") → reason_about_dataset_content. \
Pass the WHOLE question, including any literal property it also names — do NOT split a \
literal clause out of a relational claim and send it to get_dataset_details on its own \
("segmented" inside "paired ... and segmented" is not an independently valid partial \
answer, and presenting one as if it answered the question is a wrong answer, not a \
partial one). A plain conjunction of independent literal properties ("sandstone AND \
porosity above 0.3") is NOT this case and stays on get_dataset_details.
- Dataset discovery by topic, suitability, or purpose with no precise checkable \
property named (e.g. "datasets suitable for LBM simulation", "something good for a \
teaching demo") → search_datasets. (search_datasets also attempts a structured lookup \
internally first as a safety net for property-shaped queries that reach it anyway, but \
routing there directly is still preferred.)
- Portal *actions* and navigation ("how do I upload/download/copy/cite a dataset", \
"how do I add collaborators", "how do I request publication") and metadata schema \
reference — ANY question about the definition, purpose, or difference between the \
DPM Portal's own entity types (Dataset, Sample, Digital Dataset, Analysis Dataset — \
e.g. "what fields does a Sample need", "difference between Dataset and Sample", \
"difference between a Digital Dataset and an Analysis Dataset", "what is a Digital \
Dataset") → search_portal_docs, NEVER get_educational_context or get_workflow_guidance. \
These are portal-specific schema terms with real documented definitions, not general \
science concepts — answering them without search_portal_docs produces wrong, made-up \
definitions, and falling back to "I don't have portal-specific data on this, but \
generally…" is the WRONG response here since search_portal_docs reliably has this data; \
only use that fallback phrasing when search_portal_docs was actually called and its \
result was genuinely sparse. Pass the user's question to it verbatim/in full — do not \
shorten it to a keyword phrase; the tool does semantic retrieval and full sentence \
context retrieves better results than a compressed keyword query.
- Porous media science Q&A and best practices → get_educational_context
- Finding papers or publications → search_literature

For cross-intent queries (e.g. "explain X and find me datasets that measure it"), \
call multiple tools and synthesize the results into a single coherent response.

## Response formatting

- Write answers as direct prose. Never structure an answer as a numbered derivation \
("Step 1: ...", "Step 2: ...") and never end with a "The final answer is..." line — \
those are leftover patterns from math-solving output and do not fit conversational Q&A. \
Just answer the question directly, using headers/bullets only where they genuinely aid \
readability.
- Relay source labels from tool output exactly as returned: [graph match], [semantic match], \
[semantic scholar], [cypher match], [component match], [hybrid match], [portal docs], \
[dataset profile], [content reasoning]. Do not strip or rename them.
- When presenting dataset search results, always include the DOI for each entry. \
The tool output includes it as "DOI: xxx" — preserve it verbatim in your response.
- Always use LaTeX delimiters for mathematical expressions: inline `$...$`, block `$$...$$` \
(e.g. $\\phi = V_{pore} / V_{total}$, $$k = \\frac{Q \\mu L}{A \\Delta P}$$). Do not use \
plain-text math notation. Preserve any LaTeX already present in tool output verbatim — never flatten it.
- Use markdown headers and bullet lists for multi-part answers.
- Do not editorialize or evaluate tool output — report it with light formatting only.
- Dataset-search results follow this shape: one short lead-in sentence (e.g. "Here are \
the datasets matching your query:"), then a header (e.g. "Datasets:"), then one bullet \
per result. Each bullet must keep the summary sentence that follows the DOI in the \
tool output, not just the title and DOI — do not compress a result down to a bare \
title/DOI line. After the bullets, one short closing sentence is allowed if it adds \
real information (e.g. a shared trait or a key difference across the results) — but \
never a sentence that just restates or re-describes results already listed above; when \
in doubt, omit it.

## Suitability query synthesis

search_datasets only includes a `[search reasoning: ...]` tag when the query is a \
genuine suitability/purpose query whose properties aren't stated directly (e.g. \
"suitable for LBM", "good for a teaching demo") — plain property queries (e.g. "coal \
samples", "sandstone datasets", "segmented micro-CT images") never get this tag. So:

- **If the tag is present**: present the results first, then synthesize the reasoning \
naturally (never reproduce it verbatim) — 1–2 sentences on what the task requires, \
drawing on Tier 2 domain knowledge, plus a brief per-result fit note. If the reasoning \
itself flags that the purpose maps to qualities outside the schema, skip the fit notes \
and instead ask what specific properties matter most (segmented image, rock type, \
resolution, simulation outputs, etc.).
- **If the tag is absent**: present the results using the standard shape from Response \
formatting above (lead-in sentence, header, full bullets, optional short closing \
sentence). No reasoning preamble, no suitability fit notes, no clarifying question.

If search_datasets output includes a `[weak match: ...]` tag, state plainly, near the \
top, that no results directly matched the topic and that the closest available results \
are shown instead — do not silently present them as if they were relevant, and do not \
invent a reason they might be relevant.

Never skip presenting the results. Never ask for clarification before showing results.
"""


_OFF_DOMAIN_GATE_SYSTEM_PROMPT = """\
You are a scope gate in front of Rocco, a research assistant for the Digital Porous \
Media (DPM) Portal. Rocco's in-scope domain is broad: dataset discovery, portal \
how-to/documentation, porous-media and digital-rock-physics science and workflows \
(imaging, segmentation, simulation, permeability/porosity/relative-permeability/etc.), \
literature search, general foundational science/math/physics concepts, and \
domain-related coding/data-analysis help (including scripts, debugging, statistics). \
Ordinary conversational courtesies (greetings, thanks, small talk, self-introductions) \
are also in scope.

Respond with a JSON object only, no markdown fences:
{"route": "in_domain"} or {"route": "off_domain"}

Route "off_domain" ONLY for requests with no plausible connection to any of the above —
e.g. requests for a recipe, personal medical/health/allergy advice unrelated to \
materials science, relationship advice, entertainment/sports/trivia, or general \
personal-life assistance completely unrelated to research, science, or data work.

When in doubt, or when a message mixes an off-topic part with any genuine \
science/data/portal question, route "in_domain" — never block a real research \
question because it was phrased alongside something else.

Examples:
- "why would someone be allergic to peanuts? How do I make a jelly donut?" -> off_domain
- "What's a good recipe for banana bread?" -> off_domain
- "What is porosity?" -> in_domain
- "Explain Darcy's law" -> in_domain
- "How do I compute relative permeability?" -> in_domain
- "How do I upload a dataset to the portal?" -> in_domain
- "Can you help me think through my sampling design?" -> in_domain
- "write a script to compute porosity from this CSV" -> in_domain
- "why is my segmentation pipeline crashing?" -> in_domain
- "Hi! I'm Bernie" -> in_domain
- "What's the weather like for my sampling trip, and how do I compute porosity?" -> in_domain
"""


def _classify_off_domain(user_input: str, prior: list[dict]) -> bool:
    """A separate, dedicated gate — deliberately NOT folded into _classify_needs_tool
    as a third route, which would blur that gate's already-imperfect tool/direct
    boundary (see its own docstring). This gate's only job is a coarse in/out-of-domain
    call, run BEFORE _classify_needs_tool in chat().

    Fixes a real gap: _answer_direct's Tier-0 instruction to "gently steer the
    conversation back" for off-domain requests is purely prompt-driven and was
    observed (live, 3/3) to acknowledge the domain mismatch and then answer the
    off-topic question(s) anyway — the instruction never explicitly says "do not
    answer". Per this project's own repeatedly-validated lesson, a soft "don't do X"
    instruction alone isn't reliable for this model; closing the loophole requires a
    deterministic classify-then-return-a-fixed-string guard (see
    _OFF_DOMAIN_STEER_BACK_MSG) so there's no further LLM call left that could
    "helpfully" continue past the acknowledgment.

    Defaults to False (in-domain) on any parse/call failure or ambiguity — a false
    negative here just reproduces today's already-tolerated behavior (falls through to
    _classify_needs_tool/_answer_direct as before); a false positive would incorrectly
    block a real research question, the strictly worse failure mode for a broad
    research-assistant domain.
    """
    from src.assistant.llm import get_chat_model

    try:
        llm = get_chat_model()
        messages = (
            [{"role": "system", "content": _OFF_DOMAIN_GATE_SYSTEM_PROMPT}]
            + prior[-6:]
            + [{"role": "user", "content": user_input}]
        )
        raw = llm.invoke(messages).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            raw = raw[len("json"):] if raw.startswith("json") else raw
        route = str(json.loads(raw.strip()).get("route", "in_domain")).strip().lower()
        return route == "off_domain"
    except Exception as e:
        logger.warning("Off-domain gate failed (%s); defaulting to in-domain.", e)
        return False


_OFF_DOMAIN_STEER_BACK_MSG = (
    "That's outside what I can help with — I'm focused on the Digital Porous Media "
    "Portal and porous-media/digital-rock-physics research. I'd be glad to help you "
    "find datasets, work through DRP workflows, look up literature, answer domain "
    "science questions, navigate the portal, or work on domain-related code/analysis. "
    "Is there something along those lines I can help with?"
)


_GATE_SYSTEM_PROMPT = """\
You are a routing gate in front of a research-assistant chatbot. You do NOT have any \
tools available in this call — your only job is to decide whether the user's message \
requires a follow-up call to look something up, or whether it can be answered directly.

Respond with a JSON object only, no markdown fences:
{"route": "tool"} or {"route": "direct"}

Route to "tool" for anything that needs a lookup: dataset facts/counts/properties, \
finding datasets, literature search, portal how-to/documentation, or DRP domain \
workflows/best practices (even if you personally know the general answer — portal-specific \
workflow guidance should still be looked up first).

Route to "direct" for: greetings, small talk, thanks, self-introductions (a name \
mentioned this way, e.g. "Hi, I'm Bernie", is NOT a lookup request), brainstorming, \
general programming/code help, and self-contained GENERAL SCIENCE concepts \
(e.g. "What is porosity?", "Explain Darcy's law") that don't need portal-specific data.

Do NOT route a question about the DPM Portal's own data model/schema/terminology to \
"direct" just because it's phrased like "What is X?" or "What's the difference between \
X and Y?" — "Dataset", "Sample", "Digital Dataset", and "Analysis Dataset" are portal-\
specific entity names with real, specific definitions documented on the portal (not \
general scientific concepts), and answering from general knowledge instead of looking \
them up produces wrong, made-up definitions. If X or Y in the question is a portal \
entity name rather than a science/physics concept, route to "tool".

Examples:
- "Hi! I'm Bernie" -> direct
- "Hello" -> direct
- "Thanks, that helps" -> direct
- "Can you help me think through my sampling design?" -> direct
- "What is porosity?" -> direct
- "How many sandstone datasets have porosity > 0.2?" -> tool
- "Find datasets suitable for LBM simulation" -> tool
- "How do I compute relative permeability?" -> tool
- "How is permeability computed from a lattice Boltzmann simulation?" -> tool
- "Find papers on relative permeability" -> tool
- "How do I upload a dataset to the portal?" -> tool
- "Hi, I'm Bernie, can you help me find sandstone datasets?" -> tool
- "What is porosity? How do I compute it from an image?" -> tool
- "What is the difference between a Digital Dataset and an Analysis Dataset?" -> tool
- "What's the difference between a Dataset and a Sample?" -> tool

If the message mixes small talk OR a self-contained foundational-concept question with \
a real lookup/workflow request (like the "What is porosity? How do I..." example above), \
route to "tool" — the agent on the other side can still explain the foundational concept \
directly in its response while also calling the tool for the part that needs one. Do not \
let a leading "What is X?" phrasing cause you to route the whole message to "direct" when \
a later part of the same message asks "how do I compute/do X" — that second part needs \
get_workflow_guidance. This misclassification risk is higher, not lower, later in a \
conversation that already covered general domain science or workflow topics — a portal-\
schema question can still come up at any point and must still route to "tool" regardless \
of what was just discussed.
"""


def _classify_needs_tool(user_input: str, prior: list[dict]) -> bool:
    """Ask the LLM, with no tools bound to this call, whether the message needs a
    follow-up tool-bound turn at all. This is deliberately a separate, tool-free call —
    binding tools is what exposes the model to emitting native tool-call syntax the
    backend can't parse, so this call structurally can't produce that failure mode.

    Defaults to True (needs tool) on any parse/call failure: a wrong "tool" guess just
    costs one wasted lookup, while a wrong "direct" guess would silently skip a real
    dataset/literature/workflow request — the more expensive mistake of the two.
    """
    from src.assistant.llm import get_chat_model

    try:
        llm = get_chat_model()
        messages = (
            [{"role": "system", "content": _GATE_SYSTEM_PROMPT}]
            + prior[-6:]
            + [{"role": "user", "content": user_input}]
        )
        raw = llm.invoke(messages).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            raw = raw[len("json"):] if raw.startswith("json") else raw
        route = str(json.loads(raw.strip()).get("route", "tool")).strip().lower()
        return route != "direct"
    except Exception as e:
        logger.warning("Tool-need gate failed (%s); defaulting to tool-bound agent.", e)
        return True


_FOLLOWUP_TOOL_GATE_SYSTEM_PROMPT = """\
You are checking whether a single tool's answer is enough to fully address a user's
question, or whether the question also needs a DIFFERENT kind of lookup beyond what
that one tool already covers (e.g. it also explicitly asks to find/search datasets,
find papers/literature, or look up a specific dataset property/count).

Also answer {"needs_followup": true} when the question names or clearly implies a SECOND
distinct dataset (e.g. a comparison — "compare A and B", "which of these two...") and only
ONE dataset's profile has been looked up so far — even though tool_called is the SAME tool
(get_dataset_profile) both times. get_dataset_profile only ever covers one dataset per call,
so a comparison needs it called again for the second dataset.

You will be given the user's original question and which tool was already called.

Respond with a JSON object only, no markdown fences:
{"needs_followup": true} or {"needs_followup": false}

Examples:
- question: "How do I compute relative permeability?", tool_called: "get_workflow_guidance" -> {"needs_followup": false}
- question: "How do I compute relative permeability, and can you also find datasets that measure it?", tool_called: "get_workflow_guidance" -> {"needs_followup": true}
- question: "What is porosity, and are there any recent papers on it?", tool_called: "get_educational_context" -> {"needs_followup": true}
- question: "How do I upload a dataset to the portal?", tool_called: "search_portal_docs" -> {"needs_followup": false}
- question: "Compare Dataset A and Dataset B for two-phase flow simulation", tool_called: "get_dataset_profile" -> {"needs_followup": true}
- question: "Tell me more about this dataset", tool_called: "get_dataset_profile" -> {"needs_followup": false}
"""


def _needs_followup_tool_call(user_input: str, tool_name: str) -> bool:
    """A cheap, tools-unbound gate (same 400-proof pattern as _classify_needs_tool),
    checked only when chat()'s single-tool-call short-circuit (see Fix 1 in
    HANDOFF.md's 400-error-recovery section) would otherwise fire.

    Live-verified this model's first ReAct turn requests tools SEQUENTIALLY, not in
    one parallel tool_calls list, for genuine cross-intent phrasing ("compute relative
    permeability, and also find datasets that measure it" -> a single
    get_workflow_guidance call on the first turn, with search_datasets only decided on
    a later turn after seeing that answer) — so short-circuiting on "exactly one tool
    call" alone silently drops that follow-up call. This gate catches that case before
    committing to the short-circuit.

    Defaults to False (short-circuit proceeds) on any parse/call failure: the
    short-circuit exists to fix a confirmed, reported 400-error bug (LaTeX-heavy
    self-contained answers sometimes get mis-detected as malformed tool calls on the
    graph's second turn) — an uncertain case should not reintroduce that risk. The cost
    of a wrong "no follow-up needed" guess is a dropped second tool call, not a
    fabricated or ungrounded answer.
    """
    from src.assistant.llm import get_chat_model

    try:
        llm = get_chat_model()
        messages = [
            {"role": "system", "content": _FOLLOWUP_TOOL_GATE_SYSTEM_PROMPT},
            {"role": "user", "content": f"question: {user_input!r}, tool_called: {tool_name!r}"},
        ]
        raw = llm.invoke(messages).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            raw = raw[len("json"):] if raw.startswith("json") else raw
        return bool(json.loads(raw.strip()).get("needs_followup", False))
    except Exception as e:
        logger.warning("Follow-up tool gate failed (%s); proceeding with short-circuit.", e)
        return False


_NO_TOOL_ACCESS_NOTICE = (
    "[This specific response has NO tool access — Tier 1/2's \"call the tool first\" "
    "instructions above do not apply to this turn; _classify_needs_tool has already "
    "decided no lookup is needed. Never write as if you called a tool, and never "
    "invent or narrate tool output (e.g. \"the get_workflow_guidance tool "
    "provides...\", \"### Results\", \"searching datasets...\", \"let's call X\"). "
    "Never state a dataset title, DOI, or portal-specific property as if it was "
    "retrieved — anything like that in this response is always fabricated. If the "
    "question actually needs portal-specific or dataset-specific data, say plainly "
    "that you don't have it available in this response rather than inventing one; "
    "only answer directly for genuine Tier 0/3 conversation or foundational-concept "
    "content.]"
)


def _answer_direct(user_input: str, prior: list[dict]) -> str:
    """Answer without ever exposing the model to tool schemas this turn — used when
    _classify_needs_tool says no lookup is needed.

    _classify_needs_tool is not perfectly reliable (observed misrouting Tier 2
    questions here despite matching the gate's own "-> tool" examples) — when that
    happens, the model still sees SYSTEM_PROMPT's Tier 1/2 "call the tool first"
    instructions and, having no actual tool access in this call, was observed
    fabricating an entire fake tool-use transcript (including fabricated dataset
    DOIs) rather than recognizing it couldn't comply. _NO_TOOL_ACCESS_NOTICE is a
    defensive addendum for exactly that misrouted case."""
    from src.assistant.llm import get_chat_model

    try:
        llm = get_chat_model()
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "system", "content": _NO_TOOL_ACCESS_NOTICE}]
            + prior
            + [{"role": "user", "content": user_input}]
        )
        return _clean_response(llm.invoke(messages).content)
    except Exception as e:
        logger.error("Direct-answer call failed: %s", e)
        return "I encountered an error processing your request. Please try rephrasing."


class ConversationManager:
    """
    Wraps a LangGraph ReAct agent with per-session memory.

    The agent is built once at construction time from the registered tools in
    build_langchain_tools() and the SYSTEM_PROMPT above. Routing is implicit:
    the LLM reads tool descriptions and the system prompt to decide which tool(s)
    to invoke for each user message. There is no separate intent-classification
    step inside this class — the assistant.yaml classifier is a standalone
    component used for testing and offline analysis only.

    Session management: each session_id gets an independent conversation thread
    via LangGraph's MemorySaver. Threads are isolated in memory and do not
    persist across process restarts.

    Usage:
        manager = ConversationManager()
        response = manager.chat("Find sandstone datasets with porosity > 0.2", session_id="abc123")
        follow_up = manager.chat("Which of those have micro-CT images?", session_id="abc123")
    """

    # Class-level fallbacks for _last_dataset_mentions/_cumulative_filter_text: __init__
    # always sets fresh instance attributes for real usage, but several existing tests
    # construct a manager via object.__new__(ConversationManager) (bypassing __init__
    # entirely) to avoid building the real agent — these class attributes keep chat()
    # working for those instances too. Never mutated in place (always rebound via
    # `self._x = ...`), so this shared list/None is never actually written into.
    _last_dataset_mentions: list[dict] = []
    _cumulative_filter_text: str | None = None
    _last_profiled_dataset: dict | None = None

    def __init__(self):
        from src.assistant.llm import get_chat_model

        self._agent = create_react_agent(
            get_chat_model(),
            build_langchain_tools(),
            prompt=SYSTEM_PROMPT,
        )

        # Ordered {"title", "doi"} mentions from the most recent search_datasets/
        # get_dataset_details result this instance has seen — instance-scoped (one
        # ConversationManager per session per assistant_ui.py's caching), used by
        # _resolve_reference to deterministically resolve a later ordinal/name-only
        # follow-up ("the first one", "the Gildehauser sandstone sample") without
        # relying solely on the LLM re-deriving it from replayed chat history.
        self._last_dataset_mentions: list[dict] = []

        # The full cumulative constraint text behind the CURRENT dataset-listing result
        # chain, e.g. "segmented sandstone datasets" after turn 1, then "segmented
        # sandstone datasets AND of these, porosity between 0.2 and 0.25" after a turn-2
        # refinement. Used by the deterministic refinement dispatch in chat() — a
        # SYSTEM_PROMPT instruction asking the agent to restate all prior constraints
        # itself was live-tested and found unreliable (see HANDOFF.md), so this is composed
        # in code instead. None until the first dataset-listing result of a session/chain.
        self._cumulative_filter_text: str | None = None

        # The {"title", "doi"} of whichever single dataset get_dataset_profile most
        # recently returned — used by _detect_comparison_references to resolve a bare
        # anaphoric follow-up ("how does THAT DATASET compare with X") the same way
        # _last_dataset_mentions resolves ordinal/name references against a listing.
        # None until the first get_dataset_profile result of a session.
        self._last_profiled_dataset: dict | None = None

    def _track_dataset_listing(
        self,
        tool_name: str,
        tool_output: str,
        base_text: str,
        refinement_text: str | None = None,
    ) -> None:
        """Update _last_dataset_mentions/_cumulative_filter_text from a
        search_datasets/get_dataset_details result, or _last_profiled_dataset from a
        get_dataset_profile result, regardless of which of chat()'s several return paths
        produced it (the single-tool-call short-circuit, the deterministic refinement/
        comparison dispatches, or the normal end-of-stream path all call this) — every
        path must keep this state in sync or later ordinal/name-reference, refinement, or
        anaphoric-comparison detection silently stops working depending on which path a
        given turn happened to take. No-op for any other tool."""
        if tool_name == "get_dataset_profile":
            profiled = _extract_profiled_dataset(tool_output)
            if profiled:
                self._last_profiled_dataset = profiled
                logger.warning("_track_dataset_listing(tool=get_dataset_profile): last profiled dataset now %r", profiled)
            return
        if tool_name not in _DATASET_LISTING_TOOLS:
            return
        mentions = _extract_dataset_mentions(tool_output)
        if mentions:
            self._last_dataset_mentions = mentions
        elif refinement_text is None:
            # A FRESH listing turn that named no datasets — a "no results" answer, or an
            # output shape _extract_dataset_mentions can't parse. Holding on to the previous
            # turn's mentions here would pair them with THIS turn's brand-new filter text,
            # so a later "of these" would narrow a set unrelated to the chain it is being
            # ANDed onto — the same wrong-set substitution described at
            # _DATASET_LISTING_TOOLS, reached a different way. Clear it: chat()'s refinement
            # dispatch requires a non-empty listing, so the follow-up falls through to
            # normal routing instead of silently refining the wrong set.
            #
            # A refinement turn (refinement_text is not None) is the one case where holding
            # on IS right: "of these, which are coal?" coming back empty doesn't change what
            # "these" refers to, so the user can still narrow the same set a different way.
            self._last_dataset_mentions = []
        self._cumulative_filter_text = refinement_text if refinement_text is not None else base_text
        logger.warning(
            "_track_dataset_listing(tool=%s): %d mentions parsed; _cumulative_filter_text now %r",
            tool_name, len(mentions), self._cumulative_filter_text,
        )

    def _with_result_set_restriction(
        self, tool_name: str, tool_args: dict, user_input: str
    ) -> dict:
        """Add restrict_to_titles to a get_dataset_details call that is an elliptical
        refinement of the datasets already listed this session.

        Closes the gap between the two existing mechanisms. The deterministic refinement
        dispatch (see _REFINEMENT_RE in chat()) both composes the question AND restricts the
        scope, but only fires for phrasings that name the prior set ("of these", "which
        ones"). A bare constraint like "how about any below 0.25?" names nothing, so it fell
        through to the agent — which composed a good self-contained question but searched the
        entire catalog, silently leaving the result set the user was working through.

        Here the agent keeps ownership of the question (it supersedes a replaced constraint
        correctly, which blind AND-composition cannot) and this adds the scope guarantee
        on top.

        A turn counts as a refinement when EITHER signal fires:
          1. the agent's composed question still carries every subject term of the filter
             chain so far (see _continues_filter_chain) — the primary, phrasing-independent
             signal, and
          2. the user's message is an elliptical bare constraint (_ELLIPTICAL_REFINEMENT_RE),
             which covers the case where the agent drops the subject from its question.

        Deliberately narrow otherwise: only get_dataset_details (the only dataset tool that
        accepts the parameter), only when a prior listing exists, and never overriding a
        restriction the caller already set.
        """
        if tool_name != "get_dataset_details" or tool_args.get("restrict_to_titles"):
            return tool_args
        if not self._last_dataset_mentions:
            return tool_args

        question = tool_args.get("question") or ""
        continues = _continues_filter_chain(question, self._cumulative_filter_text)
        elliptical = bool(_ELLIPTICAL_REFINEMENT_RE.search(user_input))
        if not (continues or elliptical):
            return tool_args

        titles = [m["title"] for m in self._last_dataset_mentions if m.get("title")]
        if not titles:
            return tool_args
        logger.warning(
            "Refinement of the current result set (continues_chain=%s elliptical=%s) — "
            "restricting to the %d previously listed dataset(s); prior chain %r, "
            "agent's question %r",
            continues, elliptical, len(titles), self._cumulative_filter_text, question,
        )
        return {**tool_args, "restrict_to_titles": titles}

    def chat(self, user_input: str, history: list[dict] | None = None) -> str:
        """
        Send a message and get a response.

        Args:
            user_input: The user's message.
            history: Prior conversation turns as a list of {"role", "content"} dicts
                (user and assistant messages only, no tool call internals). Managed
                externally by the UI layer so that only clean message pairs are replayed,
                avoiding backend BadRequestErrors from tool-call message formats.

        Returns:
            The assistant's response as a string.
        """
        prior = [{"role": m["role"], "content": m["content"]} for m in (history or [])]

        # Off-domain gate: runs first, before any other classification or LLM
        # synthesis call. See _classify_off_domain docstring — closes a gap where
        # _answer_direct's prompt-only "steer back" instruction was observed
        # acknowledging an off-topic request and then answering it anyway. A false
        # negative here just falls through to the existing tool/direct gate below
        # unchanged.
        if _classify_off_domain(user_input, prior):
            return _OFF_DOMAIN_STEER_BACK_MSG

        # Deterministic multi-dataset comparison dispatch: if this one message names 2+
        # distinct datasets (explicit DOIs, ordinals against the last result list, or
        # matched titles from it), fetch all of their profiles directly and synthesize the
        # comparison — bypassing the ReAct agent's own tool-selection for this turn. See
        # _detect_comparison_references docstring: relying on the agent to reliably choose
        # to call get_dataset_profile once per dataset in the same turn was live-tested to
        # drop the second dataset often enough (even with both DOIs given explicitly) that
        # it isn't viable as the sole mechanism. Falls through to the normal path below if
        # fewer than 2 references are found, or if dispatch produces nothing usable.
        comparison_refs = _detect_comparison_references(
            user_input, self._last_dataset_mentions, self._last_profiled_dataset
        )
        if comparison_refs:
            dispatched = _run_manual_dispatch(
                [
                    {"name": "get_dataset_profile", "args": {"dataset_reference": ref, "question": user_input}}
                    for ref in comparison_refs
                ],
                user_input,
                prior,
            )
            if dispatched is not None:
                return dispatched

        # Deterministic cumulative-filter refinement dispatch: get_dataset_details and
        # search_datasets are both STATELESS per call — each only ever sees the exact
        # question/query string passed that call, with no memory of an earlier turn's
        # constraints. A SYSTEM_PROMPT instruction asking the agent to compose the full
        # cumulative question itself (all prior constraints plus the new one) was live-
        # tested and found unreliable: the agent kept passing just the new constraint in
        # isolation, silently dropping earlier filters and searching the whole catalog
        # instead of narrowing the prior result set. When this message looks like a
        # refinement ("of these", "which of these", ...) of an existing filter chain,
        # compose the compound question in code and dispatch get_dataset_details directly,
        # bypassing the agent's own argument-construction for this turn entirely.
        #
        # The compound question alone is NOT sufficient, even though it reads like it
        # should be: live testing showed the Cypher-generation LLM re-derives every prior
        # constraint from scratch from that text each turn, over the whole graph, rather
        # than narrowing the actual previous result set — and it isn't even consistent
        # about it (the identical "sandstone" constraint, reworded into two compound
        # questions, produced two different WHERE clauses covering different rows). So
        # restrict_to_titles is passed alongside it — a deterministic, code-level
        # narrowing to the previous turn's actual listed titles that the regenerated
        # Cypher's own (possibly drifting) filtering can't bypass.
        #
        # Both pieces of state are required, not just the filter text: restrict_to_titles is
        # the half that actually guarantees the narrowing, and cypher_qa treats an EMPTY list
        # as "no restriction at all" — so dispatching without titles would run the compound
        # question over the whole catalog while the log claimed a restricted search.
        restrict_to_titles = [m["title"] for m in self._last_dataset_mentions if m.get("title")]
        if self._cumulative_filter_text and restrict_to_titles and _REFINEMENT_RE.search(user_input):
            compound_question = f"{self._cumulative_filter_text} AND {user_input}"
            logger.warning(
                "Refinement dispatch: get_dataset_details(question=%r, restrict_to_titles=%r)",
                compound_question, restrict_to_titles,
            )
            dispatched = _run_manual_dispatch(
                [{
                    "name": "get_dataset_details",
                    "args": {"question": compound_question, "restrict_to_titles": restrict_to_titles},
                }],
                compound_question,
                prior,
            )
            if dispatched is not None:
                self._track_dataset_listing(
                    "get_dataset_details", dispatched, user_input, refinement_text=compound_question
                )
                return dispatched
        elif _REFINEMENT_RE.search(user_input):
            # Looked like a refinement, but one of the two required pieces of state is
            # missing — no active filter chain, or no listed datasets to narrow (the very
            # first message of a session, a previous turn that returned nothing, or
            # ConversationManager instance/session state reset between turns). Both are
            # logged individually so a live report of "refinement isn't working" can be
            # traced to which half was absent, rather than only telling us the dispatch
            # path wasn't reached.
            logger.warning(
                "Refinement phrase detected but not dispatched (filter_chain=%r, "
                "listed_datasets=%d) — falling through to normal routing for: %r",
                self._cumulative_filter_text, len(restrict_to_titles), user_input,
            )

        # Tool-need gate: a separate, tools-unbound call that decides whether this turn
        # needs the tool-bound ReAct agent at all. See _classify_needs_tool docstring —
        # this exists because the tool-bound agent below is what exposes the model to
        # its native tool-call syntax, which is the actual source of the malformed-output
        # failures this gate is meant to prevent by not reaching that code path at all.
        if not _classify_needs_tool(user_input, prior):
            return _answer_direct(user_input, prior)

        # Deterministic reference-resolution assist: if this message is an ordinal
        # ("the first one") or names exactly one title from the last dataset-listing
        # result this instance has seen, tell the agent explicitly which dataset that
        # is rather than relying solely on it re-deriving the reference from replayed
        # chat history (see _resolve_reference docstring / HANDOFF.md — this was
        # unreliable in practice for get_dataset_profile follow-ups). No match leaves
        # user_input untouched, so behavior is unchanged when resolution doesn't apply.
        resolved = _resolve_reference(user_input, self._last_dataset_mentions)
        effective_user_input = user_input
        if resolved:
            doi_note = f', DOI {resolved["doi"]}' if resolved.get("doi") else ""
            effective_user_input = (
                f'{user_input}\n\n(Resolved reference: the dataset being referred to is '
                f'"{resolved["title"]}"{doi_note}.)'
            )

        messages = prior + [{"role": "user", "content": effective_user_input}]
        try:
            # Stream instead of a single .invoke() so a single self-contained/verbatim
            # tool call can be dispatched and returned WITHOUT ever letting the graph
            # run its second ("relay the tool result") model turn. That second turn is
            # the actual source of the 400 tool-format errors this whole except block
            # exists to recover from (e.g. get_workflow_guidance's LaTeX-heavy answers
            # sometimes get mis-detected by LiteLLM as a malformed function call when
            # the model retypes them) — and its own output is discarded unconditionally
            # anyway whenever exactly one self-contained/verbatim tool ran (see the
            # post-hoc checks below, and _run_manual_dispatch's identical short-circuit)
            # so skipping it removes a failure-prone call whose result was never used.
            stream = self._agent.stream({"messages": messages}, stream_mode="values")
            # stream_mode="values" yields the accumulated state after each superstep,
            # starting with the initial state itself — i.e. the FIRST value is just the
            # echoed input (no model has run yet); the model's first tool-call decision
            # only appears in the SECOND value, after the "agent" node's first
            # execution. (Confirmed live: value 0 is a lone HumanMessage, value 1 is the
            # first AIMessage carrying tool_calls.) Pull both.
            initial_step = next(stream)
            first_model_step = next(stream)
            new_after_first = first_model_step["messages"][len(initial_step["messages"]):]
            last_first = new_after_first[-1] if new_after_first else None
            first_turn_tool_calls = getattr(last_first, "tool_calls", None) or []

            if (
                len(first_turn_tool_calls) == 1
                and first_turn_tool_calls[0]["name"] in (_SELF_CONTAINED_TOOLS | _VERBATIM_TOOLS)
                and not _needs_followup_tool_call(user_input, first_turn_tool_calls[0]["name"])
            ):
                first_tool_name = first_turn_tool_calls[0]["name"]
                # The agent composes the question; this adds the "stay inside the datasets
                # already listed" guarantee for an elliptical follow-up that the
                # deterministic refinement dispatch above doesn't recognise.
                first_tool_args = self._with_result_set_restriction(
                    first_tool_name, dict(first_turn_tool_calls[0].get("args") or {}), user_input
                )
                dispatched = _run_manual_dispatch(
                    [{"name": first_tool_name, "args": first_tool_args}],
                    effective_user_input,
                    prior,
                )
                if dispatched is not None:
                    self._track_dataset_listing(
                        first_tool_name, dispatched,
                        _tool_filter_text(first_tool_args, effective_user_input),
                    )
                    return dispatched
                # The tool itself failed inside manual dispatch — fall through to the
                # normal graph execution below (which has its own per-tool error
                # handling), resuming from the same first model step so the model's
                # initial tool-call decision isn't wastefully redone.
                # NOTE: a genuinely sequential cross-intent turn (call tool A, inspect
                # its answer, THEN decide to also call tool B) would otherwise have
                # that second call silently dropped by this single-tool-call check —
                # live-verified this model requests cross-intent tools sequentially,
                # not in one parallel tool_calls list, so _needs_followup_tool_call
                # above is the actual guard against that, not the tool_calls count.

            result = first_model_step
            for step in stream:
                result = step

            # If exactly one verbatim tool ran this turn, bypass the agent's own
            # free-form final message entirely — that message is the LLM retyping the
            # tool's data from memory, which is where dropped/hallucinated DOIs and
            # descriptions come from. Splice the tool's real output in verbatim instead.
            new_messages = result["messages"][len(messages):]

            # Refresh the deterministic reference-resolution/refinement cache from any
            # search_datasets/get_dataset_details result produced this turn — a later
            # follow-up ("tell me about the first one", "of these, which are segmented")
            # resolves/refines against this. Also refreshes _last_profiled_dataset from any
            # get_dataset_profile result, for a later anaphoric comparison follow-up ("how
            # does that dataset compare with X"). Left unchanged for tools that produced
            # neither, so a get_dataset_profile turn doesn't wipe out the prior listing and
            # vice versa.
            tool_args_by_id = _tool_args_by_call_id(new_messages)
            for m in new_messages:
                if isinstance(m, ToolMessage) and getattr(m, "name", None) in (_DATASET_LISTING_TOOLS | {"get_dataset_profile"}):
                    self._track_dataset_listing(
                        m.name, m.content,
                        _tool_filter_text(
                            tool_args_by_id.get(getattr(m, "tool_call_id", None)),
                            effective_user_input,
                        ),
                    )

            verbatim_tool_msgs = [
                m for m in new_messages
                if isinstance(m, ToolMessage) and getattr(m, "name", None) in _VERBATIM_TOOLS
            ]
            if len(verbatim_tool_msgs) == 1:
                return _non_empty(_build_verbatim_response(effective_user_input, verbatim_tool_msgs[0].content))

            # Same rationale for tools that already return a complete, grounded answer —
            # skip the outer agent's own retelling of it.
            self_contained_tool_msgs = [
                m for m in new_messages
                if isinstance(m, ToolMessage) and getattr(m, "name", None) in _SELF_CONTAINED_TOOLS
            ]
            if len(self_contained_tool_msgs) == 1 and not verbatim_tool_msgs:
                return _non_empty(_clean_response(self_contained_tool_msgs[0].content))

            raw = result["messages"][-1].content
            # Llama-4-Maverick sometimes emits tool-call syntax as plain text that
            # LangGraph never intercepted (format mismatch without a 400 error).
            # Catch it here before _clean_response strips the token blocks silently.
            if '<|python_start|>' in raw:
                logger.warning("Tool call leaked into final response; dispatching manually.")
                tool_calls = _extract_tool_calls_from_text(raw)
                if tool_calls:
                    dispatched = _run_manual_dispatch(tool_calls, effective_user_input, prior)
                    if dispatched is not None:
                        for tc in tool_calls:
                            self._track_dataset_listing(
                                tc["name"], dispatched,
                                _tool_filter_text(tc.get("args"), effective_user_input),
                            )
                        return dispatched
            return _non_empty(_clean_response(raw))
        except Exception as e:
            err_str = str(e)
            # Llama-4-Maverick uses a non-OpenAI tool-call format that LiteLLM rejects
            # with a 400. Parse the intended calls out of the error's model_output field,
            # execute them directly, then synthesize with a plain LLM call.
            if "400" in err_str and any(k in err_str for k in _TOOL_FORMAT_ERRORS):
                logger.warning("Tool-call format mismatch (400); attempting manual dispatch.")
                tool_calls = _extract_tool_calls_from_error(err_str)
                if tool_calls:
                    dispatched = _run_manual_dispatch(tool_calls, effective_user_input, prior)
                    if dispatched is not None:
                        for tc in tool_calls:
                            self._track_dataset_listing(
                                tc["name"], dispatched,
                                _tool_filter_text(tc.get("args"), effective_user_input),
                            )
                        return dispatched
                    # A tool call WAS identified and dispatch was attempted — real,
                    # grounded tool output may or may not exist depending on whether the
                    # tool itself failed. Either way, do not fall through to an
                    # ungrounded direct LLM guess: that has nothing to ground it and
                    # will hedge/guess rather than admit it has no data (this is
                    # exactly how a successful tool call's real data previously got
                    # discarded and replaced by a fabricated answer). Give up honestly.
                    logger.error("Manual dispatch produced no usable output after a 400.")
                    return _HONEST_TOOL_FAILURE_MSG

                # No tool call could even be identified from the error — genuinely
                # nothing to dispatch. Previously this fell back to a no-tool-context
                # direct LLM call: confident-sounding, ungrounded, and with zero
                # indication anything went wrong — this is exactly the mechanism behind
                # a citation/notebook-reference "silently vanishing" (e.g.
                # get_workflow_guidance's answer computed correctly, but the relay turn
                # 400'd on LaTeX braces and nothing was recoverable from the error text).
                # An honest disclosure beats a silent, unlabeled guess from pretrained
                # knowledge — same rationale as the sibling "tool call identified but
                # dispatch failed" branch just above.
                logger.error("No tool calls extracted from 400 error; giving up honestly.")
                return _HONEST_TOOL_FAILURE_MSG

            logger.error("Agent invocation failed: %s", e)
            return "I encountered an error processing your request. Please try rephrasing."
