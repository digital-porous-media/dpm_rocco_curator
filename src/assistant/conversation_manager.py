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
  Portal how-to / schema  search_portal_docs     (portal markdown docs — stub until pipeline built)
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
_SELF_CONTAINED_TOOLS = {"get_workflow_guidance", "get_educational_context"}

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

# Maps each tool name to its primary string parameter key.
# Used to reconstruct tool calls from the error's model_output field.
_TOOL_PARAM_KEYS: dict[str, str] = {
    "get_educational_context": "question",
    "get_workflow_guidance": "goal",
    "search_datasets": "query",
    "get_dataset_details": "question",
    "search_literature": "query",
    "search_portal_docs": "question",
}

# Returned when a tool call was identified but produced nothing usable after a 400
# tool-format error — deliberately honest rather than falling back to an ungrounded
# direct LLM guess (mirrors tools.py's _HONEST_NO_TUTORIAL_MSG pattern).
_HONEST_TOOL_FAILURE_MSG = (
    "I wasn't able to complete that lookup due to an internal issue processing the "
    "request. Could you try rephrasing your question?"
)


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


def _clean_response(text: str) -> str:
    return _strip_recap_paragraph(_TOOL_CALL_RE.sub('', text).strip())


def _extract_tool_calls_from_text(text: str) -> list[dict]:
    """
    Extract Llama-style <|python_start|>fn(key="val")<|python_end|> calls from a text string.
    Used both for leaked tool-call text in the final agent response and as Strategy 1 in the
    400 error path.
    """
    calls: list[dict] = []
    seen: set[tuple[str, str]] = set()
    python_call_re = re.compile(
        r'<\|python_start\|>\s*(\w+)\s*\(([^)]*)\)\s*<\|python_end\|>',
        re.DOTALL,
    )
    kwarg_re = re.compile(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"')
    for m in python_call_re.finditer(text):
        tool_name = m.group(1)
        if tool_name not in _TOOL_PARAM_KEYS:
            continue
        param_key = _TOOL_PARAM_KEYS[tool_name]
        kwargs = dict(kwarg_re.findall(m.group(2)))
        value = kwargs.get(param_key) or (next(iter(kwargs.values())) if kwargs else None)
        if value and (tool_name, value) not in seen:
            calls.append({"name": tool_name, "args": {param_key: value}})
            seen.add((tool_name, value))
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
    seen: set[tuple[str, str]] = set()

    # Strategy 1: Llama native <|python_start|>fn(key="val")<|python_end|> format
    calls = _extract_tool_calls_from_text(err_str)
    if calls:
        return calls

    # Strategies 2 & 3 operate on the normalized (unescaped) error string.
    # The actual err_str contains 4 literal backslashes before each quote (\\\\"),
    # e.g. \\\\"value\\\\" — stripping all double-backslash pairs normalizes it.
    normalized = err_str.replace('\\\\', '')

    for tool_name, param_key in _TOOL_PARAM_KEYS.items():
        if tool_name not in normalized:
            continue
        idx = normalized.find(tool_name)
        snippet = normalized[idx: idx + 500]

        # Strategy 2: {"type":"string","value":"..."}
        mv = re.search(r'"value":\s*"([^"]*)"', snippet)
        if mv:
            value = mv.group(1)
            if value and (tool_name, value) not in seen:
                calls.append({"name": tool_name, "args": {param_key: value}})
                seen.add((tool_name, value))
                continue

        # Strategy 3: {"param_key": "value"} — direct JSON parameter key match
        mp = re.search(rf'"{re.escape(param_key)}"\s*:\s*"([^"]*)"', snippet)
        if mp:
            value = mp.group(1)
            if value and (tool_name, value) not in seen:
                calls.append({"name": tool_name, "args": {param_key: value}})
                seen.add((tool_name, value))

    return calls

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
                logger.info("Manual dispatch: %s(%s)", call["name"], call["args"])
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
        return _clean_response(raw_results[0][1])

    # Same verbatim-passthrough rationale as the normal ReAct path in chat(): don't let
    # a second LLM call retype search_datasets' real DOIs/descriptions from memory.
    if len(raw_results) == 1 and raw_results[0][0] in _VERBATIM_TOOLS:
        return _build_verbatim_response(user_input, raw_results[0][1])

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
        return _clean_response(llm.invoke(synth_messages).content)
    except Exception as e:
        # The tool call(s) already succeeded and produced real, grounded data (results
        # is non-empty here) — a failure in this polish-only synthesis step must not
        # cause that real data to be thrown away and replaced by an ungrounded guess.
        # Fall back to the raw tool output directly rather than returning None.
        logger.error("Synthesis after manual dispatch failed: %s; returning raw tool output", e)
        return _clean_response(tool_output)


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
datasets, DRP workflows, literature, domain science, or domain-related coding/analysis help) \
rather than refusing outright or acting as a general-purpose assistant.

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

- "How to X", "How do I X", "what are the steps to X", any workflow or method question → get_workflow_guidance (always first; call search_datasets afterward only if the user also wants datasets)
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
- Dataset discovery by topic, suitability, or purpose with no precise checkable \
property named (e.g. "datasets suitable for LBM simulation", "something good for a \
teaching demo") → search_datasets. (search_datasets also attempts a structured lookup \
internally first as a safety net for property-shaped queries that reach it anyway, but \
routing there directly is still preferred.)
- Portal how-to guides and metadata schema reference → search_portal_docs
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
[semantic scholar], [cypher match], [component match], [hybrid match]. Do not strip or rename them.
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
general programming/code help, and self-contained foundational science concepts \
(e.g. "What is porosity?", "Explain Darcy's law") that don't need portal-specific data.

Examples:
- "Hi! I'm Bernie" -> direct
- "Hello" -> direct
- "Thanks, that helps" -> direct
- "Can you help me think through my sampling design?" -> direct
- "What is porosity?" -> direct
- "How many sandstone datasets have porosity > 0.2?" -> tool
- "Find datasets suitable for LBM simulation" -> tool
- "How do I compute relative permeability?" -> tool
- "Find papers on relative permeability" -> tool
- "How do I upload a dataset to the portal?" -> tool
- "Hi, I'm Bernie, can you help me find sandstone datasets?" -> tool

If the message mixes small talk with a real request (like the last example), route to "tool" \
— the agent on the other side will handle the conversational part too.
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


def _answer_direct(user_input: str, prior: list[dict]) -> str:
    """Answer without ever exposing the model to tool schemas this turn — used when
    _classify_needs_tool says no lookup is needed."""
    from src.assistant.llm import get_chat_model

    try:
        llm = get_chat_model()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + prior + [
            {"role": "user", "content": user_input}
        ]
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

    def __init__(self):
        from src.assistant.llm import get_chat_model

        self._agent = create_react_agent(
            get_chat_model(),
            build_langchain_tools(),
            prompt=SYSTEM_PROMPT,
        )

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

        # Tool-need gate: a separate, tools-unbound call that decides whether this turn
        # needs the tool-bound ReAct agent at all. See _classify_needs_tool docstring —
        # this exists because the tool-bound agent below is what exposes the model to
        # its native tool-call syntax, which is the actual source of the malformed-output
        # failures this gate is meant to prevent by not reaching that code path at all.
        if not _classify_needs_tool(user_input, prior):
            return _answer_direct(user_input, prior)

        messages = prior + [{"role": "user", "content": user_input}]
        try:
            result = self._agent.invoke({"messages": messages})

            # If exactly one verbatim tool ran this turn, bypass the agent's own
            # free-form final message entirely — that message is the LLM retyping the
            # tool's data from memory, which is where dropped/hallucinated DOIs and
            # descriptions come from. Splice the tool's real output in verbatim instead.
            new_messages = result["messages"][len(messages):]
            verbatim_tool_msgs = [
                m for m in new_messages
                if isinstance(m, ToolMessage) and getattr(m, "name", None) in _VERBATIM_TOOLS
            ]
            if len(verbatim_tool_msgs) == 1:
                return _build_verbatim_response(user_input, verbatim_tool_msgs[0].content)

            # Same rationale for tools that already return a complete, grounded answer —
            # skip the outer agent's own retelling of it.
            self_contained_tool_msgs = [
                m for m in new_messages
                if isinstance(m, ToolMessage) and getattr(m, "name", None) in _SELF_CONTAINED_TOOLS
            ]
            if len(self_contained_tool_msgs) == 1 and not verbatim_tool_msgs:
                return _clean_response(self_contained_tool_msgs[0].content)

            raw = result["messages"][-1].content
            # Llama-4-Maverick sometimes emits tool-call syntax as plain text that
            # LangGraph never intercepted (format mismatch without a 400 error).
            # Catch it here before _clean_response strips the token blocks silently.
            if '<|python_start|>' in raw:
                logger.warning("Tool call leaked into final response; dispatching manually.")
                tool_calls = _extract_tool_calls_from_text(raw)
                if tool_calls:
                    dispatched = _run_manual_dispatch(tool_calls, user_input, prior)
                    if dispatched:
                        return dispatched
            return _clean_response(raw)
        except Exception as e:
            err_str = str(e)
            # Llama-4-Maverick uses a non-OpenAI tool-call format that LiteLLM rejects
            # with a 400. Parse the intended calls out of the error's model_output field,
            # execute them directly, then synthesize with a plain LLM call.
            if "400" in err_str and any(k in err_str for k in _TOOL_FORMAT_ERRORS):
                logger.warning("Tool-call format mismatch (400); attempting manual dispatch.")
                tool_calls = _extract_tool_calls_from_error(err_str)
                if tool_calls:
                    dispatched = _run_manual_dispatch(tool_calls, user_input, prior)
                    if dispatched:
                        return dispatched
                    # A tool call WAS identified and dispatch was attempted — real,
                    # grounded tool output may or may not exist depending on whether the
                    # tool itself failed. Either way, do not fall through to the
                    # no-tool-context direct LLM call below: that has nothing to ground
                    # it and will hedge/guess rather than admit it has no data (this is
                    # exactly how a successful tool call's real data previously got
                    # discarded and replaced by a fabricated answer). Give up honestly.
                    logger.error("Manual dispatch produced no usable output after a 400.")
                    return _HONEST_TOOL_FAILURE_MSG

                logger.warning("No tool calls extracted from 400 error; falling back to direct LLM.")
                # Last resort: direct LLM call with no tool context. Only reached when no
                # tool call could even be identified from the error — there is genuinely
                # nothing to dispatch, so this is the least-bad option left.
                try:
                    from src.assistant.llm import get_chat_model
                    llm = get_chat_model()
                    synth_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
                    return _clean_response(llm.invoke(synth_messages).content)
                except Exception as e2:
                    logger.error("Synthesis fallback failed: %s", e2)

            logger.error("Agent invocation failed: %s", e)
            return "I encountered an error processing your request. Please try rephrasing."
