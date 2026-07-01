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
  Dataset discovery       search_datasets        (semantic similarity, Neo4j vector index)
  Structured queries      get_dataset_details    (Cypher QA, exact property lookups)
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

import logging
import re

from langgraph.prebuilt import create_react_agent

from src.assistant.tools import build_langchain_tools

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


def _clean_response(text: str) -> str:
    return _TOOL_CALL_RE.sub('', text).strip()


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
    for call in tool_calls:
        fn = tools_map.get(call["name"])
        if fn:
            try:
                res = fn.invoke(call["args"])
                results.append(f"--- {call['name']} ---\n{res}")
                logger.info("Manual dispatch: %s(%s)", call["name"], call["args"])
            except Exception as te:
                logger.warning("Tool %s failed in manual dispatch: %s", call["name"], te)

    if not results:
        return None

    tool_output = "\n\n".join(results)
    try:
        llm = get_chat_model()
        synth_messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + prior
            + [{"role": "user", "content": (
                f"{user_input}\n\n"
                "[Knowledge base context — do not mention internal tool names "
                "(get_workflow_guidance, get_educational_context, etc.) in your response]\n"
                f"{tool_output}"
            )}]
        )
        return _clean_response(llm.invoke(synth_messages).content)
    except Exception as e:
        logger.error("Synthesis after manual dispatch failed: %s", e)
        return None


SYSTEM_PROMPT = """\
You are Rocco, an expert research assistant for the Digital Porous Media (DPM) Portal. \
You help researchers discover datasets, understand porous media workflows, and find relevant literature.

## Knowledge tiers — follow these strictly

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

- Dataset discovery by topic or concept → search_datasets
- Exact dataset properties, counts, or filter queries → get_dataset_details
- Portal how-to guides and metadata schema reference → search_portal_docs
- Porous media science Q&A and best practices → get_educational_context
- Step-by-step DRP workflow guidance with tutorial links → get_workflow_guidance
- Finding papers or publications → search_literature

For cross-intent queries (e.g. "explain X and find me datasets that measure it"), \
call multiple tools and synthesize the results into a single coherent response.

## Response formatting

- Relay source labels from tool output exactly as returned: [graph match], [semantic match], \
[semantic scholar], [cypher match], [component match], [hybrid match]. Do not strip or rename them.
- When presenting dataset search results, always include the DOI for each entry. \
The tool output includes it as "DOI: xxx" — preserve it verbatim in your response.
- Use plain text for mathematical expressions (e.g. phi = V_pore / V_total, k = Q*mu*L/(A*dP)).
- Use markdown headers and bullet lists for multi-part answers.
- Do not editorialize or evaluate tool output — report it with light formatting only.

## Suitability query synthesis

When search_datasets returns results in response to a suitability or purpose-based query \
(e.g. "suitable for LBM", "good for relative permeability simulation"), always present \
the results first. Then apply one of the following based on the [search rationale] tag:

- **No META-PROPERTY in rationale** (purpose maps to domain/schema properties): \
Before the results, add 1–2 sentences explaining what the task requires, drawing on \
Tier 2 domain knowledge. After each result, add a brief note on why it is or isn't \
a strong match.

- **META-PROPERTY in rationale** (purpose maps to qualities not in the schema): \
Present the results as-is (no suitability annotation — you cannot verify meta-qualities). \
After all results, add: "To narrow this further, could you tell me what specific \
properties matter most? For example: do you need a segmented image, a particular rock \
type, a resolution range, or simulation outputs already included?"

Never skip presenting the results. Never ask for clarification before showing results.
"""


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
        messages = prior + [{"role": "user", "content": user_input}]
        try:
            result = self._agent.invoke({"messages": messages})
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
                else:
                    logger.warning("No tool calls extracted from 400 error; falling back to direct LLM.")

                # Last resort: direct LLM call with no tool context
                try:
                    from src.assistant.llm import get_chat_model
                    llm = get_chat_model()
                    synth_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
                    return _clean_response(llm.invoke(synth_messages).content)
                except Exception as e2:
                    logger.error("Synthesis fallback failed: %s", e2)

            logger.error("Agent invocation failed: %s", e)
            return "I encountered an error processing your request. Please try rephrasing."
