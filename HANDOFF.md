# Handoff: Rocco General Assistant — Sprint Execution (Week 1 BCC Tasks)

## Goal

Build the General Assistant module (v2.0.0) for the Rocco porous media research platform.
The intern (JRS) starts Week 2 on Jun 7 and needs a complete, tested tool interface to code against.
Bernie (BCC) owns all prompt files, educational/workflow tools, and literature routing.

---

## Current Progress

### ✅ Completed This Session (Issues #24, #29, #30)

All three Week 1 BCC issues are implemented and tested. 28 new unit tests pass.

#### #24 — Prompts (`src/prompts/`)

- **`query_expander.yaml`** — fleshed out from stub:
  - Primary purpose clarified: semantic enrichment (domain vocabulary expansion) before filter inference
  - 3 few-shot examples: rock-type query, workflow/method query, voxel-range query
  - Valid filter fields documented: `rock_type`, `modality`, `voxel_size_um`, `segmented`, `has_simulation`
  - All filters optional — only populated when clearly implied by phrasing

- **`educational.yaml`** — fleshed out from stub:
  - Tiered knowledge policy: portal facts → tools only; domain Q&A → context-first + fallback disclaimer; foundational concepts → pre-trained OK
  - LaTeX math instruction: inline `$...$`, block `$$...$$`
  - Honesty rule: never assert properties not in context
  - Template vars: `{{ context }}` (system), `{{ question }}` (user)

- **Tests:** `tests/assistant/test_prompts.py` — 12 tests, all pass (load/render validation, no LLM calls)

#### #29 — Educational tools (`src/assistant/tools.py`)

- **`expand_query(query) -> dict`** — renders `query_expander.yaml`, calls LLM, parses JSON; graceful passthrough on parse error. Not a `@tool` (called internally).
- **`get_workflow_guidance(goal) -> str`** (`@tool`) — keyword-matches `domain_workflows.yaml` (up to 3 workflows), keyword-matches `tutorials.yaml`, assembles context, calls LLM via `educational.yaml`.
- **`get_educational_context(question) -> str`** (`@tool`) — same as above but also pulls `global_best_practices` sections when query triggers relevant terms (REV, resolution, segmentation, etc.).
- `general_chat` retired from `build_langchain_tools()` (still in file for reference).

**Data quality fix:** `domain_workflows.yaml` has `- []` (nested empty list) as an `example_datasets` entry in the REV workflow — context builder now coerces to strings and filters empties.

#### #30 — Literature routing (`src/assistant/tools.py`)

- **`search_literature(query) -> str`** (`@tool`) — lazy-loads `LiteratureSearch` singleton, calls `search_external_literature(max_results=5)`, formats results with `[semantic scholar]` source labels, author truncation ("et al." after 3), and "No abstract." placeholder.
- **`build_langchain_tools()`** — now returns all 5 active tools: `search_datasets`, `get_dataset_details`, `get_workflow_guidance`, `get_educational_context`, `search_literature`.

- **Tests:** `tests/assistant/test_tools.py` — 16 tests, all pass (mocked LLM/API, no credentials needed)

### Pre-existing (not regressed)

The full test suite has 38 failures in `test_intent_classifier.py` (24) and `test_search_integration.py` (14) — both require `LLM_API_KEY` env var to be set and are live acceptance tests. These were failing before this session.

---

## What Worked

- Patching `src.assistant.llm.get_chat_model` (source module) rather than `src.assistant.tools.get_chat_model` — functions import `get_chat_model` locally inside each call, so the patch must target the module where it's defined.
- Using `.func` accessor on LangChain `@tool`-decorated functions in tests to bypass the tool wrapper and call the underlying function directly.
- Factoring keyword matching into `_match_workflows()` and `_match_tutorials()` helpers, with `_global_practices_context()` for the domain Q&A path — makes `get_workflow_guidance` and `get_educational_context` share infrastructure cleanly.
- Lazy singleton pattern for `_load_workflows()` / `_load_tutorials()` — safe to call repeatedly, no repeated disk I/O.

## What Didn't Work

- **`git stash` in a background task chain** (`git stash && conda run ... | tail -5; git stash pop`) — the stash pop appeared to fail silently, leaving files at old content. Had to manually recover with `git stash pop`. Avoid this pattern; use `git diff HEAD` or a worktree instead.
- Patching `src.assistant.tools.get_chat_model` — fails with `AttributeError` because the name isn't in the tools module namespace (it's a deferred local import). Must patch at `src.assistant.llm.get_chat_model`.

---

## Next Steps

### BCC — Still Outstanding (Week 1)

1. **`data/tutorials.yaml` remaining entries** (#12) — skeleton exists and is populated through Chapter 4. Verify all notebook paths are correct against the `dpm_teach` repo; add any missing chapters.

2. **`conversation_manager.py`** (#33) — LangGraph ReAct agent that dispatches to the 5 tools now in `build_langchain_tools()`. Intent classification via `assistant.yaml` prompt routes to the right tool. Due Week 4–5 per sprint plan.

3. **`assistant_ui.py`** (#25) — Streamlit tab shell for the General Assistant. Due Week 6.

4. **Smoke test the live tools** — run the manual tests from the plan with a real API key:
   ```python
   from src.assistant.tools import expand_query, get_workflow_guidance, get_educational_context, search_literature
   print(expand_query("sandstone with low porosity"))
   print(get_workflow_guidance("compute absolute permeability"))
   print(get_educational_context("what is a Representative Elementary Volume?"))
   print(search_literature("pore network modeling permeability"))
   ```

5. **Close GitHub issues** — #24, #29, #30 can be closed; update the project board.

### Intern (JRS) — Week 2 (Jun 7–13)

- Environment setup and codebase orientation (#20)
- Neo4j field coverage audit with `scripts/audit_schema.py` (#21)

---

## Key Files Changed This Session

| File | Change |
|------|--------|
| `src/prompts/query_expander.yaml` | Fleshed out from stub (21 → 67 lines) |
| `src/prompts/educational.yaml` | Fleshed out from stub (17 → 42 lines) |
| `src/assistant/tools.py` | 4 functions implemented; `build_langchain_tools()` updated (94 → 289 lines) |
| `tests/assistant/test_prompts.py` | New — 12 tests for prompt load/render |
| `tests/assistant/test_tools.py` | New — 16 tests for all 4 new tool functions |

## Key Files (Unchanged, Still Relevant)

- `src/assistant/literature_search.py` — fully implemented Semantic Scholar wrapper (used by `search_literature`)
- `data/domain_workflows.yaml` — 15 DRP workflow entries + global_best_practices (used by educational tools)
- `data/tutorials.yaml` — portal tutorial notebook mappings (used by `get_workflow_guidance`)
- `src/prompts/assistant.yaml` — intent classifier prompt (v0.2.1, used by `conversation_manager.py`)
- `src/assistant/llm.py` — `get_chat_model()` singleton (used by all educational tools)
- `src/assistant/graph_store.py` — Neo4j search (used by `search_datasets`, `get_dataset_details`)
- GH Project board: https://github.com/orgs/digital-porous-media/projects/3

## Branch

`feature/general-assistant` — all changes are uncommitted working tree modifications.
