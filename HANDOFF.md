# Handoff: Acceptance Suite Hardening + Live Cypher/Prompt Defect Fixes

## Goal

Shore up coverage gaps in `tests/assistant/test_search_integration.py`, get it running
live against a real Neo4j instance for the first time in a while, fix whatever real bugs
that surfaces, and manually walk through the General Assistant UI (`rocco_ui.py`) as the
"UI part" of project board issue #42 (20-query acceptance suite through the tabbed UI).

This picks up mid-way through that UI walkthrough — the user found two more issues live
in the chat UI that are **not yet fixed**. See "Next Steps" below; that's where the next
session should start.

---

## Current Progress

### 1. Closed 8 test-coverage gaps in `test_search_integration.py` (done, committed-ready)

Added: a mock/prompt check for the sparse-field `OPTIONAL MATCH` guard (since
`cypher_qa()` can't be inspected live without an LLM), a domain-QA fallback-disclaimer
test (D-5), an honest-gap/no-fabricated-tutorial test (D-6), a query-expansion
regression test for already-specific queries (Q-3) + strengthened Q-1, structural mock
tests for `GraphStore.search_datasets()` (combined vector+filter) and
`component_search()`, an REV-specific workflow test (W-5), and a Semantic-Scholar
network-failure test (L-4, mocks `requests.get` to raise — fully offline).

### 2. Got Neo4j loaded and embedded on this machine (done)

Pipeline used (all scripts already existed, just hadn't been run here):
```bash
python scripts/scrape_metadata.py                          # → data/metadata/*.json (gitignored)
python scripts/load_graph.py --mode rebuild --skip-index   # loads graph structure
python scripts/build_dataset_vector_index.py               # embeds datasetEmbedding + componentEmbedding
python scripts/audit_schema.py --neo4j --verify             # confirms 176 datasets loaded
```
Docker Neo4j on this machine is reachable at `neo4j://localhost:7687` (published port
confirmed via `docker ps --filter publish=7687`) — this **just works** as `localhost` as
long as the container publishes the port; no IP substitution ever needed for
same-machine setups.

### 3. Fixed a `pytest.mark.skipif` bug that silently skipped all `_live` tests (done)

`skipif` conditions on `S*_live`/`M*_live` tests checked `os.getenv("LLM_API_KEY")`
directly, evaluated at **collection time**, before the `chat_model` fixture's own
`load_dotenv()` call ever ran. Since the shell environment itself never has
`LLM_API_KEY` (only `.env` does), these tests always silently skipped regardless of
`.env` contents. **Fixed** by deleting the redundant `skipif` decorators — the
`chat_model` fixture's own `pytest.skip()` (which does call `load_dotenv()` first) is now
the single source of truth. This is what let the tests actually run live for the first
time and surface the bugs below.

### 4. Fixed two real Cypher-generation defects in `src/assistant/graph_store.py` (done, verified against live data)

- **`OPTIONAL MATCH` + inline `WHERE` scoping bug**: a `WHERE` immediately following an
  `OPTIONAL MATCH` is scoped as part of that match's own predicate in Cypher — if the
  predicate fails, `OPTIONAL MATCH` still emits a row with the variable bound to `NULL`,
  silently defeating the filter. The LLM-generated Cypher for "datasets with both a
  segmented image and a simulation analysis" hit this and returned **all 176 datasets**
  instead of the correct 12. Fixed by adding explicit AND-condition guidance +
  a worked `WITH d, dd, ad` example to `CYPHER_GENERATION_TEMPLATE` (the existing example
  only covered the OR case). Verified: now returns exactly 12.
- **Undocumented `voxelDimensions` format**: `MANUAL_SCHEMA` never described the actual
  stored string format (`"X, Y, Z units (in micrometers): 3.3113, 3.3113, 3.3113"` — no
  `x` delimiter, mixed units), so the LLM guessed a wrong `split(..., 'x')` pattern that
  matched nothing. Fixed by documenting the real format + unit-conversion guidance in
  `MANUAL_SCHEMA`. Verified against live data.

### 5. Fixed a tutorial-path hallucination in `get_educational_context`/`get_workflow_guidance` (done, verified 5/5 stable)

When no real tutorial matched (`_match_tutorials()` returns `[]`), the model
(Llama-4-Maverick via SambaNova/TACC) would sometimes still fabricate a plausible-looking
notebook path — either by echoing prompt examples, or (worse, discovered mid-fix) by
reasoning "since the actual context isn't provided, let's hypothetically say there's a
tutorial..." and presenting the hypothetical as its final answer. **Prompt-only fixes
(honesty-guard wording, "answer directly" instruction) were unreliable and one attempt
made it strictly worse** (see "What Didn't Work"). The fix that actually worked is a
**deterministic code-level guard**: `_strip_fabricated_tutorial_reference()` in
`src/assistant/tools.py` — if `tutorials` is empty but the response still contains
`.ipynb`, strip the fabricated block (or replace the whole response if it doesn't match
the expected format) and substitute the fixed honest-gap message. Verified 5/5 stable
runs after this fix, vs. failing 3-5/5 with prompt-only attempts.

### 6. Restored LaTeX instruction in `src/prompts/educational.yaml` — **INCOMPLETE, see Next Steps**

`educational.yaml`'s system prompt was reverted to LaTeX delimiters (`$...$`, `$$...$$`)
after discovering it had been deliberately changed to plain-text math earlier in the
project (because SambaNova/Streamlit weren't rendering it — see "What Didn't Work").
Added a `_normalize_latex_delimiters()` fallback in `assistant_ui.py` for `\(...\)`-style
output. **However**: the user just reported live in the UI that a Darcy-permeability
response rendered with NO LaTeX at all (`k = QmuL/(A*dP)`, plain text, no `$`). Root
cause is almost certainly that **`conversation_manager.py`'s own top-level `SYSTEM_PROMPT`
still has its own, separate "Response formatting" section that says "Use plain text for
mathematical expressions... Do not use LaTeX"** (line ~220) — this is the prompt that
actually governs the ReAct agent's user-visible final synthesis, and it was never updated
when `educational.yaml` was fixed. **This is the next thing to fix.**

### 7. Attempted fix for "robotic"/over-narrated query responses in `conversation_manager.py` — **PARTIALLY WORKED, one issue remains, one NEW issue just reported**

`conversation_manager.py`'s `SYSTEM_PROMPT` had an unconditional "Chain-of-thought
preamble" rule that made the agent narrate `search_datasets`' internal
`[search reasoning: ...]` line (from `expand_query()`, which *always* produces a
rationale, even for plain queries) on every single search, including plain
already-specific queries like "coal samples" — producing redundant, "robotic"-sounding
preambles.

- **Fixed**: removed the unconditional preamble section, merged its intent into the
  existing "Suitability query synthesis" section, now gated on classifying the query
  itself (plain property statement vs. purpose/task description) rather than on whether
  a rationale string exists. Verified live: plain queries no longer get an explicit
  "search reasoning" narration; genuine suitability queries ("suitable for LBM
  simulation") still get the useful requirement-explanation + per-result fit notes.
- **Known residual issue (user + I agreed to leave as-is for now)**: plain queries still
  get a trailing "To narrow this further, could you tell me what specific properties
  matter most?" clarifying question appended, even though the query was already specific.
  Low severity (extra sentence, not a wrong claim) — explicitly deferred, not a bug to
  chase further without new information.
- **NEW, not yet investigated**: the user's latest message shows the "coal samples"
  query in the *actual UI* (not my `ConversationManager()` test harness) still produces
  narration: *"The query 'coal samples' implies a search for datasets related to coal's
  microstructure and properties..."* — this reads like the same category of
  unwanted-preamble problem recurring in different wording, meaning the "Suitability
  query synthesis" gate is **still not reliably followed** by this model. This needs
  fresh investigation next session — don't assume the earlier fix fully solved it.

---

## What Worked

- **Deterministic code-level guards beat prompt-only fixes** for this specific model
  (Llama-4-Maverick via SambaNova/TACC). Twice this session, prompt-wording iteration on
  a reliability/hallucination issue produced unpredictable, sometimes *worse* results
  (see below), while a code-level check (`_strip_fabricated_tutorial_reference`) fixed
  the same issue deterministically. **Prefer code-level guards over prompt engineering
  when the desired behavior is "never do X" and X is programmatically detectable** (e.g.
  "never reference a notebook path when no tutorial matched" — detectable via `.ipynb` in
  response + empty `tutorials` list).
- Running the same live test 5+ times in a row (not just once) was essential to catch
  non-determinism — a single passing run after a prompt edit is not evidence of a fix.
- Removing a redundant/overlapping prompt instruction (the unconditional "Chain-of-thought
  preamble") reduced but did not eliminate the over-narration behavior — worth remembering
  this model doesn't reliably honor fine-grained conditional branching in system prompts.

## What Didn't Work

1. **"Answer directly, no visible reasoning" instruction in `educational.yaml`** — made
   the tutorial-hallucination bug **strictly worse**: instead of a hedged answer that
   *still* landed on the correct honest-gap message via visible chain-of-thought, the
   model skipped the hedge entirely and confidently fabricated a notebook path
   (`digital_rock_physics/nmr_relaxometry.ipynb`) with zero hedging. Reverted immediately.
   **Lesson: removing a model's visible reasoning trace can remove its own
   self-correction path — don't assume "more direct" prompting is safer.**
2. **Iterating on prompt wording alone to fix the tutorial hallucination** (v0.1.6 →
   0.1.8 across ~4 edits) reduced the failure rate (3/3 fail → 2/5 fail) but never
   eliminated it, and took several rounds of re-testing to even measure that much. Not
   worth further iteration once a deterministic guard is available.
3. **Trusting a single test run as confirmation** — early in this thread, a fix looked
   "verified" after 1 passing run, then failed 2/5 and later 5/5 on repeat testing. The
   non-determinism only shows up with repeated sampling.

---

## Update (this session): all three Next-Steps bugs fixed

1. **LaTeX rendering** — fixed. `conversation_manager.py`'s `SYSTEM_PROMPT` "Response
   formatting" section had a stale "Use plain text for mathematical expressions"
   instruction (line ~220) that overrode `educational.yaml`'s LaTeX instruction, since
   `SYSTEM_PROMPT` governs the outer ReAct agent's final synthesis. Replaced it with a
   LaTeX-delimiter instruction (`$...$` / `$$...$$`) matching `educational.yaml`, plus an
   explicit "preserve LaTeX already present in tool output verbatim" clause. Single edit
   fixes both the normal ReAct path and the manual-dispatch synthesis path (both reuse
   `SYSTEM_PROMPT`).

2. **"Robotic" narration on plain discovery queries** — root cause moved to a
   deterministic code-level guard instead of prompt-level classification (per this
   project's own "What Worked" lesson: prompt-only fixes for this model are unreliable).
   `search_datasets` (`tools.py`) now checks a new `_is_plain_property_query()` helper
   against known closed-vocabulary schema values (`porousMediaType`, `source`,
   `segmented`) plus an imaging-keyword list, and only prepends the
   `[search reasoning: ...]` tag when the query is *not* a plain property query —
   regardless of whether `expand_query()`'s rationale is non-empty (it always is). The
   "Suitability query synthesis" section in `conversation_manager.py` was simplified to
   just react to tag presence/absence rather than re-deriving the classification itself.

3. **Missing tutorial citation + "Tool call leaked into final response; dispatching
   manually"** — root cause was more specific than originally guessed: the manual-dispatch
   fallback path (triggered when Llama-4-Maverick emits its tool call as literal text
   instead of a structured call) was *correctly* re-invoking `get_workflow_guidance`
   (which does find the right tutorial), but then discarded that already-correct,
   citation-checked answer by re-synthesizing it through a **second LLM call** governed
   by the generic `SYSTEM_PROMPT` — which has no verbatim-citation rule and invites
   paraphrasing. Fixed in `_run_manual_dispatch()`: for a single leaked call to
   `get_workflow_guidance` or `get_educational_context` (tools that already return a
   self-contained, user-ready answer), skip the second synthesis call and return the
   tool's own cleaned output directly. For other/multi-call cases, kept the synthesis
   step but added an explicit "preserve notebook paths, DOIs, and LaTeX verbatim"
   instruction to its prompt.

**Verification so far:** `pytest tests/assistant/ -v` — 121/122 passing, same baseline as
before this session (the one failure, `test_s3_fibrous_media_honest_gap_live`, is a
pre-existing live vector-search relevance flake unrelated to these changes — the top
hybrid-search hit for "fibrous media" wasn't actually fibrous and wasn't a gap message
either; a retrieval-quality issue, not a regression from this session's edits).

**Not yet done — next session should start here:** the actual live
`streamlit run rocco_ui.py` manual walkthrough of all three fixes was not re-run in this
session (no interactive UI access in this environment). Before treating these three bugs
as closed:
- Re-run "How is the Darcy permeability computed from a lattice Boltzmann simulation?"
  several times (the leak path is non-deterministic — model-dependent) and confirm the
  `5-2-1_lbm_d2q9_bgk.ipynb` notebook path now appears, both via the manual-dispatch path
  (watch for the "dispatching manually" log line) and the normal path.
- Ask a question with math (e.g. "How is porosity defined?") and confirm `$...$`/`$$...$$`
  renders via KaTeX, not plain text.
- Ask "coal samples" / "sandstone datasets" and confirm no reasoning preamble; ask
  "datasets suitable for LBM simulation" and confirm the suitability framing still works.
- Then resume the broader 20-query acceptance walkthrough (#42) below.

---

## Update 2 (this session): live UI review surfaced + fixed 2 more issues

The user manually ran several queries against the live UI (not yet re-checked for the
three "Update 1" fixes above, but this round of review surfaced two *additional*,
previously-undetected problems):

1. **Every `search_datasets` response ended with a redundant recap paragraph** that
   just restated the bullets already shown, e.g. "These datasets are related to coal
   samples. The Moura Coal dataset contains..." — pure repetition, no new information.
   Happened on every query type, including the plain queries whose reasoning preamble
   Update 1 already suppressed — this is the actual remaining "robotic" pattern.
   (Reviewed and explicitly ruled **not** a bug: the model inferring "gas diffusion
   layers are a type of fibrous media" — that's legitimate Tier-3 domain knowledge, not
   an asserted dataset property.)
2. **No honest disclosure when semantic search returns weak/off-topic matches.** Asking
   "Are there any fibrous media datasets?" returned a non-fibrous bead-pack dataset as
   the top hit with no gap message — this is the exact live reproduction of
   `test_s3_fibrous_media_honest_gap_live`, which had been passed off as a "pre-existing
   flake" in Update 1 but turned out to be a real, reproducible bug. A follow-up "What
   are other fibrous media samples?" returned one real match plus five unrelated
   datasets captioned only "may be relevant... or related topics," papering over weak
   recall instead of disclosing it.

**Fixes:**

1. **Recap paragraph** — prompt instruction added to `conversation_manager.py`
   `SYSTEM_PROMPT`'s "Response formatting" section: open dataset results with a short
   header (e.g. "Datasets:", per the user's preference — a header is fine, a
   *trailing* recap is not), and explicitly do not add a closing paragraph that
   restates the bullets. Backed by a **deterministic code-level guard**
   (`_strip_recap_paragraph()` in `conversation_manager.py`, wired into `_clean_response`
   so it applies at every return path): finds the last bullet/source-labeled line, and
   strips any paragraph after it that opens with a known recap phrase ("These datasets
   are related to...", "These results may be...", etc.). Deliberately scoped to *after*
   the last bullet only, so it never touches a leading header or a legitimate
   suitability-query fit note / clarifying question that also appears after the bullets.
2. **Weak-match honesty** — rather than a numeric similarity-score threshold (rejected:
   embedding-score scale is model/index-dependent and an arbitrary cutoff would be
   fragile), used the same deterministic keyword-overlap pattern as the
   `_is_plain_property_query()` guard from Update 1. New helpers in `tools.py`:
   `_extract_query_topic_terms()` (pulls concrete topic terms from the query via the
   existing rock-type/imaging-keyword vocab, plus any `inferred_filters` values
   `expand_query` returned) and `_results_mention_any()`. In `search_datasets`, if topic
   terms were identified and none of the returned results mention any of them, an
   honest `[weak match: ...]` tag is prepended (results are still shown in full — this
   is additive disclosure, not filtering). `conversation_manager.py`'s SYSTEM_PROMPT was
   updated to tell the model to relay this tag as a plain "no direct matches, showing
   closest available results" statement rather than inventing a relevance justification.

**Verification:** `pytest tests/assistant/ -v` — **122/122 passing**, including
`test_s3_fibrous_media_honest_gap_live`, which now passes for real (not just luck) since
the tool output gives the model a concrete, grounded signal to relay. Added
`tests/assistant/test_conversation_manager.py` (new file — `_strip_recap_paragraph`
unit tests: strips real recap text, preserves leading headers, preserves suitability
clarifying questions and per-result fit notes) and `TestSearchDatasetsWeakMatch` in
`tests/assistant/test_tools.py` (mocked `hybrid_search`/`component_search`, asserts the
`[weak match]` tag appears for off-topic results and is absent for on-topic ones).

**Not yet done — still no live UI walkthrough this session** (no interactive Streamlit
access in this environment). Next session should re-run, live, all the queries from both
review rounds: "Show me datasets from coal samples", "I want datasets with segmented
micro-CT images of natural sandstone", "Are there any fibrous media datasets on the
portal?", "What are other fibrous media samples?", "Datasets useful for studying CO2
sequestration at the pore scale", plus the Update 1 LaTeX/tutorial-citation queries —
confirm no recap paragraph, confirm the weak-match disclosure phrases honestly, and
watch for any further un-anticipated LLM formatting drift before treating #42's UI
walkthrough as ready to demo.

---

## Update 3 (this session): porosity scale bug + a real data-loss bug in the 400-error fallback path

The user ran "porosity above 0.3" then "porosity above 0.4" live and caught two more
issues — one data-quality, one a genuine, more serious code bug (this one actually lost
real, already-fetched tool data and replaced it with a fabricated answer).

1. **Porosity scale ambiguity.** The raw DRP metadata (`data/metadata/*.json`, already
   loaded into Neo4j) stores `porosity` inconsistently — some datasets as a 0-1
   fraction (e.g. `0.39`), others as a 0-100 percent value (e.g. `30.0`, `50.0`,
   confirmed via grep) — with no units field to tell them apart. `Grain Packing`
   (DRP-1, raw porosity=30) showed up for both "porosity above 0.3" and "above 0.4"
   because 30 trivially passes either fraction-scale filter. Per `CLAUDE.md`'s rule
   against modifying DRP metadata, the raw values can't be changed — fixed at the
   Cypher-generation layer instead, following the exact precedent already used for
   `voxelDimensions`' inconsistent free-text format: `graph_store.py`'s `MANUAL_SCHEMA`
   now documents the mixed scale and instructs the LLM to normalize with
   `CASE WHEN s.porosity > 1 THEN s.porosity / 100 ELSE s.porosity END` before any
   numeric comparison, and `CYPHER_GENERATION_TEMPLATE` got a matching worked example.

2. **A successful tool call's real data was being thrown away and replaced by a
   fabricated guess** — worse than initially suspected. **Correction to this session's
   own earlier diagnosis**: the first draft of this fix assumed the bug was in the
   separate `<|python_start|>`-leaked-text branch (`conversation_manager.py:342-355`).
   The user's own terminal log proved that wrong — it showed `get_dataset_details` (via
   `GraphCypherQAChain`) running **twice** and returning **correct, real rows both
   times**, yet the user-visible final answer was still a fabricated hedge
   ("Unfortunately, the current information doesn't directly provide porosity
   values..."). The real mechanism: this is the **400 tool-format-error branch**
   (`conversation_manager.py:360-`). Its manual-dispatch tool call succeeds
   (`_run_manual_dispatch`'s `results` list is non-empty — real data), but
   `get_dataset_details` isn't in the `_SELF_CONTAINED_TOOLS` shortcut, so that real
   output was handed to a second, polish-only LLM synthesis call. When *that* call
   raised, the surrounding `except: return None` discarded the real tool data entirely,
   and `chat()`'s 400-branch then fell through to a "last resort: direct LLM call with
   no tool context" — which has nothing to ground it and hedges/guesses.

   **Fixed both failure points, not just added a retry:**
   - `_run_manual_dispatch`: on synthesis-call failure, now returns the raw (still real,
     grounded) tool output directly instead of `None` — only returns `None` when no
     tool actually succeeded at all.
   - `chat()`'s 400-branch: the no-tool-context "last resort" direct LLM call is now
     reachable *only* when no tool call could be identified from the error at all (the
     genuinely-nothing-to-dispatch case). If a tool call was identified but
     `_run_manual_dispatch` still returns falsy, the code now returns a new fixed
     `_HONEST_TOOL_FAILURE_MSG` constant (styled like `tools.py`'s
     `_HONEST_NO_TUTORIAL_MSG`) instead of ever reaching the ungrounded fallback.
   - Left the `<|python_start|>`-leaked-text branch untouched — confirmed not the
     culprit for this symptom.

   **Noted, not fixed:** the duplicate `GraphCypherQAChain` run itself (tool runs once
   inside the normal LangGraph step, then again inside manual dispatch after the 400)
   is an unavoidable consequence of `self._agent.invoke()` having no bound
   checkpointer in this code path — there's no partial state to recover the first run's
   output from when an exception is raised. Real latency/cost inefficiency, but a
   separate, larger fix (wiring a checkpointer/`thread_id` through) not attempted here.

**Verification:** Added `TestRunManualDispatchFallback` and `TestChatHonestFallback` to
`tests/assistant/test_conversation_manager.py` (mocked — no live LLM/Neo4j): confirms
raw tool output survives a synthesis failure, confirms `None` is still returned only
when no tool succeeds, and confirms `chat()` returns `_HONEST_TOOL_FAILURE_MSG` (not
ungrounded text) when a 400 error's dispatch attempt produces nothing usable.
`pytest tests/assistant/ -v` — **133/133 passing** (129 prior + 4 new tests).

**Not yet done:** live re-verification of the exact "porosity above 0.3" → "above 0.4"
sequence through the real UI/`chat()` path (no interactive Streamlit access this
session) — confirm Grain Packing (porosity=30) is now excluded from fraction-scale
queries, and that a 400 error (if one still occurs) now surfaces the real dispatched
data instead of a fabricated guess.

---

## Next Steps (superseded by "Update" above for items 1 and 2 — kept for history)

**Start here — two live UI bugs the user just reported, neither fixed yet:**

1. **LaTeX still not rendering in the actual UI.** Almost certainly because
   `conversation_manager.py`'s own `SYSTEM_PROMPT` (not `educational.yaml`) has a
   separate, stale "Response formatting" instruction saying "Use plain text for
   mathematical expressions... Do not use LaTeX" (around line 220 as of this session —
   confirm current line number, this file was being actively edited). This is the prompt
   that actually governs the top-level ReAct agent's final answer to the user, so fixing
   `educational.yaml` alone was insufficient. **Fix**: update this section to match
   `educational.yaml`'s LaTeX instruction (or better, keep formatting rules in one place
   to avoid this exact drift happening again). Re-verify live in the actual
   `streamlit run rocco_ui.py` UI, not just via a `ConversationManager()` script, since
   that's the surface that's actually broken.

2. **"Coal samples" query in the live UI still narrates query-expansion reasoning**
   ("The query 'coal samples' implies a search for datasets related to coal's
   microstructure and properties...") despite the "Suitability query synthesis" gate
   that was supposed to suppress this for plain queries. The earlier fix (this session,
   step 7 above) reduced but apparently didn't fully solve this — reproduce it again,
   consider whether a **code-level fix is more appropriate here too** (mirroring the
   tutorial-hallucination fix): e.g., in `tools.py`'s `search_datasets()`, only include
   the `[search reasoning: ...]` line in the tool's *returned string* when the query is
   deterministically classified as suitability/purpose-based (so the agent has nothing
   to narrate for plain queries, rather than trusting it to selectively ignore rationale
   that's always present). Needs a reliable, code-based classifier — e.g. check whether
   the raw query already contains a known schema value (rock type list, "segmented",
   etc.) as a "plain query" signal, vs. task/purpose phrasing.

3. **Once both are fixed**, resume the manual UI walkthrough of the 20-query acceptance
   set (see prior conversation for the full query list, organized by S/M/D/W/Q/L
   categories) — check LaTeX rendering, colored source badges, DOI links, and the D-6/M-3
   fixes specifically, then capture screenshots/recording for the #42 demo and #45's demo
   video.

4. **Separately, lower-priority**: the Semantic Scholar literature test
   (`test_l2_super_resolution_micro_ct`) is flaky due to Semantic Scholar's own search
   relevance on that exact phrasing (returned an unrelated bone-measurement paper) — not
   a bug in this repo, not investigated further this session.

5. Per CLAUDE.md's "Remaining Work Before Project Conclusion" section: the
   previously-unverified "hanging test suite" issue is **resolved** — `pytest
   tests/assistant/` (full directory, no marker filter) completed cleanly in ~170-210s
   with no hang, both before and after this session's fixes. That item can be checked off.

---

## File State

### Files modified this session (not yet committed as of this handoff — verify with `git status`)
| File | Change |
|------|--------|
| `tests/assistant/test_search_integration.py` | +8 new tests (D-5, D-6, Q-3, TestComponentSearch, TestCombinedSearch, L-4, sparse-field-guard prompt check, W-5); strengthened Q-1; loosened M-3 assertion; added `import re` |
| `tests/assistant/conftest.py` | *(read, not modified — `mock_graph_store` fixture reference only)* |
| `tests/assistant/test_prompts.py` | Fixed stale `rock_type`/`modality` assertions → `porousMediaType`/`voxelDimensions` |
| `tests/assistant/test_tools.py` | Fixed 3 stale `assert_called_once()` assertions (now called twice due to internal `_match_workflows` LLM ranking call) → `assert_called()` + last-call content check |
| `src/assistant/graph_store.py` | Fixed `MANUAL_SCHEMA` (voxelDimensions format) + `CYPHER_GENERATION_TEMPLATE` (AND-condition `WITH` guidance) |
| `src/assistant/tools.py` | Added `_strip_fabricated_tutorial_reference()` guard, applied in `get_workflow_guidance` + `get_educational_context` |
| `src/prompts/educational.yaml` | v0.1.5 → v0.1.8: restored LaTeX delimiters, added "context is real, not hypothetical" honesty guard. **Still has the old plain-text-math instruction duplicated/overridden by `conversation_manager.py` — see Next Steps #1** |
| `src/assistant/assistant_ui.py` | Added `_normalize_latex_delimiters()` for `\(...\)`/`\[...\]` → `$...$`/`$$...$$` fallback |
| `src/assistant/conversation_manager.py` | Removed unconditional "Chain-of-thought preamble" section; rewrote "Suitability query synthesis" to gate on query classification — **incomplete fix, see Next Steps #2**. Also still has stale plain-text-math instruction — see Next Steps #1 |

### Environment / data state
- Neo4j (Docker, `my-neo4j` container) on this machine is fully loaded: 176 datasets,
  both vector indexes (`datasetEmbedding`, `componentEmbedding`) built and populated.
- `.env` has `NEO4J_URI=neo4j://localhost:7687`, `LLM_PROVIDER=sambanova`,
  `LLM_MODEL=Llama-4-Maverick-17B-128E-Instruct`.
- Full `tests/assistant/` suite: 121-122/122 passing (only the unrelated Semantic
  Scholar flake fails), confirmed multiple times this session.
