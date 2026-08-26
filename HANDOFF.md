# Handoff: Dataset Detail Follow-Up Queries (`get_dataset_profile`)

## Goal

Give the General Assistant a way to answer *"tell me more about this dataset"* /
*"give me more details on the first one"*-style follow-ups after a dataset search — with a
fuller profile of the one dataset the user means (organizational structure, file types, reuse
suitability, etc.) — instead of the old behavior, which re-ran a search/lookup and returned the
same title/DOI/one-line-summary shape as the original result.

Scope grew beyond that original ask during planning to also cover: general **and**
specific-field detail questions, pronoun/positional reference resolution ("this", "it", "the
first one"), organizational structure (`Sample -> DigitalDataset -> AnalysisDataset`),
file-type/"how do I read this in Python" reasoning, reuse-suitability reasoning with
chain-of-thought, and multi-dataset comparison.

**Status: implemented, live-verified, and MERGED.** Everything below is real, done work — not a
plan. Originally developed on `feature/dataset-details`; the work landed in commits `6efdf89`
through `ae1b95c`, and the follow-on documentation audit is a separate commit on top. The
"Next Steps" section at the bottom still lists genuinely open verification items — read it as a
to-do list, not as a description of uncommitted work.

---

## Current Progress

### Feature implementation (done)

- **`GraphStore.get_dataset_profile(reference)`** (`src/assistant/graph_store.py`) — resolves a
  dataset by datasetNumber / DOI / title (three tiers, case-insensitive, DOI-prefix-stripped),
  then fetches the full `Dataset` node plus every `PART_OF` sub-node (`Sample`,
  `DigitalDataset`, `AnalysisDataset`, `RelatedPublication`, and — speculatively, **not yet
  verified against the live schema** — `RelatedSoftware`/`RelatedDataset`) plus `INPUT_FOR`
  pipeline edges. Returns `DatasetProfileMatch` / `DatasetProfileAmbiguous` / `None`.
- **`get_dataset_profile(dataset_reference, question)` tool** (`src/assistant/tools.py`) — a
  **self-contained tool** (own internal grounded LLM synthesis pass, not a raw verbatim splice)
  because it has to reason over the data: concise high-level overview for general "tell me
  more" questions, full detail for specific-field questions, file-format/"how do I read this in
  Python" reasoning (with a real, non-fabricated TACC Corral archive URL derived from
  `datasetNumber` — see `_corral_archive_url()`), and reuse-suitability judgments.
- **New prompt** `src/prompts/dataset_profile.yaml` — tiered knowledge policy (portal facts
  context-only, file-format/suitability reasoning may draw on general knowledge but must say
  so), concise-overview-vs-specific-field framing, chain-of-thought internally but only the
  useful conclusion shown, honesty about missing fields.
- **`conversation_manager.py`** — `get_dataset_profile` added to `_SELF_CONTAINED_TOOLS`; new
  `SYSTEM_PROMPT` routing rules (follow-up detail questions route here, resolve the reference
  from history first; comparisons call this tool once per dataset); `_TOOL_PARAM_KEYS` extended
  from `dict[str, str]` to `dict[str, list[str]]` so the 400-error tool-call-recovery path can
  reconstruct a tool with **two** required args (this tool is the first one); and
  `_FOLLOWUP_TOOL_GATE_SYSTEM_PROMPT` extended with a rule + example for "same tool, second
  dataset" (a comparison) so it isn't wrongly short-circuited after the first profile call.
- **Comparisons of 2+ datasets** reuse `get_dataset_profile` called once per dataset — no
  separate comparison tool. This relies on the *already-existing* mechanism: when 2+ tool calls
  happen in one turn, the single-call short-circuit never fires and the outer ReAct agent's own
  final-message synthesis combines the results.
- **Tests** — new coverage in `tests/assistant/test_graph_store.py` (tier matching, ambiguous/
  not-found, sub-node + edge assembly, embedding-strip, `USE_NEO4J=false`),
  `tests/assistant/test_tools.py` (tool behavior, context-building, node-count cap, embedding
  strip), `tests/assistant/test_conversation_manager.py` (multi-arg extraction, followup-gate
  comparison case), `tests/assistant/test_search_integration.py` (two-turn "search → tell me
  more" and comparison flows, mocked + live-tier). **189 tests passing**, no regressions.
- **Docs** — new `docs/user_guide/dataset_profiles.rst` (full capability page); updated
  `assistant.rst`, `architecture.rst` (Request Lifecycle diagram + dropdowns), `index.rst`
  toctree, `quickstart_assistant.rst`, cross-links from `dataset_discovery.rst`/
  `structured_queries.rst`. Sphinx build verified clean (no new errors/broken refs).
- **`Tasks.md` / `HANDOFF.md`** — the old "Future Feature: Dataset Detail Follow-Up Queries"
  backlog entry in `Tasks.md` is marked `— IMPLEMENTED` with a summary of what changed vs. the
  original proposal (this file's prior revision had that same planning content — superseded by
  this rewrite).

### Production bug found and fixed this session (done)

After implementing the feature, the user hit a real error while using the General Assistant
chat UI:

```
litellm.ContextWindowExceededError: ... maximum context length is 131072 tokens.
However, your messages resulted in 254690 tokens.
```

**Root cause:** `get_dataset_profile`'s Cypher query did `RETURN d, collect(DISTINCT s), ...`
— the entire Neo4j node. Every dataset embedded by `scripts/build_dataset_vector_index.py`
(per `CLAUDE.md`, that's all 176 datasets + 3,273 sub-nodes) carries a `datasetEmbedding`
(`Dataset`) or `componentEmbedding` (`Sample`/`DigitalDataset`/`AnalysisDataset`) property — a
**4096-float vector**. That vector was flowing straight into the LLM context as a literal float
list, easily enough alone to blow the context window on a **single call**, independent of
dataset "size" by file/sample count — which is why it hit Bentheimer Sandstone (DOI
`10.17612/P77P49`, a small dataset by file count, but fully embedded) and not necessarily every
dataset.

**Fix, two layers (both landed in the current uncommitted diff):**
1. **Wire-level** (`graph_store.py`) — the full-profile Cypher query now uses map projection
   (`d{.*, datasetEmbedding: null}`, `s{.*, componentEmbedding: null}`, same for `dd`/`ad`) so
   the vectors never leave Neo4j.
2. **Python-level backstop** (`tools.py`) — `_is_embedding_like(key, value)` strips any
   property literally named `datasetEmbedding`/`componentEmbedding`, **and** (guard against any
   other renamed/future embedding field) any list of 16+ numeric values, before it reaches the
   prompt context. Applied in both `_render_node_list()` and the top-level dataset dict filter
   in `_build_profile_context()`.

A **second, unrelated defensive fix** landed in the same pass, found during investigation
before the real cause was pinned down: `_render_node_list()`/`_render_pipeline_edges()` now cap
at `_MAX_NODES_PER_TYPE = 25` sub-nodes/pipeline-chains per type, with an honest "...and N more
not shown" note (never silent). This guards against a dataset with an unusually large number of
sub-nodes doing the same thing the embeddings did — it wasn't the actual reported bug, but it's
a real, cheap-to-keep guard against the same failure mode from a different angle.

Regression tests added for both: `test_embedding_vectors_are_stripped_from_context` /
`test_full_profile_query_nulls_embedding_vectors` (embedding fix) and
`test_large_sub_node_count_is_capped_not_unbounded` (node-count cap).

**Not yet re-confirmed live**: the user had not yet retried in the chat UI as of the end of
this session to confirm the fix resolves the actual reported error. This is the most important
next step (see below).

---

## Follow-up Session: Three More Live Bugs Found and Fixed

The user drove real multi-turn conversations through the chat UI (with a live Neo4j instance)
and reported the raw logs/transcripts back. All three bugs below were root-caused from those
transcripts, fixed, and — unlike the section above — **live-verified working** (either by the
user directly in the chat UI, or by me re-running the exact query/transcript against the real
`GraphStore`/`ConversationManager`). All fixes are in the same uncommitted
`feature/dataset-details` diff.

### Bug 1 — Cumulative-filter refinement silently re-scanned the whole graph each turn

**Symptom:** A 3-turn refinement chain ("find sandstone datasets" → "which of these are
segmented?" → "which have porosity < 0.25?") looked like it was narrowing correctly, but the
underlying mechanism was fragile: the existing `_cumulative_filter_text` approach composes a
compound natural-language question and re-sends it to `get_dataset_details` each turn, which
generates **brand-new Cypher over the entire graph** every time — it never actually queries only
the previously-listed dataset IDs. Live evidence this was a real (not just theoretical) risk:
the *same* "sandstone" constraint, reworded into two different compound questions across two
turns, produced two different WHERE clauses (`s IS NULL OR toLower(...)  = 'sandstone'` vs.
`s IS NOT NULL AND toLower(...) = 'sandstone'`) — meaning a refinement's result set is not
guaranteed to be an actual subset of the prior turn's, only however the regenerated Cypher
happens to phrase it that time.

**Fix:** `GraphStore.cypher_qa()` (`graph_store.py`) gained an optional `restrict_to_titles:
list[str]` param — after the freshly generated Cypher runs, rows are deterministically
post-filtered in Python to only those whose title is in this list. `get_dataset_details` (tools.py)
passes it through. `conversation_manager.py`'s refinement dispatch now passes
`restrict_to_titles=[titles from self._last_dataset_mentions]` alongside the existing compound
question — the compound question still helps the Cypher-generation LLM pick the right new
property to filter on, but the actual "stay within the previously shown set" guarantee no longer
depends on it being consistent.

### Bug 2 — Dataset comparison lost one dataset when referred to anaphorically

**Symptom:** After "Tell me about the Gildehauser Sandstone?" (a `get_dataset_profile` call),
the next turn — "How does **that dataset** compare with the downscaling-based segmentation
one?" — returned a comparison that only had data on the downscaling dataset, with the model
stating "the context provided doesn't contain any information about the Gildehauser Sandstone
dataset," despite Gildehauser having just been discussed.

**Root cause:** `_detect_comparison_references()` (the deterministic dispatch that fetches both
profiles and bypasses the ReAct agent's own unreliable multi-tool-call behavior) only resolves
references against `_last_dataset_mentions` — the most recent **listing**. It had no concept of
"the dataset the user was just individually shown a profile for." A bare anaphoric reference
("that dataset") matched nothing, so only 1 of the 2 needed references was found, the function's
`len(refs) >= 2` gate failed, and the turn fell through to the unreliable single-tool-call agent
path — which called `get_dataset_profile` for only the downscaling dataset.

**Fix:** Added `_extract_profiled_dataset()` (parses `get_dataset_profile`'s own
code-generated `[dataset profile] Title (DOI: ...)` header — same pattern as the existing
listing-mention parser) and `self._last_profiled_dataset`, refreshed on every dispatch path that
can produce a `get_dataset_profile` result. `_detect_comparison_references()` now accepts
`last_profiled` and a new `_DATASET_ANAPHORA_RE` (`"that dataset"`, `"this one"`, `"the other
dataset"`) — when exactly one explicit reference is found and the message contains an anaphoric
phrase, the second reference resolves to `last_profiled`.

**Live-verified** by the user replaying the exact transcript in the chat UI: the comparison now
correctly pulls real properties from both datasets.

### Bug 3 — Leaked chain-of-thought scaffold in the comparison answer

**Symptom:** Once Bug 2 was fixed, the *comparison itself* was correct, but the response the
user saw was the model's raw reasoning trace — `Step 1: Identify the key characteristics of
DRP-137. ... Step 8: ... The final answer is: <the actual answer>` — not a clean answer.
`_COMPARISON_SYNTHESIS_SYSTEM_PROMPT` said a lot about *content* (preserve every fact, don't
under-report) but nothing about *output format*.

**Fix, two layers (same pattern as the embedding-vector bug above):**
1. **Prompt** — added an explicit rule to `_COMPARISON_SYNTHESIS_SYSTEM_PROMPT`: output only
   the finished comparison, no "Step N:" stages, no "The final answer is:" preamble.
2. **Code backstop** — `_strip_reasoning_scaffold()` in `conversation_manager.py`, wired into
   `_clean_response()` (so it protects every response path, not just comparisons). It requires
   **both** a `Step N:` scaffold **and** a `final answer is:` marker before stripping anything —
   this conjunction is the actual safety guard, since a legitimate `get_workflow_guidance` answer
   ("Step 1: Segment the image...") is *supposed* to have numbered steps and must never be
   touched. Falls back to the original text if the "final answer" section would be empty (never
   returns `""` — see the recurring "empty response poisons replayed history" lesson elsewhere
   in this file).

**Live-verified** by the user: confirmed the leaked scaffold is gone in the chat UI.

### Bug 4 (schema-only, not yet re-confirmed by the user in the chat UI) — Two `search_datasets` structured-lookup queries silently degraded to weak semantic search

Found from a separate transcript where two property-shaped queries returned irrelevant results
via `[hybrid match]`/`[component match]` instead of the structured Cypher path:

- *"...of the same limestone"* — `porousMediaType`'s enum has no `'limestone'` value (it's
  `carbonate`); the generated Cypher filtered on the literal string, matched zero rows, and fell
  back to semantic search.
- *"paired tomographic and segmented images"* — the Cypher-generation LLM tried to match
  "tomographic" against `DigitalDataset.fileTypes`, which is a **list**, not a string — this
  raised a live `Neo.ClientError.Statement.TypeError` (`toLower()` on a `StringArray`), and
  *that* exception is what triggered the fallback to semantic search (confirmed from the actual
  Neo4j driver log line the user shared, not just inferred).

**Fix:** three additions to `MANUAL_SCHEMA` in `graph_store.py` (the block fed to
`GraphCypherQAChain`'s Cypher-generation prompt): (1) a colloquial-rock-name → enum mapping
table (limestone/dolomite/chalk/Ketton/Estaillades → `carbonate`; quartzite/Bentheimer/Berea →
`sandstone`; etc.) with a title/description-substring fallback for names not in the table; (2) an
explicit note that there is no queryable imaging-modality field, so "tomographic"/"CT"/"micro-CT"
should never generate a WHERE clause at all (virtually every portal dataset already involves
some form of CT imaging); (3) a note that `fileTypes` is a list of file extensions, not imaging
modality, and the correct list-membership pattern (`any(f IN dd.fileTypes WHERE ...)`) if a
future query does legitimately need to filter on it.

**Live-verified by me** (not yet by the user in the chat UI) — re-ran both exact query strings
directly against `GraphStore.cypher_qa()` on the live Neo4j instance:
- The limestone query now generates `WHERE toLower(s.porousMediaType) = 'carbonate' AND
  dd.segmented = 'yes'` and returns 10 genuinely on-topic carbonate/segmented datasets.
- The "paired tomographic" query now generates `WHERE dd.segmented = 'yes'` (correctly drops the
  non-existent imaging-modality filter) and returns 10 real, DOI-backed structured results with
  no crash and no semantic-search fallback.

### Tests added this session

All in `tests/assistant/test_conversation_manager.py` unless noted:
- `TestStripReasoningScaffold` (5 tests) — scaffold-present, legitimate-numbered-workflow
  untouched, plain-answer untouched, bold-markdown variant, empty-final-answer-falls-back.
- (Bugs 1 and 4 are schema/Cypher-generation changes — verified live rather than via new unit
  tests, since they depend on the actual Cypher the LLM generates, which mocks can't exercise
  meaningfully. `restrict_to_titles` post-filtering logic itself, being pure Python, would
  benefit from a dedicated unit test — see Next Steps.)

**Full suite: 120 tests passing, no regressions** (`pytest tests/assistant/test_tools.py
tests/assistant/test_conversation_manager.py tests/assistant/test_graph_store.py -v`).

---

## What Worked

- **Reading the actual control-flow code before proposing an approach** (same lesson as the
  original planning session, held up again): tracing `graph_store.py`'s existing
  `MANUAL_SCHEMA`/relationship docstrings gave the exact confirmed `PART_OF`/`INPUT_FOR`
  relationship types to use in the new Cypher, rather than guessing.
- **Measuring instead of guessing when debugging the context-overflow report.** The first
  instinct (a large sub-node count) was checked directly by measuring `SYSTEM_PROMPT` + all
  tool description sizes (~17k chars, nowhere near the limit) *before* committing to that fix —
  which turned out to be the wrong root cause. When the user reported it recurring on a
  small-by-file-count dataset, that ruled out "sub-node count" cleanly and pointed straight at
  something size-independent: the embedding vectors carried on every embedded node.
- **Fixing at the source (Cypher) and adding a Python backstop**, not just one or the other —
  the wire-level fix is more efficient (saves Neo4j driver bandwidth too) and the Python
  backstop protects against the fix regressing or a future differently-named embedding field.
- **Writing a plan file and getting it reviewed before implementing** (this was done via
  Claude Code's plan mode in the same session) surfaced two concrete scope corrections from the
  user before any code was written: don't show empty metadata fields, and give a concise
  overview (not a field dump) for general "tell me more" questions. Both are now enforced in
  `_build_profile_context()`/the prompt, with dedicated tests.
- **(Follow-up session) Asking the user for the raw log/transcript, not just a description of
  the symptom, was what made root-causing Bugs 1, 2, and 4 fast.** The Neo4j-driver-level
  exception text for Bug 4 ("Expected a string value for `toLower`, but got: StringArray") and
  the two different generated WHERE clauses for Bug 1 were both only visible in the raw log —
  a paraphrase ("the filter isn't narrowing" / "search results aren't great") would have sent
  debugging in a much slower, more speculative direction.
- **(Follow-up session) Fixing at the Cypher-generation-prompt level (`MANUAL_SCHEMA`) rather
  than post-hoc in Python, for Bug 4.** Unlike Bugs 1–3 (which needed a code-level fix because
  the underlying model behavior genuinely can't be made reliable by prompting alone — this
  project's own repeated, hard-won lesson, see the comments throughout
  `conversation_manager.py`), Bug 4 was a genuine missing-information problem: the
  Cypher-generation LLM had no way to know `limestone → carbonate` or that `fileTypes` is a list.
  Giving it that information once, in the schema all Cypher generation is grounded in, fixed
  both queries live on the first try — a reminder that "prompting doesn't work reliably for this
  model" (true for multi-step compositional tasks like restating cumulative filters) is not a
  blanket rule; it doesn't apply to a single missing fact the model has no way to derive itself.

## What Didn't Work / Gotchas Hit This Session

- **First theory on the context-overflow bug was wrong, but not wasted.** The initial
  assumption (unbounded sub-node count) led to a real, worthwhile fix (`_MAX_NODES_PER_TYPE`
  cap) — but it wasn't *the* bug, since the user confirmed it recurred on a small dataset
  (Bentheimer Sandstone). **Lesson:** when a user reports "not for all datasets, but for some,"
  don't stop at the first plausible-sounding cause that would explain *a* context blowup — that
  phrasing is a strong signal the real cause is data-dependent in a way that doesn't correlate
  with the datasets that would be caught by an initial correctness "cap unbounded lists" fix
  (i.e. something present on many small datasets too, like the vector embeddings actually
  are — every dataset gets embedded regardless of size).
- **`git status`/file-state surprises mid-session**: `Tasks.md` briefly appeared to have
  reverted to unrelated older content on disk (an `ENOENT` on a `git status`-observed file) —
  turned out to be a transient tool-layer glitch, not a real revert; re-reading the file showed
  the edit had actually landed. **Lesson:** if a file edit seems to have vanished right after
  making it, re-read the file directly before assuming it needs to be redone — don't blindly
  retry destructive operations on a stale assumption.
- **`rocco_ui.py` has an uncommitted diff not from this session's work** (a `black`-style
  reformat plus a page-title rename from "DPM Curator" to "DPM Research Assistant"). Not
  investigated or touched — flagging so the next session doesn't assume it's part of this
  feature or accidentally reverts/re-touches it without understanding why it's there.
- (Carried over from the original planning session, still relevant if branching again from
  scratch:) **Branching off a stale local `main`** silently drops recently-merged work — always
  `git fetch origin && git log <local-branch>..origin/main` before cutting a new branch.

---

## Honest Content/Relationship Reasoning Tool — IMPLEMENTED (this session)

Bug 4 above fixed the crash/fallback for "paired tomographic and segmented images" and "...of the
same limestone," but its fix — silently dropping "tomographic" and answering with
`segmented='yes'` alone — still overclaimed: it presented a generic "has some segmented data"
list as if it had verified "paired," which it never checked. This session implemented the
designed fix. The full design is still on disk at
`/home/bchan/.claude/plans/please-review-the-handoff-fuzzy-eich.md`; what follows is what
actually landed.

### What was built

- **`reason_about_dataset_content(question)`** (`src/assistant/tools.py`) — one general mechanism
  for any question no literal field can settle. In `_SELF_CONTAINED_TOOLS` and
  `_TOOL_PARAM_KEYS`; registered in `build_langchain_tools()` (now 8 tools).
- **Precomputed fact sheets** — `Dataset.factSheet` (JSON) + `Dataset.factSheetText` (rendered
  prose), built by a new step in `scripts/build_dataset_vector_index.py`. Its own edge-preserving
  assembly (`_build_fact_sheet`/`_render_fact_sheet_text`), deliberately NOT reusing
  `_build_embedding_text` — that flattens sub-node properties into aggregated lines, discarding
  which `DigitalDataset` belongs to which `Sample`, which is the whole point here. `--only
  fact-sheets` rebuilds just this stage.
- **`factSheetEmbedding`** (vector) + **`datasetFactSheetFulltext`** (BM25) indexes.
- **`GraphStore.rank_fact_sheets()` / `fetch_fact_sheets()`** — ranking reuses the *existing*
  `hybrid_search` RRF fusion, extracted into a shared `_rrf_merge()`. No new fusion mechanism, no
  per-relationship-type Cypher condition to author for the next phrasing someone asks about.
- **`src/prompts/corpus_reasoning.yaml`** — the cited reasoning pass + the map-reduce
  batch-screening prompt used for exhaustive ("list every dataset where…") questions.
- **Deterministic `_needs_content_reasoning()` gate**, run by BOTH `get_dataset_details` and
  `search_datasets` before either commits to a Cypher answer. `search_datasets` needed it too, and
  this wasn't obvious from the plan: its own structured-first path trips `looks_structured` on the
  literal sub-clause alone ("segmented"), so routing the flagship example query there would have
  reproduced exactly the bug this tool exists to remove.
- **Grounding enforced in code, not prompt** (the project's standing lesson): a candidate with no
  citation is dropped; a candidate whose title wasn't in the shortlist actually sent is dropped as
  a likely fabrication; titles/DOIs come from graph records, never retyped by the model.

### Two real bugs found while implementing — both fixed, both live-verified

**Bug 5 — `INPUT_FOR` was documented and queried backwards everywhere.** The live graph has
`(DigitalDataset)-[:INPUT_FOR]->(Sample)` — CHILD → PARENT, "was derived from", the same
direction as `PART_OF` — confirmed independently by edge counts (1893 DigitalDataset→Sample, 983
AnalysisDataset→DigitalDataset, 55 AnalysisDataset→Sample, zero the other way) and by
`scripts/load_graph.py`'s `_establish_connection`, which writes `MERGE (s)<-[:INPUT_FOR]-(t)`
with `s` the parent in the DRP metadata's `links` list.

Because `get_dataset_profile()`'s Cypher queried it parent → child, it matched **zero** edges:
every profile's "Organizational structure" section was silently empty and every digital dataset
was reported under "no recorded sample/analysis link." `MANUAL_SCHEMA` (which grounds ALL
generated Cypher) and `docs/neo4j_schema.md` (the intern reference) had it wrong too. Fixed in
all three plus the module docstring. Live check after the fix: DRP-126 now renders
`P4-3-3-2-3-1 -> Large-Area SEM` / `-> Automated Mineralogy`; across the corpus, 35 datasets have
one sample with scans at more than one voxel dimension (was 0 before the fix — i.e. the
"same sample, different resolutions" case was entirely unrepresentable).

**Bug 6 — `get_dataset_profile()`'s Cypher was pathologically slow.** Four chained `OPTIONAL
MATCH`es cross-multiplied before `collect()`. Measured live: **28.2s** for the largest dataset
(DRP-372, 961 sub-nodes) with only the `PART_OF` joins present, and **no completion within 300s**
once the `INPUT_FOR` joins were restored — so Bug 5's fix could not ship without this one.
Decomposed into one flat query per node/edge type, assembled in Python: **0.8s** for that same
dataset, now returning 207 sample→scan and 618 scan→analysis edges where it previously returned
none. The build script uses the same decomposition in bulk (all 184 datasets in 1.3s).

**Bug 7 — the embedding endpoint limits TOTAL CHARACTERS per request, not item count.** The
first live build run failed immediately with a 500 whose body carried per-item
`{"embedding": null, "error": "unexpected_error"}` for most items in the batch. Not a length
problem per item (a single 20k-character text embeds fine) and not an item-count problem (16
short texts embed fine): measured against the live TACC/SambaNova E5-Mistral-7B-Instruct
endpoint, a request totalling ~14k characters succeeds and ~40k fails — even 2 items, if both
are large. Fact sheets are far bigger than the title+description blobs the other two embedding
passes send, so the fixed `--batch-size 16` that has always worked for them fails on the very
first fact-sheet batch.

Fixed with `_batch_by_char_budget()` (12k-character budget, oversized single sheets get their
own request) plus real failure recovery: the previous code only handled a *short return* from
the API, not an exception, so any 500 aborted the entire 184-dataset build. Now a failed batch
retries item-by-item, and an item that still fails is skipped **loudly** (named in a warning,
listed at the end) with its fact sheet still stored — only its vector-ranking contribution is
lost, since BM25 and title-restricted fetch still work for it.

**Bug 8 — the shortlist-membership guard dropped every valid candidate.** Found by running the
flagship query live: the reasoning pass returned five perfectly good, correctly-cited candidates,
and the user saw "I couldn't find a dataset that plausibly matches this." Cause: the model echoes
the fact sheet's own header format and returned `"<title> (DOI: ...)"`, which never matched the
bare title key. The guard was doing exactly its job — the key was wrong. `_match_shortlisted_record()`
now resolves by DOI first, then exact title with a trailing `(DOI: ...)` stripped, then a *unique*
containment match, each requiring an unambiguous hit so a model can still never introduce a dataset
it wasn't shown.

**Bug 9 — a truncated response was reported as "nothing matches."** Long citations over a 25-sheet
shortlist exhausted `max_tokens` mid-JSON. Two things were wrong: the parse failure was rendered
with the same message as a genuine zero-candidate result (asserting a negative finding that was
never established — precisely the overclaim class this tool exists to remove), and the complete,
cited candidates already in the truncated array were thrown away. Now `_parse_reasoning_response`
returns `None` for a genuine parse failure (reported as an internal formatting failure, explicitly
*not* as "nothing matches"), `_salvage_truncated_candidates()` recovers the complete objects from a
cut-off array — still subject to the same citation and shortlist checks, so parsing loosened but
grounding did not — and a salvaged list always carries "this list was cut off before it finished."
`max_tokens` raised to 3000 and the prompt now asks for short citations.

**Bug 10 — the embedding model also has a per-ITEM limit, and 20 datasets silently lost their
vector.** After fixing Bug 7's batching, the first full build still left 20 of 184 datasets
without a `factSheetEmbedding` (stored, BM25-rankable, but not vector-rankable). All were the
large sheets. Cause: E5-Mistral-7B-Instruct caps at 4096 tokens and
`check_embedding_ctx_length=False` means LangChain never truncates — so an over-long item just
500s. A synthetic length test had passed misleadingly because repetitive filler text tokenizes ~2x
less densely than real fact-sheet text full of numeric voxel strings and identifiers. Fixed with
`FACT_SHEET_EMBED_MAX_CHARS = 8_000` applied to the embedded copy ONLY — the stored `factSheetText`
stays complete, so BM25 and the reasoning pass still see the whole sheet and only the ranking
vector is built from the leading section. Measured: the 17 sheets that failed at full length spanned
10.5k–20.9k characters and all 17 embedded at 8k. `--retry-missing` then recovered all of them in
9.4s. **Final state: 184/184 on `factSheet`, `factSheetText`, and `factSheetEmbedding`.**

### Three follow-up fixes from the user's live chat-UI session

**Bug 11 — a newline in a citation broke the bullet.** Reported as "the rationale of the last
dataset ... is another paragraph". A citation is copied out of the fact sheet and often carries
real newlines; markdown ends a list item at the first unindented line, so everything after the
newline rendered as a loose paragraph that reads as the previous dataset's rationale having
escaped its bullet. `_one_line()` now collapses whitespace in both `reason` and `citation`. The
trailing `caveat` is also labelled and italicised (`*Note: ...*`) so it can't be misread as the
last bullet's continuation.

**Bug 12 — citations were unreadable raw fact-sheet dumps.** `_tidy_citation()` drops the fact
sheet's own section header ("Digital datasets (images/scans) (2): "), compacts the stored voxel
phrasing (`X, Y, Z units (in micrometers): 4.54, 4.54, 4.54` → `4.54 x 4.54 x 4.54 micrometers`,
with an optional-unit variant for sheets that don't record one), strips stray wrapping quotes,
and cuts over-long citations with an explicit marker. **Presentation only** — values are never
rounded, reordered, or dropped, because they are the grounding.

Also tightened `corpus_reasoning.yaml`: the model had started including datasets in order to
explain why they *don't* match (e.g. "Residual CO2 trapping... all have 'segmented: no'"), which
displays as a match and is simply wrong. The prompt now states the list is matches only and to
re-read its own `reason` before returning an entry. Verified gone live.

**Bug 13 — the cumulative filter chain was overwritten with the bare follow-up message.** In a
3-turn chain ("find sandstone" → "porosity above 0.3" → "how about below 0.25?"), the agent
correctly called the tool with `question="sandstone datasets with porosity below 0.25"`, but
`_track_dataset_listing` stored `effective_user_input` ("How about any below 0.25?") as
`_cumulative_filter_text` — so the accumulated "sandstone" constraint was silently dropped and
the *next* refinement would compose from a chain that had forgotten two turns. `_tool_filter_text()`
now prefers the tool call's own `question`/`query` argument over the raw message, applied at every
tracking call site (`_tool_args_by_call_id()` maps a `ToolMessage` back to its args on the
end-of-stream path). Falls back to the user message when a call carries no text argument, so the
deterministic dispatch paths are unchanged.

**Bug 13b — the reported bug itself: an elliptical follow-up was not restricted to the prior
result set.** Bug 13 above fixed which text got *stored* in the chain; it did not make turn 3
("how about any below 0.25?") stay inside the sandstone datasets found two turns earlier — that
turn still searched the whole catalog. Root cause: `_REFINEMENT_RE` matches phrasings that name
the prior set ("of these", "which ones" — turn 2 matched and was restricted), but turn 3 names it
nowhere, so it bypassed the deterministic dispatch and reached the agent with
`restrict_to_titles=None`.

Naively adding "how about" to `_REFINEMENT_RE` would have made it worse: that path ANDs the new
text onto the whole prior chain, and here the new constraint SUPERSEDES an earlier one on the same
property — "porosity above 0.3" AND "any below 0.25" composes to a contradiction returning nothing.

Fixed by splitting the two concerns: the **agent keeps ownership of the question** (live-observed
composing "sandstone datasets with porosity below 0.25" — it supersedes correctly, which blind
AND-composition cannot) and `_with_result_set_restriction()` injects `restrict_to_titles` into its
call.

**A first attempt at the detection was wrong and is worth recording as a lesson.** It added an
`_ELLIPTICAL_REFINEMENT_RE` tuned to the literal phrasings in the first transcript ("any below
0.25"). A second live transcript immediately defeated it: "Are there any with porosity > 0.3?"
misses because `any with` isn't at the start of the string, and "How about with porosity > 0.2?"
misses because `>` is a symbol rather than a comparison word — so BOTH follow-up turns ran
unrestricted again. That is the growing-pattern-library problem this codebase keeps rediscovering,
reintroduced in a new place.

The working signal does not look at the user's phrasing at all. `_continues_filter_chain()`
compares the AGENT's freshly composed question against the accumulated filter chain: if every
subject term of the chain survives into the new question, it is a refinement. Numbers, comparison
words, and generic "dataset(s)" are stripped, so "above 0.3" -> "> 0.2" is recognised as the same
chain while "carbonate datasets" is not. This is available on every turn regardless of how the user
words the follow-up, and it fails safe — a dropped subject means no restriction and today's
catalog-wide behavior, never a wrongly narrowed answer. The elliptical regex is kept only as a
secondary trigger for the case where the agent itself drops the subject.

Live-verified on BOTH reported transcripts (all four follow-up turns now restricted; three
topic-change controls stay catalog-wide). Transcript 2 turn 3 returns the 2 of the 10 previously
listed datasets with porosity > 0.2, instead of 8 the user had never been shown.

**Known residual limit:** `restrict_to_titles` post-filters the freshly generated Cypher's rows,
and that query is capped at `top_k=10`. A previously-listed dataset that satisfies the new
constraint but falls outside the fresh query's top 10 is dropped silently. This is inherent to the
existing `restrict_to_titles` design (Bug 1's fix), not new — but it compounds with the `top_k`
truncation noted below and is the most likely remaining source of "it forgot something".

**Bug 14 (found while reading that same log) — `s IS NULL OR` silently defeated required
filters.** Turn 1 generated `WHERE s IS NULL OR toLower(s.porousMediaType) = 'sandstone'`, which
is true for every dataset with no Sample node at all. The handoff had already noted this drift
under Bug 1 but only worked around it for refinements (`restrict_to_titles`); the root cause
polluted even a first-turn query. Fixed in `CYPHER_GENERATION_TEMPLATE` with an explicit
WRONG/RIGHT rule, and by rewriting the porosity example there to use a plain `MATCH` — modelling
the `OPTIONAL MATCH` shape for a *required* filter is what invited the `IS NULL OR`. Live-verified:
both queries from the user's transcript now generate
`MATCH (d:Dataset)<-[:PART_OF]-(s:Sample) WHERE toLower(s.porousMediaType) = 'sandstone'`.

*Not a bug:* "Companion Data for Digital Porous Media Tutorials" and "Network Generation
Comparison Forum" appearing under "sandstone datasets" looked like false positives from the
`IS NULL OR` but are genuine — they really do have sandstone `Sample` nodes ("Sandstones",
"Castlegate sandstone"). Worth knowing: **70 datasets have a sandstone sample but the chain
returns `top_k=10`**, with nothing in the output saying the list is truncated. That silent cap is
pre-existing and untouched — a candidate next fix, since a user refining against "these 10" is
unknowingly refining against a seventh of the real matches.

### Calibrated against the real corpus, not guessed

The plan sketched a top-K of ~40–60 fact sheets. Measured over all 184 live datasets, a rendered
fact sheet is a median of ~4.5k characters (p90 ~11k, max ~21k), so 40 sheets would be ~180k
characters ≈ 45k tokens — a third of the model's context on every relational question, and in
tension with the plan's own "lost in the middle" concern. Settled on **K=25** with a 120k-char
budget (~28k tokens), which typically fits the whole shortlist rather than silently cutting it to
a third. Constants and the reasoning are documented at the definition site in `tools.py`.

Also found and capped: an uncapped pipeline-chain list took one live dataset's fact sheet to
**129k characters** on its own. Every level of the fact sheet is now capped/truncated with an
explicit, never-silent count.

### Tests

- `tests/assistant/test_fact_sheet_builder.py` (new, 16 tests) — edge preservation, allowlist,
  truncation/caps, rendering.
- `tests/assistant/test_tools.py` — the gate (both directions, parametrized), the three worked
  cases from the plan's verification section (resolution, scanner-named-in-description,
  methodology-implies-pairing), both grounding guards, context budget, map-reduce, restrict-to-
  titles, and gate wiring for both `get_dataset_details` and `search_datasets`.
- `tests/assistant/test_graph_store.py` — `_rrf_merge`, `rank_fact_sheets` (incl. degradation when
  one index is missing), `fetch_fact_sheets`, plus rewritten `get_dataset_profile` tests for the
  decomposed query shape and a regression test pinning the `INPUT_FOR` direction.
- `test_prompts.py`, `test_conversation_manager.py` — prompt contract and tool registration.

**Full suite: 310 passing, 51 deselected (live-marked), no regressions.** The full-directory hang
noted in the previous handoff did **not** reproduce — `pytest tests/` completes in ~25s.

---

## Next Steps

0. ~~Build the fact sheets against the live graph~~ — **done.** All **184/184** datasets have
   `factSheet`, `factSheetText`, **and** `factSheetEmbedding`, and both indexes
   (`factSheetEmbedding`, `datasetFactSheetFulltext`) are live. Stragglers from transient endpoint
   errors can be recovered without a full 30-minute rebuild:
   ```bash
   python scripts/build_dataset_vector_index.py --only fact-sheets --retry-missing
   ```
   Fully reversible if ever needed:
   ```cypher
   MATCH (d:Dataset) REMOVE d.factSheet, d.factSheetText, d.factSheetEmbedding;
   DROP INDEX factSheetEmbedding IF EXISTS;
   DROP INDEX datasetFactSheetFulltext IF EXISTS;
   ```
0b. ~~Run the example queries end-to-end~~ — **done, live, against the real graph and LLM.** All
   verified working after Bugs 8 and 9 were fixed:
   - *"Are there paired tomographic and segmented images?"* → 5 real datasets, each citing its own
     recorded evidence (e.g. Bentheimer Sandstone for Analyzing Wetting Phenomena: `Original
     tomographic image — segmented: no` / `Segmented image — segmented: yes`, both at 4.95 µm).
     This is the query that motivated the whole feature.
   - *"Which datasets image the same sample at different resolutions?"* → correctly surfaces the
     SEM shale super-resolution dataset (quoting its 1X–16X magnifications / 9.1–145.6 nm voxel
     lengths), the multiscale 3D-printing dataset, and the 4D acid-leaching before/after scans.
   - *"Which datasets were imaged on an Xradia or Versa scanner?"* → 5 datasets, each quoting the
     scanner sentence from a Sample or DigitalDataset description. There is no queryable
     imaging-modality field at all, so this is otherwise unanswerable.
   - **Control:** *"sandstone datasets with porosity above 0.3"* correctly does NOT trip the gate —
     it still generates real Cypher (with the porosity-scale CASE) and answers honestly.
   Still worth doing: the same queries through the **chat UI** (`streamlit run rocco_ui.py`) rather
   than by calling the tool directly, to confirm agent routing picks the tool up as intended.
0c. **Re-verify `get_dataset_profile` in the chat UI** now that the `INPUT_FOR` direction is fixed
   — the organizational-structure section should be populated for the first time. Verified directly
   against the live graph already (DRP-126 renders `P4-3-3-2-3-1 -> Large-Area SEM` /
   `-> Automated Mineralogy`; DRP-372 returns 207 + 618 edges in 0.8s, was 28s+ with zero edges),
   but not yet through the UI.
1. ~~Confirm the context-overflow fix actually resolves the reported error live~~ — **done**,
   plus the three follow-up bugs above found from further live use. Bugs 1–3 above are
   live-verified by the user in the chat UI; Bug 4 (schema fixes) is live-verified by me directly
   against `GraphStore.cypher_qa()` but **not yet re-confirmed by the user in the chat UI** —
   worth a quick check next session (retry "...of the same limestone" and "paired tomographic
   and segmented images" in `streamlit run rocco_ui.py`).
2. **Verify `RelatedSoftware`/`RelatedDataset`'s relationship type empirically** — the full
   profile Cypher assumes `PART_OF` for these (matching the general schema docstring), but
   `MANUAL_SCHEMA` (the block actually fed to `GraphCypherQAChain`) never confirms it. Run
   `MATCH (n:RelatedSoftware)-[r]->() RETURN type(r), count(*)` (and the equivalent for
   `RelatedDataset`) against the live graph; `OPTIONAL MATCH` degrades safely if wrong, but the
   docstring/comments calling this out as unverified should be resolved either way.
3. **Sweep for other unbounded/embedding-carrying fields elsewhere in the codebase** — this
   session only fixed `get_dataset_profile`'s new Cypher query. `GraphStore.get_dataset()`
   (`RETURN d`, still dead/unregistered code) and any other method doing `RETURN d`/`RETURN
   node` wholesale on an embedded node label should be checked for the same
   `datasetEmbedding`/`componentEmbedding` leak risk before they're ever wired up or reused.
4. **Run the live-tier acceptance tests** with real credentials —
   `pytest tests/assistant/test_search_integration.py -k "test_p1 or test_p2" -v` (needs
   `LLM_API_KEY`/`SAMBANOVA_API_KEY` and a live Neo4j) to exercise the real two-turn "search →
   tell me more" and comparison flows end-to-end, not just the mocked smoke test.
5. **Manual smoke test** beyond the bugs already found: a specific-field follow-up, a
   file-reading/"where can I download this" question, and a suitability question — confirm
   `[dataset profile]` labels appear, DOIs match the graph, the Corral archive URL resolves for a
   real dataset, and no fabricated properties/paths appear. (Search → "tell me more", the
   3-turn refinement chain, and 2-dataset comparison are now covered by this follow-up session's
   live testing.)
6. **Add a dedicated unit test for `restrict_to_titles`** (Bug 1's fix) — the live testing
   confirmed it works end-to-end, but the post-filtering logic in `GraphStore.cypher_qa()` (empty
   intersection, no `restrict_to_titles` passed, partial-title-match cases) is pure Python and
   should have direct mocked-Cypher-row coverage in `tests/assistant/test_graph_store.py` rather
   than relying solely on live verification going forward.
7. **`docs/user_guide/dataset_profiles.rst`/`docs/neo4j_schema.md`** — consider documenting the
   colloquial-rock-name → `porousMediaType` enum mapping added to `MANUAL_SCHEMA` this session
   (limestone/dolomite/chalk → carbonate, etc.) somewhere user- or intern-facing, not just inside
   the Cypher-generation prompt string — it's a real, non-obvious gotcha for anyone writing
   Cypher against this schema by hand too.
8. **Delete or commit the scratch smoke-test scripts** left in the repo root from this session's
   live verification (`smoke_test_cypher.py`, `smoke_test_conversation.py`) — untracked,
   uncommitted, not meant to be permanent. Useful to keep as ad hoc debugging tools, but decide
   deliberately rather than letting them ride into a commit accidentally; consider moving into
   `tests/assistant/` as a real (mocked or live-tier-marked) test if they're worth keeping.
9. **Commit.** Nothing from this feature or any of the bug fixes has been committed yet —
   everything described above is uncommitted on `feature/dataset-details`. Decide whether the
   context-overflow fix and this follow-up session's four bugs should be separate commits from
   the feature itself (all found/fixed live, arguably separate logical changes) or folded in.
10. Once confirmed working, open the PR and update `Tasks.md`'s per-week tracking / project board
    per the repo's usual process (not done this session).

## Key Files to Read First in the Next Session

- `docs/user_guide/content_reasoning.rst` — the user-facing page for the new
  `reason_about_dataset_content` tool: why a literal-field answer to a relational question is a
  wrong answer rather than a partial one, the routing dividing line, the query-time sequence, and
  the fact-sheet schema. (`/home/bchan/.claude/plans/please-review-the-handoff-fuzzy-eich.md` is
  the original design it was implemented from, kept for history — where the two diverge, the code
  and this doc are correct; see "Calibrated against the real corpus" above.)
- `src/assistant/tools.py` — `_needs_content_reasoning`/`_RELATIONAL_PATTERNS` (the gate),
  `_reason_about_dataset_content`, `_render_reasoning_answer` (the two grounding guards),
  `_FACT_SHEET_SHORTLIST_K`/`_FACT_SHEET_CONTEXT_CHAR_BUDGET` (and why they're set where they are)
- `scripts/build_dataset_vector_index.py` — `_build_fact_sheet`/`_render_fact_sheet_text`,
  `IndexBuilder._fetch_profile_rows` (bulk, cross-product-free)
- `src/assistant/tools.py` — `get_dataset_profile`, `_build_profile_context`,
  `_render_node_list`, `_is_embedding_like`, `_MAX_NODES_PER_TYPE`, `_corral_archive_url`
- `src/assistant/graph_store.py` — the module docstring's `INPUT_FOR` direction warning (Bug 5);
  `GraphStore.get_dataset_profile()` and its now-decomposed per-type queries (note the
  `{.*, ...Embedding: null}` map projections and the `RelatedSoftware`/`RelatedDataset`
  verification TODO in its docstring); `_rrf_merge`/`rank_fact_sheets`/`fetch_fact_sheets`;
  `MANUAL_SCHEMA`'s colloquial-rock-name mapping, `fileTypes`-is-a-list notes (Bug 4), and
  `INPUT_FOR` direction block (Bug 5); `cypher_qa()`'s `restrict_to_titles` param (Bug 1)
- `src/prompts/dataset_profile.yaml` — the synthesis prompt's knowledge-policy/formatting rules
- `src/assistant/conversation_manager.py` — `_SELF_CONTAINED_TOOLS`, `_TOOL_PARAM_KEYS`,
  `_FOLLOWUP_TOOL_GATE_SYSTEM_PROMPT`, and the `SYSTEM_PROMPT` routing bullets; from the
  follow-up session: `_extract_profiled_dataset`/`_last_profiled_dataset`/
  `_DATASET_ANAPHORA_RE` (Bug 2), `_strip_reasoning_scaffold`/`_COMPARISON_SYNTHESIS_SYSTEM_PROMPT`
  (Bug 3), the refinement-dispatch block passing `restrict_to_titles` (Bug 1)
- `docs/user_guide/dataset_profiles.rst` — the user-facing capability page (also a good
  refresher on the feature's intended behavior/scope)
- `Tasks.md` — "Future Feature: Dataset Detail Follow-Up Queries — IMPLEMENTED" section
- `tests/assistant/test_tools.py::TestBuildProfileContext`,
  `tests/assistant/test_graph_store.py::TestGetDatasetProfile` — regression tests for the
  context-overflow bug
- `tests/assistant/test_conversation_manager.py::TestStripReasoningScaffold` — regression tests
  for Bug 3, useful as executable documentation of the "when is Step-N text legitimate vs. a
  leak" distinction
- `smoke_test_cypher.py` / `smoke_test_conversation.py` (repo root, uncommitted) — the live
  verification scripts used to confirm Bugs 1/2/3/4 against the real Neo4j instance; see Next
  Steps item 8 for what to do with them
