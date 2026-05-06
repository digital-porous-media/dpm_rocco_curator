# Extending Rocco (Description Curator)

Related: [[00_Project_Overview]]

## Current State

Rocco evaluates a dataset description against a 10-criterion rubric and enhances it using RAG from uploaded PDFs. It works but is narrow: users must already have a draft description and know to use the tool.

## Proposed Extensions

### 1. Starter Description Generator

Many researchers have no description at all. Add a "Generate Starter" flow:
- User fills a short form: rock type, imaging method, voxel size, institution, associated paper (optional)
- Rocco generates a rubric-compliant draft description as a starting point
- User then edits and runs the normal evaluate → enhance loop

**Why this matters:** Lowers the barrier to entry; the current tool assumes you already have something to improve.

### 2. Multi-Round Refinement Indicator

Currently the conversation history is preserved but the UI doesn't make iteration visible. Add:
- A score history chart showing rubric score across rounds
- "What improved" diff view between original and current description

### 3. Rubric Explainer

Before evaluating, users often don't know what a "good" description looks like for each criterion. Add a collapsible rubric guide panel that explains each criterion with a short example of a passing vs. failing statement.

### 4. Batch Mode (stretch goal)

Allow uploading a CSV of descriptions (e.g., for portal admins who want to audit multiple datasets at once). Return a scored summary table.

## Files to Modify

| File | Change |
|------|--------|
| `rocco_ui.py` | Add starter form, score history, rubric explainer panel |
| `src/editor/editor.py` | Add `generate_starter(form_data)` method |
| `src/prompts/editor.yaml` | Add a `starter_generation` prompt variant |
| `src/evaluator/rubric.json` | Add `example_pass` / `example_fail` fields per criterion |

## Tasks

See [[04_Tasks#Rocco Extensions]].
