# Contributing to Rocco

Thank you for contributing to Rocco! This guide covers both the **Description Curator** (current) and the **General Assistant** (in development). Please read the relevant sections for your task.

## Code of Conduct

We are committed to a welcoming and respectful environment. Please treat all contributors with respect.

---

## Getting Started

### Prerequisites

- Python 3.9+
- `conda` (Miniforge or Anaconda) — primary environment manager
- `git`
- An LLM API key (see `.env.example`)
- Neo4j ≥ 5.x — required only for assistant search features; see [Neo4j Setup](#neo4j-setup-assistant-only) below

### Setting Up Your Development Environment

1. **Clone the repository:**
   ```bash
   git clone git@github.com:digital-porous-media/dpm_rocco_ai.git
   cd dpm_rocco_ai
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml    # if environment.yml exists
   # — or —
   conda create -n rocco_ai python=3.11
   conda activate rocco_ai
   ```

3. **Install the package in editable mode:**
   ```bash
   conda activate rocco_ai

   # Curator only (evaluator, editor, screener)
   pip install -e ".[dev]"

   # Curator + General Assistant (adds Neo4j, LangChain, LangGraph)
   pip install -e ".[dev,graph]"
   ```

4. **Copy and configure `.env`:**
   ```bash
   cp .env.example .env
   # Edit .env — set LLM_API_KEY, LLM_MODEL, and optionally Neo4j vars
   ```

5. **Verify the setup:**
   ```bash
   python -c "import src; print('Curator: OK')"
   python -c "import os; os.environ['USE_NEO4J']='false'; from src.assistant.graph_store import GraphStore; GraphStore(); print('Assistant: OK')"
   ```

### Neo4j Setup (Assistant Only)

The assistant's dataset search requires Neo4j. Skip this if you are only working on the curator.

1. **Install and start Neo4j** (macOS via Homebrew, or download from neo4j.com):
   ```bash
   neo4j start
   # Open http://localhost:7474 — log in with neo4j/neo4j, set a new password
   ```

2. **Add credentials to `.env`:**
   ```
   USE_NEO4J=true
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=<your-password>
   ```

3. **Verify the connection:**
   ```bash
   python -c "
   from neo4j import GraphDatabase
   import os; from dotenv import load_dotenv; load_dotenv()
   d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')))
   d.verify_connectivity(); print('Connected!'); d.close()
   "
   ```

4. **Load the dataset graph and build vector indexes** (first time only):
   ```bash
   python scripts/load_graph.py --mode rebuild
   python scripts/build_dataset_vector_index.py
   ```

---

## Branch Naming Convention

Branch names follow the pattern `<type>/<short-kebab-description>`:

| Type | When to use | Example |
|------|-------------|---------|
| `feature/` | New functionality | `feature/publication-corpus` |
| `fix/` | Bug fix | `fix/graph-store-null-filter` |
| `docs/` | Documentation only | `docs/update-contributing` |
| `refactor/` | Code restructuring, no behavior change | `refactor/llm-client-pydantic` |
| `test/` | Adding or fixing tests | `test/assistant-conftest` |

**Rules:**
- Use kebab-case (hyphens, no underscores or spaces)
- Keep it short — 3–5 words after the prefix
- Reference the issue number in the branch if it helps: `feature/17-literature-search`

**Protected branches:**

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases — tag `v1.x` (curator), `v2.0.0` (unified assistant) |
| `feature/general-assistant` | Active assistant development — intern topic branches merge here |

**Do not push directly to `main` or `feature/general-assistant`.** All changes go through a PR.

**Intern workflow:**
```bash
# Always branch from feature/general-assistant for assistant work
git checkout feature/general-assistant
git pull origin feature/general-assistant
git checkout -b feature/<your-task>   # e.g. feature/publication-corpus
```

---

## Making Changes

### Code Style

**Formatter:** `black` at line length 100. **Import sorter:** `isort`. Run both before every commit:
```bash
black . --line-length 100
isort .
```

**Rules enforced by the formatter (do not override):**
- Double quotes for strings
- Trailing commas in multi-line collections
- Imports sorted: stdlib → third-party → local

**Rules not enforced by the formatter:**
- No wildcard imports (`from module import *`)
- Type hints on all public function signatures
- No f-strings in log messages (use `%s` lazy formatting: `logger.debug("loaded %s items", n)`)
- Return type annotations required on functions that return non-trivial types

**Comments:** Write a comment only when the **why** is non-obvious — a hidden constraint, a workaround for a specific bug, a non-obvious invariant. One short line max. No multi-line comment blocks. No docstrings that restate the function name.

### Assistant-Specific Constraints

- **`graph_store.py`** must accept `filters: dict` — do not hardcode field names.
- **`USE_NEO4J=false`** must keep the assistant functional (falls back to publication FAISS).
- **Session state** in `assistant_ui.py` must use `assistant_`-prefixed keys to avoid collisions with the curator's `curator_`-prefixed keys.
- **Never assert a dataset property that isn't in the graph.** If a field is missing, say so honestly — do not fabricate values.
- **No APOC.** All Cypher must run on vanilla Neo4j (local, TACC, or AuraDB). The Cypher generation prompt already enforces this.

---

## Testing

**Always run tests before committing.**

```bash
conda activate rocco_ai

# Full suite
pytest tests/ -v

# Curator only
pytest tests/test_curator_integration.py tests/test_vector_store.py tests/test_llm_client.py -v

# Assistant only
pytest tests/assistant/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Key test files

| File | What it guards |
|------|---------------|
| `tests/test_curator_integration.py` | Evaluator, editor, screener — catches `RoccoClient` interface breaks |
| `tests/test_llm_client.py` | Provider routing, configuration, backwards compat |
| `tests/test_vector_store.py` | Embedding batch handling, FAISS alignment |
| `tests/assistant/test_graph_store.py` | Neo4j vector search, structured filter queries |
| `tests/assistant/test_intent_classifier.py` | Intent classifier prompt against all 6 intents |

Tests in `tests/assistant/` use fixtures from `tests/assistant/conftest.py` — a mock Neo4j driver and a small in-memory FAISS index so Neo4j is not required to run the assistant test suite.

### Before refactoring a public interface

1. Confirm all tests pass: `pytest tests/ -v`
2. Refactor, then re-run tests
3. If you changed `RoccoClient`, `GraphStore`, or any `tools.py` function signature, the relevant test suite will catch it

---

## Submitting Changes

### Commit Message Format

```
<Type>: <short imperative description>  (≤ 72 chars total)

<Optional body — explain WHY, not WHAT. Wrap at 72 chars.>

Closes #<n>   ← or "Part of #<n>" if the issue isn't fully resolved
```

Valid types:

| Type | When to use |
|------|-------------|
| `Add` | New file, function, or feature |
| `Fix` | Bug fix |
| `Refactor` | Restructuring with no behavior change |
| `Update` | Change to existing functionality |
| `Remove` | Deleting code or files |
| `Docs` | Documentation only |
| `Test` | Adding or fixing tests |

**Examples:**
```
Add: literature_search.py with Semantic Scholar fallback routing

Closes #17
```
```
Fix: skip apoc.meta.data() call when refresh_schema=False

Neo4j Community Edition does not ship APOC. GraphStore was calling
apoc.meta.data() on init, crashing when APOC is absent.
```
```
Refactor: RoccoClient inherits BaseChatModel directly

Removes the separate RoccoChatModel adapter; curator and assistant
now share one LLM client class with zero duplication.
```

**Rules:**
- Subject line is imperative ("Add", not "Added" or "Adding")
- No period at the end of the subject line
- Body is optional; use it when the why is non-obvious
- Always include `Closes #<n>` or `Part of #<n>` when a commit addresses an issue

### PR Process

1. **Sync your branch** with the base branch before opening the PR:
   ```bash
   git fetch origin
   git rebase origin/feature/general-assistant   # or origin/main for curator fixes
   ```

2. **Run the full test suite** locally and confirm it passes:
   ```bash
   pytest tests/ -v
   ```

3. **Push your branch:**
   ```bash
   git push origin feature/<your-task>
   ```

4. **Open a PR** on GitHub targeting the correct base branch:
   - Assistant work → `feature/general-assistant`
   - Curator fixes → `main`

5. **PR description must include:**
   - What changed and why (not just "fixes bug")
   - Issue reference: `Closes #<n>` or `Part of #<n>`
   - How you tested it (command you ran, output you saw)

6. **Request a review** from the project maintainer (BC-Chang). Do not merge your own PR.

7. **Address review feedback** with new commits (do not force-push after review starts). Reply to comments with "Done" or explain why you pushed back.

8. **After approval**, the reviewer merges. You can delete your branch once merged.

---

## Adding a New LLM Provider

All LLM calls go through `RoccoClient` in `src/llm/client.py`. To add a provider:

1. Add its base URL to `PROVIDER_URLS` in `src/llm/client.py`
2. Update `.env.example` with the provider alias and example model names
3. Test with: `python -c "from src.llm.client import RoccoClient; c = RoccoClient(); print(c.call(system='ping', user='hello'))"`

---

## Reporting Bugs

Include:
- Python version (`python --version`) and OS
- Exact steps to reproduce
- Expected vs. actual behavior
- Full traceback

---

## Questions?

Open an issue on GitHub or ask in the project Slack channel.

---

Thank you for contributing to Rocco!
