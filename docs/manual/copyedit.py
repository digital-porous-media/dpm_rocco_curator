"""Google-style copy edits applied to the staged manual sources.

Nothing here touches the repository. ``scripts/build_manual.py`` calls
:func:`apply` against ``build/manual/src/`` after staging, so ``docs/`` and the
root Markdown files stay byte-for-byte unchanged and the documentation website
is unaffected.

Two kinds of edit:

* **Mechanical passes** — sentence-case headings and US spelling. These run over
  every staged document and are safe to generalize because they are constrained
  by an explicit protected-term list.
* **Targeted replacements** — banned words, first person, future tense,
  marketing voice, broken cross-references, structural fixes, and one redaction.
  Each is an exact string match against a named file. Every entry marked
  ``required`` must apply, so if upstream text changes the build fails loudly
  instead of silently skipping the edit.

Reference: https://developers.google.com/style
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Protected terms for the sentence-case pass
# ---------------------------------------------------------------------------

# Multi-word proper nouns, checked before single tokens.
PROTECTED_PHRASES = [
    "Digital Porous Media Portal",
    "Digital Porous Media",
    "Description Curator",
    "General Assistant",
    "Content Screener",
    "Semantic Scholar",
    "Windows Subsystem for Linux",
    "Google Gemini",
    "Neo4j Browser",
    "Roboto Mono",
]

# Single tokens that keep their capitalization anywhere in a heading.
PROTECTED_TOKENS = {
    # Products and services
    "Rocco", "Neo4j", "Cypher", "Streamlit", "Sphinx", "Python", "Jupyter",
    "JupyterHub", "Ollama", "OpenAI", "Anthropic", "Claude", "Gemini", "Google",
    "DeepSeek", "HuggingFace", "SambaNova", "TACC", "Zenodo", "LangChain",
    "LangGraph", "FAISS", "GitHub", "Git", "Docker", "Linux", "Windows",
    "macOS", "Corral", "Markdown", "PageIndex", "Pygments", "Furo", "Pydantic",
    "Scholar", "Portal", "Darcy", "Archie",
    # Component names the documentation treats as proper nouns
    "Evaluator", "Editor", "Writer", "Ingestor", "Retriever", "Curator",
    "Assistant",
    # Field terms that are conventionally capitalized
    "Boltzmann",
}

# A token is left alone if it is an acronym, contains a digit, or is code.
_ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9&/.+-]*$")
_HAS_DIGIT_RE = re.compile(r"\d")


def sentence_case(title: str) -> str:
    """Convert a heading to sentence case, preserving proper nouns and acronyms."""
    # Anything in inline code, or a heading that is itself an identifier, is left
    # exactly as written.
    if "``" in title or "`" in title or "::" in title or title.startswith(("Appendix", "Part ")):
        return title

    placeholders: list[str] = []

    def _stash(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    for phrase in PROTECTED_PHRASES:
        title = re.sub(re.escape(phrase), _stash, title)

    parts = re.split(r"(\W+)", title)
    out: list[str] = []
    seen_word = False
    for part in parts:
        if not part or not part[0].isalnum():
            out.append(part)
            continue
        if "\x00" in part:
            out.append(part)
            seen_word = True
            continue
        keep = (
            not seen_word                      # first word of the heading
            or part in PROTECTED_TOKENS
            or _ACRONYM_RE.match(part)         # LLM, RAG, API, UI, DOI, Q&A
            or _HAS_DIGIT_RE.search(part)      # WSL2, CO2, E5, v1.0.0
        )
        out.append(part if keep else part.lower())
        seen_word = True

    title = "".join(out)
    for i, original in enumerate(placeholders):
        title = title.replace(f"\x00{i}\x00", original)
    return title[:1].upper() + title[1:] if title else title


# ---------------------------------------------------------------------------
# Heading passes
# ---------------------------------------------------------------------------

_UNDERLINE_CHARS = set("=-~^\"'`#*+:")


def rst_headings(text: str) -> tuple[str, int]:
    """Sentence-case RST section titles.

    A title is recognized only when it is unindented and followed by a full-width
    underline, which keeps the pass out of literal blocks and simple tables.
    """
    lines = text.split("\n")
    changed = 0
    for i in range(len(lines) - 1):
        title, under = lines[i], lines[i + 1]
        if not title.strip() or title[:1].isspace():
            continue
        if len(under) < 3 or len(set(under)) != 1 or under[0] not in _UNDERLINE_CHARS:
            continue
        if len(under) < len(title.rstrip()) - 2:
            continue
        new = sentence_case(title.rstrip())
        if new != title.rstrip():
            lines[i] = new
            lines[i + 1] = under[0] * max(len(new), 3)
            changed += 1
        elif len(under) != len(title.rstrip()):
            lines[i + 1] = under[0] * len(title.rstrip())
    return "\n".join(lines), changed


def md_headings(text: str) -> tuple[str, int]:
    """Sentence-case Markdown ATX headings, skipping fenced code blocks."""
    lines = text.split("\n")
    changed = 0
    fence = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if not m:
            continue
        new = sentence_case(m.group(2))
        if new != m.group(2):
            lines[i] = f"{m.group(1)} {new}"
            changed += 1
    return "\n".join(lines), changed


# ---------------------------------------------------------------------------
# US spelling
# ---------------------------------------------------------------------------

# Deliberately narrow. "analyses" and "analysis" are correct English and must not
# be caught by a broad -ise/-yse rule.
SPELLINGS = {
    "serialised": "serialized",
    "normalised": "normalized",
    "Recognised": "Recognized",
    "recognises": "recognizes",
    "recognised": "recognized",
    "behaviour": "behavior",
    "colour": "color",
    "organised": "organized",
    "prioritise": "prioritize",
}


def spellings(text: str) -> tuple[str, int]:
    changed = 0
    for british, american in SPELLINGS.items():
        count = text.count(british)
        if count:
            text = text.replace(british, american)
            changed += count
    return text, changed


# ---------------------------------------------------------------------------
# Targeted replacements
# ---------------------------------------------------------------------------

# (path relative to the staged tree) -> [(old, new, required), ...]
#
# `required=True` means the build fails if the text is not found, so an upstream
# edit that moves this text is reported instead of silently dropping the fix.

REPLACEMENTS: dict[str, list[tuple[str, str, bool]]] = {
    # -- Banned words (developers.google.com/style/word-list) ----------------
    "user_guide/quickstart_curator.rst": [
        ('      Simply write feedback in the "Your Feedback" text area:',
         '      Write feedback in the "Your Feedback" text area:', True),
        # Marketing voice and exclamation points.
        ("Get up and running with the Description Curator!",
         "This chapter walks you through evaluating and improving your first dataset\ndescription.", True),
        ("**That's it!** You've just improved a dataset description using AI and research documents.",
         "You have now improved a dataset description using Rocco and your own research\ndocuments.", True),
        # Comma splice.
        ("Include as much detail as possible in your feedback, Rocco will summarize and organize it.",
         "Include as much detail as possible in your feedback. Rocco summarizes and organizes it.", True),
        # Future tense.
        ("Rocco will:", "Rocco does the following:", True),
        (".. important:: Rocco will **NOT**:", ".. important:: Rocco does **not**:", True),
        ("Rocco shows citations for **all** changes and updates. Citations will show:",
         "Rocco shows citations for **all** changes and updates. Each citation shows:", True),
        ("Your browser will automatically open", "Your browser opens", False),
        # Dropped page.
        ("- **Contributing**: :doc:`../developer_guide/contributing` — report issues, contribute improvements",
         "- **Contributing**: :doc:`/contributing` — report issues, contribute improvements", True),
    ],
    "user_guide/quickstart_assistant.rst": [
        ("Get up and running with the General Assistant!",
         "This chapter walks you through your first conversation with the General\nAssistant.", True),
        ("- **Contributing**: :doc:`../developer_guide/contributing` — report issues, contribute improvements",
         "- **Contributing**: :doc:`/contributing` — report issues, contribute improvements", True),
    ],
    "user_guide/evaluator.rst": [
        ("but is easily adaptable to other domains", "but is adaptable to other domains", True),
        (":doc:`../developer_guide/api_reference` — Full class documentation",
         ":doc:`/appendix/api_reference` — Full class documentation", True),
    ],
    "user_guide/rag.rst": [
        (":doc:`../developer_guide/api_reference` — Full class documentation",
         ":doc:`/appendix/api_reference` — Full class documentation", True),
    ],
    "user_guide/writer.rst": [
        (":doc:`../developer_guide/api_reference` — Full class documentation",
         ":doc:`/appendix/api_reference` — Full class documentation", True),
    ],
    "user_guide/content_reasoning.rst": [
        ("this is not simply reusing the embedding text",
         "this is not a reuse of the embedding text", True),
    ],
    "user_guide/configuration.rst": [
        ("To switch providers at runtime, just edit ``.env`` and restart the Streamlit app.",
         "To switch providers at runtime, edit ``.env`` and restart the Streamlit app.", True),
        # First person.
        ("   We **recommend installing Ollama on WSL2 (Windows Subsystem for Linux)** if you're on Windows, "
         "rather than the Windows Desktop app, as it provides better integration with development tools.",
         "   On Windows, **install Ollama on WSL2 (Windows Subsystem for Linux)** rather than the Windows "
         "desktop app. WSL2 integrates better with development tools.", True),
        ("- Need help? Check :doc:`../developer_guide/contributing`",
         "- Need help? Check :doc:`/contributing`", True),
        # "Next Steps" is underlined with '=', the same level as the document
        # title, so it renders as a second chapter instead of a closing section.
        ("Next Steps\n==========", "Next steps\n----------", True),
    ],
    "developer_guide/architecture.rst": [
        (":doc:`../developer_guide/contributing` — Development guidelines",
         ":doc:`/contributing` — Development guidelines", True),
        ("- And more...", "- Additional helpers described in Appendix A", True),
    ],
    "developer_guide/prompts.rst": [
        (":doc:`api_reference` — Auto-generated class documentation",
         ":doc:`/appendix/api_reference` — Generated class documentation", True),
        # Two sections are underlined at document-title level, so they render as
        # separate chapters rather than sections of this one.
        ("General Assistant Prompts\n=========================",
         "General Assistant prompts\n-------------------------", True),
        ("Prompts That Live in Code, Not YAML\n==================================",
         "Prompts that live in code, not YAML\n-----------------------------------", True),
    ],

    # -- Root documents ------------------------------------------------------
    "overview.md": [
        # First person.
        ("We validated Rocco's evaluation accuracy by comparing",
         "Rocco's evaluation accuracy was validated by comparing", True),
        ("**Statistical Analysis**: We employed a cumulative link mixed model (CLMM)",
         "**Statistical analysis**: A cumulative link mixed model (CLMM) was used", True),
        ("We welcome bug reports, feature requests, and pull requests! See [CONTRIBUTING.md](CONTRIBUTING.md)",
         "Bug reports, feature requests, and pull requests are welcome. See the contributing chapter in Part V.", True),
        ("If you use Rocco in your research, please cite it:",
         "If you use Rocco in your research, cite it as follows:", True),
        # Marketing voice.
        ("**Multi-LLM Support**: Use any OpenAI-compatible LLM provider!",
         "**Multi-LLM support**: Use any OpenAI-compatible LLM provider.", True),
        ("**Ready to improve your dataset descriptions?** Start by following the "
         "[Quick Start](#quick-start) guide above!",
         "To get started, follow the installation and quickstart chapters in Part I.", True),
        # Future tense.
        ("The app will open at `http://localhost:8501`",
         "The app opens at `http://localhost:8501`", True),
        # Links to files that are not chapters of this manual.
        ("- **[CLAUDE.md](CLAUDE.md)** — Developer guide (components, patterns, implementation details)\n",
         "", True),
        ("- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines and code standards",
         "- Contribution guidelines and code standards — see Part V", True),
        ("- **[.env.example](.env.example)** — Detailed LLM provider configuration reference",
         "- Detailed LLM provider configuration reference — see Part IV", True),
        ("See [`.env.example`](.env.example) for all supported providers and their base URLs.",
         "See Part IV for all supported providers and their base URLs.", True),
        ("see the [LICENSE](LICENSE) file for details",
         "see Appendix E for details", True),
        # Shields.io badges fetch over the network and mean nothing on paper.
        ("[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]"
         "(https://opensource.org/licenses/MIT)\n", "", False),
    ],
    "contributing.md": [
        ("see [Neo4j Setup](#neo4j-setup-assistant-only) below",
         "see the Neo4j setup section below", True),
    ],
    "changelog.md": [
        ("Planned features are tracked in `CLAUDE.md` under \"Vision & Roadmap\":",
         "Planned features:", True),
        ("For detailed information about components, architecture, and development setup, "
         "see [CLAUDE.md](CLAUDE.md).",
         "For components, architecture, and development setup, see Part V.", True),
    ],
    "deployment.md": [
        ("(see [Secrets & the environment file](#secrets--the-environment-file)",
         "(see the secrets section below", True),
        ("Environment variables are documented in [`.env.example`](.env.example)",
         "Environment variables are documented in `.env.example`", True),
        # Redaction. The repository keeps this TODO because it is a real action
        # item; it does not belong in a document circulated outside the team,
        # because it names both a live weakness and the file that holds it.
        ("**TODO:** `/etc/dpm_rocco/app.env` currently holds one collaborator's personal LLM API key "
         "rather than a dedicated credential. This should be replaced with a **service account.**",
         "The service reads its credentials from an environment file provisioned on the host, "
         "outside version control. Use a dedicated service account rather than an individual's "
         "personal API key.", True),
    ],
    "neo4j_schema.md": [
        ("# Neo4j Graph Schema — DRP Portal",
         "# Appendix B. Neo4j graph schema", True),
    ],
}


def apply(stage: Path) -> dict:
    """Run every copy edit over the staged tree. Returns a report."""
    report = {"headings": 0, "spellings": 0, "replacements": 0, "missing": []}

    for rel, edits in REPLACEMENTS.items():
        path = stage / rel
        if not path.exists():
            report["missing"].append(f"{rel} (file not staged)")
            continue
        text = path.read_text(encoding="utf-8")
        for old, new, required in edits:
            if old in text:
                text = text.replace(old, new)
                report["replacements"] += 1
            elif required:
                report["missing"].append(f"{rel}: {old.splitlines()[0][:72]}")
        path.write_text(text, encoding="utf-8")

    skip = {"index.rst", "conf.py"}
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name in skip:
            continue
        rel = path.relative_to(stage)
        # The manual's own front matter and appendices are already in Google style.
        if rel.parts and rel.parts[0] in {"front", "parts", "appendix", "_prompts", "_static"}:
            continue
        if path.suffix == ".rst":
            transform = rst_headings
        elif path.suffix == ".md":
            transform = md_headings
        else:
            continue
        text = path.read_text(encoding="utf-8")
        text, n_head = transform(text)
        text, n_spell = spellings(text)
        if n_head or n_spell:
            path.write_text(text, encoding="utf-8")
        report["headings"] += n_head
        report["spellings"] += n_spell

    return report


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Screenshots captured by scripts/capture_screenshots.py. They are placed only in
# the staged copy: the documentation website keeps its own two figures, so the
# live site is unaffected.
#
# (path relative to the staged tree) -> [(anchor text, figure block, mode), ...]
# mode is "replace", "after", or "before".

_S = "../_static/screenshots/manual"


def _fig(name: str, alt: str, caption: str, width: str = "100%") -> str:
    return (
        f".. figure:: {_S}/{name}\n"
        f"   :alt: {alt}\n"
        f"   :align: center\n"
        f"   :width: {width}\n\n"
        f"   {caption}\n"
    )


FIGURES: dict[str, list[tuple[str, str, str]]] = {
    "user_guide/streamlit_app.rst": [
        (".. figure:: ../_static/screenshots/main_ui.png\n"
         "   :alt: Rocco main interface\n"
         "   :align: center\n",
         _fig("02_curator_empty.png", "The Description Curator page",
              "The Description Curator. Paste a description into the text area, then "
              "select **Evaluate Description**."),
         "replace"),
        (".. figure:: ../_static/screenshots/enhancement_result.png\n"
         "   :alt: Rocco enhancement result with citations\n"
         "   :align: center\n",
         _fig("05_enhanced_citations.png", "An enhanced description with citations",
              "An enhanced description. The original and the rewrite appear side by side, "
              "with a rationale and a citation for every added statement."),
         "replace"),
        ("**2. Evaluate**",
         _fig("03_evaluation_results.png", "Evaluation results and rubric breakdown",
              "The evaluation panel: a total score out of 10, and a per-criterion "
              "breakdown you can expand for the reasoning behind each score.") + "\n",
         "before"),
        ("**3a. Upload context documents**",
         _fig("04_upload.png", "The document uploader", "Upload PDF or DOCX papers to "
              "give Rocco evidence to draw on.", "70%") + "\n",
         "before"),
        ("For each turn, you can:",
         _fig("07_manage_context.png", "The Manage Context expander",
              "The **Manage Context (Prior Turns)** expander. Each prior round is a card "
              "you can exclude, edit, or inspect.") + "\n",
         "before"),
    ],
    "user_guide/quickstart_curator.rst": [
        ("- **↻ Iterate** — try multiple rounds of feedback to perfect your description",
         "- **↻ Iterate** — try multiple rounds of feedback to perfect your description\n\n"
         + _fig("06_accept_reject.png", "Accept, reject, and edit controls",
                "Adopt the rewrite, keep the original, or edit the result before adopting it."),
         "replace"),
    ],
    "user_guide/quickstart_assistant.rst": [
        ("Step 2: (optional) configure Neo4j and Semantic Scholar",
         _fig("01_navigation.png", "The page selector",
              "Use the navigation buttons in the sidebar to switch between the General "
              "Assistant and the Description Curator.") + "\n",
         "before"),
    ],
    "user_guide/assistant.rst": [
        (".. note::",
         _fig("08_assistant_empty.png", "The General Assistant page",
              "The General Assistant. Ask a question in the chat box at the bottom of "
              "the page.") + "\n",
         "before"),
        # Placed under the capability table rather than under "Reading source
        # badges": this answer came back through the plain search route, which
        # emits no badge, so it cannot illustrate one.
        ("What the assistant can do",
         _fig("09_dataset_search.png", "A dataset search answer",
              "A dataset search answer. Results are listed with their DOIs, followed by "
              "a reminder to verify them on the portal. Compare the badged answer in "
              ":doc:`content_reasoning`, which was reasoned rather than looked up.") + "\n",
         "after_section"),
    ],
    "user_guide/dataset_profiles.rst": [
        ("What it does", _fig("10_dataset_profile.png", "A dataset profile answer",
                              "A profile answer, produced by asking a follow-up question "
                              "about a dataset from the previous turn.") + "\n", "after_section"),
    ],
    "user_guide/content_reasoning.rst": [
        ("What the user sees", _fig("11_content_reasoning.png", "A content reasoning answer",
                              "A content-reasoning answer. The ``content reasoning`` badge "
                              "marks an answer that was reasoned over fact sheets rather "
                              "than looked up, and every dataset named carries the basis "
                              "it was judged on.") + "\n",
         "after_section"),
    ],
    "user_guide/multi_turn.rst": [
        ("Narrowing a result set", _fig("12_multi_turn.png", "A multi-turn refinement",
                              "A refinement turn. \"Of these\" narrows the result set from "
                              "the previous answer rather than searching the whole catalog.") + "\n",
         "after_section"),
    ],
}


def figures(stage: Path, available: set[str]) -> tuple[int, list[str]]:
    """Place figures into the staged chapters. Returns (placed, skipped)."""
    placed, skipped = 0, []
    for rel, items in FIGURES.items():
        path = stage / rel
        if not path.exists():
            skipped.append(f"{rel} (not staged)")
            continue
        text = path.read_text(encoding="utf-8")
        for anchor, block, mode in items:
            name = re.search(r"manual/(\S+\.png)", block)
            if name and name.group(1) not in available:
                skipped.append(f"{rel}: {name.group(1)} was not captured")
                continue
            if mode == "replace":
                if anchor not in text:
                    skipped.append(f"{rel}: anchor not found for {name.group(1) if name else '?'}")
                    continue
                text = text.replace(anchor, block, 1)
            elif mode == "before":
                if anchor not in text:
                    skipped.append(f"{rel}: anchor not found for {name.group(1) if name else '?'}")
                    continue
                text = text.replace(anchor, block + "\n" + anchor, 1)
            elif mode == "after_section":
                # Insert after the underline of the named section.
                m = re.search(
                    rf"^{re.escape(anchor)}\n[-~^=]+\n+", text, re.M | re.I
                )
                if not m:
                    skipped.append(f"{rel}: section '{anchor}' not found")
                    continue
                text = text[: m.end()] + block + "\n" + text[m.end():]
            placed += 1
        path.write_text(text, encoding="utf-8")
    return placed, skipped
