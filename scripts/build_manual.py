#!/usr/bin/env python3
"""Build the single-PDF Rocco reference manual.

Pipeline:

1. **Stage**  — copy the documentation into ``build/manual/src/``. The repository's
   own sources are never modified.
2. **Overlay** — apply the copy-edited chapters from ``docs/manual/overrides/``
   and generate the chapters that are derived from code.
3. **Build**  — ``sphinx-build -b singlehtml`` over the staged tree.
4. **Print**  — render the single HTML page to PDF with WeasyPrint and
   ``docs/manual/print.css``.

Two secret scans gate the output: one over the staged tree before Sphinx runs,
and one over the finished PDF's extracted text and metadata. A PDF is a single
file that leaves the repository, so anything that leaks into it leaks completely.

Usage::

    python scripts/build_manual.py                 # full build
    python scripts/build_manual.py --only-html     # stop after Sphinx (fast CSS iteration)
    python scripts/build_manual.py --skip-screenshots
    python scripts/build_manual.py --check-drift   # report overrides whose source changed
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANUAL = REPO / "docs" / "manual"
BUILD = REPO / "build" / "manual"
STAGE = BUILD / "src"
HTML = BUILD / "html"
PDF = BUILD / "Rocco_Manual.pdf"

# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

# Documents that live outside docs/ and are pulled in as manual chapters.
# This is an explicit allowlist, never a glob: `.env` sits in the repository root
# next to README.md, so a directory sweep is the obvious way a key ends up in the
# PDF.
ROOT_DOCS = {
    "README.md": "overview.md",
    "CONTRIBUTING.md": "contributing.md",
    "DEPLOYMENT.md": "deployment.md",
    "CHANGELOG.md": "changelog.md",
    "benchmarks/README.md": "benchmarks.md",
    "LICENSE": "_license.txt",
}

# Superseded by Appendix A and by the full CONTRIBUTING.md chapter. Dropped from
# the staged tree; the chapters that linked to them are relinked in the overrides.
STAGE_DROP = [
    "developer_guide/api_reference.rst",
    "developer_guide/contributing.rst",
]

def _ignore(directory: str, names: list[str]) -> set[str]:
    """Skip the website build output and this manual's own config.

    Scoped to the top level of docs/ on purpose: a bare ``ignore_patterns("manual")``
    also matches ``docs/_static/screenshots/manual/``, which silently drops every
    screenshot and leaves the figures unresolved.
    """
    skip = {"__pycache__", ".doctrees"}
    if Path(directory).resolve() == (REPO / "docs").resolve():
        skip |= {"_build", "manual"}
    return {n for n in names if n in skip or n.endswith(".pyc")}


def stage() -> None:
    """Copy the documentation into the staging tree."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.parent.mkdir(parents=True, exist_ok=True)

    # 1. The docs/ tree, minus the website build output and this manual's own config.
    shutil.copytree(REPO / "docs", STAGE, ignore=_ignore)

    # 2. The manual's own scaffolding, overwriting docs/index.rst and docs/conf.py.
    shutil.copy2(MANUAL / "conf.py", STAGE / "conf.py")
    shutil.copy2(MANUAL / "index.rst", STAGE / "index.rst")
    for sub in ("front", "parts", "appendix"):
        shutil.copytree(MANUAL / sub, STAGE / sub, dirs_exist_ok=True)

    # 3. Root-level documents.
    for src, dest in ROOT_DOCS.items():
        shutil.copy2(REPO / src, STAGE / dest)

    # 4. Benchmark figures, referenced by the benchmarks chapter.
    figs = STAGE / "figures"
    figs.mkdir(exist_ok=True)
    for png in sorted((REPO / "benchmarks" / "figures").glob("*.png")):
        shutil.copy2(png, figs / png.name)

    # 5. Prompt files and rubric, reproduced verbatim in Appendix C.
    prompts = STAGE / "_prompts"
    prompts.mkdir(exist_ok=True)
    for yml in sorted((REPO / "src" / "prompts").glob("*.yaml")):
        shutil.copy2(yml, prompts / yml.name)
    for extra in ("rubric.json", "examples_v3.json"):
        shutil.copy2(REPO / "src" / "evaluator" / extra, prompts / extra)

    # 6. Drop superseded pages.
    for rel in STAGE_DROP:
        (STAGE / rel).unlink(missing_ok=True)


def overlay() -> list[str]:
    """Apply copy-edited chapters over the staged tree. Returns applied paths."""
    overrides = MANUAL / "overrides"
    applied = []
    for path in sorted(overrides.rglob("*")):
        if path.is_dir() or path.name.startswith("_"):
            continue
        rel = path.relative_to(overrides)
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        applied.append(str(rel))
    return applied


def fix_paths() -> None:
    """Rewrite links in the staged root documents.

    ``README.md`` lives at the repository root and points into ``docs/``. Once
    staged, the docs tree *is* the root, so those prefixes have to go.
    """
    overview = STAGE / "overview.md"
    text = overview.read_text(encoding="utf-8")
    text = text.replace("](docs/", "](")
    overview.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Generated chapters
# ---------------------------------------------------------------------------

_GROUPS = {
    "TestSemanticSearch": "Dataset discovery",
    "TestComponentSearch": "Component search",
    "TestMetadataFilter": "Structured property filters",
    "TestDomainQA": "Domain questions",
    "TestWorkflowGuidance": "Workflow guidance",
    "TestQueryExpansion": "Query expansion",
    "TestLiteratureSearch": "Literature search",
    "TestDatasetProfile": "Dataset profiles",
    "TestMultiPartQuestions": "Multi-part questions",
}


def _extract_query(fn: ast.FunctionDef) -> str | None:
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in ("query", "question", "q")
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and len(arg.value) > 14
                    and " " in arg.value
                ):
                    return arg.value
    return None


def _cell(text: str, indent: str = "       ") -> str:
    """Format a value as a list-table cell body.

    Docstrings wrap across lines. A list-table cell must be a single indented
    block, so newlines are collapsed and continuation lines are re-indented.
    """
    flat = " ".join(str(text).split())
    words, lines, current = flat.split(" "), [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > 78:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return ("\n" + indent).join(lines)


def generate_acceptance_appendix() -> int:
    """Write Appendix D from the acceptance test suite. Returns the case count."""
    source = REPO / "tests" / "assistant" / "test_search_integration.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    cases: dict[tuple[str, str], dict] = {}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            doc = ast.get_docstring(fn) or ""
            match = re.match(r"([A-Z]-\d+)", doc)
            if not match:
                continue
            key = (cls.name, match.group(1))
            entry = cases.setdefault(
                key,
                {"id": match.group(1), "desc": "", "query": None, "live": False, "unit": False},
            )
            # tests/assistant/conftest.py auto-applies the `live` marker to any
            # test requesting the chat_model fixture (conversation_manager depends
            # on it), so the marker is not visible as a decorator. Naming alone is
            # not enough either: several live cases have no _live suffix.
            args = {a.arg for a in fn.args.args}
            if args & {"chat_model", "conversation_manager"} or fn.name.endswith("_live"):
                entry["live"] = True
            else:
                entry["unit"] = True
            query = _extract_query(fn)
            if query and not entry["query"]:
                entry["query"] = query
            desc = doc.split(":", 1)[-1].strip().rstrip(".")
            desc = re.sub(r"^live\b[ —-]*", "", desc, flags=re.I)
            if len(desc) > len(entry["desc"]):
                entry["desc"] = desc

    lines = [
        "Appendix D. Acceptance query suite",
        "==================================",
        "",
        "This appendix lists the acceptance cases that verify the General",
        "Assistant end to end. They are automated in",
        "``tests/assistant/test_search_integration.py``, and this table is",
        "generated from that file, so it cannot drift from the suite.",
        "",
        "Most cases have two tiers. The unit tier runs against mock fixtures and",
        "is part of the default test run. The live tier calls the real language",
        "model, Neo4j, and Semantic Scholar endpoints, and is excluded by default.",
        "Run the live tier explicitly:",
        "",
        ".. code-block:: console",
        "",
        "   $ pytest tests/assistant/test_search_integration.py -m live -v",
        "",
    ]

    for cls_name, label in _GROUPS.items():
        group = [v for (c, _), v in sorted(cases.items()) if c == cls_name]
        if not group:
            continue
        group.sort(key=lambda e: (e["id"][0], int(e["id"].split("-")[1])))
        lines += [label, "-" * len(label), ""]
        lines += [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 8 46 34 12",
            "",
            "   * - Case",
            "     - Query",
            "     - Verifies",
            "     - Tier",
        ]
        for entry in group:
            query = entry["query"] or "(structural check; no query text)"
            tiers = [t for t, on in (("unit", entry["unit"]), ("live", entry["live"])) if on]
            tier = ", ".join(tiers) or "unit"
            desc = entry["desc"] or "—"
            desc = desc[0].upper() + desc[1:] if desc[0].islower() else desc
            lines += [
                f"   * - {entry['id']}",
                f"     - {_cell(query)}",
                f"     - {_cell(desc)}",
                f"     - {tier}",
            ]
        lines.append("")

    dest = STAGE / "appendix" / "acceptance_queries.rst"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(cases)


# ---------------------------------------------------------------------------
# Table of contents
# ---------------------------------------------------------------------------

# Sphinx's singlehtml builder replaces each toctree with the *content* of the
# documents it lists, so the build produces no link list to turn into a table of
# contents. The TOC is therefore generated here from the finished HTML and
# injected into the placeholder container in index.rst. Page numbers are filled
# in by WeasyPrint at render time via target-counter().

_HEADING_RE = re.compile(
    r'<section id="(?P<id>[^"]+)">\s*<h(?P<level>[2-4])>(?P<title>.*?)<a class="headerlink"',
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _plain(html_fragment: str) -> str:
    text = _TAG_RE.sub("", html_fragment)
    return " ".join(text.split()).replace("¶", "").strip()


_SECTION_RE = re.compile(r'<section id="(?P<id>[^"]+)">')


def dedupe_anchors() -> int:
    """Give every section a unique id. Returns the number of ids rewritten.

    Sphinx numbers section ids per source document, but the singlehtml builder
    concatenates every document into one page, so generic headings collide:
    the build produces 16 sections with id="see-also". An id is supposed to be
    unique in a document, and WeasyPrint follows the spec -- it resolves
    ``target-counter(attr(href), page)`` against the *first* element carrying
    the id. Every "See also" entry in the table of contents therefore printed
    the page of the first one, sending the reader to Part II from Part III.

    The first occurrence keeps its id. Body cross-references written by Sphinx
    already resolve there, and renaming it would leave them dangling, which
    WeasyPrint renders as page 0. Only the second and later occurrences are
    renamed, along with the headerlink each one carries, so the table of
    contents gets a distinct target for every section.
    """
    index = HTML / "index.html"
    html = index.read_text(encoding="utf-8")

    seen: dict[str, int] = {}
    out = []
    pos = 0
    renamed = 0

    for m in _SECTION_RE.finditer(html):
        anchor = m.group("id")
        n = seen.get(anchor, 0)
        seen[anchor] = n + 1
        if n == 0:
            continue

        unique = f"{anchor}--{n + 1}"
        out.append(html[pos : m.start()])
        out.append(f'<section id="{unique}">')
        renamed += 1

        # Retarget this section's own headerlink. It is the first one after the
        # section tag; stop at the next section so a nested duplicate is left
        # for its own iteration.
        tail_end = html.find("<section", m.end())
        if tail_end == -1:
            tail_end = len(html)
        tail = html[m.end() : tail_end]
        tail = tail.replace(
            f'<a class="headerlink" href="#{anchor}"',
            f'<a class="headerlink" href="#{unique}"',
            1,
        )
        out.append(tail)
        pos = tail_end

    out.append(html[pos:])
    index.write_text("".join(out), encoding="utf-8")
    return renamed


def inject_toc(depth: int = 3) -> int:
    """Build the table of contents from the rendered headings. Returns entry count."""
    index = HTML / "index.html"
    html = index.read_text(encoding="utf-8")

    entries = []
    for m in _HEADING_RE.finditer(html):
        level = int(m.group("level"))
        if level > depth + 1:
            continue
        title = _plain(m.group("title"))
        if not title:
            continue
        # "1. Part I. Getting started" — the toctree number duplicates the part
        # label, which is hidden in the body for the same reason.
        title = re.sub(r"^\d+\.\s+(?=Part\s)", "", title)
        entries.append((level, m.group("id"), title))

    parts = ['<div class="manual-toc">']
    current = 1
    for level, anchor, title in entries:
        while current < level:
            parts.append("<ul>")
            current += 1
        while current > level:
            parts.append("</ul>")
            current -= 1
        parts.append(f'<li><a href="#{anchor}">{title}</a></li>')
    while current > 1:
        parts.append("</ul>")
        current -= 1
    parts.append("</div>")
    toc_html = "".join(parts)

    # Drop the web theme's stylesheet and scripts. basic.css is written for a
    # browser and fights the print rules (figure sizing in particular); print.css
    # is meant to be the only presentational sheet. pygments.css is kept, because
    # it carries the syntax-highlighting colors.
    html = re.sub(
        r'<link[^>]+href="[^"]*(?:basic|alabaster|furo|design-style|sphinx-design|tabs|copybutton)[^"]*\.css[^>]*>',
        "", html,
    )
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
    html = re.sub(r"<script[^>]*/?>", "", html)

    placeholder = '<div class="manual-toc-placeholder docutils container">'
    start = html.find(placeholder)
    if start == -1:
        raise SystemExit("TOC placeholder not found in the built HTML")
    end = html.find("</div>", start) + len("</div>")
    index.write_text(html[:start] + toc_html + html[end:], encoding="utf-8")
    return len(entries)


# ---------------------------------------------------------------------------
# Secret scanning
# ---------------------------------------------------------------------------

_KEY_PATTERNS = [
    ("OpenAI project key", re.compile(r"sk-proj-(?!your-)[A-Za-z0-9_-]{16,}")),
    ("OpenAI key", re.compile(r"sk-(?!proj-|ant-|your-|\.\.\.)[A-Za-z0-9]{20,}")),
    ("Anthropic key", re.compile(r"sk-ant-(?!your-)[A-Za-z0-9_-]{16,}")),
    ("HuggingFace token", re.compile(r"hf_(?!your)[A-Za-z0-9]{16,}")),
    ("Google API key", re.compile(r"AIza(?!-your)[A-Za-z0-9_-]{30,}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

# Values short enough or common enough that a literal match would be noise.
_ENV_MIN_LEN = 8
_ENV_IGNORE = {"true", "false", "none", "ollama", "localhost", "neo4j", "bolt"}


def env_secrets() -> list[tuple[str, str]]:
    """Read literal secret values from the local .env, for exact-match scanning."""
    env = REPO / ".env"
    if not env.exists():
        return []
    out = []
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        name = name.strip()
        if len(value) < _ENV_MIN_LEN or value.lower() in _ENV_IGNORE:
            continue
        # Only treat credential-ish variables as secrets; URLs and model names are
        # published configuration and appear in the docs legitimately.
        if not re.search(r"(KEY|TOKEN|PASSWORD|SECRET)$", name):
            continue
        out.append((name, value))
    return out


def _entropy(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def scan_text(text: str, where: str, secrets: list[tuple[str, str]]) -> list[str]:
    """Return a list of findings. Empty means clean."""
    findings = []
    for label, pattern in _KEY_PATTERNS:
        for m in pattern.finditer(text):
            findings.append(f"{where}: {label} matched {m.group(0)[:12]}…")
    for name, value in secrets:
        if value in text:
            findings.append(f"{where}: literal value of .env variable {name} appears verbatim")
    # High-entropy standalone tokens that look like credentials.
    for token in re.findall(r"(?<![A-Za-z0-9/+_-])[A-Za-z0-9_-]{40,}(?![A-Za-z0-9/+_=-])", text):
        if _entropy(token) > 4.6 and not token.startswith(("graphviz-", "sha256-")):
            findings.append(f"{where}: high-entropy token {token[:16]}… (entropy {_entropy(token):.2f})")
    return findings


def scan_tree(secrets: list[tuple[str, str]]) -> list[str]:
    findings = []
    for path in sorted(STAGE.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        findings += scan_text(text, str(path.relative_to(STAGE)), secrets)
    if (STAGE / ".env").exists():
        findings.append("FATAL: .env was staged")
    return findings


def scan_pdf(secrets: list[tuple[str, str]]) -> list[str]:
    import pypdf

    reader = pypdf.PdfReader(str(PDF))
    findings = []
    for i, page in enumerate(reader.pages, 1):
        try:
            findings += scan_text(page.extract_text() or "", f"PDF page {i}", secrets)
        except Exception:  # noqa: BLE001 - a page that will not extract is not a leak
            continue
    findings += scan_text(json.dumps(dict(reader.metadata or {})), "PDF metadata", secrets)
    return findings


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------

MANIFEST = MANUAL / "overrides" / "_manifest.json"


def source_for(rel: str) -> Path | None:
    """Map an override path back to the repository source it was derived from."""
    reverse = {v: k for k, v in ROOT_DOCS.items()}
    if rel in reverse:
        return REPO / reverse[rel]
    candidate = REPO / "docs" / rel
    return candidate if candidate.exists() else None


def check_drift() -> int:
    """Report copy edits and figure anchors that no longer match their source.

    The copy edits are exact string matches against upstream documentation. When
    an upstream chapter is reworded, the matching edit stops applying — this
    reports that without building the whole manual. Hand-written overrides, if
    any exist, are additionally checked against the SHA-256 of the file they were
    derived from.
    """
    stage()
    applied = overlay()
    fix_paths()

    sys.path.insert(0, str(MANUAL))
    import copyedit

    stale = 0
    edits = copyedit.apply(STAGE)
    if edits["missing"]:
        print(f"{len(edits['missing'])} copy edit(s) no longer match their source:")
        for item in edits["missing"]:
            print("  !", item)
        stale += len(edits["missing"])
    else:
        print(f"Copy edits: all {edits['replacements']} targeted replacements still apply.")

    shots = REPO / "docs" / "_static" / "screenshots" / "manual"
    available = {p.name for p in shots.glob("*.png")} if shots.exists() else set()
    placed, skipped = copyedit.figures(STAGE, available)
    anchor_misses = [s for s in skipped if "not captured" not in s]
    if anchor_misses:
        print(f"{len(anchor_misses)} figure anchor(s) no longer match:")
        for item in anchor_misses:
            print("  !", item)
        stale += len(anchor_misses)
    else:
        print(f"Figures: all {placed} anchors still match.")
    missing_shots = [s for s in skipped if "not captured" in s]
    if missing_shots:
        print(f"{len(missing_shots)} screenshot(s) missing — run scripts/capture_screenshots.py:")
        for item in missing_shots:
            print("  -", item)

    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for rel, recorded in sorted(manifest.items()):
            src = source_for(rel)
            if src is None:
                continue
            if hashlib.sha256(src.read_bytes()).hexdigest() != recorded:
                print(f"  !  override {rel}: source changed since it was written")
                stale += 1
        print(f"Overrides: {len(manifest)} hand-written, checked against source.")
    print("\nDrift check clean." if not stale else f"\n{stale} item(s) need attention.")
    return stale


def write_manifest(applied: list[str]) -> None:
    manifest = {}
    for rel in applied:
        src = source_for(rel)
        if src is not None:
            manifest[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Build phases
# ---------------------------------------------------------------------------


def build_html(strict: bool) -> None:
    env = dict(os.environ, ROCCO_MANUAL_REPO_ROOT=str(REPO))
    cmd = [sys.executable, "-m", "sphinx", "-b", "singlehtml", str(STAGE), str(HTML)]
    if strict:
        cmd += ["-W", "--keep-going"]
    result = subprocess.run(cmd, env=env, cwd=str(REPO))
    if result.returncode != 0:
        raise SystemExit(f"sphinx-build failed with exit code {result.returncode}")


def render_pdf() -> None:
    from weasyprint import CSS, HTML as WeasyHTML

    doc = WeasyHTML(filename=str(HTML / "index.html"), base_url=str(HTML))
    doc.write_pdf(
        str(PDF),
        stylesheets=[CSS(filename=str(MANUAL / "print.css"))],
        uncompressed_pdf=False,
    )


def set_pdf_metadata() -> None:
    """Set document metadata explicitly so nothing is inherited from the machine."""
    import pypdf

    reader = pypdf.PdfReader(str(PDF))
    writer = pypdf.PdfWriter(clone_from=reader)
    writer.add_metadata(
        {
            "/Title": "Rocco Reference Manual",
            "/Author": "Bernard Chang, Maria Esteva, Zachary Nowacek, Masa Prodanovic",
            "/Subject": "AI research assistant for the Digital Porous Media Portal",
            "/Keywords": "dataset curation, retrieval-augmented generation, porous media",
            "/Creator": "Rocco documentation build",
            "/Producer": "WeasyPrint",
        }
    )
    tmp = PDF.with_suffix(".tmp.pdf")
    with tmp.open("wb") as fh:
        writer.write(fh)
    tmp.replace(PDF)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-html", action="store_true", help="stop after Sphinx")
    parser.add_argument("--skip-screenshots", action="store_true")
    parser.add_argument("--check-drift", action="store_true")
    parser.add_argument("--no-strict", action="store_true", help="do not pass -W to Sphinx")
    args = parser.parse_args()

    if args.check_drift:
        return 1 if check_drift() else 0

    secrets = env_secrets()
    print(f"[1/6] Staging documentation into {STAGE.relative_to(REPO)}")
    stage()
    applied = overlay()
    fix_paths()
    print(f"      {len(applied)} override(s) applied")

    sys.path.insert(0, str(MANUAL))
    import copyedit

    edits = copyedit.apply(STAGE)
    print(
        f"      copy edits: {edits['headings']} heading(s) sentence-cased, "
        f"{edits['spellings']} spelling(s), {edits['replacements']} targeted replacement(s)"
    )
    shots = REPO / "docs" / "_static" / "screenshots" / "manual"
    available = {p.name for p in shots.glob("*.png")} if shots.exists() else set()
    placed, shot_skipped = copyedit.figures(STAGE, available)
    print(f"      figures: {placed} placed, {len(shot_skipped)} skipped")
    for item in shot_skipped:
        print(f"        - skipped: {item}")

    if edits["missing"]:
        print("\nCOPY EDIT FAILED — required replacements did not match:\n")
        for item in edits["missing"]:
            print("  -", item)
        return 3
    n_cases = generate_acceptance_appendix()
    print(f"      Appendix D generated from {n_cases} acceptance cases")

    print(f"[2/6] Scanning staged tree ({len(secrets)} .env value(s) in the deny list)")
    findings = scan_tree(secrets)
    if findings:
        print("\nSECRET SCAN FAILED — build stopped:\n")
        for f in findings:
            print("  -", f)
        return 2
    print("      clean")

    print("[3/6] Building single-page HTML")
    build_html(strict=not args.no_strict)
    n_dupes = dedupe_anchors()
    if n_dupes:
        print(f"      {n_dupes} duplicate section id(s) made unique")
    n_toc = inject_toc()
    print(f"      table of contents generated with {n_toc} entries")
    if args.only_html:
        print(f"      {HTML / 'index.html'}")
        return 0

    print("[4/6] Rendering PDF")
    render_pdf()
    set_pdf_metadata()

    print("[5/6] Scanning finished PDF")
    findings = scan_pdf(secrets)
    if findings:
        print("\nSECRET SCAN FAILED on the rendered PDF — output withheld:\n")
        for f in findings:
            print("  -", f)
        PDF.unlink(missing_ok=True)
        return 2
    print("      clean")

    print("[6/6] Recording override manifest")
    write_manifest(applied)

    import pypdf

    reader = pypdf.PdfReader(str(PDF))
    size_mb = PDF.stat().st_size / 1_048_576
    print(f"\nBuilt {PDF.relative_to(REPO)} — {len(reader.pages)} pages, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
