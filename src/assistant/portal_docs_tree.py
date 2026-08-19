"""
Heading-tree builder for the dpm_docs portal documentation corpus.

Prototype (branch: experiment/pageindex-prototype). Builds a hierarchical tree
directly from each markdown page's `#`/`##`/.../`######` heading structure — no
chunking, no embeddings. This is the "index" half of a hand-rolled, PageIndex-style
retrieval approach (https://github.com/VectifyAI/PageIndex): PageIndex itself spends
an LLM call detecting table-of-contents structure in unstructured PDFs; the dpm_docs
markdown pages already have that structure explicit in their heading markup, so no
detection step is needed — the tree falls directly out of parsing.

See src/assistant/portal_docs_retrieval.py for the "retrieval" half (LLM-reasoning
node selection over this tree) and HANDOFF.md's PageIndex prototype section for the
full background/rationale.

Built at query/import time (not persisted as an artifact like the FAISS index) —
parsing ~13 files/~90 headings of already-downloaded markdown is negligible compared
to the embedding-based index's actual expensive step (per-chunk embedding API calls).
Revisit persistence only if profiling shows this assumption wrong, or once/if this
graduates out of prototype status and something wants a stable node-id contract
across process restarts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Reuse the existing, already-tuned heading/image/title-extraction logic from the
# FAISS index builder rather than reimplementing it — see that module's docstrings
# for the rationale behind each of these (figure-placeholder substitution, H1-as-
# page-title convention, "copy"-file exclusion).
from scripts.build_portal_docs_index import (  # noqa: E402
    DPM_DOCS_PAGES_BASE,
    _extract_page_title,
    _IMAGE_RE,
)

# Absolute path, not build_portal_docs_index.DATA_DIR's relative "data/portal_docs" —
# matches tools._get_portal_docs_store's convention so this doesn't break if the
# process's cwd isn't the project root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "portal_docs"

# Unlike build_portal_docs_index._HEADING_SPLIT_RE (##/### only, flattened
# sections), the tree needs every heading level so it can nest them — dpm_docs pages
# go as deep as H5 (see upload_data.md's "Curate Your Dataset" reference section).
_ANY_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.*)$")
_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class DocNode:
    """One heading-delimited section of a dpm_docs page, plus its position in the
    page's heading hierarchy."""

    node_id: str
    title: str
    level: int
    page_title: str
    doc_url: str
    text: str = ""
    full_text: str = ""
    parent_id: str | None = None
    children: list["DocNode"] = field(default_factory=list)


def _clean_title(raw: str) -> str:
    """Strip markdown emphasis markup (`**bold**`, `` `code` ``) from a heading's
    display text — upload_data.md's H5s are written as
    "**Source 1: Natural (Earth)**", and leaving that in would put noisy markdown
    punctuation in front of the LLM node-selector (see portal_docs_retrieval.py)."""
    return _MARKDOWN_EMPHASIS_RE.sub("", raw).strip()


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or "section"


def _fenced_line_flags(lines: list[str]) -> list[bool]:
    """Return, per line, whether that line's content sits inside a ``` fenced code
    block. dpm_docs indents its fences under numbered-list steps (see
    tools_lbpm_docs.md), so a column-0 heading regex never matches inside them today
    — but tracking fence state explicitly is cheap insurance against a future page
    with an unindented fence containing a `#`-prefixed line (e.g. a shell comment)
    that would otherwise false-match as a heading."""
    in_fence = False
    flags = []
    for line in lines:
        flags.append(in_fence)
        if line.strip().startswith("```"):
            in_fence = not in_fence
    return flags


def _find_headings(text: str) -> list[tuple[int, str, int, int]]:
    """Return (level, cleaned_title, own_text_start, heading_match_start) for every
    real (non-fenced) heading in `text`, in document order."""
    lines = text.split("\n")
    fenced = _fenced_line_flags(lines)

    headings = []
    for m in _ANY_HEADING_RE.finditer(text):
        line_idx = text.count("\n", 0, m.start())
        if fenced[line_idx]:
            continue
        level = len(m.group(1))
        title = _clean_title(m.group(2))
        headings.append((level, title, m.end(), m.start()))
    return headings


def parse_markdown_tree(path: Path, rel_path: str) -> list[DocNode]:
    """Parse one dpm_docs markdown file into a forest of DocNodes — normally a
    single-element list (one page = one H1 root), but two elements for a file with
    more than one H1 (see cite.md: "How to Cite the ... Portal" and "How to Cite a
    Dataset from the Portal" are two genuinely distinct topics mistakenly both marked
    H1 — the tree splits them into two page nodes rather than silently merging their
    content the way the flat FAISS chunker's H2/H3-only split does today)."""
    raw = path.read_text(encoding="utf-8")
    raw = _IMAGE_RE.sub(r"[Figure: \1]", raw)

    slug = Path(rel_path).stem
    fallback_title = slug.replace("_", " ").title()
    doc_url = f"{DPM_DOCS_PAGES_BASE}/{slug}/"

    headings = _find_headings(raw)

    if not headings:
        # Stub page with no headings at all (e.g. new_tutorials.md's "Coming soon!"
        # admonition) — one page-level node holding the whole body.
        page_title = _extract_page_title(raw, fallback=fallback_title)
        return [
            DocNode(
                node_id=f"{rel_path}#{_slugify(page_title)}",
                title=page_title,
                level=1,
                page_title=page_title,
                doc_url=doc_url,
                text=raw.strip(),
                full_text=raw.strip(),
            )
        ]

    # Own text of heading i is everything between its heading line and the next
    # heading of ANY level (a child heading starting immediately ends the parent's
    # own text, by construction — this is what makes full_text's simple concatenation
    # correct later).
    own_texts = []
    for i, (_level, _title, start, _match_start) in enumerate(headings):
        end = headings[i + 1][3] if i + 1 < len(headings) else len(raw)
        own_texts.append(raw[start:end].strip())

    roots: list[DocNode] = []
    stack: list[tuple[int, DocNode, list[str]]] = []  # (level, node, [rel_path])
    node_paths: dict[int, list[str]] = {}

    for i, (level, title, _start, _match_start) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()

        parent = stack[-1][1] if stack else None
        parent_slug_path = stack[-1][2] if stack else []
        slug_path = parent_slug_path + [_slugify(title)]
        node_id = f"{rel_path}#{'/'.join(slug_path)}"

        page_title = parent.page_title if parent is not None else title
        node = DocNode(
            node_id=node_id,
            title=title,
            level=level,
            page_title=page_title,
            doc_url=doc_url,
            text=own_texts[i],
            parent_id=parent.node_id if parent is not None else None,
        )

        if parent is not None:
            parent.children.append(node)
        else:
            roots.append(node)

        stack.append((level, node, slug_path))
        node_paths[i] = slug_path

    # Leading content before the very first heading (rare — dpm_docs pages start
    # with their H1 — but handle it rather than silently dropping it, mirroring
    # build_portal_docs_index._split_into_sections' equivalent "" section).
    preamble = raw[: headings[0][3]].strip()
    if preamble:
        preamble_node = DocNode(
            node_id=f"{rel_path}#preamble",
            title=fallback_title,
            level=1,
            page_title=fallback_title,
            doc_url=doc_url,
            text=preamble,
        )
        roots.insert(0, preamble_node)

    def _roll_up(node: DocNode) -> str:
        """Concatenate a node's own text with every descendant's rolled-up text,
        reintroducing each child's own heading line first — without this, a
        multi-child section's full_text is a flat, unlabeled blob with no signal
        distinguishing a short conceptual overview (the parent's own text) from the
        (often much longer) field-reference detail that follows in its children,
        which biases synthesis toward the latter (see HANDOFF.md's PageIndex
        prototype "Update 10" section for the concrete case this fixes)."""
        parts = [node.text] if node.text else []
        for child in node.children:
            child.full_text = _roll_up(child)
            if child.full_text:
                heading = f"{'#' * child.level} {child.title}"
                parts.append(f"{heading}\n\n{child.full_text}")
        return "\n\n".join(parts)

    for root in roots:
        root.full_text = _roll_up(root)

    return roots


def build_forest(md_dir: Path = DATA_DIR) -> list[DocNode]:
    """Parse every markdown page under md_dir into page-root DocNodes. Same file
    discovery/exclusion convention as PortalDocsIndexBuilder.run (sorted, "copy"
    files excluded — see build_portal_docs_index.fetch_dpm_docs's docstring)."""
    md_files = sorted(
        p for p in Path(md_dir).rglob("*.md") if "copy" not in p.stem.lower()
    )
    forest: list[DocNode] = []
    for f in md_files:
        rel_path = str(f.relative_to(md_dir))
        try:
            forest.extend(parse_markdown_tree(f, rel_path))
        except Exception as e:
            logger.warning("Failed to parse tree for %s (%s); skipping", rel_path, e)
    return forest


def flatten(forest: list[DocNode]) -> list[DocNode]:
    """Pre-order walk of every node (all levels, roots and descendants) — the flat
    list handed to the LLM node-selector in portal_docs_retrieval.py."""
    out: list[DocNode] = []

    def _walk(node: DocNode) -> None:
        out.append(node)
        for child in node.children:
            _walk(child)

    for root in forest:
        _walk(root)
    return out


_portal_docs_tree: list[DocNode] | None = None


def get_portal_docs_tree() -> list[DocNode]:
    """Lazy singleton — build the forest once per process, cache in a module
    global (mirrors tools._get_portal_docs_store's lazy-singleton pattern)."""
    global _portal_docs_tree
    if _portal_docs_tree is None:
        _portal_docs_tree = build_forest(DATA_DIR)
    return _portal_docs_tree
