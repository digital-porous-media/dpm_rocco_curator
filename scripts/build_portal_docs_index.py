"""
Build the FAISS vector index over DPM Portal documentation.

Fetches markdown pages from the public `dpm_docs` GitHub repo
(https://github.com/digital-porous-media/dpm_docs), chunks them, embeds them
with the shared assistant embeddings model, and saves a FAISS index that
`src.assistant.tools.search_portal_docs` queries at runtime.

All chunks are tagged `doc_type="markdown_page"` (dpm_docs how-to/reference
pages). An earlier version of this index also included the Turhan (2024)
master's thesis as a second source ("doc_type": "thesis"), but it was
dropped: only a ~14-page section of the 120-page thesis was on-topic, that
section's content is functionally redundant with dpm_docs' own "Curate Your
Dataset" reference section, and the other ~90% of the thesis (background
chapters, an unrelated case-study section) was pure noise competing for
top-k retrieval slots against real portal-doc chunks.

Figures: dpm_docs uses standard `![alt](path)` markdown image syntax. Image
content itself cannot be embedded (no vision-captioning/OCR pipeline exists
in this repo) — image markup is replaced with a `[Figure: alt]` placeholder
so alt-text captions still contribute some signal. On the step-heavy how-to
pages, alt text is often generic (e.g. "Upload Step 3"), so retrieval on
those pages will surface accurate surrounding prose but cannot describe what
a screenshot actually shows.

Freshness: dpm_docs updates roughly every 3-6 months. Each build records the
synced commit SHA in `data/portal_docs_index/_meta.json`. Run with `--check`
to compare against the current dpm_docs HEAD without doing a full rebuild.

Usage:
    python scripts/build_portal_docs_index.py
    python scripts/build_portal_docs_index.py --check
    python scripts/build_portal_docs_index.py --skip-fetch
    python scripts/build_portal_docs_index.py --skip-verify

Prerequisites:
    - LLM_API_KEY + embedding provider configured in .env (see src/llm/embeddings.py)
    - Network access to api.github.com and raw.githubusercontent.com
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPM_DOCS_REPO = "digital-porous-media/dpm_docs"
DPM_DOCS_BRANCH = "main"
DPM_DOCS_PAGES_BASE = "https://digital-porous-media.github.io/dpm_docs"

DATA_DIR = Path("data/portal_docs")
INDEX_DIR = Path("data/portal_docs_index")
META_PATH = INDEX_DIR / "_meta.json"

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_HEADING_SPLIT_RE = re.compile(r"(?m)^(#{2,3})\s+(.*)$")
_H1_RE = re.compile(r"(?m)^#\s+(.+)$")


# ---------------------------------------------------------------------------
# GitHub fetch
# ---------------------------------------------------------------------------

def _github_api_get(path: str) -> dict:
    resp = requests.get(f"https://api.github.com/repos/{DPM_DOCS_REPO}/{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_latest_commit_sha() -> str:
    """Current HEAD commit SHA of dpm_docs' default branch (one API call)."""
    data = _github_api_get(f"commits/{DPM_DOCS_BRANCH}")
    return data["sha"]


def fetch_dpm_docs(dest_dir: Path) -> list[Path]:
    """Download dpm_docs markdown pages into dest_dir, preserving relative paths.

    Only files under docs/ are fetched (matches mkdocs.yml's nav) — repo-root
    meta files (README, CONTRIBUTING, DEPLOYING, DEVELOPING) aren't portal
    content. Files with "copy" in the name (docs/index copy.md, docs/some-page
    copy.md) are stray duplicates in the source repo, also excluded.
    """
    tree = _github_api_get(f"git/trees/{DPM_DOCS_BRANCH}?recursive=1")
    md_paths = [
        item["path"]
        for item in tree.get("tree", [])
        if item["path"].startswith("docs/")
        and item["path"].endswith(".md")
        and "copy" not in Path(item["path"]).stem.lower()
    ]

    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rel_path in md_paths:
        raw_url = f"https://raw.githubusercontent.com/{DPM_DOCS_REPO}/{DPM_DOCS_BRANCH}/{rel_path}"
        resp = requests.get(raw_url, timeout=30)
        if resp.status_code != 200:
            print(f"  Warning: failed to fetch {rel_path} ({resp.status_code}), skipping")
            continue
        out_path = dest_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(resp.text, encoding="utf-8")
        written.append(out_path)
    print(f"Fetched {len(written)} markdown pages into {dest_dir}")
    return written


# ---------------------------------------------------------------------------
# Chunking — markdown pages
# ---------------------------------------------------------------------------

def _extract_page_title(text: str, fallback: str) -> str:
    m = _H1_RE.search(text)
    return m.group(1).strip() if m else fallback


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (section_heading, section_body) pairs on ##/###
    headings. Content before the first such heading gets an empty section label."""
    matches = list(_HEADING_SPLIT_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(2).strip(), text[start:end]))
    return sections


_MD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""]
)


def chunk_markdown_file(path: Path, rel_path: str) -> list[Document]:
    """Chunk one dpm_docs markdown file into Documents tagged with source,
    page_title, section, doc_url, and doc_type="markdown_page".

    Each chunk's embedded text is prefixed with a "{page_title} — {section}" context
    header (same pattern as the componentEmbedding index's parent-context header, see
    CLAUDE.md "Vector Indexes"). Without it, a short procedural chunk pulled out of its
    page — e.g. "3. From the dropdown list, select `Dataset`. 4. Fill in..." — carries
    no textual signal that it's part of an upload walkthrough, and loses out in
    similarity search to unrelated prose that happens to repeat the query's literal
    words more densely. The header is embedded, not just stored as metadata, precisely
    so it contributes to the similarity score.
    """
    raw = path.read_text(encoding="utf-8")
    # Strip image syntax to a plain-text placeholder — see module docstring
    # "Figures" section for rationale.
    raw = _IMAGE_RE.sub(r"[Figure: \1]", raw)

    slug = Path(rel_path).stem
    page_title = _extract_page_title(raw, fallback=slug.replace("_", " ").title())
    doc_url = f"{DPM_DOCS_PAGES_BASE}/{slug}/"

    docs = []
    for heading, body in _split_into_sections(raw):
        body = body.strip()
        if not body:
            continue
        header = f"{page_title} — {heading}" if heading else page_title
        for chunk_text in _MD_SPLITTER.split_text(body):
            docs.append(
                Document(
                    page_content=f"{header}\n\n{chunk_text}",
                    metadata={
                        "source": rel_path,
                        "page_title": page_title,
                        "section": heading,
                        "doc_url": doc_url,
                        "doc_type": "markdown_page",
                    },
                )
            )
    return docs


# ---------------------------------------------------------------------------
# Freshness metadata
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_freshness() -> None:
    """--check: compare stored source_commit against current dpm_docs HEAD,
    without fetching or rebuilding anything."""
    if not META_PATH.exists():
        print(
            f"No index metadata found at {META_PATH}. Run a full build first: "
            "python scripts/build_portal_docs_index.py"
        )
        return

    meta = json.loads(META_PATH.read_text())
    stored_sha = meta.get("source_commit")

    try:
        current_sha = get_latest_commit_sha()
    except Exception as e:
        print(f"Could not reach GitHub API to check freshness: {e}")
        return

    if stored_sha and stored_sha == current_sha:
        print(f"Index is current (dpm_docs HEAD: {current_sha[:8]}).")
    else:
        print(
            f"Index is STALE. Indexed commit: {stored_sha[:8] if stored_sha else '?'}, "
            f"current HEAD: {current_sha[:8]}. Re-run without --check to rebuild."
        )


# ---------------------------------------------------------------------------
# IndexBuilder
# ---------------------------------------------------------------------------

class PortalDocsIndexBuilder:
    """Fetches, chunks, embeds, and saves the portal docs + thesis FAISS index."""

    def __init__(self):
        from src.assistant.llm import get_embeddings_model
        from src.ingestor.embedder import DocumentEmbedder
        from src.retriever.retriever import VectorStoreManager

        # Resolve the real Embeddings instance (not the _LazyEmbeddings proxy) —
        # FAISS does an isinstance(..., Embeddings) check internally that the
        # proxy fails, silently falling back to treating it as a plain callable.
        self._embedder = DocumentEmbedder(embeddings=get_embeddings_model())
        self._manager = VectorStoreManager(self._embedder)

    def run(self, skip_fetch: bool = False, skip_verify: bool = False) -> None:
        if not skip_fetch:
            fetch_dpm_docs(DATA_DIR)
        else:
            print(f"Skipping fetch — reusing existing files in {DATA_DIR}")

        md_files = sorted(DATA_DIR.rglob("*.md"))
        if not md_files:
            raise RuntimeError(
                f"No markdown files found in {DATA_DIR}; run without --skip-fetch first"
            )

        chunks: list[Document] = []
        for f in md_files:
            rel_path = str(f.relative_to(DATA_DIR))
            chunks.extend(chunk_markdown_file(f, rel_path))

        print(f"Built {len(chunks)} chunks from {len(md_files)} markdown pages")

        self._manager.create_from_documents(chunks)
        self._manager.save(INDEX_DIR)
        print(f"Saved FAISS index to {INDEX_DIR}")

        self._write_meta()

        if not skip_verify:
            self._verify()

    def _write_meta(self) -> None:
        try:
            sha = get_latest_commit_sha()
        except Exception as e:
            print(f"Warning: could not resolve dpm_docs HEAD commit ({e}); _meta.json will omit source_commit")
            sha = None
        meta = {"source_commit": sha, "synced_at": _utc_now_iso()}
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(meta, indent=2))

    def _verify(self) -> None:
        print("\nRound-trip verification...")
        try:
            results = self._manager.similarity_search("How do I upload a dataset?", k=3)
            if results:
                print(f"  OK — got {len(results)} result(s). Top: {results[0].metadata.get('page_title', '?')!r}")
                for r in results:
                    print(f"    - [{r.metadata.get('doc_type')}] {r.metadata.get('page_title')}")
            else:
                print("  Warning: search returned 0 results (unexpected for this query).")
        except Exception as exc:
            print(f"  Verification failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch, chunk, and index DPM Portal documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check whether the built index is stale vs. current dpm_docs HEAD, without rebuilding",
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Reuse existing data/portal_docs/ files instead of re-fetching",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the round-trip similarity_search() smoke test",
    )
    args = parser.parse_args()

    if args.check:
        check_freshness()
        return

    builder = PortalDocsIndexBuilder()
    builder.run(skip_fetch=args.skip_fetch, skip_verify=args.skip_verify)


if __name__ == "__main__":
    main()
