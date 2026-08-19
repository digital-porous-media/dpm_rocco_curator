"""
Sync DPM Portal documentation markdown from the public `dpm_docs` GitHub repo
(https://github.com/digital-porous-media/dpm_docs) into `data/portal_docs/docs/`.

`src.assistant.portal_docs_tree` parses these files at runtime into a heading
tree for `search_portal_docs` (see that module's docstring) — there is no
separate build/index step; re-running this script and restarting the process
is all that's needed to pick up new/changed dpm_docs pages.

(This script used to also build a FAISS vector index over chunked versions of
these pages — that retrieval path was replaced by the heading-tree approach
and removed; this file now only does the fetch.)

Freshness: dpm_docs updates roughly every 3-6 months. Use `--check` to compare
the local docs against dpm_docs' current HEAD commit without re-fetching.

Usage:
    python scripts/sync_dpm_docs.py
    python scripts/sync_dpm_docs.py --check

Prerequisites:
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

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPM_DOCS_REPO = "digital-porous-media/dpm_docs"
DPM_DOCS_BRANCH = "main"
DPM_DOCS_PAGES_BASE = "https://digital-porous-media.github.io/dpm_docs"

DATA_DIR = Path("data/portal_docs")
# Last-synced commit SHA, so --check can report staleness without a full
# re-fetch. Not tied to any build artifact — just a bookkeeping file living
# alongside the synced markdown.
_SYNC_META_PATH = DATA_DIR / "_sync_meta.json"

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
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


def _extract_page_title(text: str, fallback: str) -> str:
    m = _H1_RE.search(text)
    return m.group(1).strip() if m else fallback


# ---------------------------------------------------------------------------
# Freshness metadata
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_sync_meta() -> None:
    try:
        sha = get_latest_commit_sha()
    except Exception as e:
        print(f"Warning: could not resolve dpm_docs HEAD commit ({e}); _sync_meta.json will omit source_commit")
        sha = None
    meta = {"source_commit": sha, "synced_at": _utc_now_iso()}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _SYNC_META_PATH.write_text(json.dumps(meta, indent=2))


def check_freshness() -> None:
    """--check: compare stored source_commit against current dpm_docs HEAD,
    without fetching anything."""
    if not _SYNC_META_PATH.exists():
        print(
            f"No sync metadata found at {_SYNC_META_PATH}. Run a sync first: "
            "python scripts/sync_dpm_docs.py"
        )
        return

    meta = json.loads(_SYNC_META_PATH.read_text())
    stored_sha = meta.get("source_commit")

    try:
        current_sha = get_latest_commit_sha()
    except Exception as e:
        print(f"Could not reach GitHub API to check freshness: {e}")
        return

    if stored_sha and stored_sha == current_sha:
        print(f"Docs are current (dpm_docs HEAD: {current_sha[:8]}).")
    else:
        print(
            f"Docs are STALE. Synced commit: {stored_sha[:8] if stored_sha else '?'}, "
            f"current HEAD: {current_sha[:8]}. Re-run without --check to sync."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync DPM Portal documentation markdown from GitHub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check whether the local docs are stale vs. current dpm_docs HEAD, without syncing",
    )
    args = parser.parse_args()

    if args.check:
        check_freshness()
        return

    fetch_dpm_docs(DATA_DIR)
    _write_sync_meta()


if __name__ == "__main__":
    main()
