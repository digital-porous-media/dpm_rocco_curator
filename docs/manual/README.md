# Building the reference manual

`scripts/build_manual.py` produces `build/manual/Rocco_Manual.pdf`: the complete
documentation as one printable volume, with a table of contents, screenshots,
and the generated code reference as appendices.

This directory is excluded from the documentation website build
(`exclude_patterns` in `../conf.py`). Nothing here affects the published site.

## Prerequisites

```bash
pip install -e ".[docs,manual]"
python -m playwright install chromium   # only needed to recapture screenshots
```

The build also needs Graphviz (`dot`) for the architecture diagrams, and system
Pango and Cairo libraries, which WeasyPrint uses for text layout.

## Build

```bash
python scripts/build_manual.py               # full build
python scripts/build_manual.py --only-html   # stop after Sphinx, for fast CSS work
python scripts/build_manual.py --check-drift # report copy edits that stopped matching
```

## How it works

The build never modifies the repository's documentation. It stages a copy into
`build/manual/src/`, edits the copy, and builds from there:

1. **Stage** — copy `docs/` plus the root documents (`README.md`,
   `CONTRIBUTING.md`, `DEPLOYMENT.md`, `CHANGELOG.md`, `LICENSE`,
   `benchmarks/README.md`), the prompt files, and the benchmark figures. The
   file list is an explicit allowlist, never a directory sweep — `.env` sits
   beside `README.md` in the repository root.
2. **Copy edit** — `copyedit.py` applies the Google style pass: sentence-case
   headings, US spelling, and a table of exact replacements covering banned
   words, first person, future tense, marketing voice, broken cross-references,
   and one redaction in the deployment chapter.
3. **Build** — `sphinx-build -b singlehtml` with `-W`, so a dead cross-reference
   fails the build. A reader cannot fix a broken link in a printed document.
4. **Print** — WeasyPrint renders the single HTML page with `print.css`, which
   supplies the page geometry, running heads, folios, table of contents leaders,
   and PDF bookmarks.

## Files

| Path | Purpose |
| --- | --- |
| `conf.py` | Print-specific Sphinx overrides, layered on `../conf.py`. |
| `index.rst` | Master document: cover, table of contents placeholder, part toctrees. |
| `print.css` | Page geometry, typography, and print styling. |
| `copyedit.py` | The Google style pass and figure placement. |
| `front/`, `parts/`, `appendix/` | Manual-only chapters and part title pages. |
| `overrides/` | Optional hand-written replacements for whole chapters. Empty by default. |
| `fonts/` | Vendored Roboto and Roboto Mono (Apache-2.0). |

## Screenshots

```bash
python scripts/capture_screenshots.py             # all 12
python scripts/capture_screenshots.py --only curator
```

This launches the real app and drives it, so it needs a working `.env` and, for
the assistant captures, a running Neo4j. Output goes to
`../_static/screenshots/manual/`. The two older screenshots one level up belong
to the website and are left alone.

Before each capture, the script checks that no credential value from `.env`
appears in the page text or in any input. A hit aborts that capture rather than
writing the file.

## Secret scanning

Two scans gate the output, and either one fails the build:

- the staged tree, before Sphinx runs;
- the finished PDF's extracted text and metadata, after WeasyPrint runs. This is
  the one that catches a leak arriving through a screenshot.

Both check provider key patterns, high-entropy tokens, and the literal values of
the credential variables in the local `.env`.

## Editing

- **Wording in a chapter** — edit the real file under `docs/`. The manual picks
  it up on the next build.
- **A Google style fix the manual needs but the website should not take** — add
  an entry to `REPLACEMENTS` in `copyedit.py`. Mark it `required` so the build
  fails if the upstream text later changes.
- **A whole chapter rewritten for print** — put the full file in `overrides/`,
  mirroring its staged path.
- **A new figure** — add it to `FIGURES` in `copyedit.py` with the anchor text it
  attaches to.
