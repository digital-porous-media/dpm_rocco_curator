# Zenodo Publication Checklist

Tasks to complete before publishing Rocco v1.0.0 to Zenodo. Organized by priority.

---

## Completed ✅

- [x] **Prepare for GitHub-Zenodo automated integration**
  - [x] Remove reserved DOI references from README, CITATION.cff, codemeta.json
  - [x] All `USER/dpm-rocco-curator` placeholders → `digital-porous-media/dpm-rocco-curator` in README (lines 34, 179, 220, 234–235)
  - [x] Year updated to 2025 in BibTeX

- [x] **Replace ASCII architecture diagram with graphviz SVG**
  ✅ Created `docs/architecture_diagram.svg` from Sphinx graphviz code
  ✅ Updated README line 114 to reference SVG instead of ASCII diagram

- [x] **Set up GitHub Actions docs deployment**
  ✅ Created workflow to build Sphinx docs and deploy to GitHub Pages
  ✅ GH Actions workflow builds documentation with graphviz support
  ✅ GitHub Pages enabled and configured for automatic deployment

---

## Blockers — Must Complete Before Zenodo Deposit

- [x] **Create `CITATION.cff`**
  Required for Zenodo to auto-populate the citation record.
  Required fields: `cff-version`, `message`, `title`, `authors` (real names + ORCIDs if available), `version: 1.0.0`, `repository-code`, `license: MIT`, `abstract`

- [x] **Create `codemeta.json`**
  CodeMeta is the standard for research software metadata.
  Required fields: `@context`, `@type: SoftwareSourceCode`, `name`, `author` (real names), `version`, `description`, `license` (SPDX URL), `codeRepository`, `datePublished`, `keywords`, `programmingLanguage`
  Optional but useful: `developmentStatus: active`, `relatedLink` (docs URL)
  **Note:** `referencePublication` (paper DOI) can be omitted now and added later via Zenodo's web UI once the paper is published — no re-release needed.

- [x] **Identify and fill in real author names/ORCIDs**
  Replace `"DPM Rocco Contributors"` with real names in:
  - [x] `CITATION.cff` (new file above)
  - [x] `codemeta.json` (new file above)
  - [x] `pyproject.toml` `authors` field
  - [x] `LICENSE` copyright line
  - [x] `docs/conf.py` `author` variable
  - [x] `README.md` BibTeX `author` field

- [x] **Fix placeholder GitHub URLs in `README.md`**
  ✅ Replaced all `USER/dpm-rocco-curator` → `digital-porous-media/dpm_rocco_curator`
  Lines updated: 34, 179, 220, 234–235

- [x] **Fix placeholder URL in `CONTRIBUTING.md`**
  ✅ Line 19: `yourusername/dpm-rocco-curator` → `digital-porous-media/dpm_rocco_curator`

- [x] **Add "Evaluations" section to README**
  ✅ Added with links to benchmarks/ subfolder containing detailed study data and methodology.

- [x] **Create benchmarks/ folder structure**
  ✅ Created folder skeleton with README.md, data/, figures/ subdirectories
  - [x] Clean analysis code from evaluation scripts and create `statistical_evaluation.ipynb`
  - [x] Generate/organize raw data files (`evaluation_results.xlsx`, `dataset_descriptions.md`)
  - [x] Export figures from paper as SVG and PNG (saved in figures/ folder)
  - [x] ✅ Update README.md Evaluations section to add link: `*Full study data and analysis code in [`benchmarks/`](benchmarks/) folder.*`

---

## Should Fix Before Deposit

- [x] **Create `CHANGELOG.md`**
  ✅ Created with comprehensive v1.0.0 release notes covering evaluator, editor, RAG pipeline, content screener, Streamlit UI, and future roadmap.

- [x] **Clean up `Chatbot/` folder**
  Legacy standalone LangGraph chatbot from before the `src/` architecture. Not integrated into the package.
  Either:
  - Remove entirely, or
  - Move to `archive/` or `legacy/` folder with a README note explaining it's deprecated

  **Caution:** `Chatbot/passwords.py` — verify it contains no hardcoded credentials before taking action.

- [x] **Clean up `CurationTools/` folder**
  Experimental notebooks and data artifacts (pre-production prototypes for curation pipeline development).
  Either:
  - Remove entirely, or
  - Move to `archive/` or `legacy/` folder

  **Caution:** `CurationTools/credentials.py` — verify no hardcoded credentials before taking action.

- [x] **Verify `.env` is in `.gitignore`**
  Confirm that `.env` is listed in `.gitignore` and not tracked by git (contains API keys and should never be committed).

- [x] **Clean up root-level stray files**
  These don't belong at the project root:
  - [x] `DPMP-461_description.txt`, `DPMP-523_description.txt`, `description.txt` — scratch/ticket artifacts → remove
  - [x] `pdf_rag_pipeline.py`, `evaluate_description.py` — early prototype scripts → move to `scripts/` or remove
  - [x] `test_rag_pipeline.py` — should be in `tests/` → move or remove
  - [x] Root-level `__pycache__/` — delete

---

## After GitHub v1.0.0 Release & Zenodo Deposit

- [ ] **Add Zenodo concept DOI badge and metadata**
  After Zenodo auto-archives v1.0.0 and mints a concept DOI:
  - [ ] Add DOI badge to top of `README.md` (near License/Python badges)
  - [ ] Add `doi` field to BibTeX in Citation section
  - [ ] Add `repository-artifact` URL to `CITATION.cff`
  - [ ] Add `identifier` field to `codemeta.json`
  - [ ] Create v1.0.1 release with the metadata

- [ ] **Optionally add Zenodo DOI to `pyproject.toml`**
  Add Zenodo URL to `[project.urls]` section once the DOI is minted (e.g., `Zenodo = "https://zenodo.org/records/..."`).

---

## No Changes Needed

✅ **Sphinx documentation** — Complete and builds successfully. No stubs or placeholder pages.
