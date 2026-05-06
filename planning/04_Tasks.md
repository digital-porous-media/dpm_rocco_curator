# Tasks & Week-by-Week Plan

Related: [[00_Project_Overview]]

---

## Rocco Extensions

These are lower-priority improvements to the existing curator. Either intern can pick these up during slower weeks or as stretch goals.

- [ ] Add "Generate Starter Description" form + LLM generation flow
- [ ] Add rubric explainer panel (collapsible, per-criterion with examples)
- [ ] Add score history chart across evaluation rounds
- [ ] Add before/after description diff view
- [ ] (Stretch) Batch evaluation mode from CSV upload

---

## Intern A — Search Backend

Builds the data retrieval layer (FAISS + Neo4j) and search tools for the unified assistant. See [[02_General_Assistant]] for full context.

### Week 1–2: Data Audit & Graph Exploration

- [ ] Query the DPM Portal API to fetch all 176+ dataset descriptions — sample 20 entries, note how many have sparse descriptions (< 3 sentences)
- [ ] Connect to the existing Neo4j graph: query what fields are actually populated on `Sample` and `DigitalDataset` nodes — assess coverage of `porousMediaType`, `voxelDimensions`, `imagingEquipmentAndModel`, `segmented`, `porosity`
- [ ] Extract all `RelatedPublication` abstracts from Neo4j — note how many datasets have linked papers
- [ ] Document findings: which datasets have rich descriptions vs. rich structured fields vs. both vs. neither — this informs the ranking logic
- [ ] Get Neo4j credentials from supervisor (follow standard credential management patterns)

### Week 3–4: Core Backend

- [ ] Write `src/assistant/dataset_index.py` — wraps `VectorStoreManager` for the description corpus
- [ ] Write `src/assistant/publication_corpus.py` — extract publication abstracts from Neo4j, embed them
- [ ] Write `scripts/build_assistant_index.py` → embed descriptions + publication abstracts → `data/assistant_vector_store/`
- [ ] Write `src/assistant/graph_store.py` — clean Neo4j Cypher queries for structured field filtering (rock type, modality, voxel size, segmentation, imaging equipment); consult Neo4j best practices documentation for production-ready patterns
- [ ] Write `search_datasets()`, `search_publications()`, `query_graph()`, and `merge_results()` in `src/assistant/tools.py`
- [ ] Write unit tests: FAISS-only path, Neo4j-only path, merge dedup + ranking logic, publication matching
- [ ] Add `USE_NEO4J` flag to `.env` — system works in FAISS-only mode when flag is false

### Week 5–6: Integration & Testing

- [ ] Coordinate with Intern B on tool interface (parameter names, return formats) — finalize `tools.py` spec
- [ ] Test 20 representative queries: sparse-description datasets (verify Neo4j picks them up), rich-description datasets (verify FAISS ranks them well), publication matches, requests for missing fields (verify no crash)
- [ ] Verify result structure matches what `assistant.py` expects (dataset ID, score, source label, publication link where applicable)
- [ ] Document Neo4j query patterns and FAISS index rebuild steps

### Week 7–8: Polish & Documentation

- [ ] Add source labels to results: `[description match]`, `[metadata match]`, `[publication match]`, `[both]`
- [ ] Ensure merge logic handles edge cases (e.g., same dataset appearing in both FAISS and Neo4j)
- [ ] Write docstrings for `dataset_index.py`, `graph_store.py`, `publication_corpus.py`
- [ ] Final index rebuild; verify data completeness

---

## Intern B — Education Backend & Conversation Manager

Builds the education tools, query expansion, conversation orchestration, and UI for the unified assistant. See [[02_General_Assistant]] for full context.

### Week 1–2: UX & Prompt Design

- [ ] Study `rocco_ui.py` — understand session state patterns and how PDF upload + RAG currently works
- [ ] Prototype `assistant_ui.py` as a Streamlit chat interface (no backend yet) — validate layout with supervisor (input box, message history, expandable sources/learn-more sections)
- [ ] Draft `src/prompts/educational.yaml` — domain Q&A + workflow synthesis prompts
- [ ] Draft `src/prompts/query_expander.yaml` — LLM prompt for query expansion (given intent, output expanded query + inferred filters + rationale)
- [ ] Catalogue DPM Portal tutorial URLs: verify which goals have Jupyter notebook tutorials vs. HTML docs pages; note exact URLs — the `tutorials.yaml` should reference only verified URLs, not assumed ones
- [ ] Write `data/tutorials.yaml` — map at least 20 user goals to verified portal tutorial URLs (`notebook_url` and/or `doc_url`), tools, and example dataset IDs (extended schema: see [[02_General_Assistant]])
- [ ] Write `data/domain_workflows.yaml` — 10–15 general workflow entries covering: LBM theory + requirements, pore network modeling, image segmentation, DNS, permeability/porosity analysis, drainage/imbibition simulation, imaging parameter trade-offs (voxel size vs. FOV). Method-focused, not tied to specific external software unless officially on the portal.

### Week 3–4: Core Backend

- [ ] Write `get_educational_context(topic)` in `src/assistant/tools.py` — LLM call with educational prompt + `domain_workflows.yaml` + optional RAG from uploaded paper
- [ ] Write `get_workflow_guidance(goal)` — `tutorials.yaml` lookup + LLM synthesis; surface notebook and doc links
- [ ] Write `search_tutorials(query)` in `src/assistant/tools.py` — semantic search over `data/tutorial_vector_store/` (portal tutorial content)
- [ ] Write `scripts/build_tutorial_index.py` — scrape/download portal tutorial notebook and doc content, chunk and embed into `data/tutorial_vector_store/`
- [ ] Write `expand_query(user_message)` in `src/assistant/tools.py` — LLM domain reasoning → expanded query + inferred filters + rationale
- [ ] Write `search_external_literature(query)` in `src/assistant/tools.py` — call Semantic Scholar API (no key required for basic use); return title, authors, abstract snippet, DOI/link; label results `[external literature]` to distinguish from portal publications
- [ ] Write `src/assistant/literature_search.py` — Semantic Scholar/arXiv API wrapper with rate limiting and response parsing
- [ ] Write `src/assistant/assistant.py` — unified conversation manager: intent classify → query expand (if search) → tool dispatch → merge results → synthesize response
- [ ] Test education tools in isolation: 10 domain Q&A questions, 10 workflow guidance (verify tutorial links returned), 10 query expansion calls (verify filters are reasonable), 5 external literature searches (verify Semantic Scholar returns relevant papers)

### Week 5–6: Integration & UI Connection

- [ ] Connect `assistant_ui.py` to live `assistant.py` backend (chat input → backend intent routing → response rendering)
- [ ] Add PDF upload to assistant UI (reuse `DocumentIngestor` + per-session FAISS; identical to curator flow)
- [ ] Add all intent types to the conversation flow: `search`, `filter`, `educational`, `workflow`, `dataset_explain`, `literature`
- [ ] Test mixed-intent conversation: verify session state isolation, PDF context is used, conversation history flows correctly
- [ ] Add "Sources" expandable panel (dataset IDs, source labels, similarity scores, publication links, portal links; external literature links labeled separately)
- [ ] Add "Learn more" expandable (portal tutorial notebook/doc links, example tools, dataset recommendations)

### Week 7–8: Integration into Rocco & Polish

- [ ] Coordinate with existing `rocco_ui.py`: add tabs (`["Dataset Curator", "General Assistant"]`), namespace session state keys with `curator_` prefix, import and wire `assistant_ui.py` module into tab 2
- [ ] Add sidebar example query suggestions in chat interface ("Try: Find carbonate micro-CT datasets", "Explain voxel size", "How do I measure porosity?")
- [ ] Verify session state does NOT collide between curator tab and assistant tab (separate dictionaries)
- [ ] Write docstrings for `assistant.py`, `query_expander.yaml`, `tutorials.yaml`, `domain_workflows.yaml`
- [ ] (Stretch) Add literature connection: if a paper is uploaded, try to match it to a DPM dataset by abstract similarity or DOI

---

## Shared / Coordination Points

| Milestone | Both interns | Verification |
|-----------|-------------|--------------|
| **Week 2 end** | Align on `src/assistant/tools.py` interface and `assistant_ui.py` layout — Intern A specifies search tool signatures (inputs/outputs); Intern B specifies UI expectations (message format, expandables); decide on session state key naming | Both can run isolated tests of their respective tools without the other's code |
| **Week 4 end** | Demo to supervisor: all backend tools working in isolation (A: search/graph/merge/publication corpus; B: educate/guide/expand/tutorial search/external literature); `assistant_ui.py` prototype ready with mock data | `pytest tests/assistant/` passes; manual test of each tool function independently; `data/tutorial_vector_store/` and `data/assistant_vector_store/` both populated |
| **Week 6 end** | `assistant.py` conversation manager integrates all tools from both interns; `assistant_ui.py` connected to live backend; both integrated into `rocco_ui.py` as a new tab; full app runs without session state collision | `streamlit run rocco_ui.py` → both tabs load; curator tab + assistant tab each work without affecting the other; run 10 mixed queries |
| **Week 8 end** | Final demo: 20–30 representative queries (mixed intent including tutorial search, external literature, workflow guidance); final index rebuild; documentation complete | Test suite includes search, education, degradation (sparse metadata), cross-intent queries, tutorial retrieval, external literature labeled correctly |
