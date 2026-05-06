# DPM Portal AI Assistant — Project Overview

## Vision

Expand **Rocco** from a description curator into a suite of AI tools that help researchers discover, understand, and work with datasets on the [Digital Porous Media Portal](https://digitalporousmedia.org).

## The Challenge: Incomplete Metadata

Portal datasets are published and cannot be modified. Many metadata fields (porosity, permeability, saturation) are absent or inconsistent. Any assistant built on this data must be **honest about what it doesn't know** — never infer or fabricate missing values.

## Two Work Streams

| Stream | Scope | Owner(s) |
|--------|-------|----------|
| [[01_Rocco_Extension]] | Extend the existing description curator | Ongoing / both interns |
| [[02_General_Assistant]] | **Unified** assistant for dataset discovery + education, with graceful degradation when metadata gaps appear | Intern A (search backend) + Intern B (education backend) |

> **Note:** The General Assistant combines semantic search, metadata filtering, domain Q&A, and workflow guidance into one conversational interface. See [[02_General_Assistant]] for the unified architecture. For historical reference, detailed search and education approaches are documented separately in `docs/02_Search_and_Discovery_Assistant.md` and `docs/03_Educational_and_Research_Assistant.md`.

## Shared Principles

1. **Honest by default** — return `null` for missing fields; never hallucinate metadata values
2. **Build on what exists** — reuse the FAISS vector store, LLM client, and prompt YAML system in `src/`
3. **One coherent assistant, not separate tools** — search, filtering, and education blend in one conversation manager that gracefully handles metadata gaps
4. **Graceful degradation** — when metadata is sparse, the system blends search results with educational context rather than failing silently

## Key Existing Assets

- `src/retriever/retriever.py` — FAISS vector store (reuse for all search)
- `src/llm/client.py` — LLM client (provider-agnostic; supports OpenAI, Anthropic, Ollama, etc.)
- `src/prompts/` — YAML prompt templates with Jinja2 rendering
- DPM Portal API — 176+ datasets with titles, descriptions, and metadata (external corpus)

## Timeline

Two interns, approximately 8 weeks each. See [[04_Tasks]] for week-by-week breakdown.
