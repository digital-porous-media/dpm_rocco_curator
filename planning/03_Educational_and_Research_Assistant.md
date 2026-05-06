# Educational & Research Assistant

Related: [[00_Project_Overview]], [[02_General_Assistant]]

> **Note:** This document captures the original education-only design. The current architecture is a **unified assistant** — see [[02_General_Assistant]] for the live spec. Key additions beyond what's described here: portal tutorial index (Jupyter notebooks + HTML docs), domain workflow knowledge base (`domain_workflows.yaml`), and live literature search via Semantic Scholar API.

**Assigned to:** Intern B  
**Goal:** Help users understand porous media concepts, imaging methods, and analysis workflows — even when portal metadata is incomplete.

---

## Problem

Many DPM Portal users are students or researchers new to digital rock physics. They arrive knowing what they want to study (e.g., "permeability from micro-CT") but not how to get started. The portal has no guidance layer — just a data catalog. This assistant fills that gap.

## What This Assistant Can Do

### Domain Knowledge Q&A
Answer questions using embedded knowledge about porous media, not portal-specific data:
- *"What is a typical porosity range for Berea sandstone?"*
- *"What does voxel size mean for my simulation?"*
- *"What's the difference between micro-CT and FIB-SEM?"*
- *"How does imaging voltage affect image quality?"*

The LLM draws on training knowledge + optional RAG from ingested reference papers. It does **not** invent dataset-specific values.

### Workflow Guidance
Map user goals to concrete next steps:
- *"I want to measure porosity"* → tutorial link + recommended analysis approach + example datasets used for this task
- *"I want to run a LBM flow simulation"* → recommended tools, preprocessing steps, example datasets
- *"How do I segment a micro-CT image?"* → link to DPM tutorial, recommended software

Backed by a handwritten `data/tutorials.yaml` that maps goals to URLs and example dataset IDs.

### Literature Connections
If an uploaded paper is provided, the assistant can:
- Summarize the methods and materials used
- Link to the corresponding DPM dataset if one is referenced
- Answer questions about the paper's methodology

This reuses the existing RAG pipeline from `rocco_ui.py` (PDF → FAISS → retrieval).

### "What can I do with this dataset?"
Given a dataset ID, the assistant explains:
- What type of analysis is appropriate given the imaging modality and resolution
- Which DPM tutorials or tools are relevant
- What published papers have used similar data

---

## Architecture

```
assistant_ui.py (Streamlit tab)
    ↓
src/assistant/edu_assistant.py     — conversation manager
    ↓
src/assistant/tools.py
    ├── get_educational_context(topic)    → LLM + educational.yaml + optional RAG
    └── get_workflow_guidance(goal)       → data/tutorials.yaml lookup + LLM synthesis
```

### RAG for Reference Papers

Reuses the **existing** `DocumentIngestor` + `VectorStoreManager` pipeline from `rocco_ui.py`:
- User uploads a paper (PDF)
- Chunks are embedded and stored in a per-session FAISS index
- Educational Q&A is grounded in retrieved chunks + LLM domain knowledge

### Conversation Flow

1. **Intent classify** — `{"intent": "domain_qa|workflow|paper_qa|dataset_explain", "params": {...}}`
2. **Tool dispatch** — `get_educational_context` or `get_workflow_guidance`
3. **Synthesize** — LLM assembles grounded response; for domain_qa it may draw on training knowledge freely; for portal data it must cite source
4. **Render** — chat response + "Learn more" expandable with tutorial links

---

## New Files

| File | Purpose |
|------|---------|
| `src/assistant/edu_assistant.py` | Conversation manager for this tab |
| `src/prompts/educational.yaml` | Domain Q&A and workflow guidance prompts |
| `data/tutorials.yaml` | Handwritten: ~20 user goals → tutorial URLs + example dataset IDs |

## Reuse

- `src/ingestor/document_ingestor.py` → PDF chunking (identical to existing curator flow)
- `src/retriever/retriever.py` → per-session FAISS for uploaded papers
- `src/llm/client.py` → `LLMClient.send_prompt`
- `src/prompts/loader.py` → `load_prompt` / `render`
- `src/assistant/tools.py` → shared with Search & Discovery intern (coordinate on this file)

## `data/tutorials.yaml` Structure

```yaml
- goal: "measure porosity"
  keywords: ["porosity", "pore volume", "void fraction"]
  tutorial_url: "https://digitalporousmedia.org/tutorials/porosity"
  recommended_tools: ["ImageJ", "Dragonfly", "porespy"]
  example_datasets: ["DPMP-461", "DPMP-523"]
  notes: "Requires image segmentation first."

- goal: "run LBM simulation"
  keywords: ["lattice boltzmann", "LBM", "permeability simulation", "flow simulation"]
  tutorial_url: "https://digitalporousmedia.org/tutorials/lbm"
  recommended_tools: ["palabos", "OpenLB"]
  example_datasets: ["DPMP-461"]
  notes: "Segmented binary images required. Voxel size should be < 5 µm for pore-scale accuracy."
```

## Tasks

See [[04_Tasks#Intern-B-Educational-and-Research]].
