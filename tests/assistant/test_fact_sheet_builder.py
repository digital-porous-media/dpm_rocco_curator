"""
Unit tests for the fact-sheet assembly in scripts/build_dataset_vector_index.py.

Pure functions only — no Neo4j, no embedding calls, no credentials (the script's
neo4j/LLM imports live inside IndexBuilder.__init__, so importing the module is safe).

What these pin down is the reason the fact sheet exists at all: it must preserve WHICH
DigitalDatasets belong to WHICH Sample. The aggregated shape used for embeddings
(_build_embedding_text) deliberately flattens that away, which makes "does this sample
have scans at two different resolutions?" unanswerable — the exact question
reason_about_dataset_content has to answer.
"""

import json

import pytest

from scripts.build_dataset_vector_index import (
    FACT_SHEET_DATASET_DESC_CHARS,
    FACT_SHEET_MAX_NODES_PER_TYPE,
    _build_fact_sheet,
    _render_fact_sheet_text,
)


def _row(**overrides) -> dict:
    """A two-resolution profile row: one Sample feeding two DigitalDatasets at different
    voxel sizes, one of which feeds an AnalysisDataset."""
    row = {
        "dataset": {
            "datasetNumber": 11,
            "title": "Fontainebleau Multiscale Imaging",
            "doi": "10.17612/AAA111",
            "description": "A quartz sandstone core imaged at two magnifications.",
            "license": "CC-BY-4.0",
        },
        "samples": [{
            "identifier": "s1", "title": "Core A", "porousMediaType": "sandstone",
            "description": "Acquired with an Xradia Versa micro-CT scanner.",
            "porosity": 0.21,
        }],
        "digitalDatasets": [
            {"identifier": "dd1", "title": "Core A coarse scan",
             "voxelDimensions": "X, Y, Z units (in micrometers): 5.2, 5.2, 5.2",
             "segmented": "no", "numberOfFiles": 900, "fileTypes": ["tiff"]},
            {"identifier": "dd2", "title": "Core A fine scan",
             "voxelDimensions": "X, Y, Z units (in micrometers): 1.1, 1.1, 1.1",
             "segmented": "yes", "numberOfFiles": 1800, "fileTypes": ["raw"]},
        ],
        "analysisDatasets": [
            {"identifier": "ad1", "title": "Pore network of fine scan", "type": "geometric_analysis"},
        ],
        "relatedPublications": [
            {"title": "Multiscale imaging of Fontainebleau", "abstract": "We image one core twice."},
        ],
        "sampleToDigitalEdges": [
            {"sample": "s1", "digitalDataset": "dd1"},
            {"sample": "s1", "digitalDataset": "dd2"},
        ],
        "digitalToAnalysisEdges": [{"digitalDataset": "dd2", "analysisDataset": "ad1"}],
    }
    row.update(overrides)
    return row


class TestBuildFactSheet:
    def test_keeps_per_subnode_properties_rather_than_aggregating_them(self):
        sheet = _build_fact_sheet(_row())
        voxels = [dd["voxelDimensions"] for dd in sheet["digitalDatasets"]]

        # Both resolutions survive — an aggregation that kept only the first would make
        # "same sample, different resolutions" unanswerable.
        assert len(voxels) == 2
        assert any("5.2" in v for v in voxels)
        assert any("1.1" in v for v in voxels)

    def test_preserves_which_scan_belongs_to_which_sample(self):
        sheet = _build_fact_sheet(_row())
        chains = sheet["pipelines"]

        assert {"sample": "Core A", "digitalDataset": "Core A coarse scan"} in chains
        assert {"sample": "Core A", "digitalDataset": "Core A fine scan",
                "analysisDataset": "Pore network of fine scan"} in chains

    def test_edges_are_labelled_by_title_not_bare_identifier(self):
        sheet = _build_fact_sheet(_row())
        assert all("dd" != c["digitalDataset"][:2] for c in sheet["pipelines"])
        assert "Core A" in json.dumps(sheet["pipelines"])

    def test_drops_fields_with_no_inferential_value(self):
        sheet = _build_fact_sheet(_row())
        blob = json.dumps(sheet)

        assert "numberOfFiles" not in blob
        assert "fileTypes" not in blob
        assert "license" not in blob

    def test_keeps_free_text_where_instruments_are_actually_named(self):
        """The scanner model lives in a Sample's description, not in any structured
        field — dropping descriptions would make that class of question unanswerable."""
        sheet = _build_fact_sheet(_row())
        assert "Xradia Versa" in sheet["samples"][0]["description"]

    def test_empty_values_are_omitted_entirely(self):
        row = _row(samples=[{"identifier": "s1", "title": "Core A",
                             "porousMediaType": None, "description": ""}])
        sheet = _build_fact_sheet(row)

        assert "porousMediaType" not in sheet["samples"][0]
        assert "description" not in sheet["samples"][0]

    def test_long_description_is_truncated_with_an_explicit_marker(self):
        row = _row()
        row["dataset"]["description"] = "word " * 3000
        sheet = _build_fact_sheet(row)

        assert len(sheet["description"]) < FACT_SHEET_DATASET_DESC_CHARS + 100
        assert "truncated" in sheet["description"]

    def test_code_blocks_are_stripped_from_descriptions(self):
        row = _row()
        row["dataset"]["description"] = (
            "A sandstone core.\n```\nimport numpy as np\nvol = np.fromfile('x.raw')\n```\nEnd."
        )
        sheet = _build_fact_sheet(row)

        assert "import numpy" not in sheet["description"]
        assert "A sandstone core." in sheet["description"]

    def test_sub_node_lists_are_capped_with_an_honest_count(self):
        row = _row(digitalDatasets=[
            {"identifier": f"dd{i}", "title": f"Scan {i}", "segmented": "no"}
            for i in range(FACT_SHEET_MAX_NODES_PER_TYPE + 7)
        ], sampleToDigitalEdges=[], digitalToAnalysisEdges=[])
        sheet = _build_fact_sheet(row)

        assert len(sheet["digitalDatasets"]) == FACT_SHEET_MAX_NODES_PER_TYPE
        assert sheet["omittedCounts"]["digitalDatasets"] == 7

    def test_digital_dataset_with_no_sample_link_is_surfaced(self):
        row = _row(sampleToDigitalEdges=[{"sample": "s1", "digitalDataset": "dd1"}])
        sheet = _build_fact_sheet(row)

        assert sheet["unlinkedDigitalDatasets"] == ["Core A fine scan"]

    def test_is_json_serializable(self):
        assert json.loads(json.dumps(_build_fact_sheet(_row()), default=str))

    def test_handles_a_dataset_with_no_sub_nodes(self):
        sheet = _build_fact_sheet({
            "dataset": {"datasetNumber": 5, "title": "Bare Dataset", "doi": "10.1/bare",
                        "description": "Nothing attached."},
        })
        assert sheet["title"] == "Bare Dataset"
        assert sheet["samples"] == [] and sheet["pipelines"] == []


class TestRenderFactSheetText:
    def test_renders_both_resolutions_and_the_structure(self):
        text = _render_fact_sheet_text(_build_fact_sheet(_row()))

        assert "Fontainebleau Multiscale Imaging" in text
        assert "10.17612/AAA111" in text
        assert "5.2" in text and "1.1" in text
        assert "Core A -> Core A fine scan -> Pore network of fine scan" in text

    def test_renders_publication_abstract(self):
        text = _render_fact_sheet_text(_build_fact_sheet(_row()))
        assert "We image one core twice." in text

    def test_counts_include_omitted_nodes(self):
        row = _row(digitalDatasets=[
            {"identifier": f"dd{i}", "title": f"Scan {i}"}
            for i in range(FACT_SHEET_MAX_NODES_PER_TYPE + 3)
        ], sampleToDigitalEdges=[], digitalToAnalysisEdges=[])
        text = _render_fact_sheet_text(_build_fact_sheet(row))

        assert f"({FACT_SHEET_MAX_NODES_PER_TYPE + 3})" in text
        assert "and 3 more not listed here" in text

    def test_bare_dataset_renders_without_error(self):
        text = _render_fact_sheet_text(_build_fact_sheet({
            "dataset": {"datasetNumber": 5, "title": "Bare Dataset", "doi": None,
                        "description": "Nothing attached."},
        }))
        assert "Bare Dataset" in text
        assert "DOI" not in text  # never render an empty DOI


class TestBatchByCharBudget:
    """The embedding endpoint limits TOTAL characters per request, not item count — a
    fixed batch size of 16 (fine for the smaller title+description blobs the other passes
    send) failed on the very first fact-sheet batch against the live endpoint."""

    def test_splits_when_the_budget_would_be_exceeded(self):
        from scripts.build_dataset_vector_index import _batch_by_char_budget
        texts = ["x" * 400] * 10
        batches = _batch_by_char_budget(texts, budget=1000)

        assert [len(b) for b in batches] == [2, 2, 2, 2, 2]
        assert sorted(i for b in batches for i in b) == list(range(10))

    def test_returns_indices_not_texts(self):
        from scripts.build_dataset_vector_index import _batch_by_char_budget
        batches = _batch_by_char_budget(["a", "b", "c"], budget=1000)
        assert batches == [[0, 1, 2]]

    def test_oversized_single_text_gets_its_own_batch_not_dropped(self):
        """A sheet larger than the whole budget must still be embedded — the endpoint
        accepts one large item, it just won't accept several at once."""
        from scripts.build_dataset_vector_index import _batch_by_char_budget
        texts = ["small", "y" * 50_000, "small"]
        batches = _batch_by_char_budget(texts, budget=1000)

        assert [1] in batches
        assert sorted(i for b in batches for i in b) == [0, 1, 2]

    def test_empty_input(self):
        from scripts.build_dataset_vector_index import _batch_by_char_budget
        assert _batch_by_char_budget([], budget=1000) == []

    def test_every_batch_is_within_budget_for_real_sized_sheets(self):
        from scripts.build_dataset_vector_index import (
            _batch_by_char_budget, FACT_SHEET_EMBED_CHAR_BUDGET,
        )
        # Median ~4.5k, p90 ~11k, max ~21k measured across the live 184-dataset corpus.
        texts = ["z" * n for n in (4500, 4500, 11000, 21000, 4500, 900)]
        for batch in _batch_by_char_budget(texts):
            total = sum(len(texts[i]) for i in batch)
            assert total <= FACT_SHEET_EMBED_CHAR_BUDGET or len(batch) == 1
