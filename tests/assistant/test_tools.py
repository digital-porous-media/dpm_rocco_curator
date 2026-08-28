"""
Unit tests for src/assistant/tools.py.

All LLM calls and external API calls are mocked — no credentials required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.assistant.literature_search import Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_chat_model(return_text: str) -> MagicMock:
    """Return a mock RoccoClient whose send_prompt() returns return_text."""
    mock = MagicMock()
    mock.send_prompt.return_value = return_text
    return mock


def _fake_paper(title: str = "Test Paper", doi: str = "10.1234/test") -> Paper:
    return Paper(
        title=title,
        authors=["Author A", "Author B", "Author C", "Author D"],
        abstract="An abstract about porous media.",
        year=2023,
        doi=doi,
        citation_count=10,
        pdf_url=None,
        url="https://example.com",
    )


# ---------------------------------------------------------------------------
# expand_query
# ---------------------------------------------------------------------------

class TestExpandQuery:
    def test_returns_dict_with_expected_keys(self):
        payload = json.dumps({
            "expanded_query": "sandstone porous media low porosity",
            "inferred_filters": {"rock_type": "sandstone"},
            "rationale": "Added domain terms.",
        })
        with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(payload)):
            from src.assistant.tools import expand_query
            result = expand_query("sandstone with low porosity")

        assert isinstance(result, dict)
        assert "expanded_query" in result
        assert "inferred_filters" in result
        assert "rationale" in result

    def test_parse_error_returns_passthrough(self):
        with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model("not json at all")):
            from src.assistant.tools import expand_query
            result = expand_query("my original query")

        assert result["expanded_query"] == "my original query"
        assert result["inferred_filters"] == {}
        assert "rationale" in result

    def test_expanded_query_differs_from_input(self):
        payload = json.dumps({
            "expanded_query": "sandstone porous medium low porosity quartz clastic",
            "inferred_filters": {"rock_type": "sandstone"},
            "rationale": "Expanded.",
        })
        with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(payload)):
            from src.assistant.tools import expand_query
            result = expand_query("sandstone")

        assert len(result["expanded_query"]) > len("sandstone")


# ---------------------------------------------------------------------------
# get_workflow_guidance
# ---------------------------------------------------------------------------

class TestGetWorkflowGuidance:
    def test_returns_string_for_known_keyword(self):
        with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model("Step 1: ...")):
            from src.assistant.tools import get_workflow_guidance
            result = get_workflow_guidance.func("permeability")  # .func bypasses @tool wrapper

        assert isinstance(result, str)
        assert len(result) > 0

    def test_llm_called_for_final_answer(self):
        """
        get_workflow_guidance makes two send_prompt calls: an internal
        _match_workflows() semantic ranking call, then the final answer-synthesis
        call. Assert the LLM was invoked and the final call is the answer call
        (its user prompt echoes the question) rather than asserting an exact
        count, since the ranking call is an internal implementation detail.
        """
        mock_llm = _mock_chat_model("Guidance response.")
        with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
            from src.assistant.tools import get_workflow_guidance
            get_workflow_guidance.func("lattice Boltzmann permeability")

        mock_llm.send_prompt.assert_called()
        last_call = mock_llm.send_prompt.call_args
        last_user_arg = last_call[0][0] if last_call[0] else last_call[1].get("user")
        assert "lattice Boltzmann permeability" in last_user_arg

    def test_no_keyword_match_still_calls_llm(self):
        """Unrecognised queries fall back to LLM with empty context (pre-trained knowledge)."""
        mock_llm = _mock_chat_model("I don't have portal-specific data on this, but generally…")
        with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
            from src.assistant.tools import get_workflow_guidance
            result = get_workflow_guidance.func("completely unrelated xyz query")

        mock_llm.send_prompt.assert_called()
        assert isinstance(result, str)

    def test_context_includes_workflow_info_for_keyword_match(self):
        """When a keyword matches, the context passed to the LLM contains workflow content."""
        mock_llm = _mock_chat_model("Response.")
        with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
            from src.assistant.tools import get_workflow_guidance
            get_workflow_guidance.func("segmentation threshold")

        call_kwargs = mock_llm.send_prompt.call_args
        context_arg = call_kwargs[1].get("context") or call_kwargs[0][1]
        assert "Workflow" in context_arg or "segmentation" in context_arg.lower()


class TestStripFabricatedTutorialReference:
    """Guard against notebook-path hallucination, including when it's mixed in
    alongside genuinely retrieved tutorials (not just the no-match case)."""

    def test_no_tutorials_matched_strips_fabricated_block(self):
        from src.assistant.tools import _strip_fabricated_tutorial_reference, _HONEST_NO_TUTORIAL_MSG
        response = (
            "Here you go:\n"
            "**Goal:** Compute tortuosity\n"
            "**Notebook:** `4_image_processing/4-2-5_distance_transform.ipynb`\n"
        )
        result = _strip_fabricated_tutorial_reference(response, tutorials=[])
        assert ".ipynb" not in result
        assert _HONEST_NO_TUTORIAL_MSG in result

    def test_fabricated_path_stripped_even_with_real_tutorial_present(self):
        from src.assistant.tools import _strip_fabricated_tutorial_reference
        real_tutorials = [{"goal": "Simulate LBM permeability", "notebook": "5_simulation/5-2-1_lbm_d2q9_bgk.ipynb"}]
        response = (
            "**Goal:** Simulate LBM permeability\n"
            "**Notebook:** `5_simulation/5-2-1_lbm_d2q9_bgk.ipynb`\n"
            "**Goal:** Compute tortuosity\n"
            "**Notebook:** `4_image_processing/4-2-5_distance_transform.ipynb`\n"
        )
        result = _strip_fabricated_tutorial_reference(response, real_tutorials)
        assert "5-2-1_lbm_d2q9_bgk.ipynb" in result
        assert "4-2-5_distance_transform.ipynb" not in result

    def test_inline_fabricated_path_outside_block_format_is_stripped(self):
        from src.assistant.tools import _strip_fabricated_tutorial_reference
        real_tutorials = [{"goal": "Simulate LBM permeability", "notebook": "5_simulation/5-2-1_lbm_d2q9_bgk.ipynb"}]
        response = (
            "See 5_simulation/5-2-1_lbm_d2q9_bgk.ipynb for permeability, and "
            "5_simulation/5-2-3_lbm_d3q7_conductivity.ipynb for conductivity."
        )
        result = _strip_fabricated_tutorial_reference(response, real_tutorials)
        assert "5-2-1_lbm_d2q9_bgk.ipynb" in result
        assert "5-2-3_lbm_d3q7_conductivity.ipynb" not in result

    def test_real_paths_untouched(self):
        from src.assistant.tools import _strip_fabricated_tutorial_reference
        real_tutorials = [{"goal": "g", "notebook": "5_simulation/5-2-1_lbm_d2q9_bgk.ipynb"}]
        response = "**Goal:** g\n**Notebook:** `5_simulation/5-2-1_lbm_d2q9_bgk.ipynb`\n"
        result = _strip_fabricated_tutorial_reference(response, real_tutorials)
        assert result == response.strip()


class TestEnsureAllTutorialsMentioned:
    """Complement to _strip_fabricated_tutorial_reference: when more than one
    tutorial matches (e.g. both Minkowski Functionals and Connected Components for
    "compute porosity from an image"), educational.yaml's "list every matched
    tutorial explicitly" instruction was observed dropping one anyway (live, 2/4 runs,
    same query/context/prompt — pure model content-selection variance). Deterministic
    append closes the gap the same way the strip guard closes hallucination."""

    def test_appends_missing_tutorial_not_mentioned_by_the_model(self):
        from src.assistant.tools import _ensure_all_tutorials_mentioned
        tutorials = [
            {"goal": "Characterize pore morphology with Minkowski functionals",
             "notebook": "3_morphological_characterization/3-2_minkowski_functionals.ipynb"},
            {"goal": "Identify connected pore space and measure connectivity",
             "notebook": "3_morphological_characterization/3-6_connected_components.ipynb"},
        ]
        response = (
            "**Goal:** Identify connected pore space and measure connectivity\n"
            "**Notebook:** `3_morphological_characterization/3-6_connected_components.ipynb`\n"
        )
        result = _ensure_all_tutorials_mentioned(response, tutorials)
        assert "3-6_connected_components.ipynb" in result
        assert "3-2_minkowski_functionals.ipynb" in result

    def test_noop_when_all_tutorials_already_mentioned(self):
        from src.assistant.tools import _ensure_all_tutorials_mentioned
        tutorials = [{"goal": "g", "notebook": "5_simulation/5-2-1_lbm_d2q9_bgk.ipynb"}]
        response = "**Goal:** g\n**Notebook:** `5_simulation/5-2-1_lbm_d2q9_bgk.ipynb`\n"
        assert _ensure_all_tutorials_mentioned(response, tutorials) == response

    def test_noop_when_no_tutorials_matched(self):
        from src.assistant.tools import _ensure_all_tutorials_mentioned
        response = "We don't currently have a dedicated tutorial for this topic."
        assert _ensure_all_tutorials_mentioned(response, []) == response

    def test_get_workflow_guidance_wires_in_the_guard(self):
        """End-to-end: get_workflow_guidance must apply the completeness guard on top
        of whatever the synthesis LLM returns."""
        from src.assistant.tools import get_workflow_guidance
        tutorials = [
            {"goal": "Characterize pore morphology with Minkowski functionals",
             "notebook": "3_morphological_characterization/3-2_minkowski_functionals.ipynb"},
            {"goal": "Identify connected pore space and measure connectivity",
             "notebook": "3_morphological_characterization/3-6_connected_components.ipynb"},
        ]
        incomplete_response = (
            "**Goal:** Identify connected pore space and measure connectivity\n"
            "**Notebook:** `3_morphological_characterization/3-6_connected_components.ipynb`\n"
        )
        with patch("src.assistant.tools._match_tutorials", return_value=tutorials), \
             patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(incomplete_response)):
            result = get_workflow_guidance.func("compute porosity from an image")

        assert "3-6_connected_components.ipynb" in result
        assert "3-2_minkowski_functionals.ipynb" in result


# ---------------------------------------------------------------------------
# get_educational_context
# ---------------------------------------------------------------------------

class TestGetEducationalContext:
    def test_returns_string(self):
        with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model("Porosity is...")):
            from src.assistant.tools import get_educational_context
            result = get_educational_context.func("What is porosity?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_llm_called_for_final_answer(self):
        """
        get_educational_context also makes an internal _match_workflows() ranking
        call before the answer-synthesis call — see TestGetWorkflowGuidance's
        equivalent test for why this doesn't assert an exact call count.
        """
        mock_llm = _mock_chat_model("Educational response.")
        with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
            from src.assistant.tools import get_educational_context
            get_educational_context.func("what is REV?")

        mock_llm.send_prompt.assert_called()
        last_call = mock_llm.send_prompt.call_args
        last_user_arg = last_call[0][0] if last_call[0] else last_call[1].get("user")
        assert "what is REV?" in last_user_arg

    def test_rev_query_includes_global_best_practices(self):
        """A query about REV should pull the 'representativeness' global best practice."""
        mock_llm = _mock_chat_model("Response.")
        with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
            from src.assistant.tools import get_educational_context
            get_educational_context.func("What is the Representative Elementary Volume?")

        call_kwargs = mock_llm.send_prompt.call_args
        context_arg = call_kwargs[1].get("context") or call_kwargs[0][1]
        assert "representativeness" in context_arg.lower() or "REV" in context_arg


# ---------------------------------------------------------------------------
# get_dataset_profile
# ---------------------------------------------------------------------------

def _fake_profile():
    from src.assistant.graph_store import DatasetProfileMatch
    return DatasetProfileMatch(
        dataset={"datasetNumber": 42, "title": "Bentheimer Sandstone", "doi": "10.1234/drp42", "description": "A sandstone dataset."},
        samples=[{"identifier": "s1", "title": "Core 1", "porousMediaType": "sandstone", "porosity": 0.21, "waterDepth": None}],
        digital_datasets=[{"identifier": "dd1", "title": "Scan 1", "fileTypes": ["tiff"], "imageFormat": "TIFF", "imageByteOrder": ""}],
        analysis_datasets=[{"identifier": "ad1", "title": "PNM 1", "type": "geometric_analysis"}],
        related_publications=[],
        related_software=[],
        related_datasets=[],
        sample_to_digital_edges=[{"sample": "s1", "digitalDataset": "dd1"}],
        digital_to_analysis_edges=[{"digitalDataset": "dd1", "analysisDataset": "ad1"}],
    )


class TestGetDatasetProfile:
    def test_general_profile_returns_string_with_source_label(self):
        mock_llm = _mock_chat_model("This dataset is a Bentheimer sandstone core scan.")
        with patch("src.assistant.tools._get_graph_store") as mock_factory:
            mock_factory.return_value.get_dataset_profile.return_value = _fake_profile()
            with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
                from src.assistant.tools import get_dataset_profile
                result = get_dataset_profile.func("Bentheimer Sandstone", "tell me more about this dataset")

        assert "[dataset profile]" in result
        assert "Bentheimer Sandstone" in result
        assert "10.1234/drp42" in result

    def test_llm_called_with_question_and_context(self):
        mock_llm = _mock_chat_model("Response.")
        with patch("src.assistant.tools._get_graph_store") as mock_factory:
            mock_factory.return_value.get_dataset_profile.return_value = _fake_profile()
            with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
                from src.assistant.tools import get_dataset_profile
                get_dataset_profile.func("Bentheimer Sandstone", "what is its porosity?")

        call_kwargs = mock_llm.send_prompt.call_args
        user_arg = call_kwargs[0][0]
        context_arg = call_kwargs[1].get("context") or call_kwargs[0][1]
        assert "what is its porosity?" in user_arg
        assert "porosity" in context_arg  # real field from the mocked profile

    def test_file_reasoning_question_passes_file_types_in_context(self):
        mock_llm = _mock_chat_model("Response.")
        with patch("src.assistant.tools._get_graph_store") as mock_factory:
            mock_factory.return_value.get_dataset_profile.return_value = _fake_profile()
            with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
                from src.assistant.tools import get_dataset_profile
                get_dataset_profile.func("Bentheimer Sandstone", "how do I read this dataset's files in Python?")

        context_arg = mock_llm.send_prompt.call_args[1].get("context")
        assert "tiff" in context_arg.lower()
        assert "imageFormat" in context_arg
        # datasetNumber 42 -> a real, constructed archive URL should be present.
        assert "archive/DRP-42/" in context_arg

    def test_ambiguous_match_returns_disambiguation_without_llm_call(self):
        from src.assistant.graph_store import DatasetProfileAmbiguous
        ambiguous = DatasetProfileAmbiguous(candidates=[
            {"datasetNumber": 1, "title": "Bentheimer A", "doi": "10.1/a"},
            {"datasetNumber": 2, "title": "Bentheimer B", "doi": "10.1/b"},
        ])
        mock_llm = _mock_chat_model("should not be called")
        with patch("src.assistant.tools._get_graph_store") as mock_factory:
            mock_factory.return_value.get_dataset_profile.return_value = ambiguous
            with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
                from src.assistant.tools import get_dataset_profile
                result = get_dataset_profile.func("Bentheimer", "tell me more")

        mock_llm.send_prompt.assert_not_called()
        assert "Bentheimer A" in result and "Bentheimer B" in result

    def test_not_found_returns_message_without_llm_call(self):
        mock_llm = _mock_chat_model("should not be called")
        with patch("src.assistant.tools._get_graph_store") as mock_factory:
            mock_factory.return_value.get_dataset_profile.return_value = None
            with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
                from src.assistant.tools import get_dataset_profile
                result = get_dataset_profile.func("Nonexistent Dataset", "tell me more")

        mock_llm.send_prompt.assert_not_called()
        assert "No dataset was found" in result
        assert "Nonexistent Dataset" in result

    def test_use_neo4j_false_graceful_fallback(self):
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.tools._graph_store", None):
                with patch("src.assistant.graph_store.GraphStore.get_dataset_profile", return_value=None):
                    from src.assistant.tools import get_dataset_profile
                    result = get_dataset_profile.func("Bentheimer Sandstone", "tell me more")

        assert isinstance(result, str)
        assert "No dataset was found" in result


class TestBuildProfileContext:
    def test_omits_empty_none_and_falsy_fields(self):
        from src.assistant.tools import _build_profile_context
        context = _build_profile_context(_fake_profile())

        # waterDepth: None and imageByteOrder: "" in the fixture must never appear.
        assert "waterDepth" not in context
        assert "imageByteOrder" not in context

    def test_populated_fields_present(self):
        from src.assistant.tools import _build_profile_context
        context = _build_profile_context(_fake_profile())

        assert "porosity" in context
        assert "fileTypes" in context
        assert "Core 1" in context
        assert "Scan 1" in context
        assert "PNM 1" in context

    def test_organizational_structure_chain_rendered(self):
        from src.assistant.tools import _build_profile_context
        context = _build_profile_context(_fake_profile())
        assert "Core 1" in context and "Scan 1" in context and "PNM 1" in context
        assert "->" in context

    def test_large_sub_node_count_is_capped_not_unbounded(self):
        """A dataset with far more sub-nodes than typical must not blow up context size —
        this is the guard against the reported context-window-exceeded failure, which
        reproduced with no prior conversation history (a single oversized profile call is
        enough)."""
        from src.assistant.tools import _build_profile_context, _MAX_NODES_PER_TYPE
        from src.assistant.graph_store import DatasetProfileMatch

        many_samples = [
            {"identifier": f"s{i}", "title": f"Core {i}", "porousMediaType": "sandstone"}
            for i in range(500)
        ]
        profile = DatasetProfileMatch(
            dataset={"datasetNumber": 1, "title": "Huge Collection", "doi": "10.1/huge"},
            samples=many_samples,
            digital_datasets=[],
            analysis_datasets=[],
            related_publications=[],
            related_software=[],
            related_datasets=[],
            sample_to_digital_edges=[],
            digital_to_analysis_edges=[],
        )
        context = _build_profile_context(profile)

        assert context.count("Core ") <= _MAX_NODES_PER_TYPE
        assert "more samples not shown" in context.lower()

    def test_embedding_vectors_are_stripped_from_context(self):
        """Reproduces the reported context-window-exceeded bug: a real dataset's Dataset/
        Sample/DigitalDataset nodes carry a 4096-float datasetEmbedding/componentEmbedding
        vector (populated by scripts/build_dataset_vector_index.py for every dataset/
        sub-node) which, left in, is alone enough to blow the model's context window on a
        single call — no large sub-node count or prior history required."""
        from src.assistant.tools import _build_profile_context
        from src.assistant.graph_store import DatasetProfileMatch

        fake_vector = [0.123456] * 4096
        profile = DatasetProfileMatch(
            dataset={
                "datasetNumber": 10, "title": "Bentheimer Sandstone", "doi": "10.17612/P77P49",
                "datasetEmbedding": fake_vector,
            },
            samples=[{"identifier": "s1", "title": "Core 1", "componentEmbedding": fake_vector}],
            digital_datasets=[{"identifier": "dd1", "title": "Scan 1", "componentEmbedding": fake_vector}],
            analysis_datasets=[],
            related_publications=[],
            related_software=[],
            related_datasets=[],
            sample_to_digital_edges=[],
            digital_to_analysis_edges=[],
        )
        context = _build_profile_context(profile)

        assert "0.123456" not in context
        assert "datasetEmbedding" not in context
        assert "componentEmbedding" not in context
        # Real metadata on the same nodes must still survive the strip.
        assert "Bentheimer Sandstone" in context
        assert "Core 1" in context
        assert "Scan 1" in context
        # A stripped 4096-float vector must not leave the context anywhere near that size.
        assert len(context) < 5000


# ---------------------------------------------------------------------------
# reason_about_dataset_content
# ---------------------------------------------------------------------------

# Three fact sheets covering the three cases this tool exists for: a resolution
# comparison across one sample's sub-nodes, a scanner named only in free text, and a
# pairing implied by methodology (a segmented and an unsegmented scan of one sample).
# None of these is answerable by a literal field lookup.
_RESOLUTION_SHEET = {
    "datasetNumber": 11,
    "title": "Fontainebleau Multiscale Imaging",
    "doi": "10.17612/AAA111",
    "factSheetText": (
        "Dataset 11: Fontainebleau Multiscale Imaging (DOI: 10.17612/AAA111)\n"
        "Description: A quartz sandstone core imaged at two magnifications.\n\n"
        "Samples (1):\n- Core A — porousMediaType: sandstone\n\n"
        "Digital datasets (images/scans) (2):\n"
        "- Core A coarse scan — voxelDimensions: X, Y, Z units (in micrometers): 5.2, 5.2, 5.2\n"
        "- Core A fine scan — voxelDimensions: X, Y, Z units (in micrometers): 1.1, 1.1, 1.1\n\n"
        "Structure (Sample -> Digital dataset -> Analysis dataset):\n"
        "- Core A -> Core A coarse scan\n- Core A -> Core A fine scan"
    ),
}
_SCANNER_SHEET = {
    "datasetNumber": 22,
    "title": "Estaillades Carbonate Core",
    "doi": "10.17612/BBB222",
    "factSheetText": (
        "Dataset 22: Estaillades Carbonate Core (DOI: 10.17612/BBB222)\n"
        "Description: A carbonate core from southern France.\n\n"
        "Samples (1):\n- Estaillades plug — porousMediaType: carbonate\n"
        "  Description: The plug was acquired with an Xradia Versa micro-CT scanner at 40 kV."
    ),
}
_PAIRING_SHEET = {
    "datasetNumber": 33,
    "title": "Berea Segmentation Benchmark",
    "doi": "10.17612/CCC333",
    "factSheetText": (
        "Dataset 33: Berea Segmentation Benchmark (DOI: 10.17612/CCC333)\n"
        "Description: Greyscale and hand-segmented volumes of one Berea plug.\n\n"
        "Samples (1):\n- Berea plug — porousMediaType: sandstone\n\n"
        "Digital datasets (images/scans) (2):\n"
        "- Berea greyscale volume — segmented: no\n"
        "- Berea segmented volume — segmented: yes\n\n"
        "Structure (Sample -> Digital dataset -> Analysis dataset):\n"
        "- Berea plug -> Berea greyscale volume\n- Berea plug -> Berea segmented volume"
    ),
}
_ALL_SHEETS = [_RESOLUTION_SHEET, _SCANNER_SHEET, _PAIRING_SHEET]


def _mock_fact_sheet_store(records=None, ranked=None):
    """A mocked GraphStore whose fact-sheet ranking/fetch return the fixtures above."""
    store = MagicMock()
    records = _ALL_SHEETS if records is None else records
    store.rank_fact_sheets.return_value = (
        ranked if ranked is not None else [r["datasetNumber"] for r in records]
    )
    store.fetch_fact_sheets.return_value = records
    return store


class TestNeedsContentReasoning:
    """The deterministic gate that decides 'plain literal field' vs. 'everything else'.
    A wrong call here reproduces the original bug (a bare segmented='yes' list presented
    as an answer to a 'paired' question), so both directions are pinned down."""

    @pytest.mark.parametrize("question", [
        "Are there paired tomographic and segmented images?",
        "Which datasets image the same sample at different resolutions?",
        "Find datasets with a segmented version of the same scan",
        "Are there datasets with corresponding simulation outputs?",
        "Which datasets have data derived from the same core?",
        "Do any datasets have both greyscale and segmented images?",
        "Show me before and after segmentation volumes",
    ])
    def test_relational_questions_fire(self, question):
        from src.assistant.tools import _needs_content_reasoning
        assert _needs_content_reasoning(question) is True

    @pytest.mark.parametrize("question", [
        "sandstone datasets with porosity above 0.3",
        "How many segmented carbonate datasets are there?",
        "datasets by Jane Doe",
        "Show me coal samples with voxel size smaller than 2 microns",
        "Which datasets are segmented?",
        "Find datasets suitable for LBM simulation",
    ])
    def test_plain_and_suitability_questions_do_not_fire(self, question):
        from src.assistant.tools import _needs_content_reasoning
        assert _needs_content_reasoning(question) is False

    @pytest.mark.parametrize("noun", [
        "volumes", "stacks", "tomograms", "reconstructions", "segmentations", "files", "data",
    ])
    def test_both_x_and_y_fires_regardless_of_the_artifact_noun(self, noun):
        """Live gap: "both grayscale and segmented VOLUMES" did not fire while the
        near-identical "...IMAGES" did, so an equally relational question fell through to
        Cypher and produced a plausible-looking partial answer with nothing signalling that
        the "both" half was never checked. The head noun must not decide this."""
        from src.assistant.tools import _needs_content_reasoning
        assert _needs_content_reasoning(
            f"List every dataset that has both grayscale and segmented {noun}"
        ) is True

    def test_the_noun_list_still_brakes_a_plain_two_item_conjunction(self):
        """The noun list is the only thing stopping "both X and Y" from matching any
        conjunction at all — widening it must not cost that brake."""
        from src.assistant.tools import _needs_content_reasoning
        assert _needs_content_reasoning(
            "datasets with a voxel size below 5 microns and porosity above 0.2"
        ) is False


class TestReasonAboutDatasetContent:
    def _run(self, question, llm_payload, store=None):
        with patch("src.assistant.tools._get_graph_store",
                   return_value=store or _mock_fact_sheet_store()):
            with patch("src.assistant.llm.get_chat_model",
                       return_value=_mock_chat_model(llm_payload)):
                from src.assistant.tools import reason_about_dataset_content
                return reason_about_dataset_content.func(question)

    def test_resolution_case_surfaces_with_citation_and_framing(self):
        payload = json.dumps({"candidates": [{
            "title": "Fontainebleau Multiscale Imaging",
            "reason": "Core A has two scans recorded at different voxel sizes.",
            "citation": "Core A coarse scan — 5.2 micrometers; Core A fine scan — 1.1 micrometers",
        }]})
        result = self._run("datasets imaging the same sample at different resolutions", payload)

        assert "[content reasoning]" in result
        assert "I can't confirm this from a database field" in result
        assert "Fontainebleau Multiscale Imaging" in result
        assert "10.17612/AAA111" in result
        assert "Basis:" in result
        assert "1.1 micrometers" in result

    def test_scanner_named_in_description_case(self):
        payload = json.dumps({"candidates": [{
            "title": "Estaillades Carbonate Core",
            "reason": "The sample description names the scanner directly.",
            "citation": "The plug was acquired with an Xradia Versa micro-CT scanner at 40 kV.",
        }]})
        result = self._run("which datasets were imaged on the same Xradia scanner?", payload)

        assert "I can't confirm this from a database field" in result
        assert "Estaillades Carbonate Core" in result
        assert "Xradia Versa" in result

    def test_methodology_implies_pairing_case(self):
        payload = json.dumps({"candidates": [{
            "title": "Berea Segmentation Benchmark",
            "reason": "One sample has both an unsegmented and a segmented volume recorded.",
            "citation": "Berea greyscale volume — segmented: no; Berea segmented volume — segmented: yes",
        }]})
        result = self._run("are there paired tomographic and segmented images?", payload)

        assert "I can't confirm this from a database field" in result
        assert "Berea Segmentation Benchmark" in result
        assert "segmented: yes" in result

    def test_uncited_candidate_is_dropped(self):
        """No citation, no candidate — enforced in code, not left to the prompt."""
        payload = json.dumps({"candidates": [
            {"title": "Fontainebleau Multiscale Imaging", "reason": "Seems likely.", "citation": ""},
            {"title": "Berea Segmentation Benchmark", "reason": "Has both volumes.",
             "citation": "Berea segmented volume — segmented: yes"},
        ]})
        result = self._run("paired segmented images", payload)

        assert "Fontainebleau Multiscale Imaging" not in result
        assert "Berea Segmentation Benchmark" in result

    def test_candidate_not_in_shortlist_is_dropped(self):
        """The model must not be able to introduce a dataset it was never shown."""
        payload = json.dumps({"candidates": [
            {"title": "Totally Invented Dataset", "reason": "Made up.",
             "citation": "some plausible-sounding quote"},
        ]})
        result = self._run("paired segmented images", payload)

        assert "Totally Invented Dataset" not in result
        assert "couldn't find a dataset that plausibly matches" in result

    def test_no_candidates_returns_honest_empty_answer(self):
        result = self._run("paired segmented images", json.dumps({"candidates": []}))
        assert "[content reasoning]" in result
        assert "couldn't find a dataset that plausibly matches" in result

    def test_unparseable_llm_response_is_not_reported_as_no_matches(self):
        """A parse failure must not be presented as "nothing matches" — that states a
        negative finding that was never established, the exact overclaim this tool exists
        to remove."""
        result = self._run("paired segmented images", "I think probably Berea, honestly")
        assert "couldn't find a dataset that plausibly matches" not in result
        assert "internal formatting failure" in result

    def test_title_with_doi_appended_still_resolves(self):
        """Live regression: the model echoes the fact sheet's own header format and returns
        "<title> (DOI: ...)". Matching that literally against the bare title dropped EVERY
        correctly-cited candidate, turning a good answer into a false "nothing matches"."""
        payload = json.dumps({"candidates": [{
            "title": "Berea Segmentation Benchmark (DOI: 10.17612/CCC333)",
            "reason": "Both volumes present.",
            "citation": "Berea segmented volume — segmented: yes",
        }]})
        result = self._run("paired segmented images", payload)

        assert "Berea Segmentation Benchmark" in result
        assert "10.17612/CCC333" in result

    def test_title_resolved_by_doi_even_when_the_title_text_is_wrong(self):
        payload = json.dumps({"candidates": [{
            "title": "Some Mangled Title (DOI: 10.17612/CCC333)",
            "reason": "Both volumes present.",
            "citation": "Berea segmented volume — segmented: yes",
        }]})
        result = self._run("paired segmented images", payload)
        assert "Berea Segmentation Benchmark" in result

    def test_truncated_title_still_resolves_when_unambiguous(self):
        payload = json.dumps({"candidates": [{
            "title": "Berea Segmentation",
            "reason": "Both volumes present.",
            "citation": "Berea segmented volume — segmented: yes",
        }]})
        result = self._run("paired segmented images", payload)
        assert "Berea Segmentation Benchmark" in result

    def test_truncated_response_salvages_complete_candidates(self):
        """Live regression: a long shortlist with long citations exhausted max_tokens partway
        through the JSON array. The candidates already emitted are complete and cited —
        discarding all of them because a later one was cut off throws away a real answer."""
        truncated = (
            '{\n  "candidates": [\n'
            '    {"title": "Berea Segmentation Benchmark", "reason": "Both volumes present.",'
            ' "citation": "Berea segmented volume — segmented: yes"},\n'
            '    {"title": "Fontainebleau Multiscale Imaging", "reason": "Two voxel sizes.",'
            ' "citation": "Core A fine scan — 1.1 micrometers"},\n'
            '    {"title": "Estaillades Carbonate Core", "reason": "cut off here", "citat'
        )
        result = self._run("paired segmented images", truncated)

        assert "Berea Segmentation Benchmark" in result
        assert "Fontainebleau Multiscale Imaging" in result
        # The incomplete third object must not appear...
        assert "cut off here" not in result
        # ...and the list must never read as complete.
        assert "cut off before it finished" in result

    def test_salvage_still_enforces_grounding(self):
        """Looser parsing must not mean looser grounding: a salvaged candidate still has to
        be cited and still has to be one of the fact sheets actually sent."""
        truncated = (
            '{\n  "candidates": [\n'
            '    {"title": "Totally Invented Dataset", "reason": "made up.",'
            ' "citation": "plausible quote"},\n'
            '    {"title": "Berea Segmentation Benchmark", "reason": "no citation", "citation": ""},\n'
            '    {"title": "Estaillades Carbonate Core", "reason": "cut'
        )
        result = self._run("paired segmented images", truncated)

        assert "Totally Invented Dataset" not in result
        assert "couldn't find a dataset that plausibly matches" in result

    def test_salvage_returns_none_when_nothing_is_complete(self):
        from src.assistant.tools import _parse_reasoning_response
        assert _parse_reasoning_response('{"candidates": [{"title": "Ber') is None

    def test_same_dataset_named_twice_is_shown_once(self):
        payload = json.dumps({"candidates": [
            {"title": "Berea Segmentation Benchmark", "reason": "a",
             "citation": "Berea segmented volume — segmented: yes"},
            {"title": "Berea Segmentation Benchmark (DOI: 10.17612/CCC333)", "reason": "b",
             "citation": "Berea greyscale volume — segmented: no"},
        ]})
        result = self._run("paired segmented images", payload)
        assert result.count("Berea Segmentation Benchmark") == 1

    def test_caveat_is_preserved(self):
        payload = json.dumps({
            "candidates": [{
                "title": "Berea Segmentation Benchmark", "reason": "Both volumes present.",
                "citation": "Berea segmented volume — segmented: yes",
            }],
            "caveat": "Whether the two volumes cover the identical field of view isn't recorded.",
        })
        result = self._run("paired segmented images", payload)
        assert "identical field of view isn't recorded" in result

    def test_titles_and_dois_come_from_the_graph_not_the_model(self):
        """A model-retyped DOI must never reach the user — the record's own DOI wins."""
        payload = json.dumps({"candidates": [{
            "title": "Berea Segmentation Benchmark",
            "doi": "10.9999/hallucinated",
            "reason": "Both volumes present.",
            "citation": "Berea segmented volume — segmented: yes",
        }]})
        result = self._run("paired segmented images", payload)

        assert "10.17612/CCC333" in result
        assert "10.9999/hallucinated" not in result

    def test_no_fact_sheets_built_returns_honest_message(self):
        store = _mock_fact_sheet_store(records=[], ranked=[])
        result = self._run("paired segmented images", "{}", store=store)
        assert "don't have the precomputed dataset fact sheets" in result

    def test_retrieval_failure_degrades_honestly(self):
        store = MagicMock()
        store.rank_fact_sheets.side_effect = RuntimeError("Neo4j unreachable")
        result = self._run("paired segmented images", "{}", store=store)
        assert "fact-sheet lookup failed" in result

    def test_restrict_to_titles_skips_ranking_and_fetches_that_set(self):
        """A refinement of an already-listed set is reasoned over exactly, not re-ranked —
        ranking could only lose members of a set that is already correct and small."""
        from src.assistant.tools import _reason_about_dataset_content
        store = _mock_fact_sheet_store(records=[_PAIRING_SHEET])
        payload = json.dumps({"candidates": [{
            "title": "Berea Segmentation Benchmark", "reason": "Both volumes present.",
            "citation": "Berea segmented volume — segmented: yes",
        }]})
        with patch("src.assistant.tools._get_graph_store", return_value=store):
            with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(payload)):
                result = _reason_about_dataset_content(
                    "of these, which have paired segmented images?",
                    restrict_to_titles=["Berea Segmentation Benchmark"],
                )

        store.rank_fact_sheets.assert_not_called()
        store.fetch_fact_sheets.assert_called_once_with(titles=["Berea Segmentation Benchmark"])
        assert "Berea Segmentation Benchmark" in result

    def test_exhaustive_question_uses_map_reduce_screening(self):
        """"list every dataset where..." can't legitimately be narrowed by ranking —
        it screens the whole corpus in batches instead."""
        from src.assistant.tools import _reason_about_dataset_content
        store = _mock_fact_sheet_store()
        screen_payload = json.dumps(["Berea Segmentation Benchmark"])
        final_payload = json.dumps({"candidates": [{
            "title": "Berea Segmentation Benchmark", "reason": "Both volumes present.",
            "citation": "Berea segmented volume — segmented: yes",
        }]})
        llm = MagicMock()
        llm.send_prompt.side_effect = [screen_payload, final_payload]

        with patch("src.assistant.tools._get_graph_store", return_value=store):
            with patch("src.assistant.llm.get_chat_model", return_value=llm):
                result = _reason_about_dataset_content(
                    "list every dataset with paired segmented images"
                )

        store.rank_fact_sheets.assert_not_called()
        store.fetch_fact_sheets.assert_called_once_with()
        assert "Berea Segmentation Benchmark" in result
        assert "Fontainebleau Multiscale Imaging" not in result

    def test_map_reduce_batches_are_char_budgeted_not_item_counted(self):
        """Fact-sheet sizes vary ~30x across the corpus. Batching by item count would let a
        batch of large sheets overflow the context builder and silently drop datasets — the
        precisely wrong failure for a question that asked for EVERY match."""
        from src.assistant.tools import _batch_records_by_chars

        records = [{"title": f"D{i}", "factSheetText": "x" * 15_000} for i in range(6)]
        batches = _batch_records_by_chars(records, budget=40_000)

        assert sum(len(b) for b in batches) == 6  # nothing dropped
        for batch in batches:
            total = sum(len(r["factSheetText"]) for r in batch)
            assert total <= 40_000 or len(batch) == 1

    def test_oversized_single_record_still_gets_screened(self):
        from src.assistant.tools import _batch_records_by_chars
        records = [{"title": "Huge", "factSheetText": "x" * 90_000}]
        assert _batch_records_by_chars(records, budget=40_000) == [records]

    def test_context_budget_truncation_is_never_silent(self):
        from src.assistant.tools import _build_fact_sheet_context, _FACT_SHEET_CONTEXT_CHAR_BUDGET
        huge = [
            {"datasetNumber": i, "title": f"Big {i}", "doi": f"10.1/{i}",
             "factSheetText": "x" * (_FACT_SHEET_CONTEXT_CHAR_BUDGET // 2)}
            for i in range(6)
        ]
        context, included = _build_fact_sheet_context(huge)

        assert len(included) < len(huge)
        assert "were NOT considered" in context
        assert len(context) < _FACT_SHEET_CONTEXT_CHAR_BUDGET * 1.5

    def test_falls_back_to_json_fact_sheet_when_text_missing(self):
        from src.assistant.tools import _fact_sheet_text
        record = {"factSheet": json.dumps({"title": "Old Sheet", "samples": [{"title": "Core 1"}]})}
        text = _fact_sheet_text(record)
        assert "Old Sheet" in text and "Core 1" in text


class TestReasoningAnswerFormatting:
    """A bullet's continuation lines must stay single lines: a newline inside the reason or
    citation ends the markdown list item, so everything after it renders as a loose paragraph
    that reads as the last dataset's rationale having escaped its bullet (reported live)."""

    def _render(self, reason, citation, caveat=None):
        from src.assistant.tools import _render_reasoning_answer
        parsed = {"candidates": [{
            "title": "Berea Segmentation Benchmark", "reason": reason, "citation": citation,
        }]}
        if caveat:
            parsed["caveat"] = caveat
        return _render_reasoning_answer("q", parsed, [_PAIRING_SHEET])

    def _continuation_lines(self, result):
        """Every non-blank line after the bullet's own first line."""
        lines = result.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("- **Berea"))
        return [l for l in lines[start + 1:] if l.strip()]

    def test_multiline_reason_stays_on_the_bullet(self):
        result = self._render("First part.\nSecond part.", "segmented: yes")
        for line in self._continuation_lines(result):
            assert line.startswith("  "), f"escaped the bullet: {line!r}"
        assert "First part. Second part." in result

    def test_multiline_citation_stays_on_the_bullet(self):
        citation = "Digital datasets (images/scans) (2):\n- greyscale — segmented: no\n- seg — segmented: yes"
        result = self._render("Both volumes present.", citation)
        for line in self._continuation_lines(result):
            assert line.startswith("  "), f"escaped the bullet: {line!r}"
        assert "greyscale — segmented: no" in result

    def test_caveat_is_labelled_so_it_cannot_read_as_the_last_bullet(self):
        result = self._render("r", "c", caveat="Some datasets may use other instruments.")
        assert "*Note: Some datasets may use other instruments.*" in result


class TestTidyCitation:
    """Citations are copied out of the fact sheet, so they arrive carrying its rendering.
    Presentation is cleaned up; the recorded values themselves are never altered."""

    def test_collapses_whitespace(self):
        from src.assistant.tools import _tidy_citation
        assert "\n" not in _tidy_citation("a\n  b\n\nc")
        assert _tidy_citation("a\n  b") == "a b"

    def test_strips_the_fact_sheets_own_section_header(self):
        from src.assistant.tools import _tidy_citation
        out = _tidy_citation("Digital datasets (images/scans) (2): Scan A — segmented: no")
        assert out.startswith("Scan A")
        assert "Digital datasets" not in out

    def test_compacts_verbose_voxel_dimension_phrasing(self):
        from src.assistant.tools import _tidy_citation
        out = _tidy_citation(
            "Micro-CT scan — voxelDimensions: X, Y, Z units (in micrometers): 4.54, 4.54, 4.54; segmented: no"
        )
        assert "4.54 x 4.54 x 4.54 micrometers" in out
        assert "X, Y, Z units" not in out
        # every recorded value survives — this is a reformat, not a paraphrase
        assert out.count("4.54") == 3
        assert "segmented: no" in out

    def test_drops_a_trailing_none_third_dimension(self):
        from src.assistant.tools import _tidy_citation
        out = _tidy_citation("voxelDimensions: X, Y, Z units (in nanometers): 134.9, 134.9, None")
        assert "134.9 x 134.9 nanometers" in out
        assert "None" not in out

    def test_strips_wrapping_quotes(self):
        from src.assistant.tools import _tidy_citation
        assert _tidy_citation('"segmented: yes"') == "segmented: yes"

    def test_long_citation_is_cut_explicitly_not_silently(self):
        from src.assistant.tools import _tidy_citation, _MAX_CITATION_CHARS
        out = _tidy_citation("word " * 500)
        assert len(out) < _MAX_CITATION_CHARS + 40
        assert "citation truncated" in out

    def test_plain_citation_passes_through_unchanged(self):
        from src.assistant.tools import _tidy_citation
        text = "The plug was acquired with an Xradia Versa micro-CT scanner at 40 kV."
        assert _tidy_citation(text) == text


class TestContentReasoningGateWiring:
    """The gate has to fire regardless of which tool the agent happened to call — that's
    the whole point of gating in code rather than trusting tool-choice routing."""

    def test_get_dataset_details_routes_relational_question_away_from_cypher(self):
        store = _mock_fact_sheet_store()
        payload = json.dumps({"candidates": [{
            "title": "Berea Segmentation Benchmark", "reason": "Both volumes present.",
            "citation": "Berea segmented volume — segmented: yes",
        }]})
        with patch("src.assistant.tools._get_graph_store", return_value=store):
            with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(payload)):
                from src.assistant.tools import get_dataset_details
                result = get_dataset_details.func("are there paired tomographic and segmented images?")

        store.cypher_qa.assert_not_called()
        assert "[content reasoning]" in result
        assert "Berea Segmentation Benchmark" in result

    def test_get_dataset_details_still_uses_cypher_for_plain_property_question(self):
        store = MagicMock()
        store.cypher_qa.return_value = "- **Some Sandstone** (DOI: 10.1/x)"
        with patch("src.assistant.tools._get_graph_store", return_value=store):
            from src.assistant.tools import get_dataset_details
            result = get_dataset_details.func("sandstone datasets with porosity above 0.3")

        store.cypher_qa.assert_called_once()
        store.rank_fact_sheets.assert_not_called()
        assert "Some Sandstone" in result

    def test_search_datasets_routes_relational_question_away_from_structured_lookup(self):
        """"paired tomographic and segmented images" trips search_datasets' own
        structured-first check on its literal sub-clause ("segmented") alone — answering
        it that way presents a generic segmented list as if "paired" had been verified."""
        store = _mock_fact_sheet_store()
        payload = json.dumps({"candidates": [{
            "title": "Berea Segmentation Benchmark", "reason": "Both volumes present.",
            "citation": "Berea segmented volume — segmented: yes",
        }]})
        with patch("src.assistant.tools._get_graph_store", return_value=store):
            with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(payload)):
                from src.assistant.tools import search_datasets
                result = search_datasets.func("paired tomographic and segmented images")

        store.cypher_qa.assert_not_called()
        store.hybrid_search.assert_not_called()
        assert "[content reasoning]" in result


class TestCorralArchiveUrl:
    def test_returns_url_for_populated_dataset_number(self):
        from src.assistant.tools import _corral_archive_url
        assert _corral_archive_url(42) == "https://web.corral.tacc.utexas.edu/digitalporousmedia/archive/DRP-42/"

    def test_returns_none_for_missing_dataset_number(self):
        from src.assistant.tools import _corral_archive_url
        assert _corral_archive_url(None) is None


# ---------------------------------------------------------------------------
# search_literature
# ---------------------------------------------------------------------------

class TestSearchLiterature:
    def test_formats_papers_with_source_label(self):
        papers = [
            _fake_paper("Paper One", doi="10.1111/one"),
            _fake_paper("Paper Two", doi="10.2222/two"),
        ]
        with patch("src.assistant.tools._get_lit_search") as mock_factory:
            mock_ls = MagicMock()
            mock_ls.search_external_literature.return_value = papers
            mock_factory.return_value = mock_ls

            from src.assistant.tools import search_literature
            result = search_literature.func("pore network permeability")

        assert "[semantic scholar]" in result
        assert "Paper One" in result
        assert "Paper Two" in result
        assert "10.1111/one" in result
        assert "10.2222/two" in result

    def test_no_results_returns_friendly_message(self):
        with patch("src.assistant.tools._get_lit_search") as mock_factory:
            mock_ls = MagicMock()
            mock_ls.search_external_literature.return_value = []
            mock_factory.return_value = mock_ls

            from src.assistant.tools import search_literature
            result = search_literature.func("zzz obscure query")

        assert "No papers found" in result

    def test_truncates_long_author_list(self):
        paper = _fake_paper("Long Author Paper")  # 4 authors → should show "et al."
        with patch("src.assistant.tools._get_lit_search") as mock_factory:
            mock_ls = MagicMock()
            mock_ls.search_external_literature.return_value = [paper]
            mock_factory.return_value = mock_ls

            from src.assistant.tools import search_literature
            result = search_literature.func("query")

        assert "et al." in result

    def test_no_abstract_shows_placeholder(self):
        paper = Paper(
            title="No Abstract Paper", authors=["A"], abstract=None,
            year=2020, doi=None, citation_count=0, pdf_url=None, url=""
        )
        with patch("src.assistant.tools._get_lit_search") as mock_factory:
            mock_ls = MagicMock()
            mock_ls.search_external_literature.return_value = [paper]
            mock_factory.return_value = mock_ls

            from src.assistant.tools import search_literature
            result = search_literature.func("query")

        assert "No abstract." in result


# ---------------------------------------------------------------------------
# build_langchain_tools
# ---------------------------------------------------------------------------

class TestSearchDatasetsWeakMatch:
    """search_datasets should honestly flag results that don't mention the query's
    topic, instead of silently presenting off-topic hits as ordinary matches."""

    def _mock_result(self, title, text, doi="10.1234/x"):
        return {"text": text, "metadata": {"title": title, "doi": doi}, "source_label": "[hybrid match]"}

    def test_flags_weak_match_when_no_result_mentions_topic(self):
        with patch("src.assistant.tools.expand_query", return_value={
            "expanded_query": "fibrous media datasets",
            "inferred_filters": {},
            "rationale": "",
        }):
            with patch("src.assistant.tools._get_graph_store") as mock_get_store:
                store = MagicMock()
                store.hybrid_search.return_value = [
                    self._mock_result("Bead Pack Geological Fabrics", "2D bead-packs generated with realistic geological features.")
                ]
                store.component_search.return_value = []
                mock_get_store.return_value = store

                summary = json.dumps(["2D bead-pack dataset with realistic geological features."])
                with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(summary)):
                    from src.assistant.tools import search_datasets
                    result = search_datasets.func("Are there any fibrous media datasets on the portal?")

        assert "[weak match" in result
        assert "fibrous" in result.lower()
        # The off-topic result is still shown, not dropped.
        assert "Bead Pack Geological Fabrics" in result

    def test_no_weak_match_flag_when_topic_is_present(self):
        with patch("src.assistant.tools.expand_query", return_value={
            "expanded_query": "coal datasets",
            "inferred_filters": {},
            "rationale": "",
        }):
            with patch("src.assistant.tools._get_graph_store") as mock_get_store:
                store = MagicMock()
                store.hybrid_search.return_value = [
                    self._mock_result("Moura Coal", "A coal sample dataset.")
                ]
                store.component_search.return_value = []
                mock_get_store.return_value = store

                summary = json.dumps(["A coal sample dataset matching the query."])
                with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(summary)):
                    from src.assistant.tools import search_datasets
                    result = search_datasets.func("Show me datasets from coal samples")

        assert "[weak match" not in result

    def test_summary_falls_back_to_snippet_when_llm_call_fails(self):
        """If the summarization LLM call fails, the result must still render — using a
        raw-text fallback snippet — rather than dropping the result or raising."""
        with patch("src.assistant.tools.expand_query", return_value={
            "expanded_query": "coal datasets",
            "inferred_filters": {},
            "rationale": "",
        }):
            with patch("src.assistant.tools._get_graph_store") as mock_get_store:
                store = MagicMock()
                store.hybrid_search.return_value = [
                    self._mock_result("Moura Coal", "A coal sample dataset used for permeability studies.")
                ]
                store.component_search.return_value = []
                mock_get_store.return_value = store

                broken_llm = MagicMock()
                broken_llm.send_prompt.side_effect = RuntimeError("LLM unavailable")
                with patch("src.assistant.llm.get_chat_model", return_value=broken_llm):
                    from src.assistant.tools import search_datasets
                    result = search_datasets.func("Show me datasets from coal samples")

        assert "Moura Coal" in result
        assert "A coal sample dataset used for permeability studies" in result


class TestBuildLangchainTools:
    def test_returns_all_tools(self):
        from src.assistant.tools import build_langchain_tools
        tools = build_langchain_tools()
        names = {t.name for t in tools}
        assert "search_datasets" in names
        assert "get_dataset_details" in names
        assert "get_dataset_profile" in names
        assert "reason_about_dataset_content" in names
        assert "get_workflow_guidance" in names
        assert "get_educational_context" in names
        assert "search_literature" in names

    def test_returns_list_of_correct_length(self):
        from src.assistant.tools import build_langchain_tools
        assert len(build_langchain_tools()) == 8


# ---------------------------------------------------------------------------
# USE_NEO4J=false smoke test
# ---------------------------------------------------------------------------

class TestUseNeo4jFalseGracefulFallback:
    """
    Smoke tests verifying the assistant falls back gracefully when Neo4j is
    disabled. With USE_NEO4J=false, graph queries return empty results and the
    assistant routes through LiteratureSearch (Semantic Scholar) instead.
    No graph queries should be attempted.
    """

    def test_search_datasets_returns_no_results_message(self):
        """search_datasets should return a friendly message, not raise, when Neo4j is off."""
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.tools._graph_store", None):
                with patch("src.assistant.graph_store.GraphStore.hybrid_search", return_value=[]):
                    with patch("src.assistant.graph_store.GraphStore.component_search", return_value=[]):
                        with patch("src.assistant.tools.expand_query", return_value={
                            "expanded_query": "sandstone",
                            "inferred_filters": {},
                            "rationale": "",
                        }):
                            from src.assistant.tools import search_datasets
                            result = search_datasets.func("sandstone datasets")

        assert isinstance(result, str)
        assert "No datasets found" in result

    def test_search_literature_works_without_neo4j(self):
        """search_literature uses Semantic Scholar only — must work regardless of USE_NEO4J."""
        papers = [_fake_paper("Pore Scale Paper", doi="10.9999/test")]
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.tools._get_lit_search") as mock_factory:
                mock_ls = MagicMock()
                mock_ls.search_external_literature.return_value = papers
                mock_factory.return_value = mock_ls

                from src.assistant.tools import search_literature
                result = search_literature.func("pore scale modeling")

        assert "[semantic scholar]" in result
        assert "Pore Scale Paper" in result

    def test_get_dataset_details_returns_string_without_neo4j(self):
        """get_dataset_details should degrade gracefully when Neo4j is disabled."""
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.tools._graph_store", None):
                with patch("src.assistant.graph_store.GraphStore.cypher_qa", return_value=""):
                    from src.assistant.tools import get_dataset_details
                    result = get_dataset_details.func("what is the porosity of DRP-1?")

        assert isinstance(result, str)

    def test_no_graph_queries_when_neo4j_disabled(self):
        """Verify execute_cypher is never called when USE_NEO4J=false."""
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.graph_store.GraphStore.execute_cypher") as mock_cypher:
                with patch("src.assistant.tools.expand_query", return_value={
                    "expanded_query": "sandstone",
                    "inferred_filters": {},
                    "rationale": "",
                }):
                    with patch("src.assistant.graph_store.GraphStore.hybrid_search", return_value=[]):
                        with patch("src.assistant.graph_store.GraphStore.component_search", return_value=[]):
                            from src.assistant.tools import search_datasets
                            search_datasets.func("sandstone datasets")

        mock_cypher.assert_not_called()
