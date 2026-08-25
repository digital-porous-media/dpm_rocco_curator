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
        assert "get_workflow_guidance" in names
        assert "get_educational_context" in names
        assert "search_literature" in names

    def test_returns_list_of_correct_length(self):
        from src.assistant.tools import build_langchain_tools
        assert len(build_langchain_tools()) == 7


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
