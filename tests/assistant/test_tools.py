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

    def test_llm_called_once(self):
        mock_llm = _mock_chat_model("Guidance response.")
        with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
            from src.assistant.tools import get_workflow_guidance
            get_workflow_guidance.func("lattice Boltzmann permeability")

        mock_llm.send_prompt.assert_called_once()

    def test_no_keyword_match_still_calls_llm(self):
        """Unrecognised queries fall back to LLM with empty context (pre-trained knowledge)."""
        mock_llm = _mock_chat_model("I don't have portal-specific data on this, but generally…")
        with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
            from src.assistant.tools import get_workflow_guidance
            result = get_workflow_guidance.func("completely unrelated xyz query")

        mock_llm.send_prompt.assert_called_once()
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

    def test_llm_called_once(self):
        mock_llm = _mock_chat_model("Educational response.")
        with patch("src.assistant.llm.get_chat_model", return_value=mock_llm):
            from src.assistant.tools import get_educational_context
            get_educational_context.func("what is REV?")

        mock_llm.send_prompt.assert_called_once()

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

class TestBuildLangchainTools:
    def test_returns_all_five_tools(self):
        from src.assistant.tools import build_langchain_tools
        tools = build_langchain_tools()
        names = {t.name for t in tools}
        assert "search_datasets" in names
        assert "get_dataset_details" in names
        assert "get_workflow_guidance" in names
        assert "get_educational_context" in names
        assert "search_literature" in names

    def test_returns_list_of_correct_length(self):
        from src.assistant.tools import build_langchain_tools
        assert len(build_langchain_tools()) == 6
