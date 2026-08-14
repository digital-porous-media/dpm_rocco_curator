"""
Unit tests for src/assistant/conversation_manager.py's response-cleanup helpers and
tool-grounding fallback behavior.

All LLM calls and tool invocations are mocked — no credentials required.
"""

from unittest.mock import MagicMock, patch

from src.assistant.conversation_manager import (
    ConversationManager,
    _HONEST_TOOL_FAILURE_MSG,
    _run_manual_dispatch,
    _strip_recap_paragraph,
)


class TestStripRecapParagraph:
    def test_strips_trailing_recap_after_bullets(self):
        text = (
            "Datasets:\n\n"
            "- Moura Coal (DOI: 10.17612/P7V888)\n"
            "A coal sample dataset.\n"
            "- DeepRock-SR (DOI: 10.17612/s3m9-e024)\n"
            "A super-resolution dataset.\n\n"
            "These datasets are related to coal samples. The Moura Coal dataset "
            "contains information about the coal sample, while the DeepRock-SR "
            "dataset contains a collection of greyscale digital rock images."
        )
        result = _strip_recap_paragraph(text)
        assert "These datasets are related to" not in result
        assert "Moura Coal (DOI: 10.17612/P7V888)" in result
        assert "DeepRock-SR (DOI: 10.17612/s3m9-e024)" in result

    def test_preserves_leading_header_before_bullets(self):
        text = "Datasets:\n\n- Moura Coal (DOI: 10.17612/P7V888)\nA coal sample dataset."
        result = _strip_recap_paragraph(text)
        assert result.startswith("Datasets:")

    def test_preserves_suitability_clarifying_question(self):
        text = (
            "- Sample A (DOI: 10.1234/a)\ndescription\n\n"
            "To narrow this further, could you tell me what specific properties "
            "matter most? For example: do you need a segmented image, a particular "
            "rock type, a resolution range, or simulation outputs already included?"
        )
        result = _strip_recap_paragraph(text)
        assert "To narrow this further" in result

    def test_preserves_per_result_fit_notes(self):
        text = (
            "- Sample A (DOI: 10.1234/a)\n"
            "This dataset is a strong match because it includes simulation outputs."
        )
        result = _strip_recap_paragraph(text)
        assert "strong match" in result

    def test_no_bullets_returns_text_unchanged(self):
        text = "There are no datasets matching that query."
        assert _strip_recap_paragraph(text) == text


class TestRunManualDispatchFallback:
    """A successful tool call's real, grounded output must never be discarded just
    because the downstream polish-synthesis LLM call fails."""

    def _mock_tool(self, name, return_value):
        tool = MagicMock()
        tool.name = name
        tool.invoke.return_value = return_value
        return tool

    def test_returns_raw_tool_output_when_synthesis_fails(self):
        tool = self._mock_tool("get_dataset_details", "Grain Packing (porosity: 30)")
        with patch("src.assistant.tools.build_langchain_tools", return_value=[tool]):
            with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
                mock_get_llm.return_value.invoke.side_effect = RuntimeError("boom")
                result = _run_manual_dispatch(
                    [{"name": "get_dataset_details", "args": {"question": "porosity > 0.3"}}],
                    "porosity above 0.3",
                    [],
                )
        assert result is not None
        assert "Grain Packing" in result

    def test_returns_synthesized_output_when_synthesis_succeeds(self):
        tool = self._mock_tool("get_dataset_details", "Grain Packing (porosity: 30)")
        with patch("src.assistant.tools.build_langchain_tools", return_value=[tool]):
            with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
                mock_get_llm.return_value.invoke.return_value = MagicMock(
                    content="Here is what I found: Grain Packing has porosity 30%."
                )
                result = _run_manual_dispatch(
                    [{"name": "get_dataset_details", "args": {"question": "porosity > 0.3"}}],
                    "porosity above 0.3",
                    [],
                )
        assert result is not None
        assert "Grain Packing" in result

    def test_returns_none_when_no_tool_succeeds(self):
        with patch("src.assistant.tools.build_langchain_tools", return_value=[]):
            result = _run_manual_dispatch(
                [{"name": "get_dataset_details", "args": {"question": "porosity > 0.3"}}],
                "porosity above 0.3",
                [],
            )
        assert result is None


class TestChatHonestFallback:
    """When a 400 tool-format error yields an identified tool call but dispatch still
    produces nothing usable, chat() must give up honestly rather than call the LLM with
    no tool context (which has no grounding and will hedge/guess)."""

    def test_returns_honest_message_when_dispatch_produces_nothing(self):
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        manager._agent.invoke.side_effect = Exception(
            "400 Invalid function calling output: JSONDecodeError near get_dataset_details"
        )
        with patch("src.assistant.conversation_manager._extract_tool_calls_from_error",
                   return_value=[{"name": "get_dataset_details", "args": {"question": "x"}}]):
            with patch("src.assistant.conversation_manager._run_manual_dispatch", return_value=None):
                result = manager.chat("porosity above 0.4")
        assert result == _HONEST_TOOL_FAILURE_MSG
