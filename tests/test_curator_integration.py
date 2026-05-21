"""Integration tests for the description curator workflow.

Tests that the evaluator, editor, and screener can all call send_prompt()
on RoccoClient without AttributeError. This catches refactoring regressions.

Run with: pytest tests/test_curator_integration.py -v
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from src.evaluator.evaluator import DescriptionEvaluator
from src.editor.editor import DescriptionEditor
from src.llm.content_screener import ContentScreener
from src.llm.client import RoccoClient
from src.common.utils import _build_rubric, _build_few_shot_examples


def make_mock_response(content: str) -> MagicMock:
    """Helper to create a mock OpenAI completion response."""
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )


@pytest.fixture
def mock_rocco_client(monkeypatch):
    """Provide a mocked RoccoClient for testing."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with patch("src.llm.client.openai.OpenAI") as mock_openai_class:
        # Setup the mock to return proper responses
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Create the RoccoClient
        client = RoccoClient(api_key="test-key", provider="openai")
        yield client


@pytest.fixture
def sample_rubric():
    """Sample evaluation rubric for testing (matches src/evaluator/rubric.json structure)."""
    return [
        {
            "criterion": "Self-Contained Description",
            "description": "Focuses on describing the dataset independently.",
            "max_score": 1
        },
        {
            "criterion": "Methodology",
            "description": "Describes the methodology for data collection.",
            "max_score": 1
        },
        {
            "criterion": "Clarity and Accessibility",
            "description": "Keeps descriptions clear and accessible.",
            "max_score": 1
        },
    ]


@pytest.fixture
def sample_examples():
    """Sample few-shot examples for testing."""
    return [
        {
            "description": "Example dataset with clear methodology and well-organized content.",
            "rubric_breakdown": [
                {"criterion": "Self-Contained Description", "score": 1, "explanation": "Dataset is well described independently."},
                {"criterion": "Context of Creation", "score": 0.5, "explanation": "Some context provided."},
                {"criterion": "Porous Media Type", "score": 1, "explanation": "Materials clearly mentioned."},
                {"criterion": "Research Problem", "score": 0.5, "explanation": "Research problem partially addressed."},
                {"criterion": "Reuse and Beneficiaries", "score": 1, "explanation": "Clear beneficiaries identified."},
                {"criterion": "Methodology", "score": 1, "explanation": "Methodology well explained."},
                {"criterion": "Contents and Organization", "score": 1, "explanation": "Data organization clearly described."},
                {"criterion": "Quality Control", "score": 0.5, "explanation": "Some QC measures mentioned."},
                {"criterion": "Clarity and Accessibility", "score": 1, "explanation": "Clear and accessible language."},
                {"criterion": "Keywords", "score": 1, "explanation": "Relevant keywords provided."}
            ],
            "comments": None
        }
    ]


class TestEvaluatorSendPrompt:
    """Test that DescriptionEvaluator can call send_prompt."""

    def test_evaluator_send_prompt_success(self, mock_rocco_client, sample_rubric, sample_examples):
        """Evaluator should successfully call send_prompt and return structured output."""
        # Mock the LLMClient's send_prompt to return valid JSON
        mock_rocco_client.llm_client.send_prompt = MagicMock(
            return_value=json.dumps({
                "rubric_breakdown": [
                    {"criterion": "Completeness", "score": 8, "explanation": "Good"},
                    {"criterion": "Clarity", "score": 7, "explanation": "Clear"},
                    {"criterion": "Organization", "score": 9, "explanation": "Well organized"},
                ],
                "comments": "Good work"
            })
        )

        evaluator = DescriptionEvaluator(
            model=mock_rocco_client,
            rubric=sample_rubric,
            examples=sample_examples
        )

        result = evaluator.evaluate("This is a test description")

        # Verify send_prompt was called
        mock_rocco_client.llm_client.send_prompt.assert_called_once()

        # Verify result structure
        assert result.total_score == 24  # 8 + 7 + 9
        assert len(result.rubric_breakdown) == 3
        assert result.comments == "Good work"

    def test_evaluator_send_prompt_no_attribute_error(self, mock_rocco_client, sample_rubric, sample_examples):
        """Evaluator should not raise AttributeError when calling send_prompt."""
        mock_rocco_client.llm_client.send_prompt = MagicMock(
            return_value=json.dumps({
                "rubric_breakdown": [
                    {"criterion": "Completeness", "score": 5, "explanation": "Okay"}
                ]
            })
        )

        evaluator = DescriptionEvaluator(
            model=mock_rocco_client,
            rubric=sample_rubric,
            examples=sample_examples
        )

        # This should not raise AttributeError: 'RoccoClient' object has no attribute 'send_prompt'
        try:
            result = evaluator.evaluate("test")
            assert result is not None
        except AttributeError as e:
            if "send_prompt" in str(e):
                pytest.fail(f"RoccoClient missing send_prompt method: {e}")
            raise


class TestEditorSendPrompt:
    """Test that DescriptionEditor can call send_prompt."""

    def test_editor_send_prompt_available(self, mock_rocco_client):
        """Editor should be able to access send_prompt on RoccoClient."""
        editor = DescriptionEditor(
            model=mock_rocco_client,
            rubric={"test": "criteria"},
            use_rag=False
        )

        # Verify send_prompt method exists
        assert hasattr(editor.model, "send_prompt")
        assert callable(editor.model.send_prompt)

    def test_editor_send_prompt_callable(self, mock_rocco_client):
        """Editor should be able to call send_prompt without error."""
        mock_rocco_client.llm_client.send_prompt = MagicMock(return_value="test response")

        editor = DescriptionEditor(
            model=mock_rocco_client,
            rubric={},
            use_rag=False
        )

        # Should not raise AttributeError
        try:
            response = editor.model.send_prompt("test prompt")
            assert response == "test response"
        except AttributeError as e:
            if "send_prompt" in str(e):
                pytest.fail(f"RoccoClient missing send_prompt method: {e}")
            raise


class TestScreenerSendPrompt:
    """Test that ContentScreener can call send_prompt."""

    def test_screener_send_prompt_success(self, mock_rocco_client):
        """Screener should successfully call send_prompt and return validation result."""
        mock_rocco_client.llm_client.send_prompt = MagicMock(
            return_value=json.dumps({
                "is_relevant": True,
                "is_accurate": True,
                "is_respectful": True,
                "is_coherent": True,
                "issues": [],
                "confidence": 0.95,
                "recommendation": "accept"
            })
        )

        screener = ContentScreener(model=mock_rocco_client)
        result = screener.screen_user_content("This is helpful feedback")

        # Verify send_prompt was called
        mock_rocco_client.llm_client.send_prompt.assert_called_once()

        # Verify result structure
        assert result["is_relevant"] is True
        assert result["is_accurate"] is True
        assert result["confidence"] == 0.95
        assert result["recommendation"] == "accept"

    def test_screener_send_prompt_no_attribute_error(self, mock_rocco_client):
        """Screener should not raise AttributeError when calling send_prompt."""
        mock_rocco_client.llm_client.send_prompt = MagicMock(
            return_value=json.dumps({
                "is_relevant": False,
                "is_accurate": False,
                "is_respectful": True,
                "is_coherent": False,
                "issues": ["Not relevant to dataset"],
                "confidence": 0.8,
                "recommendation": "reject"
            })
        )

        screener = ContentScreener(model=mock_rocco_client)

        # This should not raise AttributeError
        try:
            result = screener.screen_user_content("off-topic content")
            assert result is not None
        except AttributeError as e:
            if "send_prompt" in str(e):
                pytest.fail(f"RoccoClient missing send_prompt method: {e}")
            raise


class TestRoccoClientBackwardCompatibility:
    """Test that RoccoClient send_prompt method works as expected."""

    def test_rocco_client_has_send_prompt(self, mock_rocco_client):
        """RoccoClient should have send_prompt method."""
        assert hasattr(mock_rocco_client, "send_prompt")
        assert callable(mock_rocco_client.send_prompt)

    def test_rocco_client_send_prompt_delegates(self, mock_rocco_client):
        """RoccoClient.send_prompt should delegate to llm_client.send_prompt."""
        mock_rocco_client.llm_client.send_prompt = MagicMock(return_value="test response")

        # Call send_prompt on RoccoClient
        response = mock_rocco_client.send_prompt("test prompt", context="test context")

        # Verify it delegated to llm_client
        mock_rocco_client.llm_client.send_prompt.assert_called_once_with(
            "test prompt",
            "test context",
            None  # params default to None
        )
        assert response == "test response"

    def test_rocco_client_send_prompt_with_params(self, mock_rocco_client):
        """RoccoClient.send_prompt should pass through params correctly."""
        mock_rocco_client.llm_client.send_prompt = MagicMock(return_value="test response")

        test_params = {"response_format": {"type": "json_object"}}
        response = mock_rocco_client.send_prompt(
            "test prompt",
            context="test context",
            params=test_params
        )

        # Verify params were passed through
        mock_rocco_client.llm_client.send_prompt.assert_called_once_with(
            "test prompt",
            "test context",
            test_params
        )
