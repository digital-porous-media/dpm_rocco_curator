"""
Unit tests for LLMClient and RoccoClient.

All tests here are fully mocked — no API key or network access required.
Run with: pytest tests/test_llm_client.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, call
from src.llm.client import LLMClient, RoccoClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_completion(content: str) -> MagicMock:
    """Build a minimal mock that looks like openai.ChatCompletion."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Provider URL routing
# ---------------------------------------------------------------------------

class TestProviderURLRouting:
    """LLMClient must select the correct base_url for each LLM_PROVIDER value."""

    EXPECTED = {
        "openai":      "https://api.openai.com/v1",
        "anthropic":   "https://api.anthropic.com/v1",
        "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai/",
        "deepseek":    "https://api.deepseek.com/v1",
        "huggingface": "https://router.huggingface.co/v1",
        "ollama":      "http://localhost:11434/v1",
        "sambanova":   "https://ai.tejas.tacc.utexas.edu/v1",
    }

    @pytest.mark.parametrize("provider,expected_url", EXPECTED.items())
    def test_provider_url(self, provider, expected_url, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI") as mock_openai:
            client = LLMClient()
            assert client.api_url == expected_url
            mock_openai.assert_called_once_with(api_key="test-key", base_url=expected_url)

    def test_unknown_provider_falls_back_to_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "nonexistent_provider")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = LLMClient()
            assert client.api_url == "https://api.openai.com/v1"

    def test_llm_base_url_overrides_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://my-proxy.example.com/v1")

        with patch("openai.OpenAI") as mock_openai:
            client = LLMClient()
            assert client.api_url == "https://my-proxy.example.com/v1"
            mock_openai.assert_called_once_with(
                api_key="test-key", base_url="https://my-proxy.example.com/v1"
            )

    def test_api_url_param_takes_highest_priority(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://env-url.com/v1")

        with patch("openai.OpenAI"):
            client = LLMClient(api_url="https://param-url.com/v1")
            assert client.api_url == "https://param-url.com/v1"


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

class TestAPIKeyResolution:
    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "env-key-123")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = LLMClient()
            assert client.api_key == "env-key-123"

    def test_api_key_param_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = LLMClient(api_key="param-key")
            assert client.api_key == "param-key"

    def test_ollama_auto_sets_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = LLMClient()
            assert client.api_key == "ollama"

    def test_ollama_explicit_key_not_overridden(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_API_KEY", "custom-ollama-key")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = LLMClient()
            assert client.api_key == "custom-ollama-key"


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestModelResolution:
    def test_model_defaults_to_gpt4o_mini(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = LLMClient()
            assert client.model == "gpt-4o-mini"

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = LLMClient()
            assert client.model == "gpt-4o"

    def test_model_param_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = LLMClient(model="gpt-4-turbo")
            assert client.model == "gpt-4-turbo"


# ---------------------------------------------------------------------------
# send_prompt
# ---------------------------------------------------------------------------

class TestSendPrompt:
    def _make_client(self, monkeypatch, mock_openai_instance):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI", return_value=mock_openai_instance):
            return LLMClient()

    def test_send_prompt_no_context(self, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = make_mock_completion("hello world")

        client = self._make_client(monkeypatch, mock_instance)
        result = client.send_prompt("Say hello")

        assert result == "hello world"
        mock_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_send_prompt_with_context(self, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = make_mock_completion("evaluated!")

        client = self._make_client(monkeypatch, mock_instance)
        result = client.send_prompt("Evaluate this", context="You are an evaluator.")

        assert result == "evaluated!"
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "You are an evaluator."},
            {"role": "user", "content": "Evaluate this"},
        ]

    def test_send_prompt_extra_params(self, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = make_mock_completion("ok")

        client = self._make_client(monkeypatch, mock_instance)
        client.send_prompt("hello", params={"temperature": 0.2, "max_tokens": 100})

        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 100

    def test_send_prompt_raises_on_api_error(self, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = Exception("API error")

        client = self._make_client(monkeypatch, mock_instance)
        with pytest.raises(RuntimeError, match="LLM API error"):
            client.send_prompt("hello")

    def test_send_prompt_includes_timeout(self, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = make_mock_completion("ok")

        client = self._make_client(monkeypatch, mock_instance)
        client.timeout = 30
        client.send_prompt("hello")

        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# RoccoClient
# ---------------------------------------------------------------------------

class TestRoccoClient:
    def test_rocco_client_inherits_provider_routing(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_API_KEY", "AIza-test")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("openai.OpenAI"):
            client = RoccoClient()
            assert client.api_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
            assert client.provider == "gemini"

    def test_rocco_client_send_prompt(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = make_mock_completion("rocco response")

        with patch("openai.OpenAI", return_value=mock_instance):
            client = RoccoClient()
            result = client.send_prompt("Evaluate this description")

        assert result == "rocco response"
