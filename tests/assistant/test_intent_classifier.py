"""
Formal test suite for the intent classifier prompt.
Tests that queries are classified into the correct intent with expected confidence ranges.

Run with:
    pytest tests/assistant/test_intent_classifier.py -v
"""

import json
import os
import pytest
from src.prompts.loader import load_prompt, render


def extract_json(text: str) -> dict:
    """Extract JSON from response (may have explanations before/around JSON)."""
    # Try to parse the whole thing first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from code fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
        return json.loads(text.strip())
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
        return json.loads(text.strip())

    # Try to find JSON on the last line (after explanation)
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


@pytest.fixture(scope="module")
def chat_model():
    from dotenv import load_dotenv
    load_dotenv()
    if not (os.getenv("LLM_API_KEY") or os.getenv("SAMBANOVA_API_KEY")):
        pytest.skip("No LLM API key set — skipping live LLM tests")
    from src.assistant.llm import get_chat_model
    return get_chat_model()


@pytest.fixture
def prompt_data():
    """Load the assistant intent classifier prompt."""
    return load_prompt("assistant")


def classify_query(query: str, prompt_data: dict, chat_model) -> dict:
    """Classify a single query using the intent classifier prompt."""
    system = prompt_data["system"]
    user = render(prompt_data["user"], query=query)

    response = chat_model.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    result = extract_json(response.content)
    return result


@pytest.mark.parametrize(
    "query,expected_intent,min_confidence",
    [
        # semantic_search — finding datasets by topic/meaning
        ("Which datasets involve sandstone?", "semantic_search", 0.7),
        ("I need lattice Boltzmann simulation datasets in carbonate rocks", "semantic_search", 0.7),
        ("I need datasets with both porosity and permeability measurements", "semantic_search", 0.7),
        ("Find datasets suitable for lattice Boltzmann simulations","semantic_search", 0.7),
        ("Datasets good for machine learning velocity fields in sandstones", "semantic_search", 0.7),
        # metadata_filter — filtering by specific properties
        ("Show datasets with porosity > 0.2", "metadata_filter", 0.7),
        ("Datasets with permeability < 1 millidarcy", "metadata_filter", 0.7),
        ("Micro-CT images with resolution better than 5 microns", "metadata_filter", 0.7),
        # domain_qa — science questions
        ("What is wettability?", "domain_qa", 0.7),
        ("How does capillary pressure relate to pore size?", "domain_qa", 0.7),
        ("How do I compute permeability from a lattice Boltzmann simulation result?", "domain_qa", 0.7),
        # workflow_guidance — portal usage and workflows
        ("How do I upload a dataset?", "workflow_guidance", 0.7),
        ("What's the best way to organize core flooding data?", "workflow_guidance", 0.7),
        ("Can you walk me through the submission workflow?", "workflow_guidance", 0.7),
        ("How do I set up a lattice Boltzmann simulation for this image?", "workflow_guidance", 0.7),
        ("Does this image have enough resolution for a simulation?", "workflow_guidance", 0.7),
        ("How to read in this image in Python?", "workflow_guidance", 0.7),
        # literature_search — papers and publications
        ("Papers on water flooding in sandstone", "literature_search", 0.7),
        ("Find publications about pore-scale modeling", "literature_search", 0.7),
        ("What work has been done on machine learning for permeability prediction?", "literature_search", 0.7),
        # query_expansion — vague/ambiguous queries
        ("I need research data", "query_expansion", 0.5),
        ("Tell me about datasets", "query_expansion", 0.5),
        ("Something about rocks and water", "query_expansion", 0.5),
    ],
)
def test_intent_classification(
    chat_model, prompt_data, query: str, expected_intent: str, min_confidence: float
):
    """Test that queries are classified into the correct intent."""
    result = classify_query(query, prompt_data, chat_model)

    assert (
        result["intent"] == expected_intent
    ), f"Expected {expected_intent}, got {result['intent']}"
    assert (
        result["confidence"] >= min_confidence
    ), f"Expected confidence >= {min_confidence}, got {result['confidence']}"


def test_json_output_format(chat_model, prompt_data):
    """Test that intent classifier returns valid JSON with required fields."""
    result = classify_query("Show datasets with porosity > 0.2", prompt_data, chat_model)

    assert isinstance(result, dict), "Response must be a dictionary"
    assert "intent" in result, "Response must have 'intent' field"
    assert "confidence" in result, "Response must have 'confidence' field"
    assert isinstance(result["intent"], str), "'intent' must be a string"
    assert isinstance(result["confidence"], (int, float)), "'confidence' must be numeric"
    assert 0 <= result["confidence"] <= 1, "'confidence' must be between 0 and 1"


def test_valid_intents(chat_model, prompt_data):
    """Test that only valid intents are returned."""
    valid_intents = {
        "semantic_search",
        "metadata_filter",
        "domain_qa",
        "workflow_guidance",
        "query_expansion",
        "literature_search",
    }

    queries = [
        "Sandstone datasets",
        "Porosity > 0.2",
        "What is wettability",
        "How to upload",
        "I need data",
        "Papers on flooding",
    ]

    for query in queries:
        result = classify_query(query, prompt_data, chat_model)
        assert (
            result["intent"] in valid_intents
        ), f"Invalid intent '{result['intent']}' for query '{query}'"
