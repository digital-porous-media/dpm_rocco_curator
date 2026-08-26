"""
Tests for src/prompts/query_expander.yaml and educational.yaml.

No LLM calls — only load/render validation.
"""

import pytest
from src.prompts.loader import load_prompt, render


class TestQueryExpanderPrompt:
    def test_loads_required_keys(self):
        prompt = load_prompt("query_expander")
        assert "version" in prompt
        assert "system" in prompt
        assert "user" in prompt

    def test_user_template_contains_query_var(self):
        prompt = load_prompt("query_expander")
        assert "{{ query }}" in prompt["user"]

    def test_system_mentions_filter_fields(self):
        prompt = load_prompt("query_expander")
        system = prompt["system"]
        assert "porousMediaType" in system
        assert "voxelDimensions" in system
        assert "segmented" in system

    def test_renders_user_template(self):
        prompt = load_prompt("query_expander")
        rendered = render(prompt["user"], query="sandstone with low porosity")
        assert "sandstone with low porosity" in rendered

    def test_system_has_few_shot_examples(self):
        prompt = load_prompt("query_expander")
        # Should contain at least two example blocks
        assert prompt["system"].count("expanded_query") >= 2


class TestEducationalPrompt:
    def test_loads_required_keys(self):
        prompt = load_prompt("educational")
        assert "version" in prompt
        assert "system" in prompt
        assert "user" in prompt

    def test_system_contains_context_var(self):
        prompt = load_prompt("educational")
        assert "{{ context }}" in prompt["system"]

    def test_user_contains_question_var(self):
        prompt = load_prompt("educational")
        assert "{{ question }}" in prompt["user"]

    def test_renders_system_with_context(self):
        prompt = load_prompt("educational")
        rendered = render(prompt["system"], context="some workflow context here")
        assert "some workflow context here" in rendered

    def test_renders_user_with_question(self):
        prompt = load_prompt("educational")
        rendered = render(prompt["user"], question="What is porosity?")
        assert "What is porosity?" in rendered

    def test_system_mentions_latex(self):
        prompt = load_prompt("educational")
        assert "LaTeX" in prompt["system"] or "latex" in prompt["system"].lower()

    def test_system_mentions_knowledge_policy(self):
        prompt = load_prompt("educational")
        # Should mention the tiered policy concepts
        system = prompt["system"]
        assert "portal" in system.lower() or "context" in system.lower()
        assert "disclaimer" in system.lower() or "generally" in system.lower()


class TestCorpusReasoningPrompt:
    """The prompt behind reason_about_dataset_content. Its grounding rules are the
    first line of defence (the second is the code-level citation/shortlist check in
    tools._render_reasoning_answer), so the pieces the code depends on are pinned here."""

    def test_loads_required_keys(self):
        prompt = load_prompt("corpus_reasoning")
        assert "version" in prompt
        assert "system" in prompt
        assert "user" in prompt

    def test_system_contains_context_var(self):
        prompt = load_prompt("corpus_reasoning")
        assert "{{ context }}" in prompt["system"]

    def test_user_contains_question_var(self):
        prompt = load_prompt("corpus_reasoning")
        assert "{{ question }}" in prompt["user"]

    def test_system_demands_a_citation_per_candidate(self):
        prompt = load_prompt("corpus_reasoning")
        assert "No citation, no candidate" in prompt["system"]

    def test_system_specifies_the_json_keys_the_code_parses(self):
        prompt = load_prompt("corpus_reasoning")
        system = prompt["system"]
        for key in ("candidates", "title", "reason", "citation", "caveat"):
            assert f'"{key}"' in system

    def test_system_forbids_inventing_properties(self):
        prompt = load_prompt("corpus_reasoning")
        assert "Never invent" in prompt["system"]

    def test_map_reduce_screening_prompt_present_and_renderable(self):
        prompt = load_prompt("corpus_reasoning")
        assert "{{ context }}" in prompt["batch_screen_user"]
        rendered = render(prompt["batch_screen_user"], question="paired scans?", context="a sheet")
        assert "paired scans?" in rendered and "a sheet" in rendered

    def test_renders_system_with_context(self):
        prompt = load_prompt("corpus_reasoning")
        rendered = render(prompt["system"], context="fact sheet text here")
        assert "fact sheet text here" in rendered
