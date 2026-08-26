"""
Unit tests for src/assistant/conversation_manager.py's response-cleanup helpers and
tool-grounding fallback behavior.

All LLM calls and tool invocations are mocked — no credentials required.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from src.assistant.conversation_manager import (
    ConversationManager,
    _HONEST_TOOL_FAILURE_MSG,
    _NO_TOOL_ACCESS_NOTICE,
    _OFF_DOMAIN_STEER_BACK_MSG,
    _SELF_CONTAINED_TOOLS,
    _VERBATIM_TOOLS,
    _answer_direct,
    _build_verbatim_response,
    _classify_off_domain,
    _extract_tool_calls_from_error,
    _extract_tool_calls_from_text,
    _needs_followup_tool_call,
    _run_manual_dispatch,
    _strip_reasoning_scaffold,
    _strip_recap_paragraph,
)


class TestToolRoutingClassification:
    """search_portal_docs must be routed as self-contained, not verbatim: its raw
    retrieval is prose chunks (dpm_docs excerpts) that only answer the question
    once something synthesizes them (see src/prompts/portal_docs.yaml and
    tools.search_portal_docs) — unlike search_datasets/get_dataset_details, whose
    raw output already IS the final answer and must reach the user byte-for-byte.
    Routing it through _build_verbatim_response instead spliced dataset-flavored
    lead-in/disclaimer text around raw, sometimes off-topic chunk dumps."""

    def test_search_portal_docs_is_self_contained_not_verbatim(self):
        assert "search_portal_docs" in _SELF_CONTAINED_TOOLS
        assert "search_portal_docs" not in _VERBATIM_TOOLS

    def test_get_dataset_profile_is_self_contained_not_verbatim(self):
        assert "get_dataset_profile" in _SELF_CONTAINED_TOOLS
        assert "get_dataset_profile" not in _VERBATIM_TOOLS

    def test_search_datasets_framing_unchanged(self):
        hit = "DOI: 10.1/x\ntitle: foo"
        result = _build_verbatim_response("find sandstone datasets", hit)
        assert result.startswith("Here are the datasets matching your query:")
        assert "verify these datasets on the DPM Portal" in result


class TestStripReasoningScaffold:
    """Llama-4-Maverick intermittently answers a synthesis prompt by emitting its own
    chain-of-thought ("Step 1: ... Step 8: ... The final answer is: <answer>") instead
    of just the answer — live-observed leaking into a user-facing multi-dataset
    comparison response. Stripping requires BOTH the step scaffold and the final-answer
    marker, so legitimately numbered answers (workflow guidance) survive untouched."""

    LEAKED = (
        "Step 1: Identify the key characteristics of DRP-137.\n"
        "The downscaling dataset involves X-ray CT scans of two sandstone samples.\n\n"
        "Step 2: Identify the key characteristics of Gildehauser.\n"
        "It is based on a synchrotron beamline fast micro-CT flow experiment.\n\n"
        "The final answer is: DRP-137 and Gildehauser differ in imaging technique."
    )

    def test_strips_scaffold_keeping_only_final_answer(self):
        assert _strip_reasoning_scaffold(self.LEAKED) == (
            "DRP-137 and Gildehauser differ in imaging technique."
        )

    def test_leaves_legitimate_numbered_workflow_untouched(self):
        # A get_workflow_guidance answer legitimately has "Step N:" lines but never
        # announces a final answer — the conjunction guard must protect it.
        workflow = (
            "To compute absolute permeability:\n\n"
            "Step 1: Segment the image into pore and solid phases.\n"
            "Step 2: Extract the pore network.\n"
            "Step 3: Solve Stokes flow and apply Darcy's law."
        )
        assert _strip_reasoning_scaffold(workflow) == workflow

    def test_leaves_plain_answer_untouched(self):
        plain = "The Gildehauser Sandstone dataset has a porosity of 0.2."
        assert _strip_reasoning_scaffold(plain) == plain

    def test_bold_markdown_scaffold_variant_is_stripped(self):
        text = "**Step 1:** Look at both.\n\n**The final answer is:** They differ."
        assert _strip_reasoning_scaffold(text) == "They differ."

    def test_empty_final_answer_falls_back_to_original(self):
        # Never return "" — an empty response poisons replayed history downstream.
        text = "Step 1: Think about it.\n\nThe final answer is:"
        assert _strip_reasoning_scaffold(text) == text


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


class TestNoToolAccessNotice:
    """_classify_needs_tool occasionally misroutes a question that actually needs a
    tool (e.g. "How is Darcy permeability computed from a lattice Boltzmann
    simulation?") to _answer_direct — reproduced live 3/3 times for that exact
    question. Once there, the model still sees SYSTEM_PROMPT's Tier 1/2 "call the
    tool first" instructions and, having no real tool access in that call, was
    observed fabricating an entire fake tool-use transcript (invented
    get_workflow_guidance output, invented dataset search results, fabricated DOIs)
    rather than admitting it couldn't comply. _NO_TOOL_ACCESS_NOTICE must reach the
    model on every path that has no tools bound, so it never does this."""

    def test_answer_direct_includes_no_tool_access_notice(self):
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = MagicMock(content="Some answer.")
            _answer_direct("How is Darcy permeability computed from an LBM simulation?", [])

        messages = mock_get_llm.return_value.invoke.call_args[0][0]
        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        assert _NO_TOOL_ACCESS_NOTICE in system_contents


class TestChatHonestFallback:
    """When a 400 tool-format error yields an identified tool call but dispatch still
    produces nothing usable, chat() must give up honestly rather than call the LLM with
    no tool context (which has no grounding and will hedge/guess). Same standard now
    applies when NO tool call can even be identified from the error text at all (Fix 3:
    previously fell back to a fully ungrounded direct LLM guess with zero indication
    anything went wrong — this is the actual mechanism behind a citation/notebook
    reference silently vanishing, e.g. get_workflow_guidance's LaTeX-heavy answer
    computed correctly but the relay turn 400'd and nothing was recoverable)."""

    def test_returns_honest_message_when_dispatch_produces_nothing(self):
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        manager._agent.stream.side_effect = Exception(
            "400 Invalid function calling output: JSONDecodeError near get_dataset_details"
        )
        with patch("src.assistant.conversation_manager._extract_tool_calls_from_error",
                   return_value=[{"name": "get_dataset_details", "args": {"question": "x"}}]):
            with patch("src.assistant.conversation_manager._run_manual_dispatch", return_value=None):
                with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False):
                    result = manager.chat("porosity above 0.4")
        assert result == _HONEST_TOOL_FAILURE_MSG

    def test_returns_honest_message_when_no_tool_call_identified_from_error(self):
        """The former "last resort" branch (no tool call identifiable from the 400
        error text at all) must also give up honestly rather than making an ungrounded
        direct LLM call — and must not make any LLM call in doing so."""
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        manager._agent.stream.side_effect = Exception(
            "400 Invalid function calling output: could not parse tool call"
        )
        with patch("src.assistant.conversation_manager._extract_tool_calls_from_error", return_value=[]):
            with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False):
                with patch("src.assistant.conversation_manager._classify_needs_tool", return_value=True):
                    with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
                        result = manager.chat("How is Darcy permeability computed from an LBM simulation?")

        assert result == _HONEST_TOOL_FAILURE_MSG
        mock_get_llm.assert_not_called()


# LangGraph's stream(..., stream_mode="values") yields the FULL accumulated state
# (input messages + new turns) at each step, not just the newly-produced messages —
# these fixtures prepend a placeholder standing in for the echoed input message so
# chat()'s `result["messages"][len(messages):]` slicing lands on the right elements,
# same as it would against a real graph.
_USER_ECHO = {"role": "user", "content": "placeholder"}


class TestPreemptSecondTurn:
    """chat() streams the ReAct agent instead of a single .invoke() so that a single
    self-contained/verbatim tool call can be dispatched directly, without ever
    invoking the graph's second ("relay the tool result") model turn — that turn is
    where LaTeX-heavy self-contained answers (e.g. get_workflow_guidance's) sometimes
    get mis-detected by LiteLLM as a malformed function call and 400, and its own
    output is discarded unconditionally anyway whenever exactly one self-contained/
    verbatim tool ran (see the post-hoc verbatim/self-contained checks further down
    in chat())."""

    def _ai_message_with_tool_calls(self, tool_calls):
        msg = MagicMock()
        msg.tool_calls = tool_calls
        return msg

    def test_single_self_contained_tool_call_short_circuits_second_turn(self):
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        first_ai_msg = self._ai_message_with_tool_calls(
            [{"name": "get_workflow_guidance", "args": {"goal": "compute porosity"}}]
        )
        # stream_mode="values" yields the echoed input FIRST, then the state after
        # the agent node's first execution — two next() calls, two list entries here.
        manager._agent.stream.return_value = iter([
            {"messages": [_USER_ECHO]},
            {"messages": [_USER_ECHO, first_ai_msg]},
        ])

        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False), \
             patch("src.assistant.conversation_manager._classify_needs_tool", return_value=True), \
             patch(
                 "src.assistant.conversation_manager._run_manual_dispatch",
                 return_value="Here's the Minkowski Functionals notebook link.",
             ) as mock_dispatch:
            result = manager.chat("how to compute porosity from an image")

        assert result == "Here's the Minkowski Functionals notebook link."
        mock_dispatch.assert_called_once_with(
            [{"name": "get_workflow_guidance", "args": {"goal": "compute porosity"}}],
            "how to compute porosity from an image",
            [],
        )

    def test_dispatch_failure_falls_through_to_normal_graph_execution(self):
        """If the pre-emptive manual dispatch itself fails (tool error), fall through
        to the normal post-hoc handling on the already-streamed result rather than
        crashing or silently losing the turn."""
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        first_ai_msg = self._ai_message_with_tool_calls(
            [{"name": "get_workflow_guidance", "args": {"goal": "compute porosity"}}]
        )
        tool_msg = MagicMock(spec=ToolMessage)
        tool_msg.name = "get_workflow_guidance"
        tool_msg.content = "The Minkowski Functionals notebook explains this."
        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False), \
             patch("src.assistant.conversation_manager._classify_needs_tool", return_value=True), \
             patch("src.assistant.conversation_manager._run_manual_dispatch", return_value=None) as mock_dispatch:
            manager._agent.stream.return_value = iter([
                {"messages": [_USER_ECHO]},
                {"messages": [_USER_ECHO, first_ai_msg]},
                {"messages": [_USER_ECHO, first_ai_msg, tool_msg]},
            ])
            result = manager.chat("how to compute porosity from an image")

        mock_dispatch.assert_called_once_with(
            [{"name": "get_workflow_guidance", "args": {"goal": "compute porosity"}}],
            "how to compute porosity from an image",
            [],
        )
        assert result == "The Minkowski Functionals notebook explains this."

    def test_sequential_cross_intent_call_skips_short_circuit(self):
        """Live-verified this model requests cross-intent tools SEQUENTIALLY (one
        tool call on the first turn, deciding on a second tool only after seeing that
        answer), not in one parallel tool_calls list — so a single-tool-call first
        turn alone isn't a reliable signal that no follow-up call is coming.
        _needs_followup_tool_call must be consulted before short-circuiting, and when
        it says a follow-up is needed, the normal (non-short-circuited) graph
        execution must run so the model gets the chance to make that second call."""
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        first_ai_msg = self._ai_message_with_tool_calls(
            [{"name": "get_workflow_guidance", "args": {"goal": "compute relative permeability"}}]
        )
        final_msg = MagicMock(content="Combined answer citing both tools.")
        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False), \
             patch("src.assistant.conversation_manager._classify_needs_tool", return_value=True), \
             patch("src.assistant.conversation_manager._needs_followup_tool_call", return_value=True), \
             patch("src.assistant.conversation_manager._run_manual_dispatch") as mock_dispatch:
            manager._agent.stream.return_value = iter([
                {"messages": [_USER_ECHO]},
                {"messages": [_USER_ECHO, first_ai_msg]},
                {"messages": [_USER_ECHO, first_ai_msg, final_msg]},
            ])
            result = manager.chat(
                "How do I compute relative permeability, and can you also find datasets that measure it?"
            )

        mock_dispatch.assert_not_called()
        assert result == "Combined answer citing both tools."


class TestNeedsFollowupToolCall:
    def test_call_failure_defaults_to_no_followup_needed(self):
        """Default False (proceed with short-circuit) on failure — the short-circuit
        exists to fix a confirmed 400-error bug, so an uncertain case shouldn't
        reintroduce that risk just to preserve a maybe-needed follow-up call."""
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.side_effect = RuntimeError("boom")
            assert _needs_followup_tool_call("some query", "get_workflow_guidance") is False

    def test_plain_single_intent_question_does_not_need_followup(self):
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = MagicMock(
                content='{"needs_followup": false}'
            )
            assert _needs_followup_tool_call(
                "How do I compute relative permeability?", "get_workflow_guidance"
            ) is False

    def test_cross_intent_question_needs_followup(self):
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = MagicMock(
                content='{"needs_followup": true}'
            )
            assert _needs_followup_tool_call(
                "How do I compute relative permeability, and can you also find datasets that measure it?",
                "get_workflow_guidance",
            ) is True

    def test_dataset_comparison_with_same_tool_needs_followup(self):
        """A comparison names a second dataset but calls the SAME tool
        (get_dataset_profile) both times — the gate's added rule for this case must
        still surface needs_followup=True rather than short-circuiting after the
        first dataset's profile. See _FOLLOWUP_TOOL_GATE_SYSTEM_PROMPT's added example."""
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = MagicMock(
                content='{"needs_followup": true}'
            )
            assert _needs_followup_tool_call(
                "Compare Dataset A and Dataset B for two-phase flow simulation",
                "get_dataset_profile",
            ) is True


class TestExtractToolCallsMultiArg:
    """get_dataset_profile is the first tool with two required args — both
    reconstruction paths must recover BOTH (dataset_reference, question), not just
    one, or the 400-error recovery path silently drops the question arg."""

    def test_extract_from_text_recovers_both_args(self):
        text = (
            '<|python_start|>get_dataset_profile('
            'dataset_reference="Bentheimer Sandstone", question="what is its porosity?"'
            ')<|python_end|>'
        )
        calls = _extract_tool_calls_from_text(text)
        assert calls == [{
            "name": "get_dataset_profile",
            "args": {"dataset_reference": "Bentheimer Sandstone", "question": "what is its porosity?"},
        }]

    def test_extract_from_error_recovers_both_args_via_json_strategy(self):
        err_str = (
            'LiteLLM Exception: tool call get_dataset_profile '
            '{"dataset_reference": "Bentheimer Sandstone", "question": "what is its porosity?"}'
        )
        calls = _extract_tool_calls_from_error(err_str)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_dataset_profile"
        assert calls[0]["args"] == {
            "dataset_reference": "Bentheimer Sandstone",
            "question": "what is its porosity?",
        }

    def test_single_arg_tool_still_works(self):
        text = '<|python_start|>get_workflow_guidance(goal="compute permeability")<|python_end|>'
        calls = _extract_tool_calls_from_text(text)
        assert calls == [{"name": "get_workflow_guidance", "args": {"goal": "compute permeability"}}]


class TestPreemptSecondTurnMultiTool:
    def test_multiple_tool_calls_in_first_turn_do_not_short_circuit(self):
        """A first turn with more than one tool call (e.g. a cross-intent query) must
        NOT be short-circuited — the existing full-graph-execution path (and its
        post-hoc verbatim/self-contained checks) must still run unchanged."""
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        first_ai_msg = MagicMock()
        first_ai_msg.tool_calls = [
            {"name": "get_workflow_guidance", "args": {"goal": "compute permeability"}},
            {"name": "search_datasets", "args": {"query": "permeability datasets"}},
        ]
        final_msg = MagicMock(content="Combined answer citing both tools.")
        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False), \
             patch("src.assistant.conversation_manager._classify_needs_tool", return_value=True), \
             patch("src.assistant.conversation_manager._run_manual_dispatch") as mock_dispatch:
            manager._agent.stream.return_value = iter([
                {"messages": [_USER_ECHO]},
                {"messages": [_USER_ECHO, first_ai_msg]},
                {"messages": [_USER_ECHO, first_ai_msg, final_msg]},
            ])
            result = manager.chat("how do I compute permeability and find datasets for it?")

        mock_dispatch.assert_not_called()
        assert result == "Combined answer citing both tools."


class TestOffDomainSteerBack:
    """_classify_needs_tool has no off-domain route at all — an off-domain query like
    'why would someone be allergic to peanuts? How do I make a jelly donut?' gets
    routed 'direct' and _answer_direct's Tier-0 instruction to 'steer back' was
    followed only partially (reproduced live 3/3): the model acknowledges the mismatch,
    then answers the off-topic question(s) anyway. _classify_off_domain + a fixed,
    code-returned message closes this because no further LLM call can 'helpfully'
    continue past the fixed string."""

    def test_off_domain_query_returns_fixed_steer_back_message_with_no_agent_call(self):
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=True):
            result = manager.chat(
                "why would someone be allergic to peanuts? How do I make a jelly donut?"
            )
        assert result == _OFF_DOMAIN_STEER_BACK_MSG
        manager._agent.stream.assert_not_called()

    def test_classifier_call_failure_defaults_to_in_domain(self):
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.side_effect = RuntimeError("boom")
            assert _classify_off_domain("some query", []) is False

    def test_classifier_unparseable_response_defaults_to_in_domain(self):
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = MagicMock(content="not json at all")
            assert _classify_off_domain("some query", []) is False

    @pytest.mark.parametrize("query", [
        "What is porosity?",
        "Explain Darcy's law",
        "How do I compute relative permeability?",
        "Can you help me think through my sampling design?",
        "why is my segmentation pipeline crashing?",
        "How do I upload a dataset to the portal?",
    ])
    def test_legitimate_domain_questions_not_classified_off_domain(self, query):
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = MagicMock(
                content='{"route": "in_domain"}'
            )
            assert _classify_off_domain(query, []) is False

    def test_off_domain_example_query_classified_off_domain(self):
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = MagicMock(
                content='{"route": "off_domain"}'
            )
            assert _classify_off_domain(
                "why would someone be allergic to peanuts? How do I make a jelly donut?", []
            ) is True

    def test_mixed_off_topic_and_real_question_routes_in_domain(self):
        """A message combining off-topic chatter with a genuine domain question must
        not be blocked — mirrors _GATE_SYSTEM_PROMPT's existing mixed-query handling
        for the tool/direct gate."""
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = MagicMock(
                content='{"route": "in_domain"}'
            )
            assert _classify_off_domain(
                "sorry for the random question but also, what is porosity?", []
            ) is False

    def test_portal_how_to_question_unaffected_by_off_domain_gate(self):
        """A portal-action question must still reach the tool-bound agent path when
        the off-domain gate correctly does not fire."""
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        final_message = MagicMock(content="Here's how to upload a dataset...")
        final_message.tool_calls = []
        manager._agent.stream.return_value = iter([
            {"messages": [_USER_ECHO]},
            {"messages": [_USER_ECHO, final_message]},
        ])
        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False), \
             patch("src.assistant.conversation_manager._classify_needs_tool", return_value=True):
            manager.chat("How do I upload a dataset?")
        manager._agent.stream.assert_called_once()
