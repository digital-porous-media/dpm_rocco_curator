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
    _assemble_response,
    _needs_followup_lookup,
    _run_manual_dispatch,
    _strip_reasoning_scaffold,
    _strip_recap_paragraph,
    _uncovered_requests,
    _with_uncovered_segment,
    Segment,
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

    def test_reason_about_dataset_content_is_self_contained_not_verbatim(self):
        """Its output carries a fixed honesty framing plus citation-checked titles/DOIs
        taken from graph records. Re-synthesizing it through the outer agent would hand
        the model back the two things the tool exists to guarantee in code."""
        assert "reason_about_dataset_content" in _SELF_CONTAINED_TOOLS
        assert "reason_about_dataset_content" not in _VERBATIM_TOOLS

    def test_reason_about_dataset_content_is_recoverable_from_a_400(self):
        """Without an entry here, a tool-format 400 on this call can't be recovered and
        the turn degrades to the honest-failure message instead of a real answer."""
        from src.assistant.conversation_manager import _TOOL_PARAM_KEYS
        assert _TOOL_PARAM_KEYS["reason_about_dataset_content"] == ["question"]

    def test_search_datasets_framing_unchanged(self):
        hit = "DOI: 10.1/x\ntitle: foo"
        result = _build_verbatim_response("find sandstone datasets", hit)
        assert result.startswith("Here are the datasets matching your query:")
        assert "verify these datasets on the DPM Portal" in result


class TestCumulativeFilterTracking:
    """Live report: a 3-turn chain ("find sandstone" -> "porosity above 0.3" -> "how about
    below 0.25?") lost the sandstone constraint. The agent itself composed the right
    self-contained question ("sandstone datasets with porosity below 0.25"), but the tracked
    filter chain was overwritten with the bare user message ("How about any below 0.25?"),
    so the NEXT refinement built its compound question from a chain that had forgotten two
    turns of context."""

    def test_prefers_the_tools_own_question_over_the_raw_message(self):
        from src.assistant.conversation_manager import _tool_filter_text
        assert _tool_filter_text(
            {"question": "sandstone datasets with porosity below 0.25"},
            "How about any below 0.25?",
        ) == "sandstone datasets with porosity below 0.25"

    def test_prefers_the_tools_own_query_for_search_datasets(self):
        from src.assistant.conversation_manager import _tool_filter_text
        assert _tool_filter_text({"query": "sandstone dataset"}, "raw") == "sandstone dataset"

    def test_falls_back_to_the_user_message_when_no_text_arg(self):
        from src.assistant.conversation_manager import _tool_filter_text
        assert _tool_filter_text({"top_k": 5}, "find sandstone") == "find sandstone"
        assert _tool_filter_text(None, "find sandstone") == "find sandstone"

    def test_blank_arg_falls_back_rather_than_wiping_the_chain(self):
        from src.assistant.conversation_manager import _tool_filter_text
        assert _tool_filter_text({"question": "   "}, "find sandstone") == "find sandstone"

    def test_tracked_chain_keeps_the_constraint_the_tool_actually_used(self):
        """End-to-end on the tracker itself: the stored chain must be the tool's question."""
        from src.assistant.conversation_manager import ConversationManager, _tool_filter_text
        mgr = object.__new__(ConversationManager)
        listing = "- **Gildehauser Sandstone** (DOI: 10.17612/P7WW95)"
        mgr._track_dataset_listing(
            "get_dataset_details", listing,
            _tool_filter_text({"question": "sandstone datasets with porosity below 0.25"},
                              "How about any below 0.25?"),
        )
        assert "sandstone" in mgr._cumulative_filter_text
        assert mgr._cumulative_filter_text != "How about any below 0.25?"


class TestEllipticalRefinementRestriction:
    """Reported live: a 3-turn chain ("find sandstone" -> "which ones have porosity above
    0.3?" -> "how about any below 0.25?") answered turn 3 across the WHOLE catalog instead of
    the sandstone datasets found two turns earlier. Turn 2 matched _REFINEMENT_RE ("which
    ones") and was restricted; turn 3 named the prior set nowhere, so it bypassed that path
    and reached the agent unrestricted."""

    def _manager(self, titles=("Gildehauser Sandstone", "Bentheimer Sandstone")):
        mgr = object.__new__(ConversationManager)
        mgr._last_dataset_mentions = [{"title": t, "doi": None} for t in titles]
        return mgr

    def test_elliptical_followup_gets_restricted_to_the_prior_listing(self):
        mgr = self._manager()
        args = mgr._with_result_set_restriction(
            "get_dataset_details",
            {"question": "sandstone datasets with porosity below 0.25"},
            "How about any below 0.25?",
        )
        assert args["restrict_to_titles"] == ["Gildehauser Sandstone", "Bentheimer Sandstone"]
        # The agent's own question is left alone — it supersedes the replaced constraint
        # correctly, which blind AND-composition cannot.
        assert args["question"] == "sandstone datasets with porosity below 0.25"

    def test_topic_change_is_not_restricted(self):
        """"What about carbonate datasets?" names a new subject and must search the whole
        catalog — restricting it to a prior sandstone listing would return nothing."""
        mgr = self._manager()
        args = mgr._with_result_set_restriction(
            "get_dataset_details", {"question": "carbonate datasets"},
            "What about carbonate datasets?",
        )
        assert "restrict_to_titles" not in args

    def test_fresh_search_is_not_restricted(self):
        mgr = self._manager()
        args = mgr._with_result_set_restriction(
            "get_dataset_details", {"question": "coal datasets"}, "Find me coal datasets",
        )
        assert "restrict_to_titles" not in args

    def test_no_prior_listing_means_no_restriction(self):
        mgr = object.__new__(ConversationManager)
        mgr._last_dataset_mentions = []
        args = mgr._with_result_set_restriction(
            "get_dataset_details", {"question": "q"}, "How about any below 0.25?",
        )
        assert "restrict_to_titles" not in args

    def test_never_overrides_a_restriction_already_set(self):
        mgr = self._manager()
        args = mgr._with_result_set_restriction(
            "get_dataset_details",
            {"question": "q", "restrict_to_titles": ["Only This One"]},
            "How about any below 0.25?",
        )
        assert args["restrict_to_titles"] == ["Only This One"]

    def test_only_applies_to_the_tool_that_accepts_the_parameter(self):
        mgr = self._manager()
        args = mgr._with_result_set_restriction(
            "search_datasets", {"query": "sandstone"}, "How about any below 0.25?",
        )
        assert "restrict_to_titles" not in args

    def test_does_not_collide_with_the_deterministic_refinement_dispatch(self):
        """_REFINEMENT_RE phrasings keep taking the compound-question path (which composes
        AND restricts); these two mechanisms must not both claim the same message."""
        from src.assistant.conversation_manager import (
            _REFINEMENT_RE, _ELLIPTICAL_REFINEMENT_RE,
        )
        assert _REFINEMENT_RE.search("of these, which are segmented?")
        assert not _ELLIPTICAL_REFINEMENT_RE.search("of these, which are segmented?")
        assert _ELLIPTICAL_REFINEMENT_RE.search("How about any below 0.25?")
        assert not _REFINEMENT_RE.search("How about any below 0.25?")


class TestContinuesFilterChain:
    """The phrasing-independent refinement signal. Recognising refinement from the USER's
    wording failed repeatedly — "of these", "which ones", "any below 0.25", "are there any
    with porosity > 0.3", "how about with porosity > 0.2" are one intent worded five ways,
    and each new transcript brought a phrasing the pattern list lacked. The agent's own
    composed question carries the accumulated constraints on every turn, so compare that."""

    def _c(self, new_q, prior):
        from src.assistant.conversation_manager import _continues_filter_chain
        return _continues_filter_chain(new_q, prior)

    def test_narrowing_the_same_subject_continues(self):
        assert self._c("sandstone datasets with porosity > 0.3", "sandstone dataset")

    def test_changing_only_the_threshold_continues(self):
        """The number is what changes between refinement turns, so it must not count."""
        assert self._c("sandstone datasets with porosity > 0.2",
                       "sandstone datasets with porosity > 0.3")

    def test_changing_the_comparison_direction_continues(self):
        """"above 0.3" -> "below 0.25" is still the same chain."""
        assert self._c("sandstone datasets with porosity below 0.25",
                       "sandstone datasets with porosity above 0.3")

    def test_symbolic_and_worded_comparisons_are_equivalent(self):
        assert self._c("sandstone datasets with porosity > 0.2",
                       "sandstone datasets with porosity above 0.3")

    def test_topic_change_does_not_continue(self):
        assert not self._c("carbonate datasets", "sandstone datasets with porosity > 0.3")

    def test_unrelated_search_does_not_continue(self):
        assert not self._c("coal datasets", "sandstone dataset")

    def test_dropping_a_prior_constraint_does_not_continue(self):
        """Losing the subject means no restriction — the safe direction, since a wrongly
        narrowed answer is worse than a catalog-wide one."""
        assert not self._c("datasets with porosity > 0.2",
                           "sandstone datasets with porosity > 0.3")

    def test_plural_forms_match(self):
        assert self._c("sandstone datasets", "sandstones dataset")

    def test_no_prior_chain_does_not_continue(self):
        assert not self._c("sandstone datasets", None)
        assert not self._c("sandstone datasets", "")

    def test_generic_chain_with_no_subject_does_not_continue(self):
        """A chain of nothing but stopwords must not make everything a refinement."""
        assert not self._c("carbonate datasets", "the datasets")


class TestRefinementRestrictionEndToEnd:
    """Both reported transcripts, replayed through the real decision function."""

    TURN1 = ["Bentheimer Sandstone", "Gildehauser Sandstone",
             "Digital Rendering of Sedimentary Relief Peels"]

    def _mgr(self, mentions, chain):
        mgr = object.__new__(ConversationManager)
        mgr._last_dataset_mentions = [{"title": t, "doi": None} for t in mentions]
        mgr._cumulative_filter_text = chain
        return mgr

    @pytest.mark.parametrize("user_msg,agent_question,prior_chain", [
        # Transcript 2
        ("Are there any with porosity > 0.3?", "sandstone datasets with porosity > 0.3",
         "sandstone dataset"),
        ("How about with porosity > 0.2?", "sandstone datasets with porosity > 0.2",
         "sandstone datasets with porosity > 0.3"),
        # Transcript 1
        ("Which ones have porosity above 0.3?", "sandstone datasets with porosity above 0.3",
         "sandstone dataset"),
        ("How about any below 0.25?", "sandstone datasets with porosity below 0.25",
         "sandstone datasets with porosity above 0.3"),
    ])
    def test_reported_refinements_are_restricted(self, user_msg, agent_question, prior_chain):
        mgr = self._mgr(self.TURN1, prior_chain)
        args = mgr._with_result_set_restriction(
            "get_dataset_details", {"question": agent_question}, user_msg
        )
        assert args.get("restrict_to_titles") == self.TURN1, f"not restricted: {user_msg!r}"
        assert args["question"] == agent_question  # agent keeps ownership of the question

    @pytest.mark.parametrize("user_msg,agent_question", [
        ("What about carbonate datasets?", "carbonate datasets"),
        ("Find me coal datasets", "coal datasets"),
        ("Show me datasets by Jane Doe", "datasets authored by Jane Doe"),
    ])
    def test_topic_changes_stay_catalog_wide(self, user_msg, agent_question):
        mgr = self._mgr(self.TURN1, "sandstone datasets with porosity > 0.3")
        args = mgr._with_result_set_restriction(
            "get_dataset_details", {"question": agent_question}, user_msg
        )
        assert "restrict_to_titles" not in args, f"wrongly restricted: {user_msg!r}"


class TestToolArgsByCallId:
    def test_maps_tool_call_ids_to_their_args(self):
        from src.assistant.conversation_manager import _tool_args_by_call_id

        class _AI:
            tool_calls = [{"id": "call_1", "name": "get_dataset_details",
                           "args": {"question": "sandstone with porosity below 0.25"}}]

        assert _tool_args_by_call_id([_AI()])["call_1"]["question"] == (
            "sandstone with porosity below 0.25"
        )

    def test_ignores_messages_without_tool_calls(self):
        from src.assistant.conversation_manager import _tool_args_by_call_id

        class _Plain:
            content = "hello"

        assert _tool_args_by_call_id([_Plain()]) == {}


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
             patch("src.assistant.conversation_manager._uncovered_requests", return_value=[]), \
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
            uncovered=[],
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
             patch("src.assistant.conversation_manager._uncovered_requests", return_value=[]), \
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
            uncovered=[],
        )
        assert result == "The Minkowski Functionals notebook explains this."

    def test_sequential_cross_intent_call_skips_short_circuit(self):
        """Live-verified this model requests cross-intent tools SEQUENTIALLY (one
        tool call on the first turn, deciding on a second tool only after seeing that
        answer), not in one parallel tool_calls list — so a single-tool-call first
        turn alone isn't a reliable signal that no follow-up call is coming.
        The coverage gate must be consulted before short-circuiting, and when it
        reports an uncovered part that needs a LOOKUP, the normal (non-short-circuited)
        graph execution must run so the model gets the chance to make that second call.
        An uncovered part needing no lookup is handled differently — see
        TestCompoundQuestionAssembly."""
        manager = object.__new__(ConversationManager)
        manager._agent = MagicMock()
        first_ai_msg = self._ai_message_with_tool_calls(
            [{"name": "get_workflow_guidance", "args": {"goal": "compute relative permeability"}}]
        )
        final_msg = MagicMock(content="Combined answer citing both tools.")
        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False), \
             patch("src.assistant.conversation_manager._classify_needs_tool", return_value=True), \
             patch(
                 "src.assistant.conversation_manager._uncovered_requests",
                 return_value=[{"asks_for": "find datasets that measure it", "needs_lookup": True}],
             ), \
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


class TestUncoveredRequests:
    """The coverage gate reports which parts of a message the one called tool won't
    answer. Its predecessor asked for a yes/no "needs_followup" verdict and taught the
    boundary purely by example, all of which paired a tool with a SECOND TOOL — so a
    part needing no tool at all fell outside every example and the gate said no. That
    is the live-reported bug where "What is porosity and how do I compute it from a
    microCT image?" came back with only the workflow half."""

    def _llm_returning(self, payload):
        m = MagicMock()
        m.return_value.invoke.return_value = MagicMock(content=payload)
        return m

    def test_call_failure_defaults_to_nothing_uncovered(self):
        """Default [] (relay proceeds) on failure — the relay short-circuit exists to fix
        a confirmed 400-error bug, so an uncertain case shouldn't reintroduce that risk."""
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value.invoke.side_effect = RuntimeError("boom")
            assert _uncovered_requests("some query", "get_workflow_guidance") == []

    def test_malformed_entries_are_ignored(self):
        """`covered` missing defaults to True, and a blank asks_for is dropped: a
        malformed entry must not spuriously pull an extra LLM call into a single-intent
        turn."""
        payload = (
            '{"requests": [{"asks_for": "compute porosity"}, '
            '{"asks_for": "", "covered": false}, "junk"]}'
        )
        with patch("src.assistant.llm.get_chat_model", self._llm_returning(payload)), \
             patch("src.assistant.conversation_manager._tool_description", return_value="desc"):
            assert _uncovered_requests("q", "get_workflow_guidance") == []

    def test_single_intent_question_has_nothing_uncovered(self):
        payload = '{"requests": [{"asks_for": "compute relative permeability", "covered": true, "needs_lookup": false}]}'
        with patch("src.assistant.llm.get_chat_model", self._llm_returning(payload)), \
             patch("src.assistant.conversation_manager._tool_description", return_value="desc"):
            assert _uncovered_requests(
                "How do I compute relative permeability?", "get_workflow_guidance"
            ) == []

    def test_no_lookup_half_is_reported_uncovered(self):
        """The reported bug's exact shape: a Tier 3 definition alongside a Tier 2
        workflow. The definition needs no lookup, so it is answerable alongside the
        relayed tool output rather than requiring a second tool call."""
        payload = (
            '{"requests": ['
            '{"asks_for": "define porosity", "covered": false, "needs_lookup": false},'
            '{"asks_for": "compute porosity from a microCT image", "covered": true, "needs_lookup": false}'
            ']}'
        )
        with patch("src.assistant.llm.get_chat_model", self._llm_returning(payload)), \
             patch("src.assistant.conversation_manager._tool_description", return_value="desc"):
            uncovered = _uncovered_requests(
                "What is porosity and how do I compute it from a microCT image?",
                "get_workflow_guidance",
            )
        assert [u["asks_for"] for u in uncovered] == ["define porosity"]
        assert _needs_followup_lookup(uncovered) is False

    def test_lookup_half_suppresses_the_short_circuit(self):
        payload = (
            '{"requests": ['
            '{"asks_for": "compute relative permeability", "covered": true, "needs_lookup": false},'
            '{"asks_for": "find datasets that measure it", "covered": false, "needs_lookup": true}'
            ']}'
        )
        with patch("src.assistant.llm.get_chat_model", self._llm_returning(payload)), \
             patch("src.assistant.conversation_manager._tool_description", return_value="desc"):
            uncovered = _uncovered_requests(
                "How do I compute relative permeability, and can you also find datasets that measure it?",
                "get_workflow_guidance",
            )
        assert _needs_followup_lookup(uncovered) is True

    def test_tool_description_is_injected_not_restated(self):
        """Coverage is judged against the tool's LIVE description from tools.py, so the
        gate never carries a second hand-written copy of each tool's scope to drift."""
        captured = {}

        def _capture(messages):
            captured["system"] = messages[0]["content"]
            return MagicMock(content='{"requests": []}')

        mock_llm = MagicMock()
        mock_llm.return_value.invoke.side_effect = _capture
        with patch("src.assistant.llm.get_chat_model", mock_llm):
            _uncovered_requests("q", "get_workflow_guidance")

        assert "get_workflow_guidance" in captured["system"]
        # The real description's own exclusion clause must be present verbatim.
        assert "search_portal_docs" in captured["system"]


class TestAssembleResponse:
    def test_orders_segments_by_request_position(self):
        out = _assemble_response([
            Segment(kind="verbatim", content="tool answer", order=1),
            Segment(kind="generated", content="definition", order=0),
        ])
        assert out == "definition\n\ntool answer"

    def test_equal_orders_keep_append_order(self):
        out = _assemble_response([
            Segment(kind="generated", content="first"),
            Segment(kind="generated", content="second"),
        ])
        assert out == "first\n\nsecond"

    def test_empty_segment_is_dropped_not_left_as_a_gap(self):
        """A generated segment comes back empty when its LLM call failed. The verbatim
        tool output is real grounded data and must still reach the user, with no blank
        gap where the other segment would have been."""
        out = _assemble_response([
            Segment(kind="generated", content="   ", order=0),
            Segment(kind="verbatim", content="tool answer", order=1),
        ])
        assert out == "tool answer"

    def test_all_empty_falls_back_rather_than_returning_empty_string(self):
        """An empty response appended to the UI's history poisons every later turn's
        replayed context — same reason _non_empty exists."""
        assert _assemble_response([Segment(kind="generated", content="")]) == _HONEST_TOOL_FAILURE_MSG

    def test_verbatim_content_is_not_reworded(self):
        tool_bytes = "- **Estaillades Carbonate** (DOI: 10.17612/abc-123)"
        out = _assemble_response([
            Segment(kind="generated", content="Porosity is void volume over total volume.", order=0),
            Segment(kind="verbatim", content=tool_bytes, order=1),
        ])
        assert tool_bytes in out


class TestCompoundQuestionAssembly:
    """End-to-end for the reported bug: one tool call plus a no-tool half must produce
    BOTH, with the tool's bytes untouched. Before the assembler, the relay path returned
    the tool's output as the entire response and the definition half — already computed
    by the model — was discarded with nothing revealing the loss."""

    def test_relay_keeps_tool_bytes_and_gains_the_uncovered_half(self):
        tool_output = "Segment the image, then count pore voxels.\n\n  - **connected components** — path.ipynb"
        with patch("src.assistant.conversation_manager._answer_uncovered", return_value="Porosity is the void fraction."):
            out = _with_uncovered_segment(
                tool_output,
                [{"asks_for": "define porosity", "needs_lookup": False}],
                "What is porosity and how do I compute it from a microCT image?",
                [],
            )
        assert out.startswith("Porosity is the void fraction.")
        assert tool_output.strip() in out

    def test_nothing_uncovered_relays_tool_bytes_unchanged(self):
        """The single-segment case — every turn that already behaved correctly. No extra
        LLM call, no wrapping, byte-identical output."""
        with patch("src.assistant.conversation_manager._answer_uncovered") as mock_answer:
            out = _with_uncovered_segment("tool answer", [], "q", [])
        mock_answer.assert_not_called()
        assert out == "tool answer"

    def test_lookup_parts_are_never_answered_from_model_knowledge(self):
        """A needs_lookup part is the graph's job, not the generated segment's. Filling
        it here would fabricate exactly the dataset facts the lookup was meant to supply."""
        with patch("src.assistant.conversation_manager._answer_uncovered") as mock_answer:
            out = _with_uncovered_segment(
                "tool answer",
                [{"asks_for": "find datasets measuring it", "needs_lookup": True}],
                "q",
                [],
            )
        mock_answer.assert_not_called()
        assert out == "tool answer"

    def test_failed_uncovered_answer_still_relays_the_tool_output(self):
        with patch("src.assistant.conversation_manager._answer_uncovered", return_value=""):
            out = _with_uncovered_segment(
                "tool answer", [{"asks_for": "define porosity", "needs_lookup": False}], "q", [],
            )
        assert out == "tool answer"


class TestFourHundredRecoveryCoverage:
    """The 400 path never consulted the old gate at all, so a correctly-detected
    multi-part turn still collapsed to the one recovered call's output. This is the path
    the reported failure actually went through (its log line reads "Tool-call format
    mismatch (400); attempting manual dispatch")."""

    def test_single_recovered_relay_call_is_coverage_checked(self):
        from src.assistant.conversation_manager import _recovered_call_coverage

        with patch(
            "src.assistant.conversation_manager._uncovered_requests",
            return_value=[{"asks_for": "define porosity", "needs_lookup": False}],
        ) as mock_gate:
            out = _recovered_call_coverage(
                [{"name": "get_workflow_guidance", "args": {"goal": "compute porosity"}}],
                "What is porosity and how do I compute it?",
            )
        mock_gate.assert_called_once()
        assert [u["asks_for"] for u in out] == ["define porosity"]

    def test_multiple_recovered_calls_skip_the_gate(self):
        """Two or more recovered calls go through synthesis, which already sees the full
        question — nothing to assemble, and no reason to pay for the gate."""
        from src.assistant.conversation_manager import _recovered_call_coverage

        with patch("src.assistant.conversation_manager._uncovered_requests") as mock_gate:
            out = _recovered_call_coverage(
                [
                    {"name": "get_dataset_profile", "args": {"dataset_reference": "A", "question": "q"}},
                    {"name": "get_dataset_profile", "args": {"dataset_reference": "B", "question": "q"}},
                ],
                "compare A and B",
            )
        mock_gate.assert_not_called()
        assert out == []


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


class TestContentReasoningIsTrackedAsADatasetListing:
    """Reported live: "are there any datasets where there are both raw and segmented
    images?" answered correctly via reason_about_dataset_content (12 datasets), then
    "What are the lithologies of these?" answered about a completely different set.

    Root cause: reason_about_dataset_content was absent from _DATASET_LISTING_TOOLS, so the
    turn was invisible to _track_dataset_listing. That didn't merely fail to record the 12 —
    it left the PREVIOUS chain ("datasets suitable for training a segmentation model", 10
    titles) sitting there looking current, so the follow-up's _REFINEMENT_RE match refined a
    result set the user had moved on from, with nothing in the answer revealing it."""

    _CONTENT_REASONING_OUTPUT = (
        "[content reasoning] I can't confirm this from a database field — here's what "
        "reasoning over the available facts and descriptions suggests.\n"
        "\n"
        "- **Estaillades Carbonate #2** (DOI: 10.17612/P7C09J)\n"
        "  Both raw and segmented images are available.\n"
        "  *Basis:* Segmented — segmented: yes\n"
        "- **Mt. Simon Sandstone with Mineral Map** (DOI: 10.17612/sx6n-kn96)\n"
        "  The dataset includes a segmented image with mineral map.\n"
        "  *Basis:* Mt Simon Data with Mineral Map — segmented: yes\n"
    )

    def _stale_manager(self):
        """A manager holding the tracked state of an earlier, unrelated listing turn."""
        mgr = object.__new__(ConversationManager)
        mgr._last_dataset_mentions = [
            {"title": "Grain Packing", "doi": None},
            {"title": "Trabecular bone in the femoral head of strepsirrhine primates", "doi": None},
        ]
        mgr._cumulative_filter_text = "datasets suitable for training a segmentation model"
        return mgr

    def test_content_reasoning_is_a_tracked_listing_tool(self):
        from src.assistant.conversation_manager import _DATASET_LISTING_TOOLS
        assert "reason_about_dataset_content" in _DATASET_LISTING_TOOLS

    def test_content_reasoning_result_replaces_the_prior_listing(self):
        mgr = self._stale_manager()
        mgr._track_dataset_listing(
            "reason_about_dataset_content", self._CONTENT_REASONING_OUTPUT,
            "datasets with both raw and segmented images",
        )
        assert [m["title"] for m in mgr._last_dataset_mentions] == [
            "Estaillades Carbonate #2", "Mt. Simon Sandstone with Mineral Map",
        ]
        assert mgr._cumulative_filter_text == "datasets with both raw and segmented images"

    def test_dois_survive_tracking_for_a_later_reference(self):
        """Titles alone aren't enough — an ordinal/name follow-up resolves to a DOI."""
        mgr = self._stale_manager()
        mgr._track_dataset_listing(
            "reason_about_dataset_content", self._CONTENT_REASONING_OUTPUT, "q",
        )
        assert mgr._last_dataset_mentions[0]["doi"] == "10.17612/P7C09J"

    def test_followup_refines_the_content_reasoning_set_not_the_stale_one(self):
        """The end-to-end failure: the refinement dispatch must carry the 2 datasets the
        content-reasoning turn actually named, never the earlier turn's 10."""
        mgr = self._stale_manager()
        mgr._track_dataset_listing(
            "reason_about_dataset_content", self._CONTENT_REASONING_OUTPUT,
            "datasets with both raw and segmented images",
        )
        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False), \
             patch("src.assistant.conversation_manager._run_manual_dispatch") as mock_dispatch:
            mock_dispatch.return_value = "- **Estaillades Carbonate #2** (DOI: 10.17612/P7C09J)"
            mgr.chat("What are the lithologies of these?")

        calls = mock_dispatch.call_args[0][0]
        assert [c["name"] for c in calls] == ["get_dataset_details"]
        args = calls[0]["args"]
        assert args["restrict_to_titles"] == [
            "Estaillades Carbonate #2", "Mt. Simon Sandstone with Mineral Map",
        ]
        assert "both raw and segmented images" in args["question"]
        assert "segmentation model" not in args["question"]


class TestStaleListingIsNotLeftLookingCurrent:
    """The other half of the same defect: _track_dataset_listing used to update
    _cumulative_filter_text unconditionally while only updating _last_dataset_mentions when
    the turn parsed some, so a fresh turn that named nothing paired NEW filter text with an
    OLD result set."""

    def test_fresh_listing_turn_with_no_results_clears_the_prior_set(self):
        mgr = object.__new__(ConversationManager)
        mgr._last_dataset_mentions = [{"title": "Grain Packing", "doi": None}]
        mgr._cumulative_filter_text = "sandstone datasets"
        mgr._track_dataset_listing(
            "search_datasets", "No datasets found matching that query.", "coal datasets",
        )
        assert mgr._last_dataset_mentions == []

    def test_empty_refinement_turn_keeps_the_set_being_narrowed(self):
        """"Of these, which are coal?" coming back empty does not change what "these"
        refers to — the user can still narrow the same set a different way."""
        mgr = object.__new__(ConversationManager)
        mgr._last_dataset_mentions = [{"title": "Grain Packing", "doi": None}]
        mgr._track_dataset_listing(
            "get_dataset_details", "No datasets found.", "of these which are coal",
            refinement_text="sandstone datasets AND of these which are coal",
        )
        assert [m["title"] for m in mgr._last_dataset_mentions] == ["Grain Packing"]

    def test_refinement_is_not_dispatched_without_titles_to_restrict_to(self):
        """cypher_qa treats an empty restrict_to_titles as "no restriction at all", so
        dispatching without titles would run the compound question over the whole catalog
        while the log claimed a restricted search. Fall through to normal routing instead."""
        mgr = object.__new__(ConversationManager)
        mgr._last_dataset_mentions = []
        mgr._cumulative_filter_text = "sandstone datasets"
        mgr._agent = MagicMock()
        final_message = MagicMock(content="Here are some datasets...")
        final_message.tool_calls = []
        mgr._agent.stream.return_value = iter([
            {"messages": [_USER_ECHO]},
            {"messages": [_USER_ECHO, final_message]},
        ])
        with patch("src.assistant.conversation_manager._classify_off_domain", return_value=False), \
             patch("src.assistant.conversation_manager._classify_needs_tool", return_value=True), \
             patch("src.assistant.conversation_manager._run_manual_dispatch") as mock_dispatch:
            mgr.chat("Which of these are sandstone?")
        mock_dispatch.assert_not_called()
        mgr._agent.stream.assert_called_once()


class TestParseJsonObject:
    """Both malformed shapes here were observed live from this model on the coverage
    gate, ~1-in-3 across repeated runs of the SAME input, with correct judgment every
    time — the verdict was right and the envelope was unparseable. That matters more
    than it sounds: json.loads failing makes the gate report "nothing uncovered", so a
    format slip silently became a dropped half."""

    def test_plain_object(self):
        from src.assistant.conversation_manager import _parse_json_object

        assert _parse_json_object('{"requests": []}') == {"requests": []}

    def test_trailing_comma_before_bracket(self):
        from src.assistant.conversation_manager import _parse_json_object

        raw = '{"requests": [{"asks_for": "a", "covered": true},]}'
        assert _parse_json_object(raw)["requests"][0]["asks_for"] == "a"

    def test_reasoning_prose_before_a_fenced_block(self):
        from src.assistant.conversation_manager import _parse_json_object

        raw = (
            "- **needs_lookup**: True. A comparison would require both datasets.\n\n"
            "Given the analysis, the JSON response should be:\n\n"
            '```json\n{"requests": [{"asks_for": "Dataset B", "covered": false}]}\n```'
        )
        assert _parse_json_object(raw)["requests"][0]["asks_for"] == "Dataset B"

    def test_unrecoverable_returns_none(self):
        from src.assistant.conversation_manager import _parse_json_object

        assert _parse_json_object("I'm not sure how to answer that.") is None
        assert _parse_json_object("") is None

    def test_json_array_is_not_accepted_as_an_object(self):
        """A bare array has no "requests" key to read, so it must not be mistaken for a
        valid envelope."""
        from src.assistant.conversation_manager import _parse_json_object

        assert _parse_json_object('[{"asks_for": "a"}]') is None


class TestArgumentEvidenceGuard:
    """The gate's step 2 ("do the arguments actually ask for this?") is verified in code.
    Asking the model to supply supporting argument text was not enough on its own — live,
    3/3 runs, it quoted the USER'S question instead: evidence "how do I compute it from a
    microCT image" against arguments {"question": "What is porosity?"}. So the model
    proposes the evidence and code checks it actually occurs."""

    def _gate(self, payload, args):
        with patch("src.assistant.llm.get_chat_model") as mock_get_llm, \
             patch("src.assistant.conversation_manager._tool_description", return_value="desc"):
            mock_get_llm.return_value.invoke.return_value = MagicMock(content=payload)
            return _uncovered_requests("What is porosity and how do I compute it?", "t", args)

    def test_evidence_absent_from_arguments_downgrades_to_uncovered(self):
        payload = (
            '{"requests": [{"asks_for": "compute porosity", "covered": true, '
            '"needs_lookup": true, "argument_evidence": "how do I compute it"}]}'
        )
        uncovered = self._gate(payload, {"question": "What is porosity?"})
        assert [u["asks_for"] for u in uncovered] == ["compute porosity"]
        # needs_lookup must survive the downgrade, or a workflow half would be answered
        # from model knowledge instead of by the tool that has the verified tutorials.
        assert uncovered[0]["needs_lookup"] is True

    def test_evidence_present_in_arguments_stays_covered(self):
        payload = (
            '{"requests": [{"asks_for": "compute porosity", "covered": true, '
            '"needs_lookup": true, "argument_evidence": "how do I compute it"}]}'
        )
        assert self._gate(payload, {"question": "What is porosity and how do I compute it?"}) == []

    def test_evidence_match_ignores_case_and_whitespace(self):
        payload = (
            '{"requests": [{"asks_for": "compute porosity", "covered": true, '
            '"needs_lookup": true, "argument_evidence": "How Do   I Compute It"}]}'
        )
        assert self._gate(payload, {"question": "what is porosity and how do i compute it?"}) == []

    def test_guard_is_skipped_when_no_arguments_were_recorded(self):
        """A recovery path that couldn't parse the args has nothing to check against;
        showing the model "{}" would fail every request instead."""
        payload = (
            '{"requests": [{"asks_for": "compute porosity", "covered": true, '
            '"needs_lookup": true, "argument_evidence": ""}]}'
        )
        assert self._gate(payload, {}) == []

    def test_guard_never_upgrades_an_uncovered_verdict(self):
        """It can only move covered -> uncovered, so its worst case is a redundant extra
        answer, never a dropped part."""
        payload = (
            '{"requests": [{"asks_for": "define porosity", "covered": false, '
            '"needs_lookup": false, "argument_evidence": "what is porosity"}]}'
        )
        uncovered = self._gate(payload, {"question": "What is porosity?"})
        assert [u["asks_for"] for u in uncovered] == ["define porosity"]
