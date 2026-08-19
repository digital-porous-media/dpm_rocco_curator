"""
Unit tests for search_portal_docs' PageIndex-style implementation:
src.assistant.portal_docs_tree (heading-tree builder) and
src.assistant.portal_docs_retrieval (LLM-reasoning node selection + synthesis) —
this is search_portal_docs' entire implementation (see tools.search_portal_docs,
a thin delegating wrapper). Replaced an earlier FAISS/chunk-based retrieval path;
see HANDOFF.md for that history.

All LLM calls are mocked via src.assistant.llm.get_chat_model — no real network/
embeddings/index required. Tree-building tests use small synthetic markdown files
(via tmp_path) for fast, isolated unit coverage, plus one smoke test against the
real data/portal_docs/docs/*.md corpus to catch drift if dpm_docs' actual heading
structure changes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.assistant.portal_docs_tree import (
    DocNode,
    build_forest,
    flatten,
    parse_markdown_tree,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _mock_chat_model(response_text: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.send_prompt.return_value = response_text
    return mock_llm


class TestParseMarkdownTree:
    def test_simple_nesting_builds_correct_parent_child_links_and_full_text(self, tmp_path):
        path = _write(
            tmp_path,
            "page.md",
            "# Page Title\n\nIntro text.\n\n## Section A\n\nA body.\n\n### Sub A1\n\nSub body.\n",
        )
        roots = parse_markdown_tree(path, "page.md")

        assert len(roots) == 1
        page = roots[0]
        assert page.title == "Page Title"
        assert page.text == "Intro text."
        assert len(page.children) == 1

        section_a = page.children[0]
        assert section_a.title == "Section A"
        assert section_a.text == "A body."
        assert len(section_a.children) == 1

        sub_a1 = section_a.children[0]
        assert sub_a1.title == "Sub A1"
        assert sub_a1.text == "Sub body."
        assert sub_a1.parent_id == section_a.node_id

        # full_text rolls up own text + every descendant's, in document order,
        # with each child's own heading reintroduced first — without this, a
        # multi-child section's full_text is an unlabeled blob that gives an LLM
        # synthesizing an answer no signal distinguishing a short conceptual
        # overview (the parent's own text) from the field-reference detail that
        # follows (see HANDOFF.md's PageIndex prototype "Update 10" section).
        assert "Intro text." in page.full_text
        assert "A body." in page.full_text
        assert "Sub body." in page.full_text
        assert "### Sub A1" in section_a.full_text
        assert section_a.full_text == "A body.\n\n### Sub A1\n\nSub body."

    def test_heading_level_jump_still_nests_correctly(self, tmp_path):
        """H1 straight to H4 (no H2/H3 in between) — the level-aware stack must
        still nest H4 under H1 rather than crashing or attaching it as a sibling
        root."""
        path = _write(
            tmp_path,
            "jump.md",
            "# Root\n\nRoot body.\n\n#### Deep Section\n\nDeep body.\n",
        )
        roots = parse_markdown_tree(path, "jump.md")

        assert len(roots) == 1
        assert len(roots[0].children) == 1
        deep = roots[0].children[0]
        assert deep.title == "Deep Section"
        assert deep.level == 4
        assert deep.parent_id == roots[0].node_id

    def test_bold_and_backtick_markdown_stripped_from_title(self, tmp_path):
        path = _write(
            tmp_path,
            "styled.md",
            "# Page\n\nBody.\n\n##### **Source 1: Natural (`Earth`)**\n\nSource body.\n",
        )
        roots = parse_markdown_tree(path, "styled.md")

        titles = [n.title for n in flatten(roots)]
        assert "Source 1: Natural (Earth)" in titles
        assert not any("*" in t or "`" in t for t in titles)

    def test_zero_heading_file_yields_one_fallback_page_node(self, tmp_path):
        path = _write(tmp_path, "stub_page.md", '!!! caution "Coming soon!"\n    Under development.\n')
        roots = parse_markdown_tree(path, "stub_page.md")

        assert len(roots) == 1
        assert roots[0].title == "Stub Page"  # fallback: filename -> Title Case
        assert roots[0].level == 1
        assert "Coming soon" in roots[0].full_text
        assert roots[0].children == []

    def test_two_h1_headings_in_one_file_become_two_page_roots(self, tmp_path):
        """cite.md-shaped input: two H1s for what should be two distinct pages —
        the tree must split them, not silently merge the second H1's content into
        the first page's body the way the flat FAISS chunker does today."""
        path = _write(
            tmp_path,
            "cite.md",
            "# Cite the Portal\n\nPlatform citation text.\n\n"
            "# Cite a Dataset\n\nDataset citation text.\n\n"
            "## Using the DOI\n\nDOI body.\n",
        )
        roots = parse_markdown_tree(path, "cite.md")

        assert len(roots) == 2
        assert roots[0].title == "Cite the Portal"
        assert roots[0].full_text.strip() == "Platform citation text."
        assert roots[1].title == "Cite a Dataset"
        assert "Dataset citation text." in roots[1].full_text
        assert "DOI body." in roots[1].full_text
        # The platform-citation page must not absorb the dataset-citation content.
        assert "Dataset citation text." not in roots[0].full_text

    def test_hash_inside_unindented_fenced_code_block_is_not_a_heading(self, tmp_path):
        path = _write(
            tmp_path,
            "fenced.md",
            "# Page\n\nBody.\n\n```bash\n# not a real heading, just a shell comment\necho hi\n```\n\n"
            "## Real Section\n\nReal body.\n",
        )
        roots = parse_markdown_tree(path, "fenced.md")

        titles = [n.title for n in flatten(roots)]
        assert "not a real heading, just a shell comment" not in titles
        assert "Real Section" in titles

    def test_node_ids_are_unique_for_same_titled_siblings_under_different_parents(self, tmp_path):
        """Two different sections both containing a heading titled "Common Fields"
        (as upload_data.md's "2. Sample" and "3. Digital Dataset" both do) must get
        distinct node ids, since the id encodes the full ancestor path."""
        path = _write(
            tmp_path,
            "shared_titles.md",
            "# Page\n\n## Section One\n\n#### Common Fields\n\nOne fields.\n\n"
            "## Section Two\n\n#### Common Fields\n\nTwo fields.\n",
        )
        roots = parse_markdown_tree(path, "shared_titles.md")

        common_field_nodes = [n for n in flatten(roots) if n.title == "Common Fields"]
        assert len(common_field_nodes) == 2
        assert common_field_nodes[0].node_id != common_field_nodes[1].node_id


class TestBuildForestRealCorpus:
    def test_real_corpus_parses_without_exceptions_and_looks_sane(self):
        """Smoke test against the real data/portal_docs/docs/*.md files — no
        assertions on exact structure (that would need hand-maintaining a fixture
        for every real quirk), just: doesn't crash, node count is in the expected
        ballpark, every node has a non-empty doc_url."""
        forest = build_forest()
        if not forest:
            pytest.skip("data/portal_docs/docs not present in this environment")

        flat = flatten(forest)
        assert 60 <= len(flat) <= 150
        assert all(n.doc_url for n in flat)
        assert all(n.node_id for n in flat)
        # cite.md's known two-H1 anomaly should surface as two distinct page roots.
        cite_roots = [r for r in forest if r.node_id.startswith("docs/cite.md#") or r.node_id.startswith("cite.md#")]
        assert len(cite_roots) == 2


class TestSelectNodesForQuery:
    def _nodes(self):
        return [
            DocNode(node_id="a", title="1. Dataset", level=3, page_title="Upload", doc_url="u"),
            DocNode(node_id="b", title="2. Sample", level=3, page_title="Upload", doc_url="u"),
            DocNode(node_id="c", title="3. Digital Dataset", level=3, page_title="Upload", doc_url="u"),
            DocNode(node_id="d", title="4. Analysis Dataset", level=3, page_title="Upload", doc_url="u"),
        ]

    def test_well_formed_json_array_returned_in_order(self):
        from src.assistant.portal_docs_retrieval import select_nodes_for_query

        with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model('["c", "d"]')):
            ids = select_nodes_for_query("difference between digital and analysis dataset", self._nodes())

        assert ids == ["c", "d"]

    def test_json_wrapped_in_markdown_fence_is_stripped(self):
        from src.assistant.portal_docs_retrieval import select_nodes_for_query

        with patch(
            "src.assistant.llm.get_chat_model",
            return_value=_mock_chat_model('```json\n["a"]\n```'),
        ):
            ids = select_nodes_for_query("what is a dataset", self._nodes())

        assert ids == ["a"]

    def test_malformed_response_falls_back_to_keyword_match(self):
        from src.assistant.portal_docs_retrieval import select_nodes_for_query

        with patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model("not json at all")):
            ids = select_nodes_for_query("digital dataset", self._nodes())

        assert "c" in ids

    def test_unknown_id_in_response_is_silently_dropped(self):
        from src.assistant.portal_docs_retrieval import select_nodes_for_query

        with patch(
            "src.assistant.llm.get_chat_model",
            return_value=_mock_chat_model('["a", "not-a-real-id", "b"]'),
        ):
            ids = select_nodes_for_query("dataset and sample", self._nodes())

        assert ids == ["a", "b"]

    def test_comparison_query_can_return_both_relevant_entities(self):
        """Regression guard for HANDOFF.md Update 7 issue #1: a mocked response
        returning both the DigitalDataset and AnalysisDataset ids must both survive
        into the final result, not get collapsed to a single best match."""
        from src.assistant.portal_docs_retrieval import select_nodes_for_query

        with patch(
            "src.assistant.llm.get_chat_model",
            return_value=_mock_chat_model('["c", "d"]'),
        ):
            ids = select_nodes_for_query(
                "What about the difference between a Digital and Analysis Dataset?", self._nodes()
            )

        assert "c" in ids and "d" in ids


class TestSelectNodesKeywordBodyTextSignal:
    """Regression coverage for HANDOFF.md's PageIndex prototype "Update 10" bug:
    a query naming a body-level field ("Reference Sample") that is never its own
    heading used to get routed to an unrelated, same-word section ("2. Sample")
    because node selection only ever saw section titles. Uses the real corpus
    (build_forest/flatten), like TestBuildForestRealCorpus, so this is checked
    against real content rather than a synthetic fixture that could hide the same
    title/body mismatch."""

    def _real_nodes(self):
        forest = build_forest()
        if not forest:
            pytest.skip("data/portal_docs/docs not present in this environment")
        return flatten(forest)

    def test_keyword_fallback_finds_reference_sample_under_digital_dataset_not_sample(self):
        from src.assistant.portal_docs_retrieval import _select_nodes_keyword

        nodes = self._real_nodes()
        ids = _select_nodes_keyword(
            "What should I put in the Reference Sample field?", nodes, 4
        )

        assert any("3-digital-dataset" in nid for nid in ids), ids
        # The wrong-entity section ("2. Sample") must not outrank the real hit.
        digital_rank = next(i for i, nid in enumerate(ids) if "3-digital-dataset" in nid)
        sample_ranks = [i for i, nid in enumerate(ids) if "2-sample" in nid and "3-digital-dataset" not in nid]
        assert not sample_ranks or digital_rank < min(sample_ranks)

    def test_index_line_surfaces_bolded_field_names_not_just_titles(self):
        from src.assistant.portal_docs_retrieval import _index_line

        nodes = self._real_nodes()
        core_info = next(
            n for n in nodes if n.node_id.endswith("core-digital-dataset-information")
        )
        line = _index_line(core_info)

        assert "Reference Sample" in line
        assert "Is Segmented" in line


class TestFormatPortalDocNodeOverviewLabeling:
    """Regression coverage for HANDOFF.md's PageIndex prototype "Update 11" bug:
    a node with children rendered its short conceptual overview and its much
    larger rolled-up field-reference detail as one undifferentiated blob, biasing
    synthesis toward the field list purely by text volume."""

    def _node_with_children(self, **overrides):
        child = DocNode(
            node_id="parent#child",
            title="Core Info",
            level=4,
            page_title="Page",
            doc_url="u",
            text="*   **Name***: field detail.",
            full_text="*   **Name***: field detail.",
        )
        defaults = dict(
            node_id="parent",
            title="3. Digital Dataset",
            level=3,
            page_title="Page",
            doc_url="u",
            text="A digital dataset is linked to a Sample.",
            full_text="A digital dataset is linked to a Sample.\n\n#### Core Info\n\n*   **Name***: field detail.",
            children=[child],
        )
        defaults.update(overrides)
        return DocNode(**defaults)

    def test_node_with_children_gets_overview_and_details_labels(self):
        from src.assistant.portal_docs_retrieval import _format_portal_doc_node

        rendered = _format_portal_doc_node(self._node_with_children())

        assert "Overview: A digital dataset is linked to a Sample." in rendered
        assert "Reference details" in rendered
        assert "Core Info" in rendered

    def test_leaf_node_is_unchanged_no_spurious_labels(self):
        from src.assistant.portal_docs_retrieval import _format_portal_doc_node

        leaf = DocNode(
            node_id="leaf",
            title="Step 2",
            level=2,
            page_title="Page",
            doc_url="u",
            text="Click Download.",
            full_text="Click Download.",
        )
        rendered = _format_portal_doc_node(leaf)

        assert "Overview:" not in rendered
        assert "Reference details" not in rendered
        assert "Click Download." in rendered

    def test_leaf_node_with_inline_field_bullets_still_gets_split(self):
        """Regression coverage for HANDOFF.md's PageIndex prototype "Update 12" bug:
        "4. Analysis Dataset" has no sub-headings at all — its intro sentence and
        entire field-bullet list are both part of one leaf node's own text, so the
        original node.children-based split (Update 11) never fired for it. Splitting
        on the first bolded field bullet instead of on tree structure must catch
        this shape too."""
        from src.assistant.portal_docs_retrieval import _format_portal_doc_node

        leaf_with_inline_fields = DocNode(
            node_id="leaf-fields",
            title="4. Analysis Dataset",
            level=3,
            page_title="Page",
            doc_url="u",
            text=(
                "This section is for datasets that are the result of an analysis method.\n\n"
                "*   **Name***: The name of the analysis dataset.\n"
                "*   **Analysis Type***: The type of analysis performed."
            ),
            full_text=(
                "This section is for datasets that are the result of an analysis method.\n\n"
                "*   **Name***: The name of the analysis dataset.\n"
                "*   **Analysis Type***: The type of analysis performed."
            ),
        )

        rendered = _format_portal_doc_node(leaf_with_inline_fields)

        assert "Overview: This section is for datasets that are the result of an analysis method." in rendered
        assert "Reference details" in rendered
        assert "Analysis Type" in rendered


class TestRestrictToBestTitleMatch:
    """Regression coverage for HANDOFF.md's PageIndex prototype "Update 12" bug:
    select_nodes_for_query was observed padding its result list with weakly- or
    only-generically-related sections (e.g. "Community Data") alongside the one
    node that's an exact match for the question's named entity, diluting synthesis
    context and measurably degrading answer quality — an anti-padding prompt
    instruction reduced but did not reliably eliminate this."""

    def _nodes(self):
        return [
            DocNode(node_id="a", title="3. Digital Dataset", level=3, page_title="Upload", doc_url="u"),
            DocNode(node_id="b", title="Edit Digital Dataset Information", level=2, page_title="Manage", doc_url="u"),
            DocNode(node_id="c", title="Community Data", level=1, page_title="Community", doc_url="u"),
        ]

    def test_single_strong_title_match_restricts_to_that_node(self):
        from src.assistant.portal_docs_retrieval import _restrict_to_best_title_match

        restricted = _restrict_to_best_title_match("What is a Digital Dataset?", self._nodes())

        assert [n.node_id for n in restricted] == ["a"]

    def test_comparison_query_is_not_restricted(self):
        from src.assistant.portal_docs_retrieval import _restrict_to_best_title_match

        nodes = [
            DocNode(node_id="a", title="3. Digital Dataset", level=3, page_title="Upload", doc_url="u"),
            DocNode(node_id="d", title="4. Analysis Dataset", level=3, page_title="Upload", doc_url="u"),
        ]
        restricted = _restrict_to_best_title_match(
            "What is the difference between a Digital Dataset and an Analysis Dataset?", nodes
        )

        assert {n.node_id for n in restricted} == {"a", "d"}

    def test_no_strong_match_leaves_results_unchanged(self):
        from src.assistant.portal_docs_retrieval import _restrict_to_best_title_match

        nodes = [
            DocNode(node_id="x", title="Core Digital Dataset Information", level=4, page_title="Upload", doc_url="u"),
            DocNode(node_id="y", title="Source-Dependent Fields", level=4, page_title="Upload", doc_url="u"),
        ]
        restricted = _restrict_to_best_title_match("What should I put in the Reference Sample field?", nodes)

        assert restricted == nodes


class TestEnsureSourceUrlsPresent:
    """Regression coverage for HANDOFF.md's PageIndex prototype "Update 12" bug:
    the synthesis LLM was observed dropping the actual doc_url from its own
    "Sources:" line the large majority of live runs (9/10), even though every
    node's context includes a literal "Source: <url>" line and the prompt asks for
    doc_url — the user's reported "links to the docs are gone" symptom."""

    def _node(self, doc_url="https://example.org/page/"):
        return DocNode(
            node_id="n", title="T", level=2, page_title="P", doc_url=doc_url, text="x", full_text="x"
        )

    def test_missing_url_is_appended(self):
        from src.assistant.portal_docs_retrieval import _ensure_source_urls_present

        node = self._node()
        response = "An answer.\n\nSources: [portal docs] Page — T"
        result = _ensure_source_urls_present(response, [node])

        assert node.doc_url in result
        assert result.startswith(response)

    def test_already_present_url_is_not_duplicated(self):
        from src.assistant.portal_docs_retrieval import _ensure_source_urls_present

        node = self._node()
        response = f"An answer.\n\nSources: [portal docs] {node.doc_url}"
        result = _ensure_source_urls_present(response, [node])

        assert result == response

    def test_multiple_results_dedupe_by_url(self):
        from src.assistant.portal_docs_retrieval import _ensure_source_urls_present

        n1 = self._node("https://example.org/a/")
        n2 = self._node("https://example.org/a/")  # same page, different node
        response = "An answer."
        result = _ensure_source_urls_present(response, [n1, n2])

        assert result.count("https://example.org/a/") == 1


class TestDatasetContainerContext:
    """Regression coverage for HANDOFF.md's PageIndex prototype "Update 11" bug:
    a definitional question about a Dataset sub-entity (Sample/DigitalDataset/
    AnalysisDataset) got no relational framing, since the "1. Dataset" container
    node is a sibling, not an ancestor, of those sub-entities, and node selection
    has no mechanism to surface a sibling on its own. Uses the real corpus, like
    TestSelectNodesKeywordBodyTextSignal, so this is checked against the real
    "Curate Your Dataset" structure rather than a synthetic fixture."""

    def _real_nodes(self):
        forest = build_forest()
        if not forest:
            pytest.skip("data/portal_docs/docs not present in this environment")
        return {n.node_id: n for n in flatten(forest)}

    def test_selecting_digital_dataset_surfaces_container_overview(self):
        from src.assistant.portal_docs_retrieval import _dataset_container_context

        id_to_node = self._real_nodes()
        digital_dataset = next(
            n for nid, n in id_to_node.items() if nid.endswith("3-digital-dataset")
        )

        block = _dataset_container_context([digital_dataset], id_to_node)

        assert block is not None
        assert "main container" in block

    def test_container_already_selected_is_not_duplicated(self):
        from src.assistant.portal_docs_retrieval import _dataset_container_context

        id_to_node = self._real_nodes()
        container = next(
            n for nid, n in id_to_node.items() if nid.endswith("curate-your-dataset/1-dataset")
        )

        assert _dataset_container_context([container], id_to_node) is None

    def test_unrelated_page_selection_does_not_trigger_container_context(self):
        from src.assistant.portal_docs_retrieval import _dataset_container_context

        id_to_node = self._real_nodes()
        unrelated = DocNode(
            node_id="download_data.md#step-2",
            title="Step 2",
            level=2,
            page_title="Download or Copy a Dataset from DPMP",
            doc_url="u",
        )

        assert _dataset_container_context([unrelated], id_to_node) is None


class TestSearchPortalDocsV2:
    def _node(self, **overrides):
        defaults = dict(
            node_id="upload_data.md#step-2",
            title="Step 2: Download the Dataset",
            level=2,
            page_title="Download or Copy a Dataset from DPMP",
            doc_url="https://digital-porous-media.github.io/dpm_docs/download_data/",
            text="Click Download.",
            full_text="Click Download.",
        )
        defaults.update(overrides)
        return DocNode(**defaults)

    def test_hit_synthesizes_answer_from_selected_nodes(self):
        node = self._node()
        synthesized = f"Click the Download Dataset button.\n\nSources:\n[portal docs] {node.doc_url}"
        with patch("src.assistant.portal_docs_tree.get_portal_docs_tree", return_value=[node]), \
             patch("src.assistant.portal_docs_retrieval.select_nodes_for_query", return_value=[node.node_id]), \
             patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(synthesized)):
            from src.assistant.portal_docs_retrieval import search_portal_docs_v2
            result = search_portal_docs_v2("What are the steps to download a dataset?")

        # The model's own "Sources:" line already contains the real doc_url verbatim
        # here, so _ensure_source_urls_present (see TestEnsureSourceUrlsPresent) has
        # nothing to append — result is unchanged.
        assert result == synthesized

    def test_empty_selection_returns_honest_gap_message_without_calling_synthesis_llm(self):
        node = self._node()
        with patch("src.assistant.portal_docs_tree.get_portal_docs_tree", return_value=[node]), \
             patch("src.assistant.portal_docs_retrieval.select_nodes_for_query", return_value=[]), \
             patch("src.assistant.llm.get_chat_model") as mock_get_chat:
            from src.assistant.portal_docs_retrieval import search_portal_docs_v2
            result = search_portal_docs_v2("zzz obscure unrelated nonsense query")

        assert "No portal documentation found" in result
        # Same implementation-detail leak guard as test_portal_docs.py's equivalent
        # FAISS-path test.
        assert "get_workflow_guidance(" not in result
        assert "get_educational_context(" not in result
        mock_get_chat.assert_not_called()

    def test_no_tree_available_returns_honest_message(self):
        with patch("src.assistant.portal_docs_tree.get_portal_docs_tree", return_value=[]):
            from src.assistant.portal_docs_retrieval import search_portal_docs_v2
            result = search_portal_docs_v2("how do I upload a dataset?")

        assert "not yet available" in result

    def test_figure_guard_strips_fabricated_screenshot_mention(self):
        node = self._node(full_text="No figure placeholder here.")
        response_with_fabricated_mention = "Click Download. See the screenshot on the linked page for this step."
        with patch("src.assistant.portal_docs_tree.get_portal_docs_tree", return_value=[node]), \
             patch("src.assistant.portal_docs_retrieval.select_nodes_for_query", return_value=[node.node_id]), \
             patch(
                 "src.assistant.llm.get_chat_model",
                 return_value=_mock_chat_model(response_with_fabricated_mention),
             ):
            from src.assistant.portal_docs_retrieval import search_portal_docs_v2
            result = search_portal_docs_v2("How do I download a dataset?")

        assert "screenshot" not in result.lower()

    def test_genuine_figure_present_but_unmentioned_is_left_as_is(self):
        """_ensure_figure_reference (which used to proactively append a screenshot
        note when a figure was present but the model didn't mention it) was removed
        as too vague to be worth manufacturing — search_portal_docs_v2 no longer
        appends anything in this case, it just doesn't strip a genuine mention if the
        model did include one (see test_figure_guard_strips_fabricated_screenshot_mention
        for the still-active anti-fabrication half of the guard)."""
        node = self._node(full_text="[Figure: Download Step 2]\n\nClick Download.")
        mocked_response = f"Click Download.\n\nSources: [portal docs] {node.doc_url}"
        with patch("src.assistant.portal_docs_tree.get_portal_docs_tree", return_value=[node]), \
             patch("src.assistant.portal_docs_retrieval.select_nodes_for_query", return_value=[node.node_id]), \
             patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(mocked_response)):
            from src.assistant.portal_docs_retrieval import search_portal_docs_v2
            result = search_portal_docs_v2("How do I download a dataset?")

        assert result == mocked_response


class TestSearchPortalDocsDelegation:
    """tools.search_portal_docs is a thin wrapper — confirm it delegates straight
    to search_portal_docs_v2 with the question passed through unchanged."""

    def test_delegates_to_v2(self):
        with patch(
            "src.assistant.portal_docs_retrieval.search_portal_docs_v2", return_value="v2 answer"
        ) as mock_v2:
            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("How do I upload a dataset?")

        mock_v2.assert_called_once_with("How do I upload a dataset?")
        assert result == "v2 answer"


class TestFigureReferenceGuards:
    """_strip_fabricated_figure_reference — relocated here (from the deleted FAISS-
    path test_portal_docs.py) since the function itself moved into
    portal_docs_retrieval.py once the FAISS path (its only other caller) was
    removed; this logic was never FAISS-specific."""

    def test_strip_removes_fabricated_screenshot_mention(self):
        from src.assistant.portal_docs_retrieval import _strip_fabricated_figure_reference

        response = (
            "To create a dataset, fill out the required fields.\n\n"
            "See the screenshot on the linked page for this step.\n\n"
            "Sources: https://example.com/upload_data/"
        )
        cleaned = _strip_fabricated_figure_reference(response, has_figure=False)

        assert "screenshot" not in cleaned.lower()
        assert "To create a dataset, fill out the required fields." in cleaned
        assert "Sources: https://example.com/upload_data/" in cleaned

    def test_strip_leaves_response_untouched_when_figure_present(self):
        from src.assistant.portal_docs_retrieval import _strip_fabricated_figure_reference

        response = "Step 2 text.\n\nSee the screenshot on the linked page for this step."
        assert _strip_fabricated_figure_reference(response, has_figure=True) == response

    def test_strip_is_a_noop_when_no_screenshot_mention_exists(self):
        from src.assistant.portal_docs_retrieval import _strip_fabricated_figure_reference

        response = "A Sample is the physical specimen being studied."
        assert _strip_fabricated_figure_reference(response, has_figure=False) == response

    # _ensure_figure_reference (which proactively appended "See the screenshot on the
    # linked page for this step." when a figure was present but unmentioned) was
    # removed — judged too vague to be worth manufacturing on the model's behalf.
    # _strip_fabricated_figure_reference (tested above) is the only guard that
    # remains: it still prevents the model from fabricating a screenshot mention when
    # no figure is actually present, but no longer force-adds one when a figure is.
