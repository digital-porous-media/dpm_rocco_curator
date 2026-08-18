"""
Unit tests for src.assistant.tools.search_portal_docs (issue #53).

All FAISS/embeddings access is mocked via the lazy singleton
`_get_portal_docs_store`, and all LLM synthesis is mocked via
`src.assistant.llm.get_chat_model` — no real index, credentials, or network
access required.

search_portal_docs is a self-contained tool (see
conversation_manager._SELF_CONTAINED_TOOLS): it retrieves chunks via FAISS, then
hands them to an LLM (src/prompts/portal_docs.yaml) to synthesize an answer,
rather than returning the raw chunks directly. Raw-chunk pass-through was tried
first and produced disconnected, sometimes off-topic chunk dumps instead of an
answer to the user's question.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def _fake_markdown_doc(
    page_title: str = "Upload a Dataset",
    section: str = "Step 1",
    doc_url: str = "https://digital-porous-media.github.io/dpm_docs/upload_data/",
    text: str = "Navigate to the upload page and click 'New Dataset'.",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "docs/upload_data.md",
            "page_title": page_title,
            "section": section,
            "doc_url": doc_url,
            "doc_type": "markdown_page",
        },
    )


def _mock_chat_model(response_text: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.send_prompt.return_value = response_text
    return mock_llm


class TestDedupeBySection:
    def test_drops_later_chunks_from_the_same_section(self):
        """A long section gets split into multiple overlapping chunks by the
        markdown splitter (see build_portal_docs_index._MD_SPLITTER) — two chunks
        sharing (page_title, section) are near-duplicate overlap splits of the same
        passage, not distinct information, so only the best-scoring one should
        survive dedup."""
        from src.assistant.tools import _dedupe_by_section

        dup_a = _fake_markdown_doc(section="1. Dataset", text="first half of the section...")
        dup_b = _fake_markdown_doc(section="1. Dataset", text="...second half of the section")
        distinct = _fake_markdown_doc(section="2. Sample", text="A sample is a physical specimen.")
        scored = [(dup_a, 0.40), (dup_b, 0.48), (distinct, 0.55)]

        deduped = _dedupe_by_section(scored)

        assert [doc.page_content for doc, _ in deduped] == [
            "first half of the section...",
            "A sample is a physical specimen.",
        ]

    def test_same_section_name_on_different_pages_is_not_deduped(self):
        """Dedup key must be (page_title, section), not section alone — two
        different pages can legitimately share a generic heading like "Overview"."""
        from src.assistant.tools import _dedupe_by_section

        doc_a = _fake_markdown_doc(page_title="Upload a Dataset", section="Overview", text="upload overview")
        doc_b = _fake_markdown_doc(page_title="Manage Your Data", section="Overview", text="manage overview")
        scored = [(doc_a, 0.40), (doc_b, 0.45)]

        deduped = _dedupe_by_section(scored)

        assert len(deduped) == 2


class TestAnchorChunksForQuery:
    def test_returns_anchor_for_named_schema_entity(self):
        """A query naming "DigitalDataset" should fetch that section's chunk via the
        FAISS metadata filter, regardless of what similarity_search_with_score (the
        ranked path) would have returned — this is a targeted lookup, not a ranked
        search."""
        from src.assistant.tools import _anchor_chunks_for_query, _PORTAL_DOCS_PAGE

        anchor_doc = _fake_markdown_doc(
            page_title=_PORTAL_DOCS_PAGE, section="3. Digital Dataset", text="digital dataset definition"
        )
        mock_store = MagicMock()
        mock_store.vector_store.similarity_search_with_score.return_value = [(anchor_doc, 0.6)]

        anchors = _anchor_chunks_for_query(mock_store, "What is a DigitalDataset?")

        assert anchors == [anchor_doc]
        _, kwargs = mock_store.vector_store.similarity_search_with_score.call_args
        assert kwargs["filter"] == {"page_title": _PORTAL_DOCS_PAGE, "section": "3. Digital Dataset"}

    def test_no_anchor_when_query_names_no_schema_entity(self):
        from src.assistant.tools import _anchor_chunks_for_query

        mock_store = MagicMock()
        anchors = _anchor_chunks_for_query(mock_store, "How do I reset my password?")

        mock_store.vector_store.similarity_search_with_score.assert_not_called()
        assert anchors == []

    def test_multiple_named_entities_each_get_an_anchor(self):
        from src.assistant.tools import _anchor_chunks_for_query, _PORTAL_DOCS_PAGE

        def fake_search(question, k, filter, fetch_k):
            doc = _fake_markdown_doc(page_title=filter["page_title"], section=filter["section"], text=filter["section"])
            return [(doc, 0.5)]

        mock_store = MagicMock()
        mock_store.vector_store.similarity_search_with_score.side_effect = fake_search

        anchors = _anchor_chunks_for_query(
            mock_store, "What is the difference between a Dataset, Sample, and DigitalDataset?"
        )

        sections = {doc.metadata["section"] for doc in anchors}
        assert sections == {"1. Dataset", "2. Sample", "3. Digital Dataset"}

    def test_returns_empty_list_when_store_has_no_vector_store_attribute(self):
        from src.assistant.tools import _anchor_chunks_for_query

        anchors = _anchor_chunks_for_query(object(), "What is a Sample?")

        assert anchors == []

    def test_swallows_filter_lookup_failure(self):
        from src.assistant.tools import _anchor_chunks_for_query

        mock_store = MagicMock()
        mock_store.vector_store.similarity_search_with_score.side_effect = RuntimeError("boom")

        anchors = _anchor_chunks_for_query(mock_store, "What is a Sample?")

        assert anchors == []


class TestFigureReferenceGuards:
    def test_strip_removes_fabricated_screenshot_mention(self):
        from src.assistant.tools import _strip_fabricated_figure_reference

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
        from src.assistant.tools import _strip_fabricated_figure_reference

        response = "Step 2 text.\n\nSee the screenshot on the linked page for this step."
        assert _strip_fabricated_figure_reference(response, has_figure=True) == response

    def test_strip_is_a_noop_when_no_screenshot_mention_exists(self):
        from src.assistant.tools import _strip_fabricated_figure_reference

        response = "A Sample is the physical specimen being studied."
        assert _strip_fabricated_figure_reference(response, has_figure=False) == response

    def test_ensure_appends_note_when_figure_present_but_unmentioned(self):
        from src.assistant.tools import _ensure_figure_reference, _FIGURE_APPEND_NOTE

        response = "To add collaborators, select Manage Authors."
        result = _ensure_figure_reference(response, has_figure=True)

        assert result.startswith(response)
        assert _FIGURE_APPEND_NOTE in result

    def test_ensure_is_a_noop_when_no_figure_present(self):
        from src.assistant.tools import _ensure_figure_reference

        response = "To add collaborators, select Manage Authors."
        assert _ensure_figure_reference(response, has_figure=False) == response

    def test_ensure_is_a_noop_when_already_mentioned(self):
        from src.assistant.tools import _ensure_figure_reference

        response = "Step 2 text.\n\nSee the screenshot on the linked page for this step."
        assert _ensure_figure_reference(response, has_figure=True) == response


class TestSearchPortalDocs:
    def test_hit_synthesizes_answer_from_retrieved_chunks(self):
        doc = _fake_markdown_doc()
        synthesized = "To upload a dataset, navigate to the upload page and click 'New Dataset'.\n\nSources:\n[portal docs] Upload a Dataset — https://digital-porous-media.github.io/dpm_docs/upload_data/"
        with patch("src.assistant.tools._get_portal_docs_store") as mock_factory, \
             patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(synthesized)):
            mock_store = MagicMock()
            mock_store.similarity_search_with_score.return_value = [(doc, 0.5)]
            mock_factory.return_value = mock_store

            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("How do I upload a dataset?")

        assert result == synthesized
        assert "[portal docs]" in result

    def test_truncated_argument_is_expanded_before_retrieval(self):
        """The ReAct agent routinely compresses a full question into a keyword phrase
        before calling this tool (e.g. "How do I upload a dataset?" -> "upload a
        dataset") — _expand_portal_query must run on whatever argument it's given and
        the store must be queried with the expanded form, not the raw fragment."""
        doc = _fake_markdown_doc()
        with patch("src.assistant.tools._get_portal_docs_store") as mock_factory, \
             patch(
                 "src.assistant.tools._expand_portal_query",
                 return_value="How do I upload a dataset to the DPM Portal?",
             ) as mock_expand, \
             patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model("answer")):
            mock_store = MagicMock()
            mock_store.similarity_search_with_score.return_value = [(doc, 0.5)]
            mock_factory.return_value = mock_store

            from src.assistant.tools import _PORTAL_DOCS_CANDIDATE_K, search_portal_docs
            search_portal_docs.func("upload a dataset")

        mock_expand.assert_called_once_with("upload a dataset")
        mock_store.similarity_search_with_score.assert_called_once_with(
            "How do I upload a dataset to the DPM Portal?", k=_PORTAL_DOCS_CANDIDATE_K
        )

    def test_miss_returns_honest_gap_message_without_calling_llm(self):
        """Note: "without calling llm" refers to the synthesis call — query expansion
        (_expand_portal_query) always runs and is stubbed out here as a passthrough so
        it doesn't need its own LLM mock."""
        with patch("src.assistant.tools._get_portal_docs_store") as mock_factory, \
             patch("src.assistant.tools._expand_portal_query", side_effect=lambda q: q), \
             patch("src.assistant.llm.get_chat_model") as mock_get_chat:
            mock_store = MagicMock()
            mock_store.similarity_search_with_score.return_value = []
            mock_factory.return_value = mock_store

            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("zzz obscure unrelated nonsense query")

        assert "No portal documentation found" in result
        mock_get_chat.assert_not_called()

    def test_weak_matches_above_score_threshold_treated_as_miss(self):
        """FAISS similarity_search always returns top-k regardless of relevance —
        results scoring above the no-match threshold (empirically calibrated;
        see _NO_MATCH_SCORE_THRESHOLD) must be filtered out, not surfaced as a
        real answer."""
        doc = _fake_markdown_doc()
        with patch("src.assistant.tools._get_portal_docs_store") as mock_factory, \
             patch("src.assistant.tools._expand_portal_query", side_effect=lambda q: q), \
             patch("src.assistant.llm.get_chat_model") as mock_get_chat:
            mock_store = MagicMock()
            mock_store.similarity_search_with_score.return_value = [(doc, 1.3)]
            mock_factory.return_value = mock_store

            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("zzz obscure unrelated nonsense query")

        assert "No portal documentation found" in result
        mock_get_chat.assert_not_called()

    def test_index_not_built_returns_graceful_message_without_calling_llm(self):
        """search_portal_docs must not crash if the index hasn't been built yet."""
        with patch("src.assistant.tools._get_portal_docs_store", return_value=None), \
             patch("src.assistant.llm.get_chat_model") as mock_get_chat:
            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("How do I upload a dataset?")

        assert isinstance(result, str)
        assert "not yet available" in result
        mock_get_chat.assert_not_called()

    def test_search_portal_docs_works_without_neo4j(self):
        """search_portal_docs is FAISS-only — must work regardless of USE_NEO4J."""
        doc = _fake_markdown_doc()
        synthesized = "Navigate to the upload page and click 'New Dataset'.\n\nSources:\n[portal docs] Upload a Dataset"
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.tools._get_portal_docs_store") as mock_factory, \
                 patch("src.assistant.llm.get_chat_model", return_value=_mock_chat_model(synthesized)):
                mock_store = MagicMock()
                mock_store.similarity_search_with_score.return_value = [(doc, 0.5)]
                mock_factory.return_value = mock_store

                from src.assistant.tools import search_portal_docs
                result = search_portal_docs.func("How do I upload a dataset?")

        assert result == synthesized
