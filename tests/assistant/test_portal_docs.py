"""
Unit tests for src.assistant.tools.search_portal_docs (issue #53).

All FAISS/embeddings access is mocked via the lazy singleton
`_get_portal_docs_store` — no real index or credentials required.
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


def _fake_thesis_doc(
    text: str = "The data model organizes samples, digital datasets, and analyses.",
    page: int = 12,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_type": "thesis",
            "page_title": "Turhan (2024) — Towards Scalable Data Model for Curation and Reusable Workflows for Porous Media Image Analysis",
            "doc_url": "https://doi.org/10.26153/tsw/59410",
            "page": page,
        },
    )


class TestSearchPortalDocs:
    def test_hit_returns_chunk_with_source_label(self):
        doc = _fake_markdown_doc()
        with patch("src.assistant.tools._get_portal_docs_store") as mock_factory:
            mock_store = MagicMock()
            mock_store.similarity_search_with_score.return_value = [(doc, 0.5)]
            mock_factory.return_value = mock_store

            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("How do I upload a dataset?")

        assert "[portal docs]" in result
        assert "Upload a Dataset" in result
        assert "Navigate to the upload page" in result
        assert doc.metadata["doc_url"] in result

    def test_thesis_hit_uses_thesis_label(self):
        doc = _fake_thesis_doc()
        with patch("src.assistant.tools._get_portal_docs_store") as mock_factory:
            mock_store = MagicMock()
            mock_store.similarity_search_with_score.return_value = [(doc, 0.5)]
            mock_factory.return_value = mock_store

            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("How is the data model structured?")

        assert "[thesis]" in result
        assert "[portal docs]" not in result
        assert "the data model organizes samples" in result.lower()
        assert "10.26153/tsw/59410" in result

    def test_miss_returns_honest_gap_message(self):
        with patch("src.assistant.tools._get_portal_docs_store") as mock_factory:
            mock_store = MagicMock()
            mock_store.similarity_search_with_score.return_value = []
            mock_factory.return_value = mock_store

            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("zzz obscure unrelated nonsense query")

        assert "[portal docs]" not in result
        assert "No portal documentation found" in result

    def test_weak_matches_above_score_threshold_treated_as_miss(self):
        """FAISS similarity_search always returns top-k regardless of relevance —
        results scoring above the no-match threshold (empirically calibrated;
        see _NO_MATCH_SCORE_THRESHOLD) must be filtered out, not surfaced as a
        real answer."""
        doc = _fake_markdown_doc()
        with patch("src.assistant.tools._get_portal_docs_store") as mock_factory:
            mock_store = MagicMock()
            mock_store.similarity_search_with_score.return_value = [(doc, 1.3)]
            mock_factory.return_value = mock_store

            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("zzz obscure unrelated nonsense query")

        assert "[portal docs]" not in result
        assert "No portal documentation found" in result

    def test_index_not_built_returns_graceful_message(self):
        """search_portal_docs must not crash if the index hasn't been built yet."""
        with patch("src.assistant.tools._get_portal_docs_store", return_value=None):
            from src.assistant.tools import search_portal_docs
            result = search_portal_docs.func("How do I upload a dataset?")

        assert isinstance(result, str)
        assert "not yet available" in result

    def test_search_portal_docs_works_without_neo4j(self):
        """search_portal_docs is FAISS-only — must work regardless of USE_NEO4J."""
        doc = _fake_markdown_doc()
        with patch.dict("os.environ", {"USE_NEO4J": "false"}):
            with patch("src.assistant.tools._get_portal_docs_store") as mock_factory:
                mock_store = MagicMock()
                mock_store.similarity_search_with_score.return_value = [(doc, 0.5)]
                mock_factory.return_value = mock_store

                from src.assistant.tools import search_portal_docs
                result = search_portal_docs.func("How do I upload a dataset?")

        assert "[portal docs]" in result
