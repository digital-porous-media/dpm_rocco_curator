"""Unit tests for src/assistant/assistant_ui.py's pure text-formatting helpers.

These are plain regex/string functions independent of Streamlit's runtime, so they're
tested directly without mocking `st`.
"""

from src.assistant.assistant_ui import (
    _labelify_sources,
    _linkify_dois,
    _linkify_urls,
    _render_response,
)


class TestLabelifySources:
    def test_portal_docs_label_becomes_a_badge(self):
        out = _labelify_sources("[portal docs] Some excerpt text")
        assert "<span" in out
        assert "portal docs" in out
        assert "[portal docs]" not in out

    def test_portal_docs_badge_has_its_own_color(self):
        out = _labelify_sources("[portal docs] text")
        # Should not fall back to the default gray used for unrecognized labels.
        assert "#888888" not in out

    def test_existing_labels_still_work(self):
        out = _labelify_sources("[graph match] some text [semantic scholar] more")
        assert "<span" in out
        assert "graph match" in out
        assert "semantic scholar" in out


class TestLinkifyUrls:
    def test_bare_url_becomes_markdown_link(self):
        text = "Sources: https://digital-porous-media.github.io/dpm_docs/upload_data/"
        out = _linkify_urls(text)
        assert (
            "[https://digital-porous-media.github.io/dpm_docs/upload_data/]"
            "(https://digital-porous-media.github.io/dpm_docs/upload_data/)" in out
        )

    def test_does_not_double_wrap_an_existing_markdown_link(self):
        # Simulates running after _linkify_dois has already produced a markdown link.
        text = "See [DOI: 10.1234/abcd](https://doi.org/10.1234/abcd) for details."
        out = _linkify_urls(text)
        assert out.count("https://doi.org/10.1234/abcd") == 1
        assert "[https://doi.org/10.1234/abcd]" not in out


class TestRenderResponsePipeline:
    def test_doi_and_bare_url_both_linked_without_conflict(self):
        text = (
            "See DOI: 10.1234/abcd for details.\n"
            "Sources: [portal docs] https://digital-porous-media.github.io/dpm_docs/upload_data/"
        )
        out = _render_response(text)
        assert "[DOI: 10.1234/abcd](https://doi.org/10.1234/abcd)" in out
        assert (
            "[https://digital-porous-media.github.io/dpm_docs/upload_data/]"
            "(https://digital-porous-media.github.io/dpm_docs/upload_data/)" in out
        )
        assert "<span" in out  # portal docs badge still applied
