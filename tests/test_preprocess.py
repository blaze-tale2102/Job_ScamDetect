"""
Unit tests for ``src.preprocess``.
"""

import pytest

from src.preprocess import (
    clean_text,
    normalize_whitespace,
    remove_emails,
    remove_punctuation,
    remove_stopwords,
    remove_urls,
    strip_html,
)


class TestStripHtml:
    def test_removes_tags(self):
        result = strip_html("<p>Hello <b>world</b></p>")
        assert "Hello" in result and "world" in result
        assert "<p>" not in result and "<b>" not in result

    def test_handles_plain_text(self):
        assert strip_html("no tags here") == "no tags here"

    def test_handles_entities(self):
        # BeautifulSoup may preserve entities; clean_text runs unescape first
        result = strip_html("<p>a &amp; b</p>")
        assert "a" in result and "b" in result


class TestRemoveUrls:
    def test_http(self):
        assert "click" in remove_urls("click https://evil.com now")

    def test_www(self):
        assert "visit" in remove_urls("visit www.example.com today")


class TestRemoveEmails:
    def test_basic(self):
        result = remove_emails("send to test@example.com please")
        assert "@" not in result


class TestRemovePunctuation:
    def test_keeps_alphanum(self):
        assert remove_punctuation("Hello, world! 123") == "Hello  world  123"


class TestRemoveStopwords:
    def test_removes_common(self):
        result = remove_stopwords("this is a test of the system")
        assert "this" not in result.split()
        assert "test" in result.split()


class TestNormalizeWhitespace:
    def test_collapses(self):
        assert normalize_whitespace("  a   b   ") == "a b"


class TestCleanText:
    def test_full_pipeline(self):
        raw = '<p>Visit https://scam.com — EARN $$$! Call NOW.</p>'
        result = clean_text(raw)
        assert "https" not in result
        assert "<p>" not in result
        assert result == result.lower()  # lowercase

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text("   ") == ""

    def test_handles_html_entities(self):
        # &amp; should be unescaped to &, then removed by punctuation stripping
        result = clean_text("&amp; hello world testing entities")
        assert "&amp;" not in result
        assert "hello" in result
