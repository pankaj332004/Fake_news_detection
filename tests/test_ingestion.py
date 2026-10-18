"""
Unit tests for article and URL ingestion, HTML extraction, and document loading.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.ingestion.document_loader import detect_input_type, load_input
from src.ingestion.html_parser import extract_article_from_html
from src.ingestion.text_input import process_text_input
from src.ingestion.url_extractor import extract_from_url
from src.schemas.request import ArticleDocument, IngestionRequest, InputType
from src.utils.exceptions import (
    EmptyContentError,
    ExtractionError,
    InputLengthExceededError,
    InputTooShortError,
)

SAMPLE_ARTICLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Scientists Discover New Exoplanet in Habitable Zone - Science Daily</title>
    <meta name="author" content="Dr. Jane Smith">
    <meta property="article:published_time" content="2026-05-12">
</head>
<body>
    <nav><a href="/">Home</a> <a href="/news">News</a></nav>
    <article>
        <h1>Scientists Discover New Exoplanet in Habitable Zone</h1>
        <p class="byline">By Dr. Jane Smith | May 12, 2026</p>
        <p>Astronomers using next-generation space telescopes have identified an Earth-sized exoplanet orbiting a nearby red dwarf star.</p>
        <p>The planet, designated Kepler-452z, sits firmly within the circumstellar habitable zone where liquid surface water could plausibly exist.</p>
        <p>Spectroscopic analysis suggests the presence of an atmosphere with detectable nitrogen and carbon dioxide signatures.</p>
    </article>
    <aside class="advertisement">
        <p>Buy cheap supplements now at discount prices!</p>
    </aside>
    <footer>
        <p>Copyright 2026 Science Daily. All rights reserved.</p>
    </footer>
</body>
</html>
"""


def test_detect_input_type():
    """Verify automatic classification of input types."""
    # URLs
    assert detect_input_type("https://www.reuters.com/world/article-123") == InputType.URL
    assert detect_input_type("http://example.org/news") == InputType.URL
    assert detect_input_type("bbc.com/news/123") == InputType.URL

    # Headline
    assert detect_input_type("Government Approves New Energy Bill in Parliament") == InputType.HEADLINE

    # Claim
    assert detect_input_type("The COVID-19 vaccine alters human DNA permanently.") == InputType.CLAIM

    # Long article
    long_article = (
        "This is paragraph one of the article discussing economic reforms.\n\n"
        "This is paragraph two outlining the detailed implications for markets.\n\n"
        "Finally, this is the concluding paragraph summarizing the expert panel opinions."
    )
    assert detect_input_type(long_article) == InputType.ARTICLE


def test_process_text_input_claim():
    """Verify processing a single claim input."""
    claim_text = "NASA confirmed secret base discovered on the far side of Mars."
    doc = process_text_input(claim_text, input_type=InputType.CLAIM)

    assert isinstance(doc, ArticleDocument)
    assert doc.input_type == InputType.CLAIM
    assert doc.text == claim_text
    assert doc.title == claim_text
    assert doc.char_count == len(claim_text)
    assert doc.word_count == len(claim_text.split())


def test_process_text_input_article():
    """Verify processing multiline article text."""
    article_text = (
        "Breaking News Headline\n\n"
        "The committee convened this morning to deliberate on the proposed legislation.\n"
        "Members voted overwhelmingly in favor of passing the statutory framework."
    )
    doc = process_text_input(article_text, input_type=InputType.ARTICLE)

    assert doc.input_type == InputType.ARTICLE
    assert doc.title == "Breaking News Headline"
    assert "committee convened" in doc.text
    assert doc.word_count > 10


def test_extract_article_from_html():
    """Verify HTML parsing and boilerplate removal."""
    extracted = extract_article_from_html(SAMPLE_ARTICLE_HTML)

    assert "Scientists Discover New Exoplanet" in (extracted["title"] or "")
    assert "Astronomers using next-generation space telescopes" in extracted["text"]
    assert "Buy cheap supplements" not in extracted["text"]  # Stripped ad
    assert "Copyright 2026" not in extracted["text"]          # Stripped footer
    assert extracted["char_count"] > 100
    assert extracted["word_count"] > 20


def test_extract_from_empty_html():
    """Verify empty or non-informative HTML raises EmptyContentError."""
    with pytest.raises(EmptyContentError):
        extract_article_from_html("<html><body></body></html>")

    with pytest.raises(EmptyContentError):
        extract_article_from_html("")


@patch("src.ingestion.url_extractor.requests.get")
def test_extract_from_url_success(mock_get):
    """Verify successful URL article extraction."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_resp.text = SAMPLE_ARTICLE_HTML
    mock_get.return_value = mock_resp

    url = "https://www.sciencedaily.com/releases/2026/exoplanet"
    doc = extract_from_url(url)

    assert doc.input_type == InputType.URL
    assert doc.source_url == url
    assert doc.source_domain == "www.sciencedaily.com"
    assert "Exoplanet" in (doc.title or "")
    assert "Kepler-452z" in doc.text


@patch("src.ingestion.url_extractor.requests.get")
def test_extract_from_url_http_404(mock_get):
    """Verify HTTP 404 raises ExtractionError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    with pytest.raises(ExtractionError, match="Page not found"):
        extract_from_url("https://www.example.com/non-existent-page")


@patch("src.ingestion.url_extractor.requests.get")
def test_extract_from_url_timeout(mock_get):
    """Verify network timeouts raise ExtractionError."""
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    with pytest.raises(ExtractionError, match="timed out"):
        extract_from_url("https://www.example.com/slow-page")


@patch("src.ingestion.url_extractor.requests.get")
def test_extract_from_url_empty_page(mock_get):
    """Verify empty web pages raise EmptyContentError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.text = "   "
    mock_get.return_value = mock_resp

    with pytest.raises(EmptyContentError):
        extract_from_url("https://www.example.com/blank")


def test_load_input_with_ingestion_request():
    """Verify end-to-end loading using IngestionRequest model."""
    req = IngestionRequest(
        content="The Federal Reserve announced an unexpected interest rate cut today.",
        input_type=InputType.HEADLINE,
        metadata={"source": "financial_wire"},
    )
    doc = load_input(req)

    assert doc.input_type == InputType.HEADLINE
    assert doc.text == req.content
    assert doc.metadata["source"] == "financial_wire"


def test_article_document_snippet():
    """Verify snippet truncation helper on ArticleDocument."""
    text = " ".join([f"word{i}" for i in range(150)])
    doc = ArticleDocument(
        raw_input=text,
        input_type=InputType.ARTICLE,
        text=text,
        char_count=len(text),
        word_count=150,
    )
    snippet = doc.get_analysis_snippet(max_words=20)
    assert len(snippet.split()) == 20
