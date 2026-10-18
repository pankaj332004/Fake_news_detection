"""
Unit tests for end-to-end AnalysisPipeline (Phase 3).
"""

from unittest.mock import MagicMock, patch
import pytest

from src.pipeline.analysis_pipeline import AnalysisPipeline, analyze_input
from src.schemas.request import IngestionRequest, InputType
from src.utils.exceptions import InvalidURLError, UnsafeURLError


def test_pipeline_analyze_claim():
    """Verify end-to-end analysis of a claim input."""
    pipeline = AnalysisPipeline()
    res = pipeline.analyze("The government announced tax cuts for middle class families.")

    assert res["status"] == "SUCCESS"
    assert res["input_type"] in ["claim", "headline"]
    assert res["verdict"] in ["Real", "Fake"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["prob_fake"] <= 1.0
    assert 0.0 <= res["prob_real"] <= 1.0


def test_pipeline_analyze_article():
    """Verify end-to-end analysis of multiline article input."""
    article = (
        "Breaking Economic Report\n\n"
        "Official census bureau statistics released on Thursday show unemployment rates dropped to record lows.\n"
        "Industry analysts attribute the change to accelerated manufacturing investments across multiple states."
    )
    res = analyze_input(article, input_type="article")

    assert res["status"] == "SUCCESS"
    assert res["input_type"] == "article"
    assert res["word_count"] > 20
    assert res["verdict"] in ["Real", "Fake"]


@patch("src.ingestion.url_extractor.requests.get")
def test_pipeline_analyze_url(mock_get):
    """Verify end-to-end analysis of a news URL with mocked response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.text = """
    <html>
      <head><title>Space Telescope Captures Distant Galaxy Formation</title></head>
      <body>
        <article>
          <h1>Space Telescope Captures Distant Galaxy Formation</h1>
          <p>New infrared imagery obtained by astronomers reveals previously unobserved star clusters dating back to the early universe.</p>
          <p>Researchers published the peer-reviewed findings in the Astrophysical Journal today.</p>
        </article>
      </body>
    </html>
    """
    mock_get.return_value = mock_resp

    res = analyze_input("https://www.astronomy-news.org/early-universe")

    assert res["status"] == "SUCCESS"
    assert res["input_type"] == "url"
    assert res["source_domain"] == "www.astronomy-news.org"
    assert "Galaxy Formation" in (res["title"] or "")
    assert res["verdict"] in ["Real", "Fake"]


def test_pipeline_blocks_unsafe_url():
    """Verify pipeline rejects unsafe private URLs."""
    pipeline = AnalysisPipeline()
    with pytest.raises(UnsafeURLError):
        pipeline.analyze("http://127.0.0.1:8000/private-news")


def test_pipeline_blocks_invalid_url():
    """Verify pipeline rejects malformed URLs."""
    pipeline = AnalysisPipeline()
    with pytest.raises(InvalidURLError):
        pipeline.analyze("ftp://ftp.example.com/fake-news.html")
