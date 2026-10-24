# -*- coding: utf-8 -*-
"""
Tests for storage, database, source verification, and image checker services.
"""

from io import BytesIO
import pytest
from PIL import Image

from frontend.components import map_to_standard_verdict
from src.services.image_checker import ImageCheckerService
from src.services.source_checker import SourceCheckerService
from src.storage.database import init_db
from src.storage.history_repository import HistoryRepository


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Ensure database tables and seeds are initialized."""
    init_db()


def test_map_to_standard_verdict():
    """Tests that pipeline verdicts map cleanly to the 4 standard verdicts."""
    assert map_to_standard_verdict("TRUE") == "Supported"
    assert map_to_standard_verdict("REAL") == "Supported"
    assert map_to_standard_verdict("FALSE") == "False"
    assert map_to_standard_verdict("REFUTED") == "False"
    assert map_to_standard_verdict("NEEDS_REVIEW") == "Misleading"
    assert map_to_standard_verdict("MIXTURE") == "Misleading"
    assert map_to_standard_verdict("UNVERIFIABLE") == "Unverified"
    assert map_to_standard_verdict("UNKNOWN") == "Unverified"


def test_history_repository_save_and_retrieve():
    """Tests saving a fact check and querying user history."""
    check_id = HistoryRepository.save_fact_check(
        user_id=1,
        input_type="claim",
        title="Test Scientific Claim",
        query_text="Testing automated fact check repository",
        verdict="TRUE",
        confidence=0.92,
        uncertainty="low",
        flagged_review=False,
        consensus_status="STRONG_AGREEMENT",
        consensus_synthesis="Verified by test evidence.",
        source_url=None,
        result_dict={"test": True},
    )
    assert check_id > 0

    history = HistoryRepository.get_user_history(user_id=1, search_query="Testing automated")
    assert len(history) >= 1
    assert history[0]["verdict"] == "TRUE"
    assert history[0]["confidence"] == 0.92


def test_history_bookmark_toggle():
    """Tests bookmarking and retrieving saved reports."""
    check_id = HistoryRepository.save_fact_check(
        user_id=1,
        input_type="claim",
        title="Bookmark Test",
        query_text="Test claim for bookmarking",
        verdict="FALSE",
        confidence=0.88,
        uncertainty="low",
        flagged_review=False,
        consensus_status="EVIDENCE_OVERRIDE",
        consensus_synthesis="Refuted by test.",
        source_url=None,
        result_dict={},
    )

    # Initially not bookmarked
    is_bm = HistoryRepository.toggle_bookmark(check_id)
    assert is_bm is True

    bookmarks = HistoryRepository.get_bookmarked_checks(user_id=1)
    bm_ids = [b["id"] for b in bookmarks]
    assert check_id in bm_ids

    # Toggle off
    is_bm_off = HistoryRepository.toggle_bookmark(check_id)
    assert is_bm_off is False


def test_admin_metrics_and_queue():
    """Tests admin operational metrics and review queue updates."""
    metrics = HistoryRepository.get_admin_metrics()
    assert metrics["total_checks"] >= 1
    assert "total_sources" in metrics

    queue = HistoryRepository.get_review_queue(status_filter="all")
    assert len(queue) >= 1

    item_id = queue[0]["id"]
    updated = HistoryRepository.update_review_item(item_id, "approved", "Approved by unit test", "admin@veritas.ai")
    assert updated is True


def test_source_checker_registered_and_unregistered():
    """Tests source verification for registered and uncatalogued publishers."""
    # 1. Registered Tier 1 Source
    res_who = SourceCheckerService.verify_source("https://www.who.int/news/item/1")
    assert res_who["success"] is True
    assert "Tier 1" in res_who["credibility_tier"]
    assert res_who["is_registered"] is True
    assert res_who["transparency_score"] >= 90

    # 2. Registered Satire Source
    res_onion = SourceCheckerService.verify_source("theonion.com")
    assert res_onion["success"] is True
    assert "Satire" in res_onion["credibility_tier"]

    # 3. Unregistered Government / Academic domain
    res_gov = SourceCheckerService.verify_source("research.state.gov")
    assert res_gov["success"] is True
    assert "Government" in res_gov["credibility_tier"] or "Tier 1" in res_gov["credibility_tier"]

    # 4. Questionable TLD
    res_spam = SourceCheckerService.verify_source("daily-breaking-news.xyz")
    assert res_spam["success"] is True
    assert "Questionable" in res_spam["credibility_tier"]
    assert len(res_spam["flags"]) > 0


def test_image_checker_inspection_and_ela():
    """Tests image forensic inspection and ELA difference analysis."""
    # Generate in-memory test image
    img = Image.new("RGB", (200, 100), color=(30, 41, 59))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()

    inspection = ImageCheckerService.inspect_image(img_bytes, filename="test_chart.jpg")
    assert inspection["success"] is True
    assert inspection["dimensions"] == "200 × 100 px"
    assert "ela_image" in inspection
    assert "ela_score" in inspection
    assert "search_links" in inspection
    assert "Google Lens" in inspection["search_links"]
    assert len(inspection["extracted_text"]) > 0
