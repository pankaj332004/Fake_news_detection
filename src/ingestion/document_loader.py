"""
Master document loader and ingestion orchestrator.
Accepts raw text or URLs, auto-detects input type, coordinates safety and extraction,
and returns structured ArticleDocument objects.
"""

import logging
import re
from typing import Any, Dict, Optional, Union

from src.ingestion.text_input import process_text_input
from src.ingestion.url_extractor import extract_from_url
from src.schemas.request import ArticleDocument, IngestionRequest, InputType
from src.utils.exceptions import IngestionError

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(
    r"^(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:\/[^\s]*)?$",
    re.IGNORECASE,
)


def detect_input_type(content: str) -> InputType:
    """
    Intelligently determines whether user input is a URL, Claim, Headline, or Article.

    Rules:
    - Starts with http:// or https://, or matches valid URL syntax without whitespace -> URL.
    - Single line and <= 150 chars -> HEADLINE or CLAIM.
    - Multiple lines or > 500 chars -> ARTICLE.
    - Otherwise -> CLAIM.
    """
    stripped = content.strip()

    # 1. URL check (starts with any scheme like http://, https://, ftp:// or matches URL pattern)
    if "://" in stripped and "\n" not in stripped and " " not in stripped:
        return InputType.URL
    if "\n" not in stripped and " " not in stripped and URL_PATTERN.match(stripped):
        return InputType.URL

    # 2. Multi-line or long article check
    line_count = len(stripped.splitlines())
    char_count = len(stripped)

    if line_count > 2 or char_count > 600:
        return InputType.ARTICLE

    # 3. Headline vs Claim heuristic
    if char_count <= 180 and not stripped.endswith("."):
        return InputType.HEADLINE

    return InputType.CLAIM


def load_input(
    content_or_request: Union[str, IngestionRequest],
    input_type: Optional[Union[InputType, str]] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
    allow_private: bool = False,
) -> ArticleDocument:
    """
    Unified entry point for loading user inputs into the pipeline.

    Args:
        content_or_request: A raw string (claim, text, or URL) or IngestionRequest.
        input_type: Optional explicit input type.
        title: Optional title/headline for text inputs.
        metadata: Optional metadata dictionary.
        timeout: Timeout in seconds for web requests.
        allow_private: Whether to allow private IPs (for testing).

    Returns:
        ArticleDocument ready for downstream processing.
    """
    if isinstance(content_or_request, IngestionRequest):
        raw_content = content_or_request.content
        resolved_type = content_or_request.input_type
        req_metadata = {**content_or_request.metadata, **(metadata or {})}
    else:
        raw_content = str(content_or_request)
        resolved_type = InputType(input_type) if input_type else None
        req_metadata = metadata or {}

    # Auto-detect if not explicitly provided
    if not resolved_type:
        resolved_type = detect_input_type(raw_content)
        logger.info(f"Auto-detected input type: '{resolved_type.value}'")

    # Route based on input type
    if resolved_type == InputType.URL:
        url = raw_content.strip()
        if "://" not in url:
            url = "https://" + url

        return extract_from_url(
            url=url,
            timeout=timeout,
            allow_private=allow_private,
        )
    else:
        return process_text_input(
            raw_text=raw_content,
            input_type=resolved_type,
            title=title,
            metadata=req_metadata,
        )
