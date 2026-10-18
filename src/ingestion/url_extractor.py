"""
URL extraction module for fetching and parsing web news articles.
Handles network requests, safety checks, HTTP status codes, and timeouts.
"""

import logging
from typing import Optional
from urllib.parse import urlparse
import requests

from src.ingestion.html_parser import extract_article_from_html
from src.schemas.request import ArticleDocument, InputType
from src.utils.exceptions import (
    EmptyContentError,
    ExtractionError,
    InputLengthExceededError,
)
from src.utils.security import (
    DEFAULT_MAX_CHARS,
    validate_input_length,
    validate_url_safety,
)

logger = logging.getLogger(__name__)

# Default realistic browser headers to prevent basic bot blocks
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def extract_from_url(
    url: str,
    timeout: int = 15,
    max_chars: int = DEFAULT_MAX_CHARS,
    allow_private: bool = False,
    session: Optional[requests.Session] = None,
) -> ArticleDocument:
    """
    Safely fetches and parses an article from a web URL.

    Args:
        url: Target HTTP/HTTPS URL.
        timeout: Request timeout in seconds.
        max_chars: Maximum character limit for the extracted text.
        allow_private: Whether to permit private IP access (for testing).
        session: Optional custom requests.Session.

    Returns:
        ArticleDocument populated with extracted text and metadata.

    Raises:
        InvalidURLError: If the URL is syntactically invalid.
        UnsafeURLError: If the URL targets private/internal IPs or hosts.
        ExtractionError: If network request fails or returns an HTTP error.
        EmptyContentError: If the page contains no readable article content.
        InputLengthExceededError: If the extracted article exceeds max_chars.
    """
    # 1. URL Safety and format verification
    validated_url = validate_url_safety(url, allow_private=allow_private)
    parsed = urlparse(validated_url)
    domain = parsed.netloc.lower()

    # 2. Fetch HTML content
    requester = session or requests
    try:
        logger.info(f"Fetching article content from: {validated_url}")
        response = requester.get(
            validated_url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.exceptions.Timeout as e:
        raise ExtractionError(f"Request timed out while connecting to {validated_url} ({timeout}s).") from e
    except requests.exceptions.SSLError as e:
        raise ExtractionError(f"SSL certificate verification failed for {validated_url}: {e}") from e
    except requests.exceptions.ConnectionError as e:
        raise ExtractionError(f"Network connection failed when reaching {validated_url}: {e}") from e
    except requests.exceptions.RequestException as e:
        raise ExtractionError(f"Failed to fetch content from {validated_url}: {e}") from e

    # 3. Validate HTTP status
    if response.status_code == 404:
        raise ExtractionError(f"Page not found (HTTP 404) at {validated_url}.")
    elif response.status_code == 403:
        raise ExtractionError(f"Access forbidden (HTTP 403) at {validated_url}.")
    elif response.status_code >= 400:
        raise ExtractionError(
            f"HTTP error {response.status_code} received from {validated_url}."
        )

    # 4. Check content type
    content_type = response.headers.get("Content-Type", "").lower()
    if "text" not in content_type and "html" not in content_type and "json" not in content_type:
        logger.warning(f"Unexpected Content-Type '{content_type}' from {validated_url}")

    html_content = response.text
    if not html_content or not html_content.strip():
        raise EmptyContentError(f"Received empty response from {validated_url}.")

    # 5. Extract readable article text
    extracted = extract_article_from_html(html_content, url=validated_url)

    text = extracted["text"]
    if not text or not text.strip():
        raise EmptyContentError(f"Could not extract readable article text from {validated_url}.")

    # 6. Check length limit
    if len(text) > max_chars:
        # Truncate gracefully or raise if strict limit is desired; we enforce max_chars with validate_input_length
        raise InputLengthExceededError(
            f"Extracted article length ({len(text)} characters) exceeds maximum limit of {max_chars} characters."
        )

    return ArticleDocument(
        raw_input=url,
        input_type=InputType.URL,
        title=extracted.get("title"),
        text=text,
        authors=extracted.get("authors", []),
        published_date=extracted.get("published_date"),
        source_url=validated_url,
        source_domain=domain,
        char_count=len(text),
        word_count=len(text.split()),
        metadata={
            "extractor": extracted.get("extractor"),
            "status_code": response.status_code,
            "content_type": content_type,
        },
    )
