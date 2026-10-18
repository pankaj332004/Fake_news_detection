"""
HTML parser module for article content extraction.
Uses trafilatura as primary high-precision extractor with BeautifulSoup fallback and metadata enrichment.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import trafilatura

from src.utils.exceptions import EmptyContentError
from src.utils.security import sanitize_text

logger = logging.getLogger(__name__)


def _extract_title_and_meta_bs4(html_content: str) -> Dict[str, Any]:
    """Extracts title, authors, and published date from HTML head/meta tags."""
    soup = BeautifulSoup(html_content, "html.parser")
    title = None

    # Priority 1: OpenGraph or Twitter title
    og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()

    # Priority 2: <h1> tag
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Priority 3: <title> tag
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

    # Clean title common website suffixes (e.g. "Article Title - BBC News" -> "Article Title")
    if title and " - " in title:
        title = title.split(" - ")[0].strip()
    elif title and " | " in title:
        title = title.split(" | ")[0].strip()

    # Authors
    authors: List[str] = []
    author_meta = (
        soup.find("meta", attrs={"name": re.compile(r"^author$", re.I)})
        or soup.find("meta", property=re.compile(r"article:author", re.I))
    )
    if author_meta and author_meta.get("content"):
        authors.append(author_meta["content"].strip())

    # Date
    date_str: Optional[str] = None
    date_meta = (
        soup.find("meta", property=re.compile(r"article:published_time|date", re.I))
        or soup.find("meta", attrs={"name": re.compile(r"^date$|pubdate", re.I)})
    )
    if date_meta and date_meta.get("content"):
        date_str = date_meta["content"].strip()

    return {
        "title": title,
        "authors": authors,
        "published_date": date_str,
    }


def extract_article_from_html(
    html_content: str,
    url: Optional[str] = None,
    min_length: int = 20,
) -> Dict[str, Any]:
    """
    Parses HTML content to extract clean article text and metadata.

    Args:
        html_content: Raw HTML text.
        url: Optional source URL to assist extraction heuristics.
        min_length: Minimum number of characters required for valid extraction.

    Returns:
        Dict containing title, text, authors, published_date, extractor, char_count, word_count.

    Raises:
        EmptyContentError: If no readable article text can be extracted.
    """
    if not html_content or not html_content.strip():
        raise EmptyContentError("HTML content is empty.")

    # Extract auxiliary metadata from HTML head/meta
    head_meta = _extract_title_and_meta_bs4(html_content)

    # 1. Primary extraction via trafilatura
    try:
        trafilatura_result = trafilatura.extract(
            html_content,
            url=url,
            output_format="json",
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )

        if trafilatura_result:
            data = json.loads(trafilatura_result)
            raw_text = data.get("text", "")
            title = data.get("title") or head_meta.get("title")
            author_str = data.get("author")
            date_str = data.get("date") or head_meta.get("published_date")

            cleaned_text = sanitize_text(raw_text)
            if len(cleaned_text) >= min_length:
                authors = [author_str] if author_str else head_meta.get("authors", [])
                return {
                    "title": title.strip() if title else None,
                    "text": cleaned_text,
                    "authors": authors,
                    "published_date": date_str,
                    "extractor": "trafilatura",
                    "char_count": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                }
    except Exception as e:
        logger.debug(f"Trafilatura extraction encountered an issue: {e}. Falling back to BeautifulSoup.")

    # 2. Fallback extraction via BeautifulSoup
    logger.info("Attempting BeautifulSoup fallback extraction...")
    return _extract_with_beautifulsoup(html_content, head_meta=head_meta, min_length=min_length)


def _extract_with_beautifulsoup(
    html_content: str,
    head_meta: Dict[str, Any],
    min_length: int = 20,
) -> Dict[str, Any]:
    """
    Fallback extractor using BeautifulSoup to extract text from common article containers.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    title = head_meta.get("title")
    authors = head_meta.get("authors", [])
    date_str = head_meta.get("published_date")

    # Remove non-content elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]):
        element.decompose()

    # Priority 1: Check standard article containers
    article_container = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"article[-_]?body|post[-_]?content|entry[-_]?content|story[-_]?content", re.I))
    )

    paragraphs = []
    target_node = article_container if article_container else soup

    for p in target_node.find_all("p"):
        p_text = p.get_text(separator=" ", strip=True)
        # Filter out tiny boilerplate / advertisement lines
        if len(p_text) > 25:
            paragraphs.append(p_text)

    combined_text = "\n\n".join(paragraphs)
    cleaned_text = sanitize_text(combined_text)

    # If still empty, get text from the whole body
    if len(cleaned_text) < min_length:
        body = soup.find("body")
        if body:
            cleaned_text = sanitize_text(body.get_text(separator="\n", strip=True))

    if len(cleaned_text) < min_length:
        raise EmptyContentError(
            f"Failed to extract readable text from HTML (extracted only {len(cleaned_text)} characters)."
        )

    return {
        "title": title,
        "text": cleaned_text,
        "authors": authors,
        "published_date": date_str,
        "extractor": "beautifulsoup",
        "char_count": len(cleaned_text),
        "word_count": len(cleaned_text.split()),
    }
