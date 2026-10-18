"""
Text input module for handling direct user claims, headlines, and pasted article text.
Validates lengths, cleans content, and constructs structured ArticleDocument instances.
"""

from typing import Any, Dict, Optional

from src.schemas.request import ArticleDocument, InputType
from src.utils.security import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    validate_input_length,
)

# Characteristic length thresholds
MAX_HEADLINE_CHARS: int = 400
MAX_CLAIM_CHARS: int = 1000


def process_text_input(
    raw_text: str,
    input_type: InputType = InputType.CLAIM,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ArticleDocument:
    """
    Processes and validates raw user text inputs (claims, headlines, articles).

    Args:
        raw_text: The user text content.
        input_type: One of InputType.CLAIM, InputType.HEADLINE, InputType.ARTICLE.
        title: Optional headline or title.
        metadata: Optional dictionary of supplementary context.
        min_chars: Minimum character length.
        max_chars: Maximum character length.

    Returns:
        Structured ArticleDocument.

    Raises:
        InputTooShortError: If text is shorter than min_chars.
        InputLengthExceededError: If text is longer than max_chars.
    """
    field_label = f"Input {input_type.value}"
    clean_text = validate_input_length(
        raw_text, min_chars=min_chars, max_chars=max_chars, field_name=field_label
    )

    resolved_title = title
    if not resolved_title:
        if input_type in (InputType.HEADLINE, InputType.CLAIM) and len(clean_text) <= MAX_HEADLINE_CHARS:
            resolved_title = clean_text
        elif input_type == InputType.ARTICLE:
            # If the article has paragraphs, the first non-empty line can serve as candidate title
            first_line = clean_text.split("\n")[0].strip()
            if 10 <= len(first_line) <= 200:
                resolved_title = first_line

    words = clean_text.split()
    return ArticleDocument(
        raw_input=raw_text,
        input_type=input_type,
        title=resolved_title,
        text=clean_text,
        char_count=len(clean_text),
        word_count=len(words),
        metadata=metadata or {},
    )
