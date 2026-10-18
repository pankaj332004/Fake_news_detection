"""
Utilities module: security, exceptions, and helpers.
"""

from src.utils.exceptions import (
    EmptyContentError,
    ExtractionError,
    FakeNewsDetectionError,
    IngestionError,
    InputLengthExceededError,
    InputTooShortError,
    InvalidURLError,
    UnsafeURLError,
)
from src.utils.security import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    MAX_URL_LENGTH,
    is_private_ip,
    sanitize_text,
    validate_input_length,
    validate_url_safety,
)

__all__ = [
    "FakeNewsDetectionError",
    "IngestionError",
    "InvalidURLError",
    "UnsafeURLError",
    "ExtractionError",
    "EmptyContentError",
    "InputLengthExceededError",
    "InputTooShortError",
    "validate_url_safety",
    "sanitize_text",
    "validate_input_length",
    "is_private_ip",
    "MAX_URL_LENGTH",
    "DEFAULT_MIN_CHARS",
    "DEFAULT_MAX_CHARS",
]
