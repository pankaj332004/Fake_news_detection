"""
Custom exceptions for Fake News Detection pipeline.
Covers ingestion, security, retrieval, and verification modules.
"""


class FakeNewsDetectionError(Exception):
    """Base exception for all domain errors in fake news detection."""

    pass


class IngestionError(FakeNewsDetectionError):
    """Base exception for data ingestion and input extraction failures."""

    pass


class InvalidURLError(IngestionError):
    """Raised when an input URL is malformed, invalid, or has an unsupported scheme."""

    pass


class UnsafeURLError(IngestionError):
    """Raised when a URL fails safety checks (e.g. SSRF targeting private IP/localhost)."""

    pass


class ExtractionError(IngestionError):
    """Raised when fetching or parsing content from a source fails."""

    pass


class EmptyContentError(IngestionError):
    """Raised when an article, web page, or text input contains no readable text."""

    pass


class InputLengthExceededError(IngestionError):
    """Raised when input text or URL exceeds configured maximum length limits."""

    pass


class InputTooShortError(IngestionError):
    """Raised when input text is too short to be meaningfully analyzed."""

    pass
