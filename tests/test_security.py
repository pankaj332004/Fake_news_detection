"""
Unit tests for security, URL safety checks, SSRF prevention, and input length guards.
"""

import pytest

from src.utils.exceptions import (
    InputLengthExceededError,
    InputTooShortError,
    InvalidURLError,
    UnsafeURLError,
)
from src.utils.security import (
    MAX_URL_LENGTH,
    is_private_ip,
    sanitize_text,
    validate_input_length,
    validate_url_safety,
)


def test_valid_public_urls():
    """Verify that legitimate public URLs pass validation."""
    valid_urls = [
        "https://www.bbc.com/news/world-123456",
        "http://reuters.com/article/example",
        "https://en.wikipedia.org/wiki/Fake_news",
        "https://sub.domain.example.co.uk/path?arg=1&val=test#section",
    ]
    for url in valid_urls:
        validated = validate_url_safety(url)
        assert validated == url


def test_blocked_schemes():
    """Verify non-HTTP(S) schemes are blocked."""
    invalid_schemes = [
        "ftp://ftp.example.com/file.txt",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<h1>Hello</h1>",
        "gopher://example.com",
    ]
    for url in invalid_schemes:
        with pytest.raises(InvalidURLError):
            validate_url_safety(url)


def test_ssrf_protection_blocked_hosts():
    """Verify localhost and private IP addresses trigger UnsafeURLError."""
    unsafe_urls = [
        "http://localhost:8080/admin",
        "http://127.0.0.1/status",
        "http://0.0.0.0/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/secrets",
        "http://192.168.1.1/router",
        "http://172.16.0.1/internal",
        "http://[::1]/admin",
    ]
    for url in unsafe_urls:
        with pytest.raises(UnsafeURLError):
            validate_url_safety(url)


def test_url_length_limit():
    """Verify URLs exceeding MAX_URL_LENGTH trigger InputLengthExceededError."""
    long_url = "https://example.com/" + ("a" * (MAX_URL_LENGTH + 10))
    with pytest.raises(InputLengthExceededError):
        validate_url_safety(long_url)


def test_empty_or_malformed_url():
    """Verify empty, whitespace, or hostless URLs raise InvalidURLError."""
    with pytest.raises(InvalidURLError):
        validate_url_safety("")
    with pytest.raises(InvalidURLError):
        validate_url_safety("   ")
    with pytest.raises(InvalidURLError):
        validate_url_safety("https://")


def test_sanitize_text():
    """Verify null bytes, control characters, and zero-width spaces are stripped."""
    dirty_text = "Headline\x00 with null byte\x08 and \u200Bzero-width space.\r\nNext line."
    clean = sanitize_text(dirty_text)
    assert "\x00" not in clean
    assert "\x08" not in clean
    assert "\u200B" not in clean
    assert clean == "Headline with null byte and zero-width space.\nNext line."


def test_validate_input_length_bounds():
    """Verify minimum and maximum length bounds for input text."""
    # Too short
    with pytest.raises(InputTooShortError):
        validate_input_length("hi", min_chars=5)

    # Within bounds
    valid = validate_input_length("Valid headline text", min_chars=5, max_chars=100)
    assert valid == "Valid headline text"

    # Too long
    with pytest.raises(InputLengthExceededError):
        validate_input_length("A" * 150, min_chars=5, max_chars=100)
