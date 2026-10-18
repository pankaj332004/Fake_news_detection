"""
Security, URL safety, and input sanitization utilities.
Provides SSRF protection, URL validation, and input length guards.
"""

import ipaddress
import re
import socket
from urllib.parse import urlparse

from src.utils.exceptions import (
    InputLengthExceededError,
    InputTooShortError,
    InvalidURLError,
    UnsafeURLError,
)

MAX_URL_LENGTH: int = 2048
DEFAULT_MIN_CHARS: int = 5
DEFAULT_MAX_CHARS: int = 50000

# Blocked schemes
ALLOWED_SCHEMES = {"http", "https"}

# Blocked hostnames
BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "169.254.169.254",  # AWS/GCP cloud metadata
}


def is_private_ip(ip_str: str) -> bool:
    """Checks whether an IP address belongs to a private, loopback, or reserved network."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        return False


def validate_url_safety(url: str, allow_private: bool = False) -> str:
    """
    Validates that a URL is well-formed, uses http/https, within length bounds,
    and does not point to internal, private, or loopback network addresses (SSRF defense).

    Args:
        url: The URL string to validate.
        allow_private: If True, bypasses private IP/SSRF checks (useful for local mock tests).

    Returns:
        The validated and stripped URL.

    Raises:
        InvalidURLError: If the URL is malformed or uses an invalid scheme.
        InputLengthExceededError: If the URL exceeds MAX_URL_LENGTH.
        UnsafeURLError: If the URL resolves to a private or disallowed host.
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidURLError("URL must be a non-empty string.")

    cleaned_url = url.strip()

    if len(cleaned_url) > MAX_URL_LENGTH:
        raise InputLengthExceededError(
            f"URL length ({len(cleaned_url)}) exceeds maximum limit of {MAX_URL_LENGTH} characters."
        )

    try:
        parsed = urlparse(cleaned_url)
    except Exception as e:
        raise InvalidURLError(f"Malformed URL: {e}") from e

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise InvalidURLError(
            f"Unsupported URL scheme '{scheme}'. Only HTTP and HTTPS are permitted."
        )

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise InvalidURLError("URL is missing a valid hostname.")

    if allow_private:
        return cleaned_url

    # Check blacklist
    if hostname in BLOCKED_HOSTNAMES:
        raise UnsafeURLError(f"Access to restricted host '{hostname}' is forbidden.")

    # Check direct IP addresses
    if is_private_ip(hostname):
        raise UnsafeURLError(
            f"Access to private or local IP address '{hostname}' is forbidden."
        )

    # Resolve DNS to check for internal / private IP addresses (SSRF protection)
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            resolved_ip = sockaddr[0]
            if is_private_ip(resolved_ip):
                raise UnsafeURLError(
                    f"URL host '{hostname}' resolved to private IP '{resolved_ip}', which is forbidden."
                )
    except socket.gaierror:
        # Host might not resolve if offline or invalid domain; we allow DNS failure
        # to be handled downstream as extraction failure rather than security block
        pass

    return cleaned_url


def sanitize_text(text: str) -> str:
    """
    Sanitizes raw text input:
    - Removes NULL bytes and non-printable control characters (except newline, carriage return, and tab).
    - Normalizes multiple consecutive whitespace characters and strips boundary whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Remove null bytes
    cleaned = text.replace("\x00", "")

    # Remove ASCII control characters except \t, \n, \r
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Normalize unicode spaces
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned)  # zero-width spaces

    # Normalize carriage returns
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    return cleaned.strip()


def validate_input_length(
    text: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    field_name: str = "Input text",
) -> str:
    """
    Validates that text meets length requirements.

    Args:
        text: Input string.
        min_chars: Minimum character length.
        max_chars: Maximum character length.
        field_name: Descriptive name for error messages.

    Returns:
        The validated text.

    Raises:
        InputTooShortError: If text is shorter than min_chars.
        InputLengthExceededError: If text is longer than max_chars.
    """
    sanitized = sanitize_text(text)
    length = len(sanitized)

    if length < min_chars:
        raise InputTooShortError(
            f"{field_name} is too short ({length} characters). Minimum required is {min_chars}."
        )

    if length > max_chars:
        raise InputLengthExceededError(
            f"{field_name} exceeds maximum length ({length} characters). Maximum allowed is {max_chars}."
        )

    return sanitized
