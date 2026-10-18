"""
Ingestion package: article and URL extraction, text input processing, and document loading.
"""

from src.ingestion.document_loader import detect_input_type, load_input
from src.ingestion.html_parser import extract_article_from_html
from src.ingestion.text_input import process_text_input
from src.ingestion.url_extractor import extract_from_url

__all__ = [
    "load_input",
    "detect_input_type",
    "extract_from_url",
    "extract_article_from_html",
    "process_text_input",
]
