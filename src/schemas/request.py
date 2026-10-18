"""
Pydantic schemas for input validation, ingestion requests, and structured document representations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class InputType(str, Enum):
    """Supported types of user inputs."""

    CLAIM = "claim"          # Short statement or factual assertion
    HEADLINE = "headline"    # News headline
    ARTICLE = "article"      # Full article raw text body
    URL = "url"              # Direct link to an online article / report


class IngestionRequest(BaseModel):
    """
    Standard request model for user inputs entering the fake news detection system.
    """

    content: str = Field(
        ...,
        description="The raw claim, headline, article body, or URL to analyze.",
        min_length=1,
    )
    input_type: Optional[InputType] = Field(
        default=None,
        description="Explicit type of input. If omitted, the system will auto-detect.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional caller metadata (e.g., author, source, timestamp).",
    )

    @field_validator("content")
    @classmethod
    def check_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Input content cannot be empty or only whitespace.")
        return stripped


class ArticleDocument(BaseModel):
    """
    Unified representation of an ingested statement, headline, or article.
    Serves as the standardized payload for all downstream stages (classification, retrieval, verification).
    """

    raw_input: str = Field(
        ...,
        description="The original user-provided input string (claim, text, or URL).",
    )
    input_type: InputType = Field(
        ...,
        description="The resolved input classification (claim, headline, article, or url).",
    )
    title: Optional[str] = Field(
        default=None,
        description="Article title or extracted headline (if applicable).",
    )
    text: str = Field(
        ...,
        description="Cleaned, human-readable primary text extracted from the input.",
    )
    authors: List[str] = Field(
        default_factory=list,
        description="List of detected or specified article authors.",
    )
    published_date: Optional[str] = Field(
        default=None,
        description="Publication date or timestamp string.",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Source URL if the document was fetched from the web.",
    )
    source_domain: Optional[str] = Field(
        default=None,
        description="Domain of the source website (e.g. 'bbc.com', 'reuters.com').",
    )
    char_count: int = Field(
        default=0,
        description="Character count of the cleaned primary text.",
    )
    word_count: int = Field(
        default=0,
        description="Word count of the cleaned primary text.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language code (e.g., 'en').",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional extraction metadata and diagnostic details.",
    )

    def get_analysis_snippet(self, max_words: int = 100) -> str:
        """Returns a condensed snippet of the text suitable for display."""
        words = self.text.split()
        if len(words) <= max_words:
            return self.text
        return " ".join(words[:max_words]) + "..."
