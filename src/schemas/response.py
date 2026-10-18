"""
Pydantic schemas for verification and analysis response objects.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VerificationVerdict(str, Enum):
    """Final verification verdict values."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class VerificationResult(BaseModel):
    """Structured LLM verification result."""

    final_verdict: VerificationVerdict = Field(
        ..., description="Final combined verdict from LLM and evidence analysis."
    )
    llm_verdict: Optional[str] = Field(
        default=None, description="Raw verdict from the LLM."
    )
    reasoning: str = Field(
        default="", description="LLM reasoning with evidence references."
    )
    evidence_citations: List[str] = Field(
        default_factory=list, description="Passage IDs cited by the LLM."
    )
    contradictions_found: bool = Field(
        default=False, description="Whether contradictory evidence was found."
    )
    limitations: str = Field(
        default="", description="Stated limitations and caveats."
    )
    overall_confidence: float = Field(
        default=0.0, description="Calibrated overall confidence (0.0-1.0)."
    )
    uncertainty_level: str = Field(
        default="high", description="Uncertainty level: low, medium, or high."
    )
    should_flag_for_review: bool = Field(
        default=False, description="Whether result should be flagged for human review."
    )
    model_name: str = Field(
        default="", description="Name of the LLM used for verification."
    )


class AnalysisResponse(BaseModel):
    """Complete analysis response combining ingestion, classification, evidence, and verification."""

    status: str = Field(default="SUCCESS", description="Pipeline execution status.")
    input_type: str = Field(..., description="Detected input type (claim/headline/article/url).")
    raw_input: str = Field(..., description="Original user input.")
    title: Optional[str] = Field(default=None, description="Extracted title/headline.")
    source_url: Optional[str] = Field(default=None, description="Source URL if applicable.")
    source_domain: Optional[str] = Field(default=None, description="Source domain.")
    authors: List[str] = Field(default_factory=list, description="Detected authors.")
    published_date: Optional[str] = Field(default=None, description="Publication date.")
    char_count: int = Field(default=0, description="Character count.")
    word_count: int = Field(default=0, description="Word count.")
    evaluated_text: str = Field(default="", description="Text used for analysis.")
    preview: str = Field(default="", description="Condensed text preview.")

    # Statistical classification (Phase 2)
    statistical_verdict: str = Field(default="", description="Baseline classifier verdict.")
    statistical_confidence: float = Field(default=0.0, description="Baseline classifier confidence.")
    prob_fake: float = Field(default=0.0, description="Probability of being fake.")
    prob_real: float = Field(default=0.0, description="Probability of being real.")

    # Evidence retrieval (Phase 4)
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Evidence retrieval results.")

    # LLM verification (Phase 5)
    verification: Optional[VerificationResult] = Field(
        default=None, description="LLM verification results."
    )
