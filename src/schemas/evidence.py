"""
Pydantic schemas for evidence passages, retrieval queries, and retrieval responses.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StanceType(str, Enum):
    """Semantic relation of an evidence passage to the target claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    UNSPECIFIED = "unspecified"


class EvidencePassage(BaseModel):
    """
    A single factual passage extracted from a trusted reference document.
    """

    passage_id: str = Field(..., description="Unique passage identifier (e.g. 'DOC-NASA-01#p0').")
    doc_id: str = Field(..., description="Parent document identifier.")
    text: str = Field(..., description="Full text content of the passage.")
    title: Optional[str] = Field(default=None, description="Title of the source document.")
    source: str = Field(..., description="Name of the authoritative publisher or institution.")
    source_url: Optional[str] = Field(default=None, description="Canonical URL or reference link.")
    category: Optional[str] = Field(default=None, description="Domain category (e.g. 'health', 'space', 'climate').")
    published_date: Optional[str] = Field(default=None, description="Publication date string.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary metadata.")


class RetrievalResult(BaseModel):
    """
    A retrieved evidence passage paired with similarity score and stance assessment.
    """

    passage: EvidencePassage = Field(..., description="The retrieved evidence passage.")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0).")
    rank: int = Field(..., description="Rank in retrieval results (1-indexed).")
    stance: StanceType = Field(default=StanceType.UNSPECIFIED, description="Relation to the claim (supports, contradicts).")
    explanation: Optional[str] = Field(default=None, description="Brief rationale for the stance alignment.")


class RetrievalResponse(BaseModel):
    """
    Complete response containing evidence retrieved for an input claim or article.
    """

    query: str = Field(..., description="The input query statement or headline.")
    total_retrieved: int = Field(default=0, description="Total number of evidence passages retrieved.")
    supporting_passages: List[RetrievalResult] = Field(default_factory=list, description="Passages that corroborate the claim.")
    contradicting_passages: List[RetrievalResult] = Field(default_factory=list, description="Passages that refute or contradict the claim.")
    all_passages: List[RetrievalResult] = Field(default_factory=list, description="All retrieved passages ordered by similarity.")
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model used for retrieval.")
    retrieval_time_ms: float = Field(default=0.0, description="Elapsed retrieval time in milliseconds.")
