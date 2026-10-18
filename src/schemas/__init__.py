"""
Schemas package: request, response, document contracts, and evidence retrieval models.
"""

from src.schemas.evidence import (
    EvidencePassage,
    RetrievalResponse,
    RetrievalResult,
    StanceType,
)
from src.schemas.request import ArticleDocument, IngestionRequest, InputType
from src.schemas.response import (
    AnalysisResponse,
    VerificationResult,
    VerificationVerdict,
)

__all__ = [
    "InputType",
    "IngestionRequest",
    "ArticleDocument",
    "EvidencePassage",
    "RetrievalResult",
    "RetrievalResponse",
    "StanceType",
    "AnalysisResponse",
    "VerificationResult",
    "VerificationVerdict",
]
