"""
Evidence Analyzer: Processes raw LLM output and retrieval results
to produce structured analysis of evidence quality, coverage, and gaps.
"""

import logging
from typing import Any, Dict, List, Optional

from src.schemas.evidence import RetrievalResponse, RetrievalResult, StanceType

logger = logging.getLogger(__name__)

# Thresholds for evidence quality assessment
MIN_PASSAGES_FOR_RELIABLE = 2
HIGH_RELEVANCE_THRESHOLD = 0.5
LOW_RELEVANCE_THRESHOLD = 0.3


class EvidenceQuality:
    """Assessment of the quality and sufficiency of retrieved evidence."""

    def __init__(
        self,
        total_passages: int = 0,
        high_relevance_count: int = 0,
        supporting_count: int = 0,
        contradicting_count: int = 0,
        neutral_count: int = 0,
        avg_similarity: float = 0.0,
        max_similarity: float = 0.0,
        has_sufficient_evidence: bool = False,
        has_contradictory_evidence: bool = False,
        has_supporting_evidence: bool = False,
        coverage_assessment: str = "none",
        quality_score: float = 0.0,
    ):
        self.total_passages = total_passages
        self.high_relevance_count = high_relevance_count
        self.supporting_count = supporting_count
        self.contradicting_count = contradicting_count
        self.neutral_count = neutral_count
        self.avg_similarity = avg_similarity
        self.max_similarity = max_similarity
        self.has_sufficient_evidence = has_sufficient_evidence
        self.has_contradictory_evidence = has_contradictory_evidence
        self.has_supporting_evidence = has_supporting_evidence
        self.coverage_assessment = coverage_assessment
        self.quality_score = quality_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_passages": self.total_passages,
            "high_relevance_count": self.high_relevance_count,
            "supporting_count": self.supporting_count,
            "contradicting_count": self.contradicting_count,
            "neutral_count": self.neutral_count,
            "avg_similarity": round(self.avg_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "has_contradictory_evidence": self.has_contradictory_evidence,
            "has_supporting_evidence": self.has_supporting_evidence,
            "coverage_assessment": self.coverage_assessment,
            "quality_score": round(self.quality_score, 3),
        }


class EvidenceAnalyzer:
    """
    Analyzes the quality, coverage, and distribution of retrieved evidence
    to inform LLM verification decisions.
    """

    def __init__(
        self,
        high_relevance_threshold: float = HIGH_RELEVANCE_THRESHOLD,
        low_relevance_threshold: float = LOW_RELEVANCE_THRESHOLD,
        min_passages_for_reliable: int = MIN_PASSAGES_FOR_RELIABLE,
    ):
        self.high_relevance_threshold = high_relevance_threshold
        self.low_relevance_threshold = low_relevance_threshold
        self.min_passages_for_reliable = min_passages_for_reliable

    def analyze(
        self,
        retrieval_response: Optional[RetrievalResponse] = None,
        all_passages: Optional[List[RetrievalResult]] = None,
        supporting_passages: Optional[List[RetrievalResult]] = None,
        contradicting_passages: Optional[List[RetrievalResult]] = None,
    ) -> EvidenceQuality:
        """
        Analyzes evidence quality from a RetrievalResponse or pre-sorted passages.

        Args:
            retrieval_response: Full retrieval response (takes priority if provided).
            all_passages: List of all retrieved passages (used if no RetrievalResponse).
            supporting_passages: Pre-filtered supporting passages.
            contradicting_passages: Pre-filtered contradicting passages.

        Returns:
            EvidenceQuality assessment object.
        """
        if retrieval_response is not None:
            all_passages = retrieval_response.all_passages
            supporting_passages = retrieval_response.supporting_passages
            contradicting_passages = retrieval_response.contradicting_passages

        all_passages = all_passages or []
        supporting_passages = supporting_passages or []
        contradicting_passages = contradicting_passages or []

        if not all_passages:
            return EvidenceQuality(coverage_assessment="none")

        # Compute similarity statistics
        scores = [r.score for r in all_passages if r.score is not None]
        avg_sim = sum(scores) / len(scores) if scores else 0.0
        max_sim = max(scores) if scores else 0.0
        high_rel = sum(1 for s in scores if s >= self.high_relevance_threshold)

        # Count stances
        n_support = len(supporting_passages)
        n_contra = len(contradicting_passages)
        n_neutral = len(all_passages) - n_support - n_contra

        # Determine sufficiency
        has_sufficient = (
            len(all_passages) >= self.min_passages_for_reliable
            and high_rel >= 1
        )

        # Coverage assessment
        if len(all_passages) == 0:
            coverage = "none"
        elif max_sim < self.low_relevance_threshold:
            coverage = "weak"
        elif high_rel >= self.min_passages_for_reliable:
            coverage = "strong"
        elif high_rel >= 1:
            coverage = "moderate"
        else:
            coverage = "weak"

        # Quality score (0.0 to 1.0) combining relevance, count, and stance clarity
        relevance_factor = min(avg_sim / 0.6, 1.0)  # Normalize against target similarity
        count_factor = min(len(all_passages) / 5.0, 1.0)  # More passages = better
        stance_clarity = 1.0 if (n_support > 0 or n_contra > 0) else 0.5
        quality_score = (relevance_factor * 0.4 + count_factor * 0.3 + stance_clarity * 0.3)

        return EvidenceQuality(
            total_passages=len(all_passages),
            high_relevance_count=high_rel,
            supporting_count=n_support,
            contradicting_count=n_contra,
            neutral_count=n_neutral,
            avg_similarity=avg_sim,
            max_similarity=max_sim,
            has_sufficient_evidence=has_sufficient,
            has_contradictory_evidence=n_contra > 0,
            has_supporting_evidence=n_support > 0,
            coverage_assessment=coverage,
            quality_score=quality_score,
        )

    def identify_evidence_gaps(
        self,
        claim: str,
        quality: EvidenceQuality,
    ) -> List[str]:
        """
        Identifies gaps or limitations in the retrieved evidence.

        Returns:
            List of human-readable gap descriptions.
        """
        gaps = []

        if quality.total_passages == 0:
            gaps.append("No evidence found — the claim is outside the current knowledge base coverage.")
            return gaps

        if quality.coverage_assessment == "weak":
            gaps.append(
                "Retrieved passages have low semantic similarity — "
                "evidence may not be directly relevant to this specific claim."
            )

        if quality.total_passages < self.min_passages_for_reliable:
            gaps.append(
                f"Only {quality.total_passages} passage(s) retrieved — "
                f"minimum {self.min_passages_for_reliable} recommended for reliable assessment."
            )

        if not quality.has_contradictory_evidence and not quality.has_supporting_evidence:
            gaps.append(
                "No passages clearly support or contradict the claim — "
                "all evidence is contextually related but stance is unclear."
            )

        if quality.has_contradictory_evidence and quality.has_supporting_evidence:
            gaps.append(
                "Mixed evidence: both supporting and contradicting passages found. "
                "Requires careful weighing of source authority and recency."
            )

        return gaps
