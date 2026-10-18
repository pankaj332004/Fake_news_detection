"""
Contradiction Detector: Distinguishes between contradictory evidence
(evidence that directly disproves a claim) and missing evidence
(absence of relevant information).

This is critical for Phase 5: the system must NOT conflate
"no evidence found" with "evidence contradicts the claim."
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from src.schemas.evidence import RetrievalResult, StanceType
from src.verification.evidence_analyzer import EvidenceQuality

logger = logging.getLogger(__name__)


class EvidenceStatus(str, Enum):
    """Categorization of evidence availability relative to a claim."""

    CONTRADICTED = "contradicted"        # Evidence directly refutes the claim
    SUPPORTED = "supported"              # Evidence supports the claim
    MIXED = "mixed"                      # Both supporting and contradicting evidence
    INSUFFICIENT = "insufficient"        # Some evidence but not enough for a verdict
    ABSENT = "absent"                    # No evidence at all


class ContradictionAnalysis:
    """Result of contradiction vs. missing evidence analysis."""

    def __init__(
        self,
        evidence_status: EvidenceStatus,
        is_contradicted: bool = False,
        is_missing: bool = False,
        contradiction_strength: float = 0.0,
        support_strength: float = 0.0,
        key_contradictions: Optional[List[str]] = None,
        key_supports: Optional[List[str]] = None,
        explanation: str = "",
    ):
        self.evidence_status = evidence_status
        self.is_contradicted = is_contradicted
        self.is_missing = is_missing
        self.contradiction_strength = contradiction_strength
        self.support_strength = support_strength
        self.key_contradictions = key_contradictions or []
        self.key_supports = key_supports or []
        self.explanation = explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_status": self.evidence_status.value,
            "is_contradicted": self.is_contradicted,
            "is_missing": self.is_missing,
            "contradiction_strength": round(self.contradiction_strength, 3),
            "support_strength": round(self.support_strength, 3),
            "key_contradictions": self.key_contradictions,
            "key_supports": self.key_supports,
            "explanation": self.explanation,
        }


class ContradictionDetector:
    """
    Analyzes whether evidence CONTRADICTS a claim vs. is simply MISSING.

    Key distinction:
    - CONTRADICTED: "Evidence says X is false" → verdict should be FALSE
    - ABSENT/INSUFFICIENT: "No evidence found about X" → verdict should be UNVERIFIABLE
    """

    def __init__(
        self,
        min_contradiction_score: float = 0.35,
        min_support_score: float = 0.35,
    ):
        """
        Args:
            min_contradiction_score: Minimum similarity score for a contradicting
                passage to count as meaningful contradiction.
            min_support_score: Minimum similarity score for a supporting
                passage to count as meaningful support.
        """
        self.min_contradiction_score = min_contradiction_score
        self.min_support_score = min_support_score

    def analyze(
        self,
        quality: EvidenceQuality,
        all_passages: Optional[List[RetrievalResult]] = None,
        supporting_passages: Optional[List[RetrievalResult]] = None,
        contradicting_passages: Optional[List[RetrievalResult]] = None,
    ) -> ContradictionAnalysis:
        """
        Determines if evidence contradicts, supports, or is missing for a claim.

        Args:
            quality: Pre-computed evidence quality assessment.
            all_passages: All retrieved passages.
            supporting_passages: Pre-filtered supporting passages.
            contradicting_passages: Pre-filtered contradicting passages.

        Returns:
            ContradictionAnalysis with status and strength metrics.
        """
        all_passages = all_passages or []
        supporting_passages = supporting_passages or []
        contradicting_passages = contradicting_passages or []

        # Case 1: No evidence at all
        if quality.total_passages == 0:
            return ContradictionAnalysis(
                evidence_status=EvidenceStatus.ABSENT,
                is_contradicted=False,
                is_missing=True,
                explanation=(
                    "No evidence passages were retrieved. The claim cannot be "
                    "verified or refuted — it falls outside the current knowledge base."
                ),
            )

        # Compute contradiction strength (weighted by relevance scores)
        strong_contradictions = [
            r for r in contradicting_passages
            if r.score >= self.min_contradiction_score
        ]
        strong_supports = [
            r for r in supporting_passages
            if r.score >= self.min_support_score
        ]

        contra_scores = [r.score for r in strong_contradictions]
        support_scores = [r.score for r in strong_supports]

        contradiction_strength = (
            sum(contra_scores) / len(contra_scores) if contra_scores else 0.0
        )
        support_strength = (
            sum(support_scores) / len(support_scores) if support_scores else 0.0
        )

        # Extract key text snippets
        key_contradictions = [
            r.passage.text[:200] for r in strong_contradictions[:3]
        ]
        key_supports = [
            r.passage.text[:200] for r in strong_supports[:3]
        ]

        # Case 2: Clear contradiction
        if len(strong_contradictions) > 0 and len(strong_supports) == 0:
            return ContradictionAnalysis(
                evidence_status=EvidenceStatus.CONTRADICTED,
                is_contradicted=True,
                is_missing=False,
                contradiction_strength=contradiction_strength,
                support_strength=0.0,
                key_contradictions=key_contradictions,
                explanation=(
                    f"{len(strong_contradictions)} passage(s) directly contradict "
                    f"the claim with average relevance score of "
                    f"{contradiction_strength:.2f}. No supporting evidence found."
                ),
            )

        # Case 3: Clear support
        if len(strong_supports) > 0 and len(strong_contradictions) == 0:
            return ContradictionAnalysis(
                evidence_status=EvidenceStatus.SUPPORTED,
                is_contradicted=False,
                is_missing=False,
                contradiction_strength=0.0,
                support_strength=support_strength,
                key_supports=key_supports,
                explanation=(
                    f"{len(strong_supports)} passage(s) support the claim with "
                    f"average relevance score of {support_strength:.2f}. "
                    f"No contradicting evidence found."
                ),
            )

        # Case 4: Mixed evidence
        if len(strong_contradictions) > 0 and len(strong_supports) > 0:
            return ContradictionAnalysis(
                evidence_status=EvidenceStatus.MIXED,
                is_contradicted=True,
                is_missing=False,
                contradiction_strength=contradiction_strength,
                support_strength=support_strength,
                key_contradictions=key_contradictions,
                key_supports=key_supports,
                explanation=(
                    f"Mixed evidence: {len(strong_supports)} supporting and "
                    f"{len(strong_contradictions)} contradicting passages found. "
                    f"Requires careful weighing."
                ),
            )

        # Case 5: Evidence retrieved but none clearly relevant enough
        if quality.coverage_assessment in ("weak", "none"):
            return ContradictionAnalysis(
                evidence_status=EvidenceStatus.INSUFFICIENT,
                is_contradicted=False,
                is_missing=True,
                explanation=(
                    f"{quality.total_passages} passage(s) retrieved but none "
                    f"with sufficient relevance scores (>{self.min_contradiction_score}). "
                    f"Evidence is insufficient to reach a verdict."
                ),
            )

        # Case 6: Neutral evidence only
        return ContradictionAnalysis(
            evidence_status=EvidenceStatus.INSUFFICIENT,
            is_contradicted=False,
            is_missing=False,
            contradiction_strength=0.0,
            support_strength=0.0,
            explanation=(
                f"{quality.total_passages} passage(s) retrieved with neutral stance. "
                f"Evidence provides context but does not clearly support or contradict."
            ),
        )
