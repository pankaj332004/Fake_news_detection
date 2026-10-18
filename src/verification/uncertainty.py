"""
Uncertainty Estimator: Quantifies confidence and uncertainty
in the LLM verification output based on evidence quality,
model self-reported confidence, and evidence-verdict alignment.
"""

import logging
from typing import Any, Dict, Optional

from src.verification.contradiction_detector import ContradictionAnalysis, EvidenceStatus
from src.verification.evidence_analyzer import EvidenceQuality

logger = logging.getLogger(__name__)


class UncertaintyEstimate:
    """Structured uncertainty assessment for a verification result."""

    def __init__(
        self,
        overall_confidence: float = 0.0,
        evidence_confidence: float = 0.0,
        model_confidence: float = 0.0,
        alignment_score: float = 0.0,
        uncertainty_level: str = "high",
        explanation: str = "",
        should_flag_for_review: bool = False,
    ):
        self.overall_confidence = overall_confidence
        self.evidence_confidence = evidence_confidence
        self.model_confidence = model_confidence
        self.alignment_score = alignment_score
        self.uncertainty_level = uncertainty_level
        self.explanation = explanation
        self.should_flag_for_review = should_flag_for_review

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_confidence": round(self.overall_confidence, 3),
            "evidence_confidence": round(self.evidence_confidence, 3),
            "model_confidence": round(self.model_confidence, 3),
            "alignment_score": round(self.alignment_score, 3),
            "uncertainty_level": self.uncertainty_level,
            "explanation": self.explanation,
            "should_flag_for_review": self.should_flag_for_review,
        }


class UncertaintyEstimator:
    """
    Computes multi-factor uncertainty estimates for LLM verification results.

    Factors:
    1. Evidence Quality: How much relevant, high-quality evidence was retrieved?
    2. Model Confidence: What confidence did the LLM self-report?
    3. Evidence-Verdict Alignment: Does the LLM's verdict match the evidence stance distribution?
    """

    # Weight factors for combining confidence signals
    EVIDENCE_WEIGHT = 0.45
    MODEL_WEIGHT = 0.25
    ALIGNMENT_WEIGHT = 0.30

    # Confidence thresholds for uncertainty levels
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4
    FLAG_FOR_REVIEW_THRESHOLD = 0.35

    def estimate(
        self,
        evidence_quality: EvidenceQuality,
        contradiction_analysis: ContradictionAnalysis,
        llm_verdict: Optional[str] = None,
        llm_confidence: Optional[float] = None,
    ) -> UncertaintyEstimate:
        """
        Computes an uncertainty estimate combining evidence quality, model confidence,
        and verdict-evidence alignment.

        Args:
            evidence_quality: Pre-computed evidence quality assessment.
            contradiction_analysis: Pre-computed contradiction/missing evidence analysis.
            llm_verdict: The LLM's output verdict (TRUE/FALSE/UNVERIFIABLE).
            llm_confidence: The LLM's self-reported confidence (0.0-1.0).

        Returns:
            UncertaintyEstimate with calibrated confidence and review flag.
        """
        # Factor 1: Evidence confidence
        evidence_conf = self._compute_evidence_confidence(evidence_quality)

        # Factor 2: Model confidence (with skepticism penalty for extreme values)
        model_conf = self._calibrate_model_confidence(llm_confidence)

        # Factor 3: Evidence-verdict alignment
        alignment = self._compute_alignment(
            evidence_quality, contradiction_analysis, llm_verdict
        )

        # Weighted combination
        overall = (
            evidence_conf * self.EVIDENCE_WEIGHT
            + model_conf * self.MODEL_WEIGHT
            + alignment * self.ALIGNMENT_WEIGHT
        )
        overall = max(0.0, min(1.0, overall))  # Clamp to [0, 1]

        # Determine uncertainty level
        if overall >= self.HIGH_CONFIDENCE_THRESHOLD:
            level = "low"
        elif overall >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            level = "medium"
        else:
            level = "high"

        # Should flag for human review?
        should_flag = overall < self.FLAG_FOR_REVIEW_THRESHOLD

        # Build explanation
        explanation = self._build_explanation(
            evidence_conf, model_conf, alignment, overall, level,
            evidence_quality, contradiction_analysis, llm_verdict,
        )

        return UncertaintyEstimate(
            overall_confidence=overall,
            evidence_confidence=evidence_conf,
            model_confidence=model_conf,
            alignment_score=alignment,
            uncertainty_level=level,
            explanation=explanation,
            should_flag_for_review=should_flag,
        )

    def _compute_evidence_confidence(self, quality: EvidenceQuality) -> float:
        """Evidence confidence based on coverage, relevance, and quantity."""
        if quality.total_passages == 0:
            return 0.1  # Near-zero but not absolute zero

        # Combine quality factors
        coverage_map = {"strong": 0.9, "moderate": 0.6, "weak": 0.3, "none": 0.1}
        coverage_score = coverage_map.get(quality.coverage_assessment, 0.1)

        # Boost if stance is clear (all supporting OR all contradicting)
        stance_clarity = 0.0
        if quality.has_supporting_evidence and not quality.has_contradictory_evidence:
            stance_clarity = 0.8
        elif quality.has_contradictory_evidence and not quality.has_supporting_evidence:
            stance_clarity = 0.8
        elif quality.has_contradictory_evidence and quality.has_supporting_evidence:
            stance_clarity = 0.4  # Mixed is less clear
        else:
            stance_clarity = 0.2  # All neutral

        return coverage_score * 0.6 + stance_clarity * 0.4

    def _calibrate_model_confidence(self, llm_confidence: Optional[float]) -> float:
        """
        Calibrates model confidence with skepticism.
        Very high self-reported confidence is penalized slightly (overconfidence bias).
        """
        if llm_confidence is None:
            return 0.5  # Default neutral confidence

        # Clamp to valid range
        conf = max(0.0, min(1.0, llm_confidence))

        # Apply mild overconfidence penalty for extreme values
        if conf > 0.95:
            conf = 0.90  # Cap overconfident models
        elif conf < 0.15:
            conf = 0.15  # Don't let models be too unconfident

        return conf

    def _compute_alignment(
        self,
        quality: EvidenceQuality,
        contradiction: ContradictionAnalysis,
        verdict: Optional[str],
    ) -> float:
        """
        Measures alignment between LLM verdict and evidence stance distribution.
        High alignment = LLM agrees with evidence; Low alignment = LLM contradicts evidence.
        """
        if verdict is None:
            return 0.5  # Can't compute alignment without verdict

        verdict_upper = verdict.upper()

        # Check alignment scenarios
        if contradiction.evidence_status == EvidenceStatus.ABSENT:
            # No evidence → only UNVERIFIABLE is aligned
            if verdict_upper == "UNVERIFIABLE":
                return 0.9  # Correct: missing evidence → unverifiable
            else:
                return 0.2  # Misaligned: making a verdict without evidence

        elif contradiction.evidence_status == EvidenceStatus.CONTRADICTED:
            # Strong contradiction → FALSE is aligned
            if verdict_upper == "FALSE":
                return 0.9
            elif verdict_upper == "UNVERIFIABLE":
                return 0.4  # Cautious but misses the contradiction
            else:
                return 0.1  # TRUE despite contradicting evidence

        elif contradiction.evidence_status == EvidenceStatus.SUPPORTED:
            # Strong support → TRUE is aligned
            if verdict_upper == "TRUE":
                return 0.9
            elif verdict_upper == "UNVERIFIABLE":
                return 0.4
            else:
                return 0.1  # FALSE despite supporting evidence

        elif contradiction.evidence_status == EvidenceStatus.MIXED:
            # Mixed evidence → any verdict is partially valid
            if verdict_upper == "UNVERIFIABLE":
                return 0.6  # Cautious approach is reasonable
            else:
                return 0.5  # Either TRUE or FALSE could be argued

        elif contradiction.evidence_status == EvidenceStatus.INSUFFICIENT:
            # Insufficient evidence → UNVERIFIABLE is most aligned
            if verdict_upper == "UNVERIFIABLE":
                return 0.8
            else:
                return 0.3

        return 0.5  # Default neutral alignment

    def _build_explanation(
        self,
        evidence_conf: float,
        model_conf: float,
        alignment: float,
        overall: float,
        level: str,
        quality: EvidenceQuality,
        contradiction: ContradictionAnalysis,
        verdict: Optional[str],
    ) -> str:
        """Builds a human-readable explanation of the uncertainty assessment."""
        parts = [f"Overall confidence: {overall:.0%} ({level} uncertainty)."]

        if quality.total_passages == 0:
            parts.append(
                "No evidence was retrieved, severely limiting verification reliability."
            )
        elif quality.coverage_assessment == "weak":
            parts.append(
                "Retrieved evidence has weak relevance, reducing confidence."
            )
        elif quality.coverage_assessment == "strong":
            parts.append(
                "Strong evidence coverage supports the reliability of this assessment."
            )

        if alignment < 0.4 and verdict:
            parts.append(
                f"Warning: The '{verdict}' verdict has low alignment with the "
                f"evidence distribution ({alignment:.0%}), suggesting potential "
                f"model hallucination."
            )

        return " ".join(parts)
