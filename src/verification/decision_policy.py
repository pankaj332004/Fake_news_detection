"""
Decision Policy: Combines statistical classification, LLM verification,
evidence analysis, and uncertainty into a final structured veracity decision.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from src.verification.contradiction_detector import ContradictionAnalysis, EvidenceStatus
from src.verification.evidence_analyzer import EvidenceQuality
from src.verification.uncertainty import UncertaintyEstimate

logger = logging.getLogger(__name__)


class FinalVerdict(str, Enum):
    """Final veracity verdict combining all signals."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class VeracityDecision:
    """
    Structured final output combining all verification components.
    """

    def __init__(
        self,
        final_verdict: FinalVerdict,
        llm_verdict: Optional[str] = None,
        statistical_verdict: Optional[str] = None,
        reasoning: str = "",
        evidence_citations: Optional[List[str]] = None,
        contradictions_found: bool = False,
        limitations: str = "",
        evidence_quality: Optional[EvidenceQuality] = None,
        contradiction_analysis: Optional[ContradictionAnalysis] = None,
        uncertainty: Optional[UncertaintyEstimate] = None,
        llm_raw_output: Optional[str] = None,
        model_name: str = "",
    ):
        self.final_verdict = final_verdict
        self.llm_verdict = llm_verdict
        self.statistical_verdict = statistical_verdict
        self.reasoning = reasoning
        self.evidence_citations = evidence_citations or []
        self.contradictions_found = contradictions_found
        self.limitations = limitations
        self.evidence_quality = evidence_quality
        self.contradiction_analysis = contradiction_analysis
        self.uncertainty = uncertainty
        self.llm_raw_output = llm_raw_output
        self.model_name = model_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_verdict": self.final_verdict.value,
            "llm_verdict": self.llm_verdict,
            "statistical_verdict": self.statistical_verdict,
            "reasoning": self.reasoning,
            "evidence_citations": self.evidence_citations,
            "contradictions_found": self.contradictions_found,
            "limitations": self.limitations,
            "evidence_quality": self.evidence_quality.to_dict() if self.evidence_quality else None,
            "contradiction_analysis": (
                self.contradiction_analysis.to_dict() if self.contradiction_analysis else None
            ),
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "model_name": self.model_name,
        }


class DecisionPolicy:
    """
    Merges LLM output, evidence quality, contradiction analysis, and uncertainty
    into a final, actionable veracity verdict.

    Decision rules:
    1. If LLM verdict aligns with evidence AND uncertainty is low → adopt LLM verdict.
    2. If LLM verdict conflicts with evidence → flag for review.
    3. If evidence is absent → always UNVERIFIABLE regardless of LLM output.
    4. If uncertainty is very high → flag for review.
    """

    def __init__(
        self,
        flag_threshold: float = 0.35,
        override_absent_evidence: bool = True,
    ):
        """
        Args:
            flag_threshold: Below this overall confidence, flag for review.
            override_absent_evidence: If True, force UNVERIFIABLE when no evidence exists.
        """
        self.flag_threshold = flag_threshold
        self.override_absent_evidence = override_absent_evidence

    def decide(
        self,
        llm_verdict: Optional[str],
        llm_confidence: Optional[float],
        llm_reasoning: str,
        llm_citations: List[str],
        llm_contradictions_found: bool,
        llm_limitations: str,
        evidence_quality: EvidenceQuality,
        contradiction_analysis: ContradictionAnalysis,
        uncertainty: UncertaintyEstimate,
        statistical_verdict: Optional[str] = None,
        llm_raw_output: Optional[str] = None,
        model_name: str = "",
    ) -> VeracityDecision:
        """
        Produces the final veracity decision.

        Args:
            llm_verdict: LLM's verdict string (TRUE/FALSE/UNVERIFIABLE).
            llm_confidence: LLM's self-reported confidence.
            llm_reasoning: LLM's reasoning text.
            llm_citations: Evidence passage IDs cited by LLM.
            llm_contradictions_found: Whether LLM found contradictions.
            llm_limitations: LLM's stated limitations.
            evidence_quality: Pre-computed evidence quality.
            contradiction_analysis: Pre-computed contradiction analysis.
            uncertainty: Pre-computed uncertainty estimate.
            statistical_verdict: Optional baseline classifier verdict.
            llm_raw_output: Raw LLM response string.
            model_name: Name of the model used.

        Returns:
            VeracityDecision with final verdict and all supporting data.
        """
        # Rule 1: Override to UNVERIFIABLE if no evidence exists
        if (
            self.override_absent_evidence
            and contradiction_analysis.evidence_status == EvidenceStatus.ABSENT
        ):
            return VeracityDecision(
                final_verdict=FinalVerdict.UNVERIFIABLE,
                llm_verdict=llm_verdict,
                statistical_verdict=statistical_verdict,
                reasoning=(
                    "No evidence passages were retrieved for this claim. "
                    "The claim cannot be verified or refuted based on available evidence. "
                    f"LLM reasoning: {llm_reasoning}"
                ),
                evidence_citations=[],
                contradictions_found=False,
                limitations=(
                    "Assessment limited by absence of relevant evidence in the "
                    "knowledge base. This does NOT mean the claim is false."
                ),
                evidence_quality=evidence_quality,
                contradiction_analysis=contradiction_analysis,
                uncertainty=uncertainty,
                llm_raw_output=llm_raw_output,
                model_name=model_name,
            )

        # Rule 2: Flag for review if uncertainty is too high
        if uncertainty.should_flag_for_review:
            return VeracityDecision(
                final_verdict=FinalVerdict.NEEDS_REVIEW,
                llm_verdict=llm_verdict,
                statistical_verdict=statistical_verdict,
                reasoning=(
                    f"High uncertainty ({uncertainty.overall_confidence:.0%} confidence). "
                    f"LLM verdict '{llm_verdict}' has low reliability. "
                    f"LLM reasoning: {llm_reasoning}"
                ),
                evidence_citations=llm_citations,
                contradictions_found=llm_contradictions_found,
                limitations=f"{llm_limitations} | Flagged for human review due to high uncertainty.",
                evidence_quality=evidence_quality,
                contradiction_analysis=contradiction_analysis,
                uncertainty=uncertainty,
                llm_raw_output=llm_raw_output,
                model_name=model_name,
            )

        # Rule 3: Flag if LLM verdict misaligns with evidence
        if uncertainty.alignment_score < 0.3:
            return VeracityDecision(
                final_verdict=FinalVerdict.NEEDS_REVIEW,
                llm_verdict=llm_verdict,
                statistical_verdict=statistical_verdict,
                reasoning=(
                    f"LLM verdict '{llm_verdict}' conflicts with evidence distribution. "
                    f"Alignment score: {uncertainty.alignment_score:.0%}. "
                    f"LLM reasoning: {llm_reasoning}"
                ),
                evidence_citations=llm_citations,
                contradictions_found=llm_contradictions_found,
                limitations=f"{llm_limitations} | Verdict-evidence misalignment detected.",
                evidence_quality=evidence_quality,
                contradiction_analysis=contradiction_analysis,
                uncertainty=uncertainty,
                llm_raw_output=llm_raw_output,
                model_name=model_name,
            )

        # Rule 4: Adopt LLM verdict with confidence
        mapped_verdict = self._map_verdict(llm_verdict)

        return VeracityDecision(
            final_verdict=mapped_verdict,
            llm_verdict=llm_verdict,
            statistical_verdict=statistical_verdict,
            reasoning=llm_reasoning,
            evidence_citations=llm_citations,
            contradictions_found=llm_contradictions_found,
            limitations=llm_limitations,
            evidence_quality=evidence_quality,
            contradiction_analysis=contradiction_analysis,
            uncertainty=uncertainty,
            llm_raw_output=llm_raw_output,
            model_name=model_name,
        )

    def _map_verdict(self, llm_verdict: Optional[str]) -> FinalVerdict:
        """Maps LLM verdict string to FinalVerdict enum."""
        if llm_verdict is None:
            return FinalVerdict.NEEDS_REVIEW

        verdict_map = {
            "TRUE": FinalVerdict.TRUE,
            "FALSE": FinalVerdict.FALSE,
            "UNVERIFIABLE": FinalVerdict.UNVERIFIABLE,
        }
        return verdict_map.get(llm_verdict.upper(), FinalVerdict.NEEDS_REVIEW)
