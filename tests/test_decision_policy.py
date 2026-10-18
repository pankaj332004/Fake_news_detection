"""
Tests for the Decision Policy module.
Validates that the policy correctly merges LLM output, evidence analysis,
contradiction detection, and uncertainty into final verdicts.
"""

import sys
from pathlib import Path

import pytest

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.verification.contradiction_detector import ContradictionAnalysis, EvidenceStatus
from src.verification.decision_policy import DecisionPolicy, FinalVerdict, VeracityDecision
from src.verification.evidence_analyzer import EvidenceQuality
from src.verification.uncertainty import UncertaintyEstimate


class TestDecisionPolicyRules:
    """Tests for each decision rule in the policy."""

    def test_rule1_absent_evidence_forces_unverifiable(self):
        """When evidence is absent, verdict MUST be UNVERIFIABLE regardless of LLM output."""
        policy = DecisionPolicy()
        quality = EvidenceQuality(total_passages=0, coverage_assessment="none")
        contra = ContradictionAnalysis(
            evidence_status=EvidenceStatus.ABSENT, is_missing=True
        )
        unc = UncertaintyEstimate(overall_confidence=0.5, alignment_score=0.5)

        decision = policy.decide(
            llm_verdict="TRUE",  # LLM says TRUE but no evidence
            llm_confidence=0.9,
            llm_reasoning="I think this is true.",
            llm_citations=[],
            llm_contradictions_found=False,
            llm_limitations="None",
            evidence_quality=quality,
            contradiction_analysis=contra,
            uncertainty=unc,
        )
        assert decision.final_verdict == FinalVerdict.UNVERIFIABLE

    def test_rule2_high_uncertainty_flags_review(self):
        """When uncertainty is very high, verdict should be NEEDS_REVIEW."""
        policy = DecisionPolicy(flag_threshold=0.35)
        quality = EvidenceQuality(total_passages=1, coverage_assessment="weak")
        contra = ContradictionAnalysis(
            evidence_status=EvidenceStatus.INSUFFICIENT, is_missing=False
        )
        unc = UncertaintyEstimate(
            overall_confidence=0.2,  # Below flag_threshold
            alignment_score=0.5,
            should_flag_for_review=True,
        )

        decision = policy.decide(
            llm_verdict="FALSE",
            llm_confidence=0.3,
            llm_reasoning="Weak evidence.",
            llm_citations=[],
            llm_contradictions_found=False,
            llm_limitations="Very uncertain.",
            evidence_quality=quality,
            contradiction_analysis=contra,
            uncertainty=unc,
        )
        assert decision.final_verdict == FinalVerdict.NEEDS_REVIEW

    def test_rule3_misaligned_verdict_flags_review(self):
        """When LLM verdict conflicts with evidence, flag for review."""
        policy = DecisionPolicy()
        quality = EvidenceQuality(
            total_passages=3,
            supporting_count=3,
            has_supporting_evidence=True,
            coverage_assessment="strong",
        )
        contra = ContradictionAnalysis(
            evidence_status=EvidenceStatus.SUPPORTED, is_missing=False
        )
        unc = UncertaintyEstimate(
            overall_confidence=0.4,
            alignment_score=0.1,  # Very low alignment → misaligned
            should_flag_for_review=False,
        )

        decision = policy.decide(
            llm_verdict="FALSE",  # Says FALSE but all evidence supports!
            llm_confidence=0.7,
            llm_reasoning="I think it's false.",
            llm_citations=[],
            llm_contradictions_found=False,
            llm_limitations="None",
            evidence_quality=quality,
            contradiction_analysis=contra,
            uncertainty=unc,
        )
        assert decision.final_verdict == FinalVerdict.NEEDS_REVIEW

    def test_rule4_aligned_verdict_adopted(self):
        """When LLM verdict aligns with evidence and uncertainty is low, adopt it."""
        policy = DecisionPolicy()
        quality = EvidenceQuality(
            total_passages=3,
            supporting_count=3,
            has_supporting_evidence=True,
            coverage_assessment="strong",
        )
        contra = ContradictionAnalysis(
            evidence_status=EvidenceStatus.SUPPORTED, is_missing=False
        )
        unc = UncertaintyEstimate(
            overall_confidence=0.8,
            alignment_score=0.9,
            should_flag_for_review=False,
        )

        decision = policy.decide(
            llm_verdict="TRUE",
            llm_confidence=0.85,
            llm_reasoning="Evidence confirms claim.",
            llm_citations=["DOC-1#p0"],
            llm_contradictions_found=False,
            llm_limitations="Limited KB.",
            evidence_quality=quality,
            contradiction_analysis=contra,
            uncertainty=unc,
        )
        assert decision.final_verdict == FinalVerdict.TRUE

    def test_false_verdict_with_contradictions(self):
        """Contradicting evidence + LLM says FALSE → adopt FALSE."""
        policy = DecisionPolicy()
        quality = EvidenceQuality(
            total_passages=2,
            contradicting_count=2,
            has_contradictory_evidence=True,
            coverage_assessment="strong",
        )
        contra = ContradictionAnalysis(
            evidence_status=EvidenceStatus.CONTRADICTED,
            is_contradicted=True,
            is_missing=False,
        )
        unc = UncertaintyEstimate(
            overall_confidence=0.8,
            alignment_score=0.9,
            should_flag_for_review=False,
        )

        decision = policy.decide(
            llm_verdict="FALSE",
            llm_confidence=0.9,
            llm_reasoning="Evidence directly refutes this claim.",
            llm_citations=["DOC-WHO-01#p0"],
            llm_contradictions_found=True,
            llm_limitations="KB coverage.",
            evidence_quality=quality,
            contradiction_analysis=contra,
            uncertainty=unc,
        )
        assert decision.final_verdict == FinalVerdict.FALSE
        assert decision.contradictions_found is True

    def test_decision_to_dict(self):
        """VeracityDecision should serialize to a complete dict."""
        quality = EvidenceQuality(total_passages=1)
        contra = ContradictionAnalysis(evidence_status=EvidenceStatus.SUPPORTED)
        unc = UncertaintyEstimate(overall_confidence=0.7)

        decision = VeracityDecision(
            final_verdict=FinalVerdict.TRUE,
            llm_verdict="TRUE",
            reasoning="Test",
            evidence_quality=quality,
            contradiction_analysis=contra,
            uncertainty=unc,
            model_name="test-model",
        )
        d = decision.to_dict()
        assert d["final_verdict"] == "TRUE"
        assert "evidence_quality" in d
        assert "contradiction_analysis" in d
        assert "uncertainty" in d
