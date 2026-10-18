"""
Tests for Phase 5 — LLM Verification Module.
Tests the full verification pipeline including:
- Prompt template construction
- LLM loader (mock mode)
- Evidence analysis
- Contradiction vs. missing evidence detection
- Uncertainty estimation
- Decision policy
- End-to-end verification flow
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.schemas.evidence import (
    EvidencePassage,
    RetrievalResponse,
    RetrievalResult,
    StanceType,
)
from src.verification.claim_extraction import ClaimExtractor
from src.verification.contradiction_detector import (
    ContradictionDetector,
    EvidenceStatus,
)
from src.verification.decision_policy import DecisionPolicy, FinalVerdict
from src.verification.evidence_analyzer import EvidenceAnalyzer
from src.verification.llm_loader import LLMBackend, LLMLoader
from src.verification.llm_verifier import (
    LLMVerifier,
    _extract_json_from_text,
    _parse_llm_response,
)
from src.verification.prompt_templates import VerificationPrompt
from src.verification.uncertainty import UncertaintyEstimator


# ============================================================
# Fixtures: reusable evidence passages for tests
# ============================================================

@pytest.fixture
def supporting_passage():
    return RetrievalResult(
        passage=EvidencePassage(
            passage_id="DOC-NASA-01#p0",
            doc_id="DOC-NASA-01",
            text="NASA's Apollo missions successfully landed humans on the Moon six times between 1969 and 1972.",
            source="NASA",
            source_url="https://nasa.gov/apollo",
            category="space",
        ),
        score=0.72,
        rank=1,
        stance=StanceType.SUPPORTS,
        explanation="Confirms lunar landings.",
    )


@pytest.fixture
def contradicting_passage():
    return RetrievalResult(
        passage=EvidencePassage(
            passage_id="DOC-WHO-01#p0",
            doc_id="DOC-WHO-01",
            text="Large-scale studies have found no link between vaccines and autism. The original study claiming this link was retracted due to fraud.",
            source="WHO",
            source_url="https://who.int/vaccines",
            category="health",
        ),
        score=0.68,
        rank=2,
        stance=StanceType.CONTRADICTS,
        explanation="Refutes vaccine-autism link.",
    )


@pytest.fixture
def neutral_passage():
    return RetrievalResult(
        passage=EvidencePassage(
            passage_id="DOC-WIKI-01#p0",
            doc_id="DOC-WIKI-01",
            text="The Moon is Earth's only natural satellite and orbits at a mean distance of approximately 384,400 km.",
            source="Encyclopedia",
            source_url="https://example.com",
            category="space",
        ),
        score=0.40,
        rank=3,
        stance=StanceType.NEUTRAL,
        explanation="Related context.",
    )


@pytest.fixture
def mock_retrieval_response(supporting_passage, contradicting_passage, neutral_passage):
    return RetrievalResponse(
        query="Vaccines cause autism",
        total_retrieved=3,
        supporting_passages=[supporting_passage],
        contradicting_passages=[contradicting_passage],
        all_passages=[supporting_passage, contradicting_passage, neutral_passage],
        model_name="all-MiniLM-L6-v2",
        retrieval_time_ms=12.5,
    )


@pytest.fixture
def empty_retrieval_response():
    return RetrievalResponse(
        query="Aliens built the pyramids",
        total_retrieved=0,
        supporting_passages=[],
        contradicting_passages=[],
        all_passages=[],
        model_name="all-MiniLM-L6-v2",
        retrieval_time_ms=3.0,
    )


# ============================================================
# Test: Prompt Templates
# ============================================================

class TestPromptTemplates:
    def test_build_evidence_block(self, supporting_passage, contradicting_passage):
        block = VerificationPrompt.build_evidence_block(
            [supporting_passage, contradicting_passage]
        )
        assert "DOC-NASA-01#p0" in block
        assert "DOC-WHO-01#p0" in block
        assert "SUPPORTS" in block
        assert "CONTRADICTS" in block

    def test_build_evidence_block_empty(self):
        block = VerificationPrompt.build_evidence_block([])
        assert "No evidence" in block

    def test_build_verification_prompt(self, supporting_passage, contradicting_passage):
        prompt = VerificationPrompt.build_verification_prompt(
            claim="Vaccines cause autism",
            evidence_passages=[supporting_passage, contradicting_passage],
            supporting_passages=[supporting_passage],
            contradicting_passages=[contradicting_passage],
            statistical_verdict="Fake",
            statistical_confidence=0.78,
        )
        assert "Vaccines cause autism" in prompt
        assert "CLAIM TO VERIFY" in prompt
        assert "RETRIEVED EVIDENCE" in prompt
        assert "JSON" in prompt
        assert "Fake" in prompt  # statistical verdict reference

    def test_build_no_evidence_prompt(self):
        prompt = VerificationPrompt.build_no_evidence_prompt("The earth is flat")
        assert "UNVERIFIABLE" in prompt
        assert "No relevant evidence" in prompt
        assert "The earth is flat" in prompt


# ============================================================
# Test: LLM Loader (Mock mode)
# ============================================================

class TestLLMLoader:
    def test_mock_load(self):
        loader = LLMLoader(backend=LLMBackend.MOCK)
        info = loader.load()
        assert info.loaded is True
        assert info.backend == LLMBackend.MOCK

    def test_mock_generate_with_evidence(self):
        loader = LLMLoader(backend=LLMBackend.MOCK)
        loader.load()
        output = loader.generate("Claim with evidence. Contradicting passages found.")
        parsed = json.loads(output)
        assert parsed["verdict"] == "FALSE"
        assert parsed["contradictions_found"] is True

    def test_mock_generate_no_evidence(self):
        loader = LLMLoader(backend=LLMBackend.MOCK)
        loader.load()
        output = loader.generate("No relevant evidence passages were found.")
        parsed = json.loads(output)
        assert parsed["verdict"] == "UNVERIFIABLE"
        assert parsed["contradictions_found"] is False

    def test_mock_generate_supporting(self):
        loader = LLMLoader(backend=LLMBackend.MOCK)
        loader.load()
        output = loader.generate("Claim with evidence passages. Supports the claim.")
        parsed = json.loads(output)
        assert parsed["verdict"] == "TRUE"

    def test_generate_without_load_raises(self):
        loader = LLMLoader(backend=LLMBackend.LOCAL)
        with pytest.raises(RuntimeError, match="not loaded"):
            loader.generate("test")


# ============================================================
# Test: Evidence Analyzer
# ============================================================

class TestEvidenceAnalyzer:
    def test_analyze_with_evidence(self, mock_retrieval_response):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(retrieval_response=mock_retrieval_response)
        assert quality.total_passages == 3
        assert quality.supporting_count == 1
        assert quality.contradicting_count == 1
        assert quality.neutral_count == 1
        assert quality.has_contradictory_evidence is True
        assert quality.has_supporting_evidence is True
        assert quality.avg_similarity > 0

    def test_analyze_empty(self, empty_retrieval_response):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(retrieval_response=empty_retrieval_response)
        assert quality.total_passages == 0
        assert quality.coverage_assessment == "none"
        assert quality.has_sufficient_evidence is False

    def test_evidence_gaps(self, mock_retrieval_response):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(retrieval_response=mock_retrieval_response)
        gaps = analyzer.identify_evidence_gaps("test claim", quality)
        # Mixed evidence should produce a gap note
        assert any("Mixed" in g or "supporting" in g.lower() for g in gaps)

    def test_quality_score_range(self, mock_retrieval_response):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(retrieval_response=mock_retrieval_response)
        assert 0.0 <= quality.quality_score <= 1.0


# ============================================================
# Test: Contradiction Detector
# ============================================================

class TestContradictionDetector:
    def test_absent_evidence(self):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(all_passages=[])
        detector = ContradictionDetector()
        result = detector.analyze(quality=quality)
        assert result.evidence_status == EvidenceStatus.ABSENT
        assert result.is_missing is True
        assert result.is_contradicted is False

    def test_contradicted_evidence(self, contradicting_passage):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(
            all_passages=[contradicting_passage],
            contradicting_passages=[contradicting_passage],
        )
        detector = ContradictionDetector()
        result = detector.analyze(
            quality=quality,
            all_passages=[contradicting_passage],
            contradicting_passages=[contradicting_passage],
        )
        assert result.evidence_status == EvidenceStatus.CONTRADICTED
        assert result.is_contradicted is True
        assert result.is_missing is False

    def test_supported_evidence(self, supporting_passage):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(
            all_passages=[supporting_passage],
            supporting_passages=[supporting_passage],
        )
        detector = ContradictionDetector()
        result = detector.analyze(
            quality=quality,
            all_passages=[supporting_passage],
            supporting_passages=[supporting_passage],
        )
        assert result.evidence_status == EvidenceStatus.SUPPORTED
        assert result.is_contradicted is False

    def test_mixed_evidence(self, supporting_passage, contradicting_passage):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(
            all_passages=[supporting_passage, contradicting_passage],
            supporting_passages=[supporting_passage],
            contradicting_passages=[contradicting_passage],
        )
        detector = ContradictionDetector()
        result = detector.analyze(
            quality=quality,
            all_passages=[supporting_passage, contradicting_passage],
            supporting_passages=[supporting_passage],
            contradicting_passages=[contradicting_passage],
        )
        assert result.evidence_status == EvidenceStatus.MIXED


# ============================================================
# Test: Uncertainty Estimator
# ============================================================

class TestUncertaintyEstimator:
    def test_high_uncertainty_no_evidence(self):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(all_passages=[])
        detector = ContradictionDetector()
        contra = detector.analyze(quality=quality)

        estimator = UncertaintyEstimator()
        unc = estimator.estimate(quality, contra, llm_verdict="UNVERIFIABLE", llm_confidence=0.3)
        assert unc.uncertainty_level in ("high", "medium")
        assert unc.overall_confidence < 0.5

    def test_low_uncertainty_strong_evidence(self, supporting_passage):
        analyzer = EvidenceAnalyzer()
        all_p = [supporting_passage] * 3  # Simulate strong coverage
        quality = analyzer.analyze(
            all_passages=all_p,
            supporting_passages=all_p,
        )
        detector = ContradictionDetector()
        contra = detector.analyze(quality=quality, all_passages=all_p, supporting_passages=all_p)

        estimator = UncertaintyEstimator()
        unc = estimator.estimate(quality, contra, llm_verdict="TRUE", llm_confidence=0.9)
        assert unc.overall_confidence > 0.5

    def test_misaligned_verdict_flagged(self, contradicting_passage):
        """LLM says TRUE but evidence contradicts → should flag for review."""
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(
            all_passages=[contradicting_passage],
            contradicting_passages=[contradicting_passage],
        )
        detector = ContradictionDetector()
        contra = detector.analyze(
            quality=quality,
            all_passages=[contradicting_passage],
            contradicting_passages=[contradicting_passage],
        )
        estimator = UncertaintyEstimator()
        unc = estimator.estimate(quality, contra, llm_verdict="TRUE", llm_confidence=0.9)
        # Alignment should be low because TRUE verdict conflicts with contradicting evidence
        assert unc.alignment_score < 0.3


# ============================================================
# Test: Decision Policy
# ============================================================

class TestDecisionPolicy:
    def test_absent_evidence_overrides_to_unverifiable(self):
        analyzer = EvidenceAnalyzer()
        quality = analyzer.analyze(all_passages=[])
        detector = ContradictionDetector()
        contra = detector.analyze(quality=quality)
        estimator = UncertaintyEstimator()
        unc = estimator.estimate(quality, contra, llm_verdict="TRUE", llm_confidence=0.8)

        policy = DecisionPolicy()
        decision = policy.decide(
            llm_verdict="TRUE",
            llm_confidence=0.8,
            llm_reasoning="Model says true.",
            llm_citations=[],
            llm_contradictions_found=False,
            llm_limitations="None",
            evidence_quality=quality,
            contradiction_analysis=contra,
            uncertainty=unc,
        )
        # Must override to UNVERIFIABLE since no evidence exists
        assert decision.final_verdict == FinalVerdict.UNVERIFIABLE

    def test_high_confidence_adopts_llm(self, supporting_passage):
        analyzer = EvidenceAnalyzer()
        all_p = [supporting_passage] * 3
        quality = analyzer.analyze(all_passages=all_p, supporting_passages=all_p)
        detector = ContradictionDetector()
        contra = detector.analyze(quality=quality, all_passages=all_p, supporting_passages=all_p)
        estimator = UncertaintyEstimator()
        unc = estimator.estimate(quality, contra, llm_verdict="TRUE", llm_confidence=0.85)

        policy = DecisionPolicy()
        decision = policy.decide(
            llm_verdict="TRUE",
            llm_confidence=0.85,
            llm_reasoning="Evidence confirms the claim.",
            llm_citations=["DOC-NASA-01#p0"],
            llm_contradictions_found=False,
            llm_limitations="Limited to current knowledge base.",
            evidence_quality=quality,
            contradiction_analysis=contra,
            uncertainty=unc,
        )
        assert decision.final_verdict == FinalVerdict.TRUE


# ============================================================
# Test: JSON Parsing
# ============================================================

class TestJSONParsing:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "verdict": "FALSE",
            "confidence": 0.85,
            "reasoning": "Evidence contradicts.",
            "evidence_citations": ["DOC-1#p0"],
            "contradictions_found": True,
            "limitations": "Limited data.",
        })
        result = _parse_llm_response(raw)
        assert result["verdict"] == "FALSE"
        assert result["confidence"] == 0.85
        assert result["contradictions_found"] is True

    def test_parse_json_in_code_fence(self):
        raw = '```json\n{"verdict": "TRUE", "confidence": 0.7, "reasoning": "OK"}\n```'
        result = _parse_llm_response(raw)
        assert result["verdict"] == "TRUE"

    def test_parse_json_with_extra_text(self):
        raw = 'Here is my analysis:\n{"verdict": "UNVERIFIABLE", "confidence": 0.3, "reasoning": "No data"}\nEnd.'
        result = _parse_llm_response(raw)
        assert result["verdict"] == "UNVERIFIABLE"

    def test_heuristic_fallback(self):
        raw = "The claim is false because evidence contradicts it."
        result = _parse_llm_response(raw)
        assert result["verdict"] == "FALSE"
        assert result["contradictions_found"] is True
        assert result["confidence"] < 0.5  # Low confidence for heuristic

    def test_heuristic_true(self):
        raw = "The claim is true based on the available evidence."
        result = _parse_llm_response(raw)
        assert result["verdict"] == "TRUE"


# ============================================================
# Test: Claim Extraction
# ============================================================

class TestClaimExtraction:
    def test_short_claim(self):
        extractor = ClaimExtractor()
        claims = extractor.extract_claims("The earth is flat.")
        assert len(claims) == 1
        assert "flat" in claims[0]

    def test_long_article(self):
        extractor = ClaimExtractor()
        article = (
            "Scientists have discovered a new species in the Amazon. "
            "The species was found in 2023 during an expedition. "
            "Researchers believe it has been undiscovered for millions of years. "
            "What caused its evolution remains unknown?"
        )
        claims = extractor.extract_claims(article)
        assert len(claims) >= 2
        # Questions should be filtered or deprioritized
        assert not any("What caused" in c for c in claims)

    def test_primary_claim(self):
        extractor = ClaimExtractor()
        text = "NASA confirmed water on Mars. The sky is blue."
        primary = extractor.get_primary_claim(text)
        assert "confirmed" in primary or "NASA" in primary


# ============================================================
# Test: End-to-End LLM Verifier (Mock mode)
# ============================================================

class TestLLMVerifier:
    def test_verify_with_contradicting_evidence(self, mock_retrieval_response):
        verifier = LLMVerifier(backend=LLMBackend.MOCK)
        decision = verifier.verify(
            claim="Vaccines cause autism",
            retrieval_response=mock_retrieval_response,
            statistical_verdict="Fake",
            statistical_confidence=0.75,
        )
        assert decision.final_verdict in (FinalVerdict.FALSE, FinalVerdict.NEEDS_REVIEW)
        assert decision.llm_verdict is not None
        assert decision.reasoning != ""

    def test_verify_no_evidence(self, empty_retrieval_response):
        verifier = LLMVerifier(backend=LLMBackend.MOCK)
        decision = verifier.verify(
            claim="Aliens built the pyramids",
            retrieval_response=empty_retrieval_response,
        )
        # Must be UNVERIFIABLE when no evidence exists
        assert decision.final_verdict == FinalVerdict.UNVERIFIABLE
        assert decision.contradiction_analysis is not None
        assert decision.contradiction_analysis.is_missing is True

    def test_verify_plain_claim(self):
        verifier = LLMVerifier(backend=LLMBackend.MOCK)
        decision = verifier.verify(
            claim="NASA landed humans on the Moon.",
        )
        # No retrieval response → no evidence → UNVERIFIABLE
        assert decision.final_verdict == FinalVerdict.UNVERIFIABLE

    def test_verify_returns_dict(self, mock_retrieval_response):
        verifier = LLMVerifier(backend=LLMBackend.MOCK)
        decision = verifier.verify(
            claim="Test claim",
            retrieval_response=mock_retrieval_response,
        )
        result_dict = decision.to_dict()
        assert "final_verdict" in result_dict
        assert "reasoning" in result_dict
        assert "evidence_citations" in result_dict
        assert "uncertainty" in result_dict
        assert "contradiction_analysis" in result_dict

    def test_distinguishes_contradiction_from_missing(
        self, contradicting_passage, empty_retrieval_response
    ):
        """KEY TEST: System must distinguish contradictory evidence from missing evidence."""
        verifier = LLMVerifier(backend=LLMBackend.MOCK)

        # Test 1: Contradicting evidence present
        contra_response = RetrievalResponse(
            query="Vaccines cause autism",
            total_retrieved=1,
            supporting_passages=[],
            contradicting_passages=[contradicting_passage],
            all_passages=[contradicting_passage],
            model_name="all-MiniLM-L6-v2",
            retrieval_time_ms=5.0,
        )
        decision_contra = verifier.verify(
            claim="Vaccines cause autism",
            retrieval_response=contra_response,
        )

        # Test 2: No evidence at all
        decision_missing = verifier.verify(
            claim="Aliens built the pyramids",
            retrieval_response=empty_retrieval_response,
        )

        # Contradiction analysis should differ
        assert decision_contra.contradiction_analysis.is_contradicted is True
        assert decision_contra.contradiction_analysis.is_missing is False

        assert decision_missing.contradiction_analysis.is_contradicted is False
        assert decision_missing.contradiction_analysis.is_missing is True

        # Verdicts should differ
        assert decision_missing.final_verdict == FinalVerdict.UNVERIFIABLE
        # Contradiction case should be FALSE or NEEDS_REVIEW (not UNVERIFIABLE)
        assert decision_contra.final_verdict != FinalVerdict.UNVERIFIABLE
