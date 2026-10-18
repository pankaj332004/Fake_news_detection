"""
Unit tests for Phase 6 Hybrid Pipeline and Result Builder.
Validates multi-model consensus, uncertainty calibration, evidence override,
and end-to-end integration across baseline, retrieval, and LLM verification.
"""

import pytest
from src.pipeline.result_builder import ConsensusStatus, HybridResultBuilder
from src.schemas.evidence import EvidencePassage, RetrievalResponse, RetrievalResult, StanceType
from src.schemas.request import ArticleDocument, InputType


@pytest.fixture
def sample_document():
    return ArticleDocument(
        raw_input="Vaccines contain microchips to monitor human populations.",
        text="Vaccines contain microchips to monitor human populations.",
        title="Microchip Claim",
        input_type=InputType.CLAIM,
    )


@pytest.fixture
def contradicting_retrieval():
    passage = EvidencePassage(
        passage_id="DOC-CDC-01#p0",
        doc_id="DOC-CDC-01",
        text="Vaccines contain biological components designed to elicit immune responses and do not contain microchips or tracking devices.",
        source="CDC",
        source_url="https://www.cdc.gov/vaccinesafety/",
        metadata={"credibility_tier": 1},
    )
    result = RetrievalResult(
        passage=passage,
        score=0.88,
        rank=1,
        stance=StanceType.CONTRADICTS,
        explanation="Refutes microchip premise directly.",
    )
    return RetrievalResponse(
        query="Vaccines contain microchips",
        total_retrieved=1,
        supporting_passages=[],
        contradicting_passages=[result],
        all_passages=[result],
        model_name="all-MiniLM-L6-v2",
        retrieval_time_ms=12.5,
    )


@pytest.fixture
def supporting_retrieval():
    passage = EvidencePassage(
        passage_id="DOC-NASA-01#p0",
        doc_id="DOC-NASA-01",
        text="Perseverance rover detected unambiguous organic molecules and ancient lakebed sediments on Mars.",
        source="NASA",
        source_url="https://www.nasa.gov/perseverance",
        metadata={"credibility_tier": 1},
    )
    result = RetrievalResult(
        passage=passage,
        score=0.92,
        rank=1,
        stance=StanceType.SUPPORTS,
        explanation="Confirms Mars lakebed discovery.",
    )
    return RetrievalResponse(
        query="NASA finds lakebed on Mars",
        total_retrieved=1,
        supporting_passages=[result],
        contradicting_passages=[],
        all_passages=[result],
        model_name="all-MiniLM-L6-v2",
        retrieval_time_ms=10.0,
    )


class TestHybridResultBuilder:
    def test_strong_agreement_fake(self, sample_document, contradicting_retrieval):
        """When baseline says Fake and LLM/Evidence says FALSE, consensus is STRONG_AGREEMENT."""
        builder = HybridResultBuilder()
        stat_pred = {"label": "Fake", "confidence": 0.82, "prob_fake": 0.82, "prob_real": 0.18}
        verif_data = {
            "final_verdict": "FALSE",
            "llm_verdict": "FALSE",
            "reasoning": "Contradicted by CDC documentation.",
            "overall_confidence": 0.85,
            "uncertainty_level": "low",
            "should_flag_for_review": False,
            "contradiction_analysis": {"evidence_status": "contradicted"},
        }
        res = builder.build(
            doc=sample_document,
            statistical_pred=stat_pred,
            retrieval_res=contradicting_retrieval,
            verification_data=verif_data,
        )

        assert res["final_verdict"] == "FALSE"
        assert res["consensus_status"] == ConsensusStatus.STRONG_AGREEMENT.value
        assert res["badge_color"] == "crimson"
        assert res["calibrated_confidence"] > 0.70
        assert "Strong consensus" in res["consensus_synthesis"]

    def test_evidence_override_statistical_real(self, sample_document, contradicting_retrieval):
        """When statistical baseline says Real (due to clean grammar) but evidence refutes it -> EVIDENCE_OVERRIDE."""
        builder = HybridResultBuilder()
        stat_pred = {"label": "Real", "confidence": 0.65, "prob_fake": 0.35, "prob_real": 0.65}
        verif_data = {
            "final_verdict": "FALSE",
            "llm_verdict": "FALSE",
            "reasoning": "Scientific evidence refutes microchip claim.",
            "overall_confidence": 0.80,
            "uncertainty_level": "medium",
            "should_flag_for_review": False,
            "contradiction_analysis": {"evidence_status": "contradicted"},
        }
        res = builder.build(
            doc=sample_document,
            statistical_pred=stat_pred,
            retrieval_res=contradicting_retrieval,
            verification_data=verif_data,
        )

        assert res["final_verdict"] == "FALSE"
        assert res["consensus_status"] == ConsensusStatus.EVIDENCE_OVERRIDE.value
        assert "Evidence override" in res["consensus_synthesis"]

    def test_ungrounded_heuristic_missing_evidence(self, sample_document):
        """When no evidence is retrieved, system flags UNGROUNDED_HEURISTIC and UNVERIFIABLE."""
        builder = HybridResultBuilder()
        empty_retrieval = RetrievalResponse(
            query="Aliens built underwater base",
            total_retrieved=0,
            supporting_passages=[],
            contradicting_passages=[],
            all_passages=[],
            retrieval_time_ms=4.0,
        )
        stat_pred = {"label": "Fake", "confidence": 0.75, "prob_fake": 0.75, "prob_real": 0.25}
        verif_data = {
            "final_verdict": "UNVERIFIABLE",
            "llm_verdict": "UNVERIFIABLE",
            "reasoning": "No relevant reference passages in database.",
            "overall_confidence": 0.30,
            "uncertainty_level": "high",
            "should_flag_for_review": False,
            "contradiction_analysis": {"evidence_status": "absent"},
        }
        res = builder.build(
            doc=sample_document,
            statistical_pred=stat_pred,
            retrieval_res=empty_retrieval,
            verification_data=verif_data,
        )

        assert res["final_verdict"] == "UNVERIFIABLE"
        assert res["consensus_status"] == ConsensusStatus.UNGROUNDED_HEURISTIC.value
        assert res["calibrated_confidence"] <= 0.55
        assert "UNVERIFIABLE" in res["consensus_synthesis"]

    def test_conflict_flagged_for_human_review(self, sample_document, contradicting_retrieval):
        """When uncertainty is high or flags review, consensus is CONFLICT_FLAGGED."""
        builder = HybridResultBuilder()
        stat_pred = {"label": "Real", "confidence": 0.55, "prob_fake": 0.45, "prob_real": 0.55}
        verif_data = {
            "final_verdict": "NEEDS_REVIEW",
            "llm_verdict": "TRUE",
            "reasoning": "Mixed signals and ambiguous claim phrasing.",
            "overall_confidence": 0.45,
            "uncertainty_level": "high",
            "should_flag_for_review": True,
            "contradiction_analysis": {"evidence_status": "mixed"},
        }
        res = builder.build(
            doc=sample_document,
            statistical_pred=stat_pred,
            retrieval_res=contradicting_retrieval,
            verification_data=verif_data,
        )

        assert res["final_verdict"] == "NEEDS_REVIEW"
        assert res["consensus_status"] == ConsensusStatus.CONFLICT_FLAGGED.value
        assert res["should_flag_for_review"] is True
        assert res["badge_color"] == "amber"
