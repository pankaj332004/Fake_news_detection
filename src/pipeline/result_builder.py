"""
Result Builder for Phase 6 Hybrid Pipeline.
Combines statistical baseline classification (Phase 2), evidence retrieval (Phase 4),
and LLM verification (Phase 5) into a unified, actionable hybrid result.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from src.schemas.request import ArticleDocument
from src.schemas.evidence import RetrievalResponse
from src.schemas.response import AnalysisResponse, VerificationResult, VerificationVerdict


class ConsensusStatus(str, Enum):
    """Consensus relationship between statistical baseline and evidence-grounded LLM."""

    STRONG_AGREEMENT = "STRONG_AGREEMENT"
    EVIDENCE_OVERRIDE = "EVIDENCE_OVERRIDE"
    UNGROUNDED_HEURISTIC = "UNGROUNDED_HEURISTIC"
    CONFLICT_FLAGGED = "CONFLICT_FLAGGED"
    PARTIAL_ALIGNMENT = "PARTIAL_ALIGNMENT"


class HybridResultBuilder:
    """
    Synthesizes multi-model pipeline components into a comprehensive hybrid verdict.
    Combines:
      - Statistical TF-IDF Classifier (stylistic patterns)
      - FAISS Evidence Retrieval (domain credibility + semantic similarity)
      - LLM Fact Verification (reasoning + contradiction detection + citations)
      - Decision Policy (uncertainty bounds + review triggers)
    """

    def __init__(
        self,
        evidence_weight: float = 0.50,
        statistical_weight: float = 0.30,
        credibility_weight: float = 0.20,
    ):
        self.evidence_weight = evidence_weight
        self.statistical_weight = statistical_weight
        self.credibility_weight = credibility_weight

    def build(
        self,
        doc: ArticleDocument,
        statistical_pred: Dict[str, Any],
        retrieval_res: Optional[RetrievalResponse] = None,
        verification_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Builds the unified analysis result dictionary with hybrid consensus and metrics.
        """
        # Statistical baseline details
        stat_label = statistical_pred.get("label", "Unknown")
        stat_conf = float(statistical_pred.get("confidence", 0.5))
        prob_fake = float(statistical_pred.get("prob_fake", 0.5))
        prob_real = float(statistical_pred.get("prob_real", 0.5))

        # Evidence retrieval details
        evidence_passages = []
        supporting_count = 0
        contradicting_count = 0
        max_similarity = 0.0
        avg_credibility = 0.0

        if retrieval_res:
            evidence_passages = [p.model_dump() for p in retrieval_res.all_passages]
            supporting_count = len(retrieval_res.supporting_passages)
            contradicting_count = len(retrieval_res.contradicting_passages)
            if retrieval_res.all_passages:
                scores = [getattr(p, "score", getattr(p, "similarity_score", 0.0)) for p in retrieval_res.all_passages]
                max_similarity = max(scores) if scores else 0.0
                cred_tiers = []
                for p in retrieval_res.all_passages:
                    tier = 1
                    if hasattr(p, "passage") and hasattr(p.passage, "metadata"):
                        tier = p.passage.metadata.get("credibility_tier", 1)
                    elif hasattr(p, "source_credibility_tier"):
                        tier = getattr(p, "source_credibility_tier", 1)
                    cred_tiers.append(tier)
                tier_weights = {1: 1.0, 2: 0.7, 3: 0.4}
                avg_credibility = sum(tier_weights.get(t, 0.5) for t in cred_tiers) / len(cred_tiers) if cred_tiers else 0.5

        # Verification details
        has_verification = verification_data is not None
        v_final = verification_data.get("final_verdict", "UNVERIFIABLE") if has_verification else "UNVERIFIABLE"
        v_llm = verification_data.get("llm_verdict", "UNVERIFIABLE") if has_verification else "UNVERIFIABLE"
        v_reasoning = verification_data.get("reasoning", "") if has_verification else ""
        v_citations = verification_data.get("evidence_citations", []) if has_verification else []
        v_limitations = verification_data.get("limitations", "") if has_verification else ""
        v_conf = float(verification_data.get("overall_confidence", stat_conf)) if has_verification else stat_conf
        v_uncertainty = verification_data.get("uncertainty_level", "high") if has_verification else "high"
        v_review = bool(verification_data.get("should_flag_for_review", False)) if has_verification else False

        contra_analysis = verification_data.get("contradiction_analysis", {}) if has_verification else {}
        evidence_status = contra_analysis.get("evidence_status", "absent" if not evidence_passages else "mixed")

        # Determine Consensus Status
        consensus, synthesis = self._determine_consensus(
            stat_label=stat_label,
            stat_conf=stat_conf,
            final_verdict=v_final,
            evidence_status=evidence_status,
            supporting_count=supporting_count,
            contradicting_count=contradicting_count,
            max_similarity=max_similarity,
            should_flag_review=v_review,
        )

        # Calibrate Final Hybrid Confidence
        if evidence_status in ("absent", "insufficient"):
            # When evidence is absent, hybrid confidence is capped and uncertainty is elevated
            calibrated_confidence = round(min(stat_conf * 0.6, 0.55), 4)
        else:
            calibrated_confidence = round(
                (self.evidence_weight * v_conf)
                + (self.statistical_weight * stat_conf)
                + (self.credibility_weight * avg_credibility),
                4,
            )

        # Determine Final Display Verdict
        # If evidence absent, verdict is UNVERIFIABLE
        if evidence_status == "absent" or v_final == "UNVERIFIABLE":
            display_verdict = "UNVERIFIABLE"
            badge_color = "slate"
        elif v_review or v_final == "NEEDS_REVIEW":
            display_verdict = "NEEDS_REVIEW"
            badge_color = "amber"
        elif v_final in ("FALSE", "Fake") or (v_final == "TRUE" and contradicting_count > 0 and v_review):
            display_verdict = "FALSE"
            badge_color = "crimson"
        elif v_final in ("TRUE", "Real"):
            display_verdict = "TRUE"
            badge_color = "emerald"
        else:
            display_verdict = v_final
            badge_color = "indigo"

        text_to_eval = doc.title if doc.title and doc.input_type.value == "url" else doc.text

        evidence_payload = {
            "total_retrieved": len(evidence_passages),
            "supporting_count": supporting_count,
            "contradicting_count": contradicting_count,
            "max_similarity": round(max_similarity, 4),
            "avg_credibility": round(avg_credibility, 4),
            "all_passages": evidence_passages,
            "retrieval_time_ms": retrieval_res.retrieval_time_ms if retrieval_res else 0.0,
        }

        return {
            "status": "SUCCESS",
            "input_type": doc.input_type.value,
            "raw_input": doc.raw_input,
            "title": doc.title,
            "source_url": doc.source_url,
            "source_domain": doc.source_domain,
            "authors": doc.authors,
            "published_date": doc.published_date,
            "char_count": doc.char_count,
            "word_count": doc.word_count,
            "evaluated_text": text_to_eval,
            "preview": doc.get_analysis_snippet(max_words=60),
            # Primary Hybrid Verdict
            "verdict": display_verdict,
            "final_verdict": display_verdict,
            "badge_color": badge_color,
            "calibrated_confidence": calibrated_confidence,
            "confidence": calibrated_confidence,
            "uncertainty_level": v_uncertainty,
            "should_flag_for_review": v_review,
            # Multi-Model Consensus Breakdown
            "consensus_status": consensus.value,
            "consensus_synthesis": synthesis,
            # Statistical Baseline (Phase 2)
            "statistical": {
                "verdict": stat_label,
                "confidence": stat_conf,
                "prob_fake": prob_fake,
                "prob_real": prob_real,
                "label_id": statistical_pred.get("label_id", 0),
            },
            # Evidence Retrieval (Phase 4)
            "evidence": evidence_payload,
            # LLM Verification (Phase 5)
            "verification": {
                "llm_verdict": v_llm,
                "final_verdict": v_final,
                "reasoning": v_reasoning,
                "evidence_citations": v_citations,
                "limitations": v_limitations,
                "uncertainty_level": v_uncertainty,
                "overall_confidence": v_conf,
                "should_flag_for_review": v_review,
                "evidence_status": evidence_status,
                "model_name": verification_data.get("model_name", "flan-t5-small") if verification_data else "none",
            },
        }

    def _determine_consensus(
        self,
        stat_label: str,
        stat_conf: float,
        final_verdict: str,
        evidence_status: str,
        supporting_count: int,
        contradicting_count: int,
        max_similarity: float,
        should_flag_review: bool,
    ) -> tuple[ConsensusStatus, str]:
        """Calculates consensus status and an explanatory human-readable synthesis."""
        stat_is_fake = stat_label.lower() in ("fake", "false")
        verdict_is_false = final_verdict in ("FALSE", "Fake")
        verdict_is_true = final_verdict in ("TRUE", "Real")

        if evidence_status in ("absent", "insufficient"):
            return (
                ConsensusStatus.UNGROUNDED_HEURISTIC,
                (
                    f"No verified evidence found in reference corpus (evidence {evidence_status}). "
                    f"Final verdict is UNVERIFIABLE. Statistical model predicts '{stat_label}' "
                    f"({stat_conf*100:.1f}%) based solely on stylistic phrasing."
                ),
            )

        if should_flag_review:
            return (
                ConsensusStatus.CONFLICT_FLAGGED,
                (
                    f"Flagged for human review. Signals exhibit high uncertainty or tension: "
                    f"Statistical={stat_label} ({stat_conf*100:.1f}%), "
                    f"Evidence Status={evidence_status} (Supports: {supporting_count}, Contradicts: {contradicting_count})."
                ),
            )

        if (stat_is_fake and verdict_is_false) or (not stat_is_fake and verdict_is_true):
            return (
                ConsensusStatus.STRONG_AGREEMENT,
                (
                    f"Strong consensus: Both the statistical classifier and evidence-grounded verification "
                    f"unanimously agree the content is {final_verdict} (top evidence similarity {max_similarity*100:.1f}%)."
                ),
            )

        # Disagreement: Evidence overrides statistical classifier
        if contradicting_count > 0 and not stat_is_fake:
            return (
                ConsensusStatus.EVIDENCE_OVERRIDE,
                (
                    f"Evidence override: While statistical styling suggested 'Real', authoritative retrieved "
                    f"evidence directly contradicts the claim ({contradicting_count} refuting passages). "
                    f"Evidence takes precedence over linguistic styling."
                ),
            )

        if supporting_count > 0 and stat_is_fake:
            return (
                ConsensusStatus.EVIDENCE_OVERRIDE,
                (
                    f"Evidence override: While stylistic classifier flagged 'Fake', authoritative evidence "
                    f"supports the claims ({supporting_count} corroborating passages). "
                    f"Factual verification overrides surface-level text style."
                ),
            )

        return (
            ConsensusStatus.PARTIAL_ALIGNMENT,
            (
                f"Hybrid assessment: Verdict is {final_verdict} based on {evidence_status} evidence "
                f"and statistical probability of {stat_conf*100:.1f}%."
            ),
        )
