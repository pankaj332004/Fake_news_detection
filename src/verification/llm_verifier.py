"""
LLM Verifier: High-level orchestrator for evidence-grounded claim verification.

Integrates:
- LLM model loading and generation (Phase 5 core)
- Prompt construction with evidence and claim
- JSON response parsing with fallback handling
- Evidence quality and contradiction analysis
- Uncertainty estimation and decision policy

Deliverable: Evidence-grounded verification results with structured reasoning,
evidence references, limitations, and explicit distinction between contradictory
evidence and missing evidence.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from src.schemas.evidence import RetrievalResponse, RetrievalResult
from src.verification.claim_extraction import ClaimExtractor
from src.verification.contradiction_detector import ContradictionDetector
from src.verification.decision_policy import DecisionPolicy, FinalVerdict, VeracityDecision
from src.verification.evidence_analyzer import EvidenceAnalyzer, EvidenceQuality
from src.verification.llm_loader import LLMBackend, LLMLoader
from src.verification.prompt_templates import VerificationPrompt
from src.verification.uncertainty import UncertaintyEstimator, UncertaintyEstimate

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to extract a JSON object from LLM text output.
    Handles models that wrap JSON in markdown code fences or extra text.
    """
    # Try direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code fence
    code_fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_fence_match:
        try:
            return json.loads(code_fence_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try extracting first {...} block
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _parse_llm_response(raw_output: str) -> Dict[str, Any]:
    """
    Parses LLM output into a structured verification result.
    Falls back to heuristic extraction if JSON parsing fails.

    Expected format:
    {
        "verdict": "TRUE" | "FALSE" | "UNVERIFIABLE",
        "confidence": float,
        "reasoning": str,
        "evidence_citations": [str],
        "contradictions_found": bool,
        "limitations": str
    }
    """
    parsed = _extract_json_from_text(raw_output)

    if parsed and "verdict" in parsed:
        # Normalize verdict
        verdict = str(parsed.get("verdict", "UNVERIFIABLE")).upper()
        if verdict not in ("TRUE", "FALSE", "UNVERIFIABLE"):
            verdict = "UNVERIFIABLE"
        parsed["verdict"] = verdict

        # Ensure all required fields
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("reasoning", "")
        parsed.setdefault("evidence_citations", [])
        parsed.setdefault("contradictions_found", False)
        parsed.setdefault("limitations", "")

        # Clamp confidence
        try:
            parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
        except (TypeError, ValueError):
            parsed["confidence"] = 0.5

        return parsed

    # Fallback: heuristic parsing from free text
    logger.warning("Could not parse JSON from LLM output. Using heuristic fallback.")
    return _heuristic_parse(raw_output)


def _heuristic_parse(text: str) -> Dict[str, Any]:
    """Extracts verdict and reasoning from non-JSON LLM output via pattern matching."""
    text_lower = text.lower()

    # Detect verdict from keywords
    if any(w in text_lower for w in ["verdict: false", "verdict is false", "claim is false", "false"]):
        verdict = "FALSE"
    elif any(w in text_lower for w in ["verdict: true", "verdict is true", "claim is true", "true"]):
        verdict = "TRUE"
    else:
        verdict = "UNVERIFIABLE"

    # Check for contradiction mentions
    contradictions = any(
        w in text_lower
        for w in ["contradict", "refute", "disprove", "debunk"]
    )

    return {
        "verdict": verdict,
        "confidence": 0.4,  # Low confidence for heuristic parse
        "reasoning": text[:500] if text else "LLM output could not be parsed into structured format.",
        "evidence_citations": [],
        "contradictions_found": contradictions,
        "limitations": "Response was extracted via heuristic parsing — structured JSON output failed.",
    }


class LLMVerifier:
    """
    High-level orchestrator for LLM-based claim verification.
    Connects all Phase 5 components into a single verify() method.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        backend: str = LLMBackend.LOCAL,
        hf_api_token: Optional[str] = None,
        auto_load: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model ID.
            backend: 'local', 'hf_api', or 'mock'.
            hf_api_token: Optional HuggingFace API token.
            auto_load: If True, load model on init.
        """
        self.llm = LLMLoader(
            model_name=model_name,
            backend=backend,
            hf_api_token=hf_api_token,
        )
        self.evidence_analyzer = EvidenceAnalyzer()
        self.contradiction_detector = ContradictionDetector()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.decision_policy = DecisionPolicy()
        self.claim_extractor = ClaimExtractor()

        if auto_load:
            self.llm.load()

    def verify(
        self,
        claim: str,
        retrieval_response: Optional[RetrievalResponse] = None,
        all_passages: Optional[List[RetrievalResult]] = None,
        supporting_passages: Optional[List[RetrievalResult]] = None,
        contradicting_passages: Optional[List[RetrievalResult]] = None,
        statistical_verdict: Optional[str] = None,
        statistical_confidence: Optional[float] = None,
    ) -> VeracityDecision:
        """
        Runs full verification pipeline: prompt construction → LLM generation →
        response parsing → evidence analysis → uncertainty estimation → final decision.

        Args:
            claim: The claim or headline to verify.
            retrieval_response: Full retrieval response (preferred).
            all_passages: All retrieved passages (alternative to retrieval_response).
            supporting_passages: Pre-filtered supporting passages.
            contradicting_passages: Pre-filtered contradicting passages.
            statistical_verdict: Optional baseline classifier verdict.
            statistical_confidence: Optional baseline classifier confidence.

        Returns:
            VeracityDecision with final verdict, reasoning, citations, and metadata.
        """
        start_time = time.perf_counter()

        # Resolve evidence sources
        if retrieval_response is not None:
            all_passages = retrieval_response.all_passages
            supporting_passages = retrieval_response.supporting_passages
            contradicting_passages = retrieval_response.contradicting_passages

        all_passages = all_passages or []
        supporting_passages = supporting_passages or []
        contradicting_passages = contradicting_passages or []

        # Step 1: Analyze evidence quality
        evidence_quality = self.evidence_analyzer.analyze(
            all_passages=all_passages,
            supporting_passages=supporting_passages,
            contradicting_passages=contradicting_passages,
        )

        # Step 2: Contradiction vs. missing evidence analysis
        contradiction_analysis = self.contradiction_detector.analyze(
            quality=evidence_quality,
            all_passages=all_passages,
            supporting_passages=supporting_passages,
            contradicting_passages=contradicting_passages,
        )

        # Step 3: Build prompt
        if evidence_quality.total_passages > 0:
            prompt = VerificationPrompt.build_verification_prompt(
                claim=claim,
                evidence_passages=all_passages,
                supporting_passages=supporting_passages,
                contradicting_passages=contradicting_passages,
                statistical_verdict=statistical_verdict,
                statistical_confidence=statistical_confidence,
            )
        else:
            prompt = VerificationPrompt.build_no_evidence_prompt(claim)

        # Step 4: Generate LLM response
        try:
            raw_output = self.llm.generate(prompt)
            logger.info(f"LLM generated {len(raw_output)} chars in {time.perf_counter()-start_time:.1f}s")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raw_output = ""

        # Step 5: Parse response
        if raw_output:
            parsed = _parse_llm_response(raw_output)
        else:
            parsed = {
                "verdict": "UNVERIFIABLE",
                "confidence": 0.1,
                "reasoning": "LLM generation failed — no output produced.",
                "evidence_citations": [],
                "contradictions_found": False,
                "limitations": "Model failed to generate a response.",
            }

        # Step 6: Uncertainty estimation
        uncertainty = self.uncertainty_estimator.estimate(
            evidence_quality=evidence_quality,
            contradiction_analysis=contradiction_analysis,
            llm_verdict=parsed["verdict"],
            llm_confidence=parsed["confidence"],
        )

        # Step 7: Final decision via policy
        decision = self.decision_policy.decide(
            llm_verdict=parsed["verdict"],
            llm_confidence=parsed["confidence"],
            llm_reasoning=parsed["reasoning"],
            llm_citations=parsed["evidence_citations"],
            llm_contradictions_found=parsed["contradictions_found"],
            llm_limitations=parsed["limitations"],
            evidence_quality=evidence_quality,
            contradiction_analysis=contradiction_analysis,
            uncertainty=uncertainty,
            statistical_verdict=statistical_verdict,
            llm_raw_output=raw_output,
            model_name=self.llm.model_name,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Verification complete in {elapsed:.1f}s — "
            f"Verdict: {decision.final_verdict.value}"
        )

        return decision


def verify_claim(
    claim: str,
    retrieval_response: Optional[RetrievalResponse] = None,
    model_name: str = "google/flan-t5-small",
    backend: str = LLMBackend.LOCAL,
    statistical_verdict: Optional[str] = None,
    statistical_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function for one-shot claim verification.

    Args:
        claim: Claim text to verify.
        retrieval_response: Retrieved evidence (optional).
        model_name: HuggingFace model ID.
        backend: LLM backend type.
        statistical_verdict: Optional baseline verdict.
        statistical_confidence: Optional baseline confidence.

    Returns:
        Dictionary with verification result.
    """
    verifier = LLMVerifier(model_name=model_name, backend=backend)
    decision = verifier.verify(
        claim=claim,
        retrieval_response=retrieval_response,
        statistical_verdict=statistical_verdict,
        statistical_confidence=statistical_confidence,
    )
    return decision.to_dict()
