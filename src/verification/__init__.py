"""
Verification package: LLM-powered evidence-grounded claim verification.
Connects retrieved evidence passages to an open-weight language model
for structured reasoning about veracity.
"""

from src.verification.llm_verifier import LLMVerifier, verify_claim
from src.verification.prompt_templates import VerificationPrompt
from src.verification.evidence_analyzer import EvidenceAnalyzer
from src.verification.contradiction_detector import ContradictionDetector
from src.verification.decision_policy import DecisionPolicy, VeracityDecision
from src.verification.uncertainty import UncertaintyEstimator

__all__ = [
    "LLMVerifier",
    "verify_claim",
    "VerificationPrompt",
    "EvidenceAnalyzer",
    "ContradictionDetector",
    "DecisionPolicy",
    "VeracityDecision",
    "UncertaintyEstimator",
]
