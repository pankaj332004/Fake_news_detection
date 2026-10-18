"""
Prompt templates for LLM-based claim verification.
Constructs structured prompts that present the claim alongside retrieved evidence
and instruct the model to return a structured JSON verdict with reasoning.
"""

from typing import List, Optional

from src.schemas.evidence import RetrievalResult, StanceType


# Maximum character budget per evidence passage in prompts
MAX_PASSAGE_CHARS = 600

# Maximum total prompt length (chars) to stay within model context windows
MAX_PROMPT_CHARS = 4000


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars with an ellipsis indicator."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


class VerificationPrompt:
    """
    Builds structured prompts for LLM verification of claims.
    Designed for open-weight instruct models (Flan-T5, Phi-3-mini, TinyLlama, etc.).
    """

    SYSTEM_PREAMBLE = (
        "You are a fact-checking assistant. Your job is to evaluate whether a claim "
        "is TRUE, FALSE, or UNVERIFIABLE based on the provided evidence passages.\n"
        "You must reason step-by-step and cite specific evidence.\n"
        "Always distinguish between CONTRADICTORY evidence (evidence that directly "
        "disproves the claim) and MISSING evidence (no relevant information found).\n"
        "You MUST respond ONLY with a valid JSON object in the exact format specified."
    )

    RESPONSE_FORMAT = """\
Respond ONLY with this JSON (no extra text):
{
  "verdict": "TRUE" | "FALSE" | "UNVERIFIABLE",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<2-4 sentence explanation referencing evidence>",
  "evidence_citations": ["<passage_id_1>", "<passage_id_2>"],
  "contradictions_found": true | false,
  "limitations": "<brief note on evidence gaps or caveats>"
}"""

    @classmethod
    def build_evidence_block(
        cls,
        evidence_passages: List[RetrievalResult],
        max_passages: int = 5,
    ) -> str:
        """
        Formats retrieved evidence passages into a numbered text block for prompt injection.

        Args:
            evidence_passages: Retrieved evidence with stance labels.
            max_passages: Maximum evidence passages to include.

        Returns:
            Formatted evidence text block.
        """
        if not evidence_passages:
            return "No evidence passages were found for this claim."

        lines = []
        for i, result in enumerate(evidence_passages[:max_passages], start=1):
            stance_label = result.stance.value.upper() if result.stance else "NEUTRAL"
            source = result.passage.source or "Unknown"
            score_pct = f"{result.score * 100:.0f}%" if result.score else "N/A"
            text = _truncate(result.passage.text, MAX_PASSAGE_CHARS)
            passage_id = result.passage.passage_id

            lines.append(
                f"[Evidence {i}] (ID: {passage_id})\n"
                f"  Stance: {stance_label} | Source: {source} | Relevance: {score_pct}\n"
                f"  Text: \"{text}\""
            )

        return "\n\n".join(lines)

    @classmethod
    def build_verification_prompt(
        cls,
        claim: str,
        evidence_passages: List[RetrievalResult],
        supporting_passages: Optional[List[RetrievalResult]] = None,
        contradicting_passages: Optional[List[RetrievalResult]] = None,
        statistical_verdict: Optional[str] = None,
        statistical_confidence: Optional[float] = None,
        max_passages: int = 5,
    ) -> str:
        """
        Constructs the full verification prompt for an LLM.

        Args:
            claim: The user's claim or headline to verify.
            evidence_passages: All retrieved evidence passages.
            supporting_passages: Pre-filtered supporting passages (optional).
            contradicting_passages: Pre-filtered contradicting passages (optional).
            statistical_verdict: Optional baseline classifier verdict (e.g. "Fake"/"Real").
            statistical_confidence: Optional baseline classifier confidence score.
            max_passages: Max evidence passages to include.

        Returns:
            Complete prompt string ready for model inference.
        """
        # Build evidence section
        evidence_block = cls.build_evidence_block(evidence_passages, max_passages)

        # Summary of evidence distribution
        n_support = len(supporting_passages) if supporting_passages else 0
        n_contra = len(contradicting_passages) if contradicting_passages else 0
        n_neutral = len(evidence_passages) - n_support - n_contra

        evidence_summary = (
            f"Evidence Summary: {len(evidence_passages)} passages retrieved "
            f"({n_support} supporting, {n_contra} contradicting, {n_neutral} neutral)"
        )

        # Optional baseline context
        baseline_context = ""
        if statistical_verdict:
            conf_str = f" ({statistical_confidence*100:.0f}% confidence)" if statistical_confidence else ""
            baseline_context = (
                f"\nNote: A statistical classifier initially rated this claim as "
                f"'{statistical_verdict}'{conf_str}. Use this only as a weak prior — "
                f"your verdict must be based on the evidence."
            )

        # Build full prompt
        prompt = f"""{cls.SYSTEM_PREAMBLE}

=== CLAIM TO VERIFY ===
"{_truncate(claim, 500)}"

=== RETRIEVED EVIDENCE ===
{evidence_summary}

{evidence_block}
{baseline_context}

=== INSTRUCTIONS ===
1. Analyze ALL evidence passages carefully.
2. Determine if the evidence SUPPORTS, CONTRADICTS, or is INSUFFICIENT to verify the claim.
3. If evidence CONTRADICTS the claim, the verdict should be FALSE.
4. If evidence SUPPORTS the claim, the verdict should be TRUE.
5. If evidence is MISSING or INSUFFICIENT (not contradictory), the verdict should be UNVERIFIABLE.
6. CRITICAL: "No evidence found" is NOT the same as "evidence contradicts the claim."

{cls.RESPONSE_FORMAT}"""

        return _truncate(prompt, MAX_PROMPT_CHARS)

    @classmethod
    def build_no_evidence_prompt(cls, claim: str) -> str:
        """
        Prompt for claims where no evidence passages were retrieved.
        The model must acknowledge the absence rather than fabricate evidence.
        """
        return f"""{cls.SYSTEM_PREAMBLE}

=== CLAIM TO VERIFY ===
"{_truncate(claim, 500)}"

=== RETRIEVED EVIDENCE ===
No relevant evidence passages were found in the knowledge base for this claim.

=== INSTRUCTIONS ===
1. Since NO evidence was found, you MUST return "UNVERIFIABLE" as your verdict.
2. Do NOT guess or use your training knowledge to verify the claim.
3. Clearly state in your reasoning that evidence is missing (not contradictory).

{cls.RESPONSE_FORMAT}"""
