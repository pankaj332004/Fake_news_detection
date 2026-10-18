"""
Claim Extraction: Extracts and normalizes atomic claims from
longer articles and compound statements for individual verification.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Maximum claim length in characters
MAX_CLAIM_LENGTH = 500
MIN_CLAIM_LENGTH = 10

# Sentence-ending patterns
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')

# Patterns indicating a factual claim (vs. opinion or question)
FACTUAL_INDICATORS = [
    r'\b(?:is|are|was|were|has|have|had|will|does|did)\b',
    r'\b(?:according to|studies show|researchers found|evidence suggests)\b',
    r'\b\d+(?:\.\d+)?%?\b',  # Numbers and percentages
    r'\b(?:confirmed|proved|discovered|announced|revealed|reported)\b',
]

# Patterns indicating non-claims (questions, opinions, commands)
NON_CLAIM_PATTERNS = [
    r'^\s*(?:what|who|where|when|why|how)\b.*\?',  # Questions
    r'^\s*(?:i think|i believe|in my opinion|personally)\b',  # Opinions
    r'^\s*(?:please|let\'s|should we|could you)\b',  # Commands/requests
]


class ClaimExtractor:
    """
    Extracts verifiable atomic claims from longer text inputs.
    Splits compound statements and filters out non-verifiable content.
    """

    def __init__(
        self,
        max_claim_length: int = MAX_CLAIM_LENGTH,
        min_claim_length: int = MIN_CLAIM_LENGTH,
    ):
        self.max_claim_length = max_claim_length
        self.min_claim_length = min_claim_length

    def extract_claims(self, text: str) -> List[str]:
        """
        Extracts a list of atomic, verifiable claims from input text.

        For short inputs (claims/headlines), returns the input as-is.
        For longer text (articles), splits into sentences and filters for
        verifiable factual assertions.

        Args:
            text: Input text (claim, headline, or article body).

        Returns:
            List of extracted claim strings suitable for verification.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # Short text: treat as a single claim
        if len(text) <= self.max_claim_length and '\n' not in text:
            sentences = SENTENCE_SPLIT_PATTERN.split(text)
            if len(sentences) <= 2:
                return [text]

        # Longer text: split into sentences
        sentences = self._split_into_sentences(text)

        # Filter for verifiable factual claims
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < self.min_claim_length:
                continue
            if len(sent) > self.max_claim_length:
                sent = sent[:self.max_claim_length]

            if self._is_verifiable_claim(sent):
                claims.append(sent)

        # If no claims pass the filter, return the first few sentences
        if not claims and sentences:
            fallback = [s.strip() for s in sentences[:3]
                        if len(s.strip()) >= self.min_claim_length]
            return fallback if fallback else [text[:self.max_claim_length]]

        return claims

    def get_primary_claim(self, text: str) -> str:
        """
        Returns the single most important claim from the input.
        For verification, this is typically the first or most assertive statement.

        Args:
            text: Input text.

        Returns:
            Primary claim string.
        """
        claims = self.extract_claims(text)
        if claims:
            # Prefer claims with factual indicators
            scored = []
            for claim in claims:
                score = sum(
                    1 for pattern in FACTUAL_INDICATORS
                    if re.search(pattern, claim, re.IGNORECASE)
                )
                scored.append((score, claim))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
        return text[:self.max_claim_length]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences."""
        # First split by newlines for paragraph boundaries
        paragraphs = text.split('\n')
        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            sents = SENTENCE_SPLIT_PATTERN.split(para)
            sentences.extend(sents)
        return sentences

    def _is_verifiable_claim(self, sentence: str) -> bool:
        """
        Determines if a sentence is a verifiable factual claim
        (vs. opinion, question, or command).
        """
        # Exclude non-claims
        for pattern in NON_CLAIM_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return False

        # Prefer sentences with factual indicators
        factual_score = sum(
            1 for pattern in FACTUAL_INDICATORS
            if re.search(pattern, sentence, re.IGNORECASE)
        )

        # Accept if it has at least one factual indicator, or is a
        # declarative sentence (not a question/command)
        if factual_score > 0:
            return True

        # Accept declarative sentences that don't end with ? or !
        if not sentence.rstrip().endswith('?'):
            return True

        return False
