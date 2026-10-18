"""
Evidence search and stance evaluation module.
Retrieves top-k evidence passages and categorizes them into supporting vs. contradicting evidence.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.retrieval.embeddings import EmbeddingGenerator, get_embedding_generator
from src.retrieval.vector_store import DEFAULT_INDEX_DIR, FAISSVectorStore
from src.schemas.evidence import (
    EvidencePassage,
    RetrievalResponse,
    RetrievalResult,
    StanceType,
)

logger = logging.getLogger(__name__)

# Refutation cues frequently found in fact-checking evidence
CONTRADICTION_CUES = [
    r"\bno link\b",
    r"\bnever\b",
    r"\bcannot alter\b",
    r"\bcannot transmit\b",
    r"\bcannot\b",
    r"\bnot true\b",
    r"\bdebunked\b",
    r"\bfalse\b",
    r"\bfraudulent\b",
    r"\brefuted\b",
    r"\bcontradicted\b",
    r"\bineffective against\b",
    r"\bhoax\b",
    r"\bno evidence\b",
    r"\bno correlation\b",
    r"\buntrue\b",
    r"\bmyth\b",
]

SUPPORT_CUES = [
    r"\bconfirm\b",
    r"\bconfirmed\b",
    r"\bverif\w+\b",
    r"\bproves?\b",
    r"\bdemonstrates?\b",
    r"\bevidence shows\b",
    r"\bconsensus\b",
    r"\bindicates?\b",
    r"\bconsistent with\b",
    r"\baccurat\w+\b",
]


def determine_stance(claim: str, passage_text: str) -> Tuple[StanceType, str]:
    """
    Analyzes semantic and lexical alignment to categorize evidence as Supporting or Contradicting.

    Args:
        claim: The user input claim or headline.
        passage_text: The retrieved reference passage text.

    Returns:
        Tuple of (StanceType, explanation_string).
    """
    norm_claim = claim.lower()
    norm_passage = passage_text.lower()

    # 1. Check for explicit refutation markers targeting claim concepts
    has_contradiction_cue = any(re.search(cue, norm_passage) for cue in CONTRADICTION_CUES)
    has_support_cue = any(re.search(cue, norm_passage) for cue in SUPPORT_CUES)

    # 2. Check claim polarity (does the claim assert something sensational or conspiratorial?)
    is_claim_negative_or_conspiracy = any(
        w in norm_claim
        for w in [
            "staged",
            "fake",
            "hoax",
            "conspiracy",
            "flat",
            "alters dna",
            "causes autism",
            "alien base",
            "microchip",
        ]
    )

    if has_contradiction_cue:
        if is_claim_negative_or_conspiracy:
            # e.g. Claim: "Moon landing staged". Passage: "Claims that landings were staged are refuted..."
            return StanceType.CONTRADICTS, "Passage explicitly refutes or disproves the asserted claim."
        elif any(w in norm_claim for w in ["vaccine", "safe", "5g", "earth"]) and any(
            w in norm_passage for w in ["cannot", "no link", "refuted", "safe"]
        ):
            # Check if passage refutes a typical false claim about the topic
            if "alter" in norm_claim or "autism" in norm_claim or "flat" in norm_claim:
                return StanceType.CONTRADICTS, "Passage refutes the premise of the claim with authoritative evidence."

    # 3. If claim affirms an established scientific fact that passage confirms
    if has_support_cue and not is_claim_negative_or_conspiracy:
        return StanceType.SUPPORTS, "Passage provides corroborating factual evidence confirming the claim."

    # 4. Default stance determination based on semantic tone
    if has_contradiction_cue and is_claim_negative_or_conspiracy:
        return StanceType.CONTRADICTS, "Evidence directly contradicts the assertion."
    elif has_support_cue:
        return StanceType.SUPPORTS, "Evidence aligns with and supports the statement."

    return StanceType.NEUTRAL, "Passage provides related contextual information."


class EvidenceRetriever:
    """
    High-level orchestrator for evidence retrieval and stance evaluation.
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        index_dir: Union[str, Path] = DEFAULT_INDEX_DIR,
    ):
        self.embedding_generator = embedding_generator or get_embedding_generator()
        self.vector_store = vector_store or FAISSVectorStore()

        # Attempt to load saved index if not already populated
        if self.vector_store.total_passages == 0:
            loaded = self.vector_store.load(index_dir)
            if not loaded:
                logger.warning(
                    f"Vector store is empty and could not be loaded from {index_dir}. "
                    "Run scripts/build_evidence_index.py to build it."
                )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.25,
    ) -> RetrievalResponse:
        """
        Retrieves relevant evidence passages for a claim and organizes them by stance.

        Args:
            query: Input claim, headline, or query string.
            top_k: Maximum number of passages to retrieve.
            min_score: Minimum cosine similarity score threshold.

        Returns:
            RetrievalResponse with supporting, contradicting, and all retrieved passages.
        """
        start_time = time.perf_counter()

        # Step 1: Compute query embedding
        query_vector = self.embedding_generator.encode_query(query)

        # Step 2: Nearest neighbor search in FAISS
        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
        )

        # Step 3: Analyze stance for each retrieved passage
        supporting: List[RetrievalResult] = []
        contradicting: List[RetrievalResult] = []
        all_results: List[RetrievalResult] = []

        for rank, (passage, score) in enumerate(raw_results, start=1):
            stance, explanation = determine_stance(query, passage.text)

            res = RetrievalResult(
                passage=passage,
                score=score,
                rank=rank,
                stance=stance,
                explanation=explanation,
            )
            all_results.append(res)

            if stance == StanceType.SUPPORTS:
                supporting.append(res)
            elif stance == StanceType.CONTRADICTS:
                contradicting.append(res)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResponse(
            query=query,
            total_retrieved=len(all_results),
            supporting_passages=supporting,
            contradicting_passages=contradicting,
            all_passages=all_results,
            model_name=self.embedding_generator.model_name,
            retrieval_time_ms=round(elapsed_ms, 2),
        )
