"""
Re-ranking utilities for retrieved evidence passages.
Combines semantic cosine similarity with source credibility weighting.
"""

from typing import List

from src.retrieval.source_validation import get_source_credibility_tier
from src.schemas.evidence import RetrievalResult


def rerank_passages(
    results: List[RetrievalResult],
    tier_boost: float = 0.05,
) -> List[RetrievalResult]:
    """
    Reranks retrieved evidence results by combining vector similarity score
    with an authoritative source credibility tier boost.

    Args:
        results: List of initial RetrievalResult items.
        tier_boost: Additional score boost for Tier 1 verified sources.

    Returns:
        Sorted list of RetrievalResult items with updated ranks.
    """
    reranked = []

    for item in results:
        tier = get_source_credibility_tier(item.passage.source_url or "")
        boost = tier_boost if "Tier 1" in tier else 0.0
        adjusted_score = min(1.0, item.score + boost)

        updated_item = item.model_copy(
            update={
                "score": round(adjusted_score, 4),
                "explanation": (item.explanation or "") + f" [Credibility: {tier}]",
            }
        )
        reranked.append(updated_item)

    # Sort descending by adjusted score
    reranked.sort(key=lambda x: x.score, reverse=True)

    # Re-assign ranks
    for rank, item in enumerate(reranked, start=1):
        item.rank = rank

    return reranked
