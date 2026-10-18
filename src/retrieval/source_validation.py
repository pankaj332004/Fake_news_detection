"""
Source validation module for reference evidence documents.
Evaluates institution credibility, canonical metadata completeness, and domain tiers.
"""

from typing import Any, Dict, List
from urllib.parse import urlparse

# Recognized high-trust institutional domains
HIGH_TRUST_DOMAINS = {
    "who.int",
    "cdc.gov",
    "nih.gov",
    "nasa.gov",
    "esa.int",
    "ipcc.ch",
    "noaa.gov",
    "bls.gov",
    "federalreserve.gov",
    "fcc.gov",
    "apnews.com",
    "reuters.com",
}


def get_source_credibility_tier(source_url: str) -> str:
    """
    Categorizes evidence sources into credibility tiers.
    Tier 1: Intergovernmental, academic, or national scientific agencies.
    Tier 2: Major accredited journalistic fact-checking networks.
    Tier 3: Standard web sources.
    """
    if not source_url:
        return "Tier 3: Standard Source"

    domain = urlparse(source_url).netloc.lower()
    # Strip leading www.
    if domain.startswith("www."):
        domain = domain[4:]

    if any(trusted in domain for trusted in HIGH_TRUST_DOMAINS):
        return "Tier 1: Authoritative Institutional / Scientific"
    elif any(domain.endswith(tld) for tld in [".gov", ".edu", ".org"]):
        return "Tier 2: Government / Academic / NGO"

    return "Tier 3: Standard Reference"


def validate_reference_corpus(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates document structure, unique IDs, and field completeness.
    """
    report = {
        "total_documents": len(documents),
        "valid": True,
        "errors": [],
        "categories": {},
    }

    seen_ids = set()
    for idx, doc in enumerate(documents):
        doc_id = doc.get("doc_id")
        if not doc_id:
            report["valid"] = False
            report["errors"].append(f"Doc at index {idx} is missing 'doc_id'.")
        elif doc_id in seen_ids:
            report["valid"] = False
            report["errors"].append(f"Duplicate doc_id '{doc_id}' at index {idx}.")
        else:
            seen_ids.add(doc_id)

        if not doc.get("text") or len(doc.get("text", "").strip()) < 50:
            report["valid"] = False
            report["errors"].append(f"Doc '{doc_id}' text is empty or too short.")

        category = doc.get("category", "general")
        report["categories"][category] = report["categories"].get(category, 0) + 1

    return report
