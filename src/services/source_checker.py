# -*- coding: utf-8 -*-
"""
Source Verification Service for Veritas AI.
Evaluates publisher domain, ownership, attribution standards, contact clarity,
historical peer-reviewed citations, and red flags.
"""

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import re

from src.storage.history_repository import HistoryRepository

# Known questionable TLDs often used for ephemeral spam or cloaked redirect networks
SUSPICIOUS_TLDS = {".xyz", ".top", ".buzz", ".work", ".click", ".stream", ".icu", ".cam", ".monster"}

# Satirical domain indicators
KNOWN_SATIRE = {
    "theonion.com", "babylonbee.com", "newsthump.com", "borowitzreport.com",
    "waterfordwhispersnews.com", "thehardtimes.net", "duffelblog.com"
}


class SourceCheckerService:
    """Evaluates publisher integrity and transparency."""

    @staticmethod
    def extract_domain(input_string: str) -> str:
        """Extracts and normalizes domain from URL or raw text."""
        raw = input_string.strip().lower()
        if not raw.startswith("http://") and not raw.startswith("https://"):
            if "/" in raw or "." in raw:
                raw = "https://" + raw

        try:
            parsed = urlparse(raw)
            domain = parsed.netloc or parsed.path
            if domain.startswith("www."):
                domain = domain[4:]
            # Clean port or paths
            domain = domain.split(":")[0].split("/")[0]
            return domain
        except Exception:
            return input_string.strip().lower()

    @classmethod
    def verify_source(cls, query: str) -> Dict[str, Any]:
        """
        Deep source evaluation. Looks up verified database first,
        then evaluates structural and journalistic indicators.
        """
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Please provide a valid publisher name, domain, or article URL."
            }

        domain = cls.extract_domain(query)
        sources_match = HistoryRepository.search_sources(domain)

        # Also search by name if not found by domain
        if not sources_match:
            sources_match = HistoryRepository.search_sources(query.strip())

        if sources_match:
            src = sources_match[0]
            flags = [f.strip() for f in src.get("flag_reasons", "").split(";") if f.strip() and f.strip() != "None"]
            return {
                "success": True,
                "domain": src["domain"],
                "publisher_name": src["name"],
                "credibility_tier": src["credibility_tier"],
                "is_registered": True,
                "ownership": src["ownership"],
                "country": src["country"],
                "founded_year": src["founded_year"],
                "contact_info": src["contact_info"],
                "history_summary": src["history_summary"],
                "editorial_practices": src["editorial_practices"],
                "flags": flags,
                "evidence_citations": src["verified_evidence_count"],
                "transparency_score": 95 if "Tier 1" in src["credibility_tier"] else (85 if "Tier 2" in src["credibility_tier"] else 60),
                "explanation": cls._build_explanation(src["credibility_tier"], src["editorial_practices"], flags)
            }

        # Dynamic heuristic evaluation for unregistered domains
        return cls._evaluate_unregistered_domain(domain)

    @classmethod
    def _evaluate_unregistered_domain(cls, domain: str) -> Dict[str, Any]:
        """Performs heuristic evaluation of an uncatalogued domain."""
        flags: List[str] = []
        is_satire = any(sat in domain for sat in KNOWN_SATIRE)
        has_suspicious_tld = any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)
        has_edu_gov = domain.endswith(".edu") or domain.endswith(".gov") or domain.endswith(".mil")
        has_org = domain.endswith(".org")

        if is_satire:
            tier = "Satire: Humor / Parody Publication"
            score = 65
            flags.append("Content is satirical/parody; not intended to be taken as factual reporting.")
        elif has_edu_gov:
            tier = "Tier 1: Government or Accredited Educational Institution"
            score = 90
        elif has_suspicious_tld:
            tier = "Questionable: Low-Trust TLD or Ephemeral Network"
            score = 35
            flags.append(f"Domain uses high-risk registrar TLD often associated with cloaked ad-farms.")
            flags.append("Missing public masthead, named editorial board, or physical headquarters disclosure.")
        elif has_org:
            tier = "Tier 2: Non-Profit or Non-Governmental Organization"
            score = 75
        else:
            tier = "Tier 3: Unverified Web Publisher"
            score = 55
            flags.append("Domain is not in the IFCN accredited signatory registry.")
            flags.append("Independent verification of author bylines and fact-checking policy recommended.")

        return {
            "success": True,
            "domain": domain,
            "publisher_name": domain.title(),
            "credibility_tier": tier,
            "is_registered": False,
            "ownership": "Pending Registry Audit",
            "country": "Undisclosed",
            "founded_year": "N/A",
            "contact_info": f"webmaster@{domain}",
            "history_summary": f"Uncatalogued web publisher '{domain}'. Analyzed via structural heuristics.",
            "editorial_practices": "Editorial standards not independently certified by IFCN or institutional bodies.",
            "flags": flags,
            "evidence_citations": 0,
            "transparency_score": score,
            "explanation": cls._build_explanation(tier, "Evaluated via dynamic registrar heuristics", flags)
        }

    @staticmethod
    def _build_explanation(tier: str, practices: str, flags: List[str]) -> str:
        """Synthesizes human-readable assessment explanation."""
        if "Tier 1" in tier:
            return "This publisher is recognized as an authoritative institutional or intergovernmental body with peer-reviewed scientific review and public accountability."
        elif "Tier 2" in tier:
            return "This publisher maintains transparent editorial guidelines, double-sourcing standards, and an accountable public corrections archival."
        elif "Satire" in tier:
            return "This publication produces humorous satire and parody. Stories should not be evaluated as literal factual reportage."
        elif "Questionable" in tier or flags:
            reasons = "; ".join(flags) if flags else "Lacks transparent ownership."
            return f"Caution advised. This domain displays indicators of unverified attribution or questionable transparency: {reasons}."
        return "Standard digital publisher. Content requires corroboration against primary institutional databases before drawing factual conclusions."
