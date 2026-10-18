"""
Source Loader module for reference evidence documents.
Loads authoritative documents, applies passage chunking, and constructs EvidencePassage instances.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from src.preprocessing.chunking import chunk_text
from src.schemas.evidence import EvidencePassage

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE_PATH = Path("data/evidence/reference_documents.json")


def load_reference_documents(
    file_path: Optional[Union[str, Path]] = None,
) -> List[dict]:
    """
    Loads raw reference documents from a JSON file.
    """
    path = Path(file_path or DEFAULT_EVIDENCE_PATH)
    if not path.exists():
        logger.warning(f"Evidence file not found at {path.resolve()}. Using empty corpus.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} reference documents from {path.resolve()}")
    return data


def create_passages_from_documents(
    documents: List[dict],
    max_chunk_chars: int = 400,
    overlap_chars: int = 60,
) -> List[EvidencePassage]:
    """
    Chunks a list of reference documents into discrete EvidencePassage objects with full metadata.
    """
    passages: List[EvidencePassage] = []

    for doc in documents:
        doc_id = doc.get("doc_id", "DOC-UNKNOWN")
        title = doc.get("title")
        source = doc.get("source", "Authoritative Source")
        source_url = doc.get("source_url")
        category = doc.get("category")
        published_date = doc.get("published_date")
        full_text = doc.get("text", "")

        chunks = chunk_text(
            full_text,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )

        for idx, chunk in enumerate(chunks):
            passage_id = f"{doc_id}#p{idx}"
            passages.append(
                EvidencePassage(
                    passage_id=passage_id,
                    doc_id=doc_id,
                    text=chunk,
                    title=title,
                    source=source,
                    source_url=source_url,
                    category=category,
                    published_date=published_date,
                    metadata={"chunk_index": idx, "total_chunks": len(chunks)},
                )
            )

    logger.info(f"Generated {len(passages)} passages from {len(documents)} documents.")
    return passages


def load_evidence_corpus(
    file_path: Optional[Union[str, Path]] = None,
    max_chunk_chars: int = 400,
    overlap_chars: int = 60,
) -> List[EvidencePassage]:
    """Convenience pipeline function to load and chunk reference documents."""
    docs = load_reference_documents(file_path)
    return create_passages_from_documents(
        docs, max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars
    )
