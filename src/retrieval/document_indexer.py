"""
Document Indexer module for constructing the FAISS evidence index.
Loads reference documents, computes dense sentence embeddings, and builds the vector index.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from src.retrieval.embeddings import DEFAULT_MODEL_NAME, get_embedding_generator
from src.retrieval.source_loader import (
    DEFAULT_EVIDENCE_PATH,
    load_evidence_corpus,
)
from src.retrieval.vector_store import DEFAULT_INDEX_DIR, FAISSVectorStore

logger = logging.getLogger(__name__)


def build_and_save_index(
    evidence_path: Optional[Union[str, Path]] = DEFAULT_EVIDENCE_PATH,
    output_dir: Union[str, Path] = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    use_mock: bool = False,
    max_chunk_chars: int = 400,
    overlap_chars: int = 60,
) -> FAISSVectorStore:
    """
    Builds a FAISS index from the reference evidence corpus and saves it to disk.

    Args:
        evidence_path: Path to reference_documents.json.
        output_dir: Destination directory for FAISS index and metadata.
        model_name: SentenceTransformers model name.
        use_mock: If True, uses synthetic embeddings (for fast offline testing).
        max_chunk_chars: Maximum character length for each passage.
        overlap_chars: Character overlap between consecutive passages.

    Returns:
        Populated FAISSVectorStore instance.
    """
    logger.info("=== Starting Evidence Vector Index Construction ===")

    # 1. Load and chunk reference documents
    passages = load_evidence_corpus(
        file_path=evidence_path,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
    )
    if not passages:
        raise ValueError(f"No evidence passages could be created from {evidence_path}.")

    # 2. Generate embeddings
    logger.info(f"Generating dense embeddings for {len(passages)} passages...")
    embedder = get_embedding_generator(model_name=model_name, use_mock=use_mock)
    texts = [p.text for p in passages]
    embeddings = embedder.encode(texts, normalize=True, show_progress_bar=True)

    # 3. Build FAISS index
    vector_store = FAISSVectorStore(dimension=embedder.dimension)
    vector_store.add_passages(passages=passages, embeddings=embeddings)

    # 4. Save to disk
    vector_store.save(output_dir=output_dir)
    logger.info(f"Successfully built and persisted FAISS index to {Path(output_dir).resolve()}")

    return vector_store
