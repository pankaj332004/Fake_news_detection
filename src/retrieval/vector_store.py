"""
FAISS Vector Store module for indexing and similarity search.
Stores dense vector embeddings with passage metadata and supports disk persistence.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import faiss
import numpy as np

from src.schemas.evidence import EvidencePassage

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("artifacts/vector_store")
FAISS_INDEX_FILENAME = "evidence_index.faiss"
METADATA_FILENAME = "evidence_passages.json"


class FAISSVectorStore:
    """
    FAISS-backed vector store for high-performance passage similarity search.
    Uses Inner Product (Cosine Similarity on L2-normalized vectors).
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.passages: List[EvidencePassage] = []

    @property
    def total_passages(self) -> int:
        return len(self.passages)

    def add_passages(
        self,
        passages: List[EvidencePassage],
        embeddings: np.ndarray,
    ) -> None:
        """
        Adds evidence passages and their corresponding float32 embeddings to the FAISS index.

        Args:
            passages: List of EvidencePassage objects.
            embeddings: Float32 numpy array of shape (N, dimension).
        """
        if len(passages) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(passages)} passages but {len(embeddings)} embeddings."
            )

        if len(passages) == 0:
            return

        emb_matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        if emb_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {emb_matrix.shape[1]}."
            )

        self.index.add(emb_matrix)
        self.passages.extend(passages)
        logger.info(f"Added {len(passages)} passages to FAISS index. Total: {self.total_passages}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_score: float = -1.0,
    ) -> List[Tuple[EvidencePassage, float]]:
        """
        Performs nearest-neighbor search for a query vector.

        Args:
            query_vector: 1D or 2D array of shape (dimension,) or (1, dimension).
            top_k: Number of nearest neighbors to retrieve.
            min_score: Minimum cosine similarity threshold.

        Returns:
            List of (EvidencePassage, score) tuples ordered from most to least similar.
        """
        if self.total_passages == 0:
            return []

        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        k = min(top_k, self.total_passages)
        scores, indices = self.index.search(q_vec, k)

        results: List[Tuple[EvidencePassage, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.passages):
                continue
            sim_score = float(score)
            if sim_score >= min_score:
                results.append((self.passages[idx], round(sim_score, 4)))

        return results

    def save(self, output_dir: Union[str, Path] = DEFAULT_INDEX_DIR) -> Path:
        """
        Persists FAISS index binary and passages metadata to output_dir.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Save FAISS binary
        index_file = out_path / FAISS_INDEX_FILENAME
        faiss.write_index(self.index, str(index_file))

        # 2. Save passages metadata
        metadata_file = out_path / METADATA_FILENAME
        passages_json = [p.model_dump() for p in self.passages]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(passages_json, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved FAISS index and {len(self.passages)} passages to {out_path.resolve()}")
        return out_path

    def load(self, index_dir: Union[str, Path] = DEFAULT_INDEX_DIR) -> bool:
        """
        Loads FAISS index and passages metadata from disk.
        """
        in_path = Path(index_dir)
        index_file = in_path / FAISS_INDEX_FILENAME
        metadata_file = in_path / METADATA_FILENAME

        if not index_file.exists() or not metadata_file.exists():
            logger.warning(f"Vector store artifacts not found at {in_path.resolve()}")
            return False

        # 1. Load FAISS binary
        self.index = faiss.read_index(str(index_file))
        self.dimension = self.index.d

        # 2. Load metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.passages = [EvidencePassage(**p) for p in data]

        logger.info(
            f"Loaded FAISS index ({self.total_passages} passages, dim={self.dimension}) from {in_path.resolve()}"
        )
        return True

    def clear(self) -> None:
        """Resets the vector index and passages collection."""
        self.index.reset()
        self.passages.clear()
