"""
Embedding Generation module for evidence retrieval.
Uses SentenceTransformers (all-MiniLM-L6-v2) with L2 normalization for cosine similarity search.
"""

import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingGenerator:
    """
    Wrapper for SentenceTransformers embedding model.
    Encodes text into normalized dense vector representations for FAISS vector search.
    """

    _instance: Optional["EmbeddingGenerator"] = None
    _model = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, use_mock: bool = False):
        self.model_name = model_name
        self.use_mock = use_mock
        self.dimension = EMBEDDING_DIMENSION

        if not self.use_mock and EmbeddingGenerator._model is None:
            self._load_model()

    def _load_model(self):
        """Loads SentenceTransformer model weights."""
        logger.info(f"Loading SentenceTransformer model '{self.model_name}' on CPU...")
        from sentence_transformers import SentenceTransformer

        EmbeddingGenerator._model = SentenceTransformer(self.model_name, device="cpu")
        logger.info(f"SentenceTransformer '{self.model_name}' loaded successfully.")

    @property
    def model(self):
        if self.use_mock:
            return None
        if EmbeddingGenerator._model is None:
            self._load_model()
        return EmbeddingGenerator._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encodes one or more texts into dense float32 vector embeddings.

        Args:
            texts: Single string or list of text strings.
            batch_size: Inference batch size.
            normalize: If True, L2-normalizes vectors so dot product equals cosine similarity.
            show_progress_bar: Whether to show tqdm progress bar.

        Returns:
            np.ndarray of shape (N, dimension) and dtype float32.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        if not text_list:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.use_mock:
            # Deterministic mock vectors for testing without loading model weights
            embeddings = []
            for t in text_list:
                # Seed hash based on text length and character sum
                np.random.seed(abs(hash(t)) % (2**31))
                vec = np.random.randn(self.dimension).astype(np.float32)
                if normalize:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                embeddings.append(vec)
            result = np.array(embeddings, dtype=np.float32)
        else:
            result = self.model.encode(
                text_list,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            ).astype(np.float32)

        return result[0] if is_single else result

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a 1D float32 normalized vector."""
        emb = self.encode(query, normalize=True)
        return emb.reshape(1, -1)


_default_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator(
    model_name: str = DEFAULT_MODEL_NAME, use_mock: bool = False
) -> EmbeddingGenerator:
    """Singleton getter for the default EmbeddingGenerator."""
    global _default_generator
    if _default_generator is None or _default_generator.use_mock != use_mock:
        _default_generator = EmbeddingGenerator(model_name=model_name, use_mock=use_mock)
    return _default_generator
