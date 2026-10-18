"""
Retrieval package for evidence document indexing, embedding generation, and similarity search.
"""

from src.retrieval.document_indexer import build_and_save_index
from src.retrieval.embeddings import (
    DEFAULT_MODEL_NAME,
    EmbeddingGenerator,
    get_embedding_generator,
)
from src.retrieval.reranker import rerank_passages
from src.retrieval.search import EvidenceRetriever, determine_stance
from src.retrieval.source_loader import (
    create_passages_from_documents,
    load_evidence_corpus,
    load_reference_documents,
)
from src.retrieval.source_validation import (
    get_source_credibility_tier,
    validate_reference_corpus,
)
from src.retrieval.vector_store import (
    DEFAULT_INDEX_DIR,
    FAISSVectorStore,
)

__all__ = [
    "build_and_save_index",
    "EvidenceRetriever",
    "determine_stance",
    "EmbeddingGenerator",
    "get_embedding_generator",
    "DEFAULT_MODEL_NAME",
    "FAISSVectorStore",
    "DEFAULT_INDEX_DIR",
    "load_evidence_corpus",
    "load_reference_documents",
    "create_passages_from_documents",
    "get_source_credibility_tier",
    "validate_reference_corpus",
    "rerank_passages",
]
