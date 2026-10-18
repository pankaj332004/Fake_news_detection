"""
Unit tests for Phase 4 Evidence Retrieval, FAISS Vector Search, and Stance Classification.
"""

from pathlib import Path
import pytest
import numpy as np

from src.preprocessing.chunking import chunk_text, split_into_sentences
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.reranker import rerank_passages
from src.retrieval.search import EvidenceRetriever, determine_stance
from src.retrieval.source_loader import (
    DEFAULT_EVIDENCE_PATH,
    create_passages_from_documents,
    load_reference_documents,
)
from src.retrieval.source_validation import (
    get_source_credibility_tier,
    validate_reference_corpus,
)
from src.retrieval.vector_store import DEFAULT_INDEX_DIR, FAISSVectorStore
from src.schemas.evidence import (
    EvidencePassage,
    RetrievalResponse,
    RetrievalResult,
    StanceType,
)


def test_split_into_sentences():
    """Verify sentence splitting respects boundaries."""
    text = "First sentence here. Second sentence starts now! Third sentence is a question?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence here."
    assert sentences[1] == "Second sentence starts now!"
    assert sentences[2] == "Third sentence is a question?"


def test_chunk_text():
    """Verify chunking produces coherent passages within size bounds."""
    text = (
        "The Earth orbits around the Sun once every 365.25 days. "
        "This orbit defines one calendar year on Earth. "
        "The axial tilt of approximately 23.5 degrees causes the annual progression of the seasons. "
        "Different parts of the planet receive varying amounts of sunlight throughout the year."
    )
    chunks = chunk_text(text, max_chunk_chars=120, overlap_chars=30)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) > 0


def test_reference_corpus_integrity():
    """Verify the default reference documents schema and uniqueness."""
    docs = load_reference_documents(DEFAULT_EVIDENCE_PATH)
    assert len(docs) >= 10
    report = validate_reference_corpus(docs)
    assert report["valid"] is True
    assert len(report["errors"]) == 0
    assert "health" in report["categories"]
    assert "space" in report["categories"]
    assert "climate" in report["categories"]


def test_create_passages_from_documents():
    """Verify conversion of reference documents into EvidencePassages with metadata."""
    sample_doc = [{
        "doc_id": "TEST-01",
        "title": "Test Title",
        "source": "Test Source",
        "source_url": "https://example.gov/test",
        "category": "science",
        "published_date": "2026-01-01",
        "text": "Sentence one is informative. Sentence two adds more context.",
    }]
    passages = create_passages_from_documents(sample_doc)
    assert len(passages) >= 1
    assert passages[0].doc_id == "TEST-01"
    assert passages[0].source == "Test Source"
    assert passages[0].source_url == "https://example.gov/test"


def test_embedding_generator_mock():
    """Verify embedding generator dimensions and normalization using mock mode."""
    embedder = EmbeddingGenerator(use_mock=True)
    vec = embedder.encode("Sample test claim", normalize=True)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Verify unit length L2 norm
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 1e-4) == 1.0


def test_faiss_vector_store_in_memory():
    """Verify FAISSVectorStore adding passages and searching with deterministic vectors."""
    store = FAISSVectorStore(dimension=384)

    passages = [
        EvidencePassage(
            passage_id="P1",
            doc_id="D1",
            text="NASA Apollo 11 landed on the moon in 1969.",
            source="NASA",
            source_url="https://nasa.gov",
        ),
        EvidencePassage(
            passage_id="P2",
            doc_id="D2",
            text="mRNA vaccines instruct cells to build a harmless spike protein.",
            source="WHO",
            source_url="https://who.int",
        ),
    ]

    # Create distinct orthogonal unit vectors
    emb1 = np.zeros(384, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(384, dtype=np.float32)
    emb2[1] = 1.0
    embs = np.vstack([emb1, emb2])

    store.add_passages(passages, embs)
    assert store.total_passages == 2

    # Query matching emb1
    q_vec = emb1.copy()
    results = store.search(q_vec, top_k=2)

    assert len(results) == 2
    top_passage, score = results[0]
    assert top_passage.doc_id == "D1"
    assert pytest.approx(score, 1e-4) == 1.0


def test_determine_stance_contradiction():
    """Verify stance detection identifies refuting evidence."""
    claim = "The moon landing was staged in a studio."
    passage = "Claims that the Apollo landings were staged are completely contradicted and refuted by rock samples and telemetry."
    stance, explanation = determine_stance(claim, passage)
    assert stance == StanceType.CONTRADICTS
    assert "refute" in explanation.lower() or "contradict" in explanation.lower()


def test_determine_stance_support():
    """Verify stance detection identifies corroborating evidence."""
    claim = "NASA radar detected water ice deposits on Mars."
    passage = "Orbital radar observations confirm that Mars possesses significant reserves of water ice beneath the surface."
    stance, explanation = determine_stance(claim, passage)
    assert stance == StanceType.SUPPORTS
    assert "confirm" in explanation.lower() or "support" in explanation.lower() or "align" in explanation.lower()


def test_source_credibility_tier():
    """Verify domain tier categorization."""
    assert "Tier 1" in get_source_credibility_tier("https://www.who.int/news/item")
    assert "Tier 1" in get_source_credibility_tier("https://mars.nasa.gov/facts")
    assert "Tier 2" in get_source_credibility_tier("https://state.gov/statement")
    assert "Tier 3" in get_source_credibility_tier("https://some-blog.xyz/news")


def test_rerank_passages():
    """Verify tier boost adjustment in reranker."""
    res1 = RetrievalResult(
        passage=EvidencePassage(
            passage_id="1", doc_id="1", text="Text 1", source="Source 1", source_url="https://blog.com"
        ),
        score=0.70,
        rank=1,
    )
    res2 = RetrievalResult(
        passage=EvidencePassage(
            passage_id="2", doc_id="2", text="Text 2", source="NASA", source_url="https://nasa.gov"
        ),
        score=0.68,
        rank=2,
    )

    reranked = rerank_passages([res1, res2], tier_boost=0.05)
    # res2 score becomes 0.68 + 0.05 = 0.73 > 0.70, so it moves to rank 1
    assert reranked[0].passage.source == "NASA"
    assert reranked[0].rank == 1


def test_evidence_retriever_saved_index():
    """Verify loading pre-built FAISS index from disk and running real queries."""
    retriever = EvidenceRetriever(index_dir=DEFAULT_INDEX_DIR)
    assert retriever.vector_store.total_passages >= 20

    # Query 1: Vaccine DNA claim
    res = retriever.retrieve("COVID-19 mRNA vaccines alter human DNA", top_k=3)
    assert isinstance(res, RetrievalResponse)
    assert res.total_retrieved > 0
    # Must retrieve WHO vaccine passage
    sources = [r.passage.source for r in res.all_passages]
    assert any("WHO" in s or "World Health" in s or "CDC" in s for s in sources)
    # Must identify contradiction
    assert len(res.contradicting_passages) > 0

    # Query 2: Water on Mars
    res_mars = retriever.retrieve("NASA discovered water ice reserves on Mars", top_k=3)
    assert res_mars.total_retrieved > 0
    top_src = res_mars.all_passages[0].passage.source
    assert "NASA" in top_src or "JPL" in top_src
    assert len(res_mars.supporting_passages) > 0
