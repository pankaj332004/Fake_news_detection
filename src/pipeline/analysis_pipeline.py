"""
End-to-End Analysis Pipeline.
Orchestrates Phase 3 Ingestion (Claims, Headlines, Articles, URLs),
Phase 4 Evidence Retrieval (FAISS vector store, supporting/contradicting passages),
and Phase 5 LLM Verification (evidence-grounded structured verification).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.classification.model_loader import DEFAULT_BASELINE_DIR
from src.classification.predict import BaselineClassifier
from src.ingestion.document_loader import load_input
from src.pipeline.result_builder import HybridResultBuilder
from src.retrieval.search import EvidenceRetriever
from src.retrieval.vector_store import DEFAULT_INDEX_DIR
from src.schemas.evidence import RetrievalResponse
from src.schemas.request import ArticleDocument, IngestionRequest, InputType
from src.verification.llm_loader import LLMBackend
from src.verification.llm_verifier import LLMVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    End-to-end pipeline connecting document ingestion, veracity classification,
    evidence passage retrieval, and LLM-based verification.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = DEFAULT_BASELINE_DIR,
        index_dir: Union[str, Path] = DEFAULT_INDEX_DIR,
        enable_retrieval: bool = True,
        enable_verification: bool = True,
        llm_model_name: str = "google/flan-t5-small",
        llm_backend: str = LLMBackend.MOCK,
    ):
        self.model_dir = Path(model_dir)
        self.index_dir = Path(index_dir)
        self.classifier = BaselineClassifier(self.model_dir)
        self.enable_retrieval = enable_retrieval
        self.enable_verification = enable_verification
        self._retriever: Optional[EvidenceRetriever] = None
        self._verifier: Optional[LLMVerifier] = None
        self.llm_model_name = llm_model_name
        self.llm_backend = llm_backend
        self.result_builder = HybridResultBuilder()

    @property
    def retriever(self) -> Optional[EvidenceRetriever]:
        if not self.enable_retrieval:
            return None
        if self._retriever is None:
            try:
                self._retriever = EvidenceRetriever(index_dir=self.index_dir)
            except Exception as e:
                logger.warning(f"Could not initialize EvidenceRetriever: {e}")
                self._retriever = None
        return self._retriever

    @property
    def verifier(self) -> Optional[LLMVerifier]:
        if not self.enable_verification:
            return None
        if self._verifier is None:
            try:
                self._verifier = LLMVerifier(
                    model_name=self.llm_model_name,
                    backend=self.llm_backend,
                    auto_load=True,
                )
            except Exception as e:
                logger.warning(f"Could not initialize LLMVerifier: {e}")
                self._verifier = None
        return self._verifier

    def analyze(
        self,
        content_or_request: Union[str, IngestionRequest],
        input_type: Optional[Union[InputType, str]] = None,
        title: Optional[str] = None,
        allow_private: bool = False,
        top_k_evidence: int = 3,
    ) -> Dict[str, Any]:
        """
        Ingests user input, runs veracity prediction, retrieves relevant evidence passages,
        and performs LLM-based verification.

        Args:
            content_or_request: Input text, URL, or IngestionRequest object.
            input_type: Optional explicit input type.
            title: Optional title/headline.
            allow_private: Whether to allow private IP requests (for testing).
            top_k_evidence: Number of evidence passages to retrieve.

        Returns:
            Dictionary with ingestion document details, classification verdict,
            evidence passages, and LLM verification results.
        """
        # Step 1: Ingest and validate document
        doc: ArticleDocument = load_input(
            content_or_request=content_or_request,
            input_type=input_type,
            title=title,
            allow_private=allow_private,
        )

        # Step 2: Select text for prediction
        text_to_evaluate = doc.title if doc.title and doc.input_type == InputType.URL else doc.text

        # Step 3: Run classification
        prediction = self.classifier.predict(text_to_evaluate)

        # Step 4: Evidence Retrieval
        evidence_data: Dict[str, Any] = {
            "total_retrieved": 0,
            "supporting_passages": [],
            "contradicting_passages": [],
            "all_passages": [],
        }
        retrieval_res: Optional[RetrievalResponse] = None

        if self.retriever and self.retriever.vector_store.total_passages > 0:
            try:
                retrieval_res = self.retriever.retrieve(
                    query=text_to_evaluate,
                    top_k=top_k_evidence,
                )
                evidence_data = {
                    "total_retrieved": retrieval_res.total_retrieved,
                    "retrieval_time_ms": retrieval_res.retrieval_time_ms,
                    "supporting_passages": [p.model_dump() for p in retrieval_res.supporting_passages],
                    "contradicting_passages": [p.model_dump() for p in retrieval_res.contradicting_passages],
                    "all_passages": [p.model_dump() for p in retrieval_res.all_passages],
                }
            except Exception as e:
                logger.warning(f"Evidence retrieval failed: {e}")

        # Step 5: LLM Verification (Phase 5)
        verification_data: Optional[Dict[str, Any]] = None
        if self.verifier:
            try:
                decision = self.verifier.verify(
                    claim=text_to_evaluate,
                    retrieval_response=retrieval_res,
                    statistical_verdict=prediction["label"],
                    statistical_confidence=prediction["confidence"],
                )
                verification_data = decision.to_dict()
            except Exception as e:
                logger.warning(f"LLM verification failed: {e}")
                verification_data = {
                    "final_verdict": "NEEDS_REVIEW",
                    "reasoning": f"LLM verification encountered an error: {e}",
                    "limitations": "Verification module failed — relying on statistical classification only.",
                }

        # Step 6: Hybrid Decision Policy & Result Synthesis (Phase 6)
        hybrid_res = self.result_builder.build(
            doc=doc,
            statistical_pred=prediction,
            retrieval_res=retrieval_res,
            verification_data=verification_data,
        )

        return {
            "status": "SUCCESS",
            "input_type": doc.input_type.value,
            "raw_input": doc.raw_input,
            "title": doc.title,
            "source_url": doc.source_url,
            "source_domain": doc.source_domain,
            "authors": doc.authors,
            "published_date": doc.published_date,
            "char_count": doc.char_count,
            "word_count": doc.word_count,
            "evaluated_text": text_to_evaluate,
            "preview": doc.get_analysis_snippet(max_words=60),
            # Backward-compatible keys
            "verdict": prediction["label"],
            "label_id": prediction["label_id"],
            "confidence": prediction["confidence"],
            "prob_fake": prediction["prob_fake"],
            "prob_real": prediction["prob_real"],
            "evidence": evidence_data,
            "verification": verification_data,
            # Phase 6 Hybrid decision & consensus keys
            "final_verdict": hybrid_res["final_verdict"],
            "hybrid_verdict": hybrid_res["verdict"],
            "badge_color": hybrid_res["badge_color"],
            "calibrated_confidence": hybrid_res["calibrated_confidence"],
            "uncertainty_level": hybrid_res["uncertainty_level"],
            "should_flag_for_review": hybrid_res["should_flag_for_review"],
            "consensus_status": hybrid_res["consensus_status"],
            "consensus_synthesis": hybrid_res["consensus_synthesis"],
            "hybrid_result": hybrid_res,
        }


def analyze_input(
    content: str,
    input_type: Optional[str] = None,
    model_dir: Union[str, Path] = DEFAULT_BASELINE_DIR,
    index_dir: Union[str, Path] = DEFAULT_INDEX_DIR,
    top_k_evidence: int = 3,
    enable_verification: bool = True,
    llm_backend: str = LLMBackend.MOCK,
) -> Dict[str, Any]:
    """Convenience helper function to analyze any input."""
    pipeline = AnalysisPipeline(
        model_dir=model_dir,
        index_dir=index_dir,
        enable_verification=enable_verification,
        llm_backend=llm_backend,
    )
    return pipeline.analyze(
        content_or_request=content,
        input_type=input_type,
        top_k_evidence=top_k_evidence,
    )


def _print_verification(verification: Optional[Dict[str, Any]]) -> None:
    """Pretty-print LLM verification results to console."""
    if not verification:
        print("\n[LLM VERIFICATION]: Disabled or unavailable.")
        return

    print("\n" + "-" * 65)
    print("[LLM VERIFICATION]")
    print("-" * 65)

    final = verification.get("final_verdict", "N/A")
    llm_v = verification.get("llm_verdict", "N/A")
    reasoning = verification.get("reasoning", "N/A")
    citations = verification.get("evidence_citations", [])
    contradictions = verification.get("contradictions_found", False)
    limitations = verification.get("limitations", "N/A")
    model = verification.get("model_name", "N/A")

    # Uncertainty info
    uncertainty = verification.get("uncertainty", {})
    overall_conf = uncertainty.get("overall_confidence", 0.0) if uncertainty else 0.0
    unc_level = uncertainty.get("uncertainty_level", "N/A") if uncertainty else "N/A"
    flagged = uncertainty.get("should_flag_for_review", False) if uncertainty else False

    # Contradiction analysis
    contra = verification.get("contradiction_analysis", {})
    evidence_status = contra.get("evidence_status", "N/A") if contra else "N/A"

    print(f"  Final Verdict:        {final}")
    print(f"  LLM Verdict:          {llm_v}")
    print(f"  Overall Confidence:   {overall_conf*100:.1f}%")
    print(f"  Uncertainty Level:    {unc_level.upper()}")
    print(f"  Evidence Status:      {evidence_status}")
    print(f"  Contradictions Found: {'Yes' if contradictions else 'No'}")
    if flagged:
        print(f"  ⚠  FLAGGED FOR HUMAN REVIEW")
    print(f"\n  Reasoning: {reasoning[:300]}")
    if citations:
        print(f"  Citations: {', '.join(citations)}")
    print(f"  Limitations: {limitations[:200]}")
    print(f"  Model: {model}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a claim, headline, article, or URL for fake news "
                    "with evidence retrieval and LLM verification."
    )
    parser.add_argument("input", type=str, help="Claim statement, article text, or URL to analyze.")
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["claim", "headline", "article", "url"],
        help="Explicitly specify input type (default: auto-detect)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum evidence passages to retrieve (default: 3)",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="mock",
        choices=["local", "hf_api", "mock"],
        help="LLM backend: 'local' (Flan-T5 on CPU), 'hf_api' (HuggingFace API), 'mock' (testing)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="google/flan-t5-small",
        help="HuggingFace model ID for LLM verification (default: google/flan-t5-small)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable LLM verification (only run classification + retrieval)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      FAKE NEWS DETECTION — END-TO-END VERIFICATION")
    print("=" * 65)

    try:
        pipeline = AnalysisPipeline(
            enable_verification=not args.no_verify,
            llm_model_name=args.llm_model,
            llm_backend=args.llm_backend,
        )
        result = pipeline.analyze(
            args.input,
            input_type=args.type,
            top_k_evidence=args.top_k,
        )

        print(f"\n[Input Classification]: {result['input_type'].upper()}")
        if result['title']:
            print(f"[Title/Headline]:       {result['title']}")
        if result['source_domain']:
            print(f"[Source Domain]:        {result['source_domain']}")
        print(f"[Word Count]:           {result['word_count']} words")
        print(f"[Text Preview]:         {result['preview']}")
        print("-" * 65)
        print(f"[STATISTICAL VERDICT]:  {result['verdict'].upper()} (Confidence: {result['confidence']*100:.1f}%)")
        print(f"[P(Fake)]:              {result['prob_fake']:.4f}")
        print(f"[P(Real)]:              {result['prob_real']:.4f}")

        # Display Evidence
        evidence = result.get("evidence", {})
        print("\n" + "-" * 65)
        print(f"[EVIDENCE RETRIEVAL]:   {evidence.get('total_retrieved', 0)} passages found")
        print("-" * 65)

        for item in evidence.get("all_passages", []):
            passage = item["passage"]
            stance = item.get("stance", "unspecified").upper()
            score = item.get("score", 0.0)
            print(f"\n* [{stance}] (Similarity: {score*100:.1f}%)")
            print(f"  Source:      {passage['source']} ({passage.get('source_url', 'N/A')})")
            print(f"  Document:    {passage.get('title', 'N/A')}")
            print(f"  Evidence:    \"{passage['text']}\"")

        # Display LLM Verification (Phase 5)
        _print_verification(result.get("verification"))

        print("=" * 65 + "\n")

    except Exception as e:
        print(f"\n[ERROR]: Ingestion or analysis failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
