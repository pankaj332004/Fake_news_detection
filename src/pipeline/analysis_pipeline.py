"""
End-to-End Analysis Pipeline.
Orchestrates Phase 3 Ingestion (Claims, Headlines, Articles, URLs) with veracity prediction.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.classification.model_loader import DEFAULT_BASELINE_DIR
from src.classification.predict import BaselineClassifier
from src.ingestion.document_loader import load_input
from src.schemas.request import ArticleDocument, IngestionRequest, InputType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    End-to-end pipeline connecting document ingestion with veracity classification.
    """

    def __init__(self, model_dir: Union[str, Path] = DEFAULT_BASELINE_DIR):
        self.model_dir = Path(model_dir)
        self.classifier = BaselineClassifier(self.model_dir)

    def analyze(
        self,
        content_or_request: Union[str, IngestionRequest],
        input_type: Optional[Union[InputType, str]] = None,
        title: Optional[str] = None,
        allow_private: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingests user input (claim, headline, article body, or URL) and runs veracity prediction.

        Args:
            content_or_request: Input text, URL, or IngestionRequest object.
            input_type: Optional explicit input type.
            title: Optional title/headline.
            allow_private: Whether to allow private IP requests (for testing).

        Returns:
            Dictionary with ingestion document details and classification verdict.
        """
        # Step 1: Ingest and validate document
        doc: ArticleDocument = load_input(
            content_or_request=content_or_request,
            input_type=input_type,
            title=title,
            allow_private=allow_private,
        )

        # Step 2: Select text for prediction
        # For long articles, we evaluate the title/headline or the primary text
        text_to_evaluate = doc.title if doc.title and doc.input_type == InputType.URL else doc.text

        # Step 3: Run classification
        prediction = self.classifier.predict(text_to_evaluate)

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
            "verdict": prediction["label"],
            "label_id": prediction["label_id"],
            "confidence": prediction["confidence"],
            "prob_fake": prediction["prob_fake"],
            "prob_real": prediction["prob_real"],
        }


def analyze_input(
    content: str,
    input_type: Optional[str] = None,
    model_dir: Union[str, Path] = DEFAULT_BASELINE_DIR,
) -> Dict[str, Any]:
    """Convenience helper function to analyze any input."""
    pipeline = AnalysisPipeline(model_dir=model_dir)
    return pipeline.analyze(content_or_request=content, input_type=input_type)


def main():
    parser = argparse.ArgumentParser(description="Analyze a claim, headline, article, or URL for fake news.")
    parser.add_argument("input", type=str, help="Claim statement, article text, or URL to analyze.")
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["claim", "headline", "article", "url"],
        help="Explicitly specify input type (default: auto-detect)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      FAKE NEWS DETECTION — END-TO-END VERIFICATION")
    print("=" * 65)

    try:
        pipeline = AnalysisPipeline()
        result = pipeline.analyze(args.input, input_type=args.type)

        print(f"\n[Input Classification]: {result['input_type'].upper()}")
        if result['title']:
            print(f"[Title/Headline]:       {result['title']}")
        if result['source_domain']:
            print(f"[Source Domain]:        {result['source_domain']}")
        print(f"[Word Count]:           {result['word_count']} words")
        print(f"[Text Preview]:         {result['preview']}")
        print("-" * 65)
        print(f"[VERDICT]:              {result['verdict'].upper()} (Confidence: {result['confidence']*100:.1f}%)")
        print(f"[P(Fake)]:              {result['prob_fake']:.4f}")
        print(f"[P(Real)]:              {result['prob_real']:.4f}")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"\n[ERROR]: Ingestion or analysis failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
