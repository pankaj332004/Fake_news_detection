"""
Master CLI script for real-world Fake News Analysis (Phase 3).
Accepts claims, headlines, full articles, and web URLs.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.pipeline.analysis_pipeline import AnalysisPipeline


def main():
    parser = argparse.ArgumentParser(description="Analyze a claim, headline, article, or URL for fake news.")
    parser.add_argument("input", type=str, help="Claim statement, raw article text, or web URL.")
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["claim", "headline", "article", "url"],
        help="Explicit input type (default: auto-detected)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      FAKE NEWS DETECTION — INPUT INGESTION & VERIFICATION")
    print("=" * 65)

    try:
        pipeline = AnalysisPipeline()
        result = pipeline.analyze(args.input, input_type=args.type)

        print(f"\n[Resolved Type]:   {result['input_type'].upper()}")
        if result['title']:
            print(f"[Headline/Title]:  {result['title']}")
        if result['source_domain']:
            print(f"[Source Domain]:   {result['source_domain']}")
        if result['authors']:
            print(f"[Authors]:         {', '.join(result['authors'])}")
        if result['published_date']:
            print(f"[Published Date]:  {result['published_date']}")
        print(f"[Length]:          {result['word_count']} words ({result['char_count']} characters)")
        print(f"[Preview]:         {result['preview']}")
        print("-" * 65)
        print(f"[VERDICT]:         {result['verdict'].upper()} (Confidence: {result['confidence']*100:.1f}%)")
        print(f"[P(Fake)]:         {result['prob_fake']:.4f}")
        print(f"[P(Real)]:         {result['prob_real']:.4f}")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"\n[ERROR]: Ingestion or analysis failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
