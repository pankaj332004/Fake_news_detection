"""
Master CLI script for real-world Fake News Analysis (Phase 3 & Phase 4).
Accepts claims, headlines, full articles, and web URLs, and retrieves supporting & contradicting evidence.
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
    parser = argparse.ArgumentParser(description="Analyze a claim, headline, article, or URL for fake news with evidence retrieval.")
    parser.add_argument("input", type=str, help="Claim statement, raw article text, or web URL.")
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["claim", "headline", "article", "url"],
        help="Explicit input type (default: auto-detected)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of evidence passages to retrieve (default: 3)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  FAKE NEWS DETECTION — INGESTION, VERACITY & EVIDENCE RETRIEVAL")
    print("=" * 65)

    try:
        pipeline = AnalysisPipeline()
        result = pipeline.analyze(args.input, input_type=args.type, top_k_evidence=args.top_k)

        print(f"\n[Resolved Input Type]: {result['input_type'].upper()}")
        if result['title']:
            print(f"[Headline/Title]:      {result['title']}")
        if result['source_domain']:
            print(f"[Source Domain]:       {result['source_domain']}")
        if result['authors']:
            print(f"[Authors]:             {', '.join(result['authors'])}")
        if result['published_date']:
            print(f"[Published Date]:      {result['published_date']}")
        print(f"[Length]:              {result['word_count']} words ({result['char_count']} characters)")
        print(f"[Preview]:             {result['preview']}")
        print("-" * 65)
        print(f"[STATISTICAL VERDICT]: {result['verdict'].upper()} (Confidence: {result['confidence']*100:.1f}%)")
        print(f"[P(Fake)]:             {result['prob_fake']:.4f}")
        print(f"[P(Real)]:             {result['prob_real']:.4f}")

        # Evidence Passages
        evidence = result.get("evidence", {})
        all_passages = evidence.get("all_passages", [])

        print("\n" + "=" * 65)
        print(f"  VERIFIED EVIDENCE RETRIEVAL ({len(all_passages)} passages found in {evidence.get('retrieval_time_ms', 0):.1f}ms)")
        print("=" * 65)

        if not all_passages:
            print("  No relevant evidence passages found above similarity threshold.")
        else:
            for item in all_passages:
                passage = item["passage"]
                stance = item.get("stance", "unspecified").upper()
                score = item.get("score", 0.0)

                icon = "[+]" if stance == "SUPPORTS" else ("[-]" if stance == "CONTRADICTS" else "[~]")
                print(f"\n{icon} Stance: {stance} | Relevance Score: {score*100:.1f}%")
                print(f"    Source:      {passage['source']}")
                if passage.get('source_url'):
                    print(f"    URL:         {passage['source_url']}")
                print(f"    Document:    {passage.get('title', 'N/A')}")
                print(f"    Passage:     \"{passage['text']}\"")

        print("\n" + "=" * 65 + "\n")

    except Exception as e:
        print(f"\n[ERROR]: Analysis failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
