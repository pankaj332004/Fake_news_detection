"""
Master CLI script for Phase 4: Building the FAISS Evidence Index.
Loads authoritative reference documents, generates dense sentence embeddings, and builds the FAISS index.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.retrieval.document_indexer import build_and_save_index
from src.retrieval.source_loader import DEFAULT_EVIDENCE_PATH
from src.retrieval.vector_store import DEFAULT_INDEX_DIR


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index from reference documents.")
    parser.add_argument(
        "--evidence-path",
        type=str,
        default=str(DEFAULT_EVIDENCE_PATH),
        help="Path to reference_documents.json file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_INDEX_DIR),
        help="Directory to save FAISS index and metadata",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock embeddings for testing",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      PHASE 4: EVIDENCE RETRIEVAL — FAISS INDEX BUILDER")
    print("=" * 65)

    try:
        vector_store = build_and_save_index(
            evidence_path=args.evidence_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            use_mock=args.use_mock,
        )

        print("\n" + "=" * 65)
        print("  FAISS EVIDENCE INDEX BUILT SUCCESSFULLY!")
        print(f"  Total Passages Indexed: {vector_store.total_passages}")
        print(f"  Vector Dimension:       {vector_store.dimension}")
        print(f"  Index Directory:        {Path(args.output_dir).resolve()}")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Failed to build evidence index: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
