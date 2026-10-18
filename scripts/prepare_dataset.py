"""
Master Pipeline Script for Phase 1: Dataset Preparation.
Runs ingestion, schema validation, and official split creation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.data.download_dataset import download_liar_dataset
from src.data.split_dataset import process_and_save_splits
from src.data.validate_dataset import print_validation_summary, validate_splits


def main():
    print("=" * 65)
    print("  PHASE 1: LIAR BENCHMARK DATASET INGESTION & PREPARATION")
    print("=" * 65)

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    # Step 1: Download
    print("\n[Step 1/3] Downloading LIAR dataset...")
    download_liar_dataset(raw_dir)

    # Step 2: Validate
    print("\n[Step 2/3] Validating raw splits & checking data leakage...")
    report = validate_splits(raw_dir)
    print_validation_summary(report)

    if report["status"] == "FAILED":
        print("ERROR: Validation failed! Aborting processing.")
        sys.exit(1)

    # Step 3: Process and Preserve Splits
    print("\n[Step 3/3] Processing labels and preserving official splits...")
    process_and_save_splits(raw_dir, processed_dir)

    print("\n" + "=" * 65)
    print("  PHASE 1 COMPLETE: DATASET READY FOR BASELINE TRAINING")
    print(f"  Artifacts saved in: {processed_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
