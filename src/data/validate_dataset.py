"""
Dataset Validation Module for LIAR Benchmark.
Verifies integrity, schema adherence, label consistency, and checks for data leakage.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path for direct script execution
sys_root = Path(__file__).resolve().parent.parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

import pandas as pd

from src.data.label_mapping import SIX_WAY_LABELS
from src.data.load_dataset import COLUMN_NAMES, load_all_raw_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_splits(raw_dir: str | Path = "data/raw") -> Dict[str, Any]:
    """
    Validates train, valid, and test splits.
    Returns a comprehensive validation report dictionary.
    """
    splits = load_all_raw_splits(raw_dir)
    report: Dict[str, Any] = {"splits": {}, "leakage": {}, "status": "PASSED"}

    split_ids = {}

    for name, df in splits.items():
        logger.info(f"Validating '{name}' split (shape: {df.shape})...")

        # 1. Column verification
        missing_cols = [c for c in COLUMN_NAMES if c not in df.columns]
        if missing_cols:
            report["status"] = "FAILED"
            logger.error(f"Missing columns in {name}: {missing_cols}")

        # 2. Check empty statements or labels
        empty_statements = (df["statement"].str.len() == 0).sum()
        empty_labels = (df["label"].str.len() == 0).sum()

        # 3. Label validity
        unique_labels = set(df["label"].unique())
        invalid_labels = unique_labels - set(SIX_WAY_LABELS)
        if invalid_labels:
            report["status"] = "FAILED"
            logger.error(f"Invalid labels in {name}: {invalid_labels}")

        label_counts = df["label"].value_counts().to_dict()

        # 4. Track IDs
        split_ids[name] = set(df["id"].unique())

        report["splits"][name] = {
            "total_rows": len(df),
            "empty_statements": int(empty_statements),
            "empty_labels": int(empty_labels),
            "unique_speakers": int(df["speaker"].nunique()),
            "label_distribution": label_counts,
            "all_valid_labels": len(invalid_labels) == 0,
        }

    # 5. Check for Data Leakage (ID overlaps across splits)
    train_val_overlap = split_ids["train"].intersection(split_ids["valid"])
    train_test_overlap = split_ids["train"].intersection(split_ids["test"])
    val_test_overlap = split_ids["valid"].intersection(split_ids["test"])

    report["leakage"] = {
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
    }

    if (
        len(train_val_overlap) > 0
        or len(train_test_overlap) > 0
        or len(val_test_overlap) > 0
    ):
        report["status"] = "WARNING_LEAKAGE"
        logger.warning(f"ID overlap detected across splits: {report['leakage']}")
    else:
        logger.info("Zero data leakage detected: train, valid, and test sets are strictly disjoint.")

    return report


def print_validation_summary(report: Dict[str, Any]):
    """Prints a formatted human-readable summary of the validation report."""
    print("\n" + "=" * 65)
    print("           LIAR DATASET VALIDATION REPORT")
    print("=" * 65)
    print(f"Overall Status: {report['status']}")
    print("-" * 65)

    for split, info in report["splits"].items():
        print(f"\nSplit: {split.upper()} (Total Statements: {info['total_rows']:,})")
        print(f"  - Empty Statements: {info['empty_statements']}")
        print(f"  - Empty Labels:     {info['empty_labels']}")
        print(f"  - Unique Speakers:  {info['unique_speakers']:,}")
        print("  - Label Breakdown:")
        total = info["total_rows"]
        for lbl, count in info["label_distribution"].items():
            pct = (count / total) * 100
            print(f"      {lbl:<15}: {count:5d} ({pct:5.1f}%)")

    print("-" * 65)
    leak = report["leakage"]
    print("Data Leakage Check:")
    print(f"  - Train <-> Valid Overlap: {leak['train_val_overlap_count']}")
    print(f"  - Train <-> Test Overlap:  {leak['train_test_overlap_count']}")
    print(f"  - Valid <-> Test Overlap:  {leak['val_test_overlap_count']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    raw_path = Path("data/raw")
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1])
    rep = validate_splits(raw_path)
    print_validation_summary(rep)
