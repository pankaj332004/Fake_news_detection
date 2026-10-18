"""
Unit tests for Phase 1 Data Ingestion, Label Mapping, and Split Preservation.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data.label_mapping import (
    SIX_WAY_LABELS,
    map_to_6way_id,
    map_to_binary,
)
from src.data.load_dataset import (
    COLUMN_NAMES,
    load_all_raw_splits,
    load_processed_split,
)
from src.data.validate_dataset import validate_splits


def test_label_mappings():
    """Verify that 6-way and binary label mappings work as specified."""
    assert len(SIX_WAY_LABELS) == 6

    # 6-way IDs
    for idx, lbl in enumerate(SIX_WAY_LABELS):
        assert map_to_6way_id(lbl) == idx

    # Binary mappings
    assert map_to_binary("pants-fire") == 1
    assert map_to_binary("false") == 1
    assert map_to_binary("barely-true") == 1
    assert map_to_binary("mostly-true") == 0
    assert map_to_binary("true") == 0
    assert map_to_binary("half-true") == 0


def test_raw_splits_integrity():
    """Verify raw splits have expected columns and records."""
    splits = load_all_raw_splits()
    assert "train" in splits
    assert "valid" in splits
    assert "test" in splits

    for name, df in splits.items():
        assert len(df) > 0
        for col in COLUMN_NAMES:
            assert col in df.columns


def test_no_data_leakage():
    """Verify that train, valid, and test sets are strictly disjoint."""
    report = validate_splits()
    assert report["status"] == "PASSED"
    leak = report["leakage"]
    assert leak["train_val_overlap_count"] == 0
    assert leak["train_test_overlap_count"] == 0
    assert leak["val_test_overlap_count"] == 0


def test_processed_splits():
    """Verify processed data files load and contain enriched target fields."""
    train = load_processed_split("train")
    assert "label_6way_id" in train.columns
    assert "label_binary" in train.columns
    assert "label_binary_name" in train.columns
    assert train["label_binary"].isin([0, 1]).all()
    assert len(train) == 10269
