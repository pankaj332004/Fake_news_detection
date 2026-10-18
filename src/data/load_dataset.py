"""
Dataset Loading Utilities for the LIAR Benchmark.
Parses raw TSVs with robust handling of tab delimiters and metadata columns.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# The canonical 14 columns described in the ACL 2017 LIAR paper
COLUMN_NAMES: List[str] = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

CREDIT_COUNT_COLS = [
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
]


def load_raw_split(
    split: str,
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Loads a specific raw split ('train', 'valid', or 'test') into a pandas DataFrame.
    Uses quoting=3 (csv.QUOTE_NONE) to ensure all rows parse without split-delimiter issues.
    """
    data_path = Path(data_dir)
    filename = f"{split}.tsv"
    file_path = data_path / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw split file '{file_path}' not found. Run download_dataset.py first."
        )

    # Read tab-separated file with no quoting interpretation to prevent quotes breaking TSV columns
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=COLUMN_NAMES,
        quoting=3,
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    # Clean numeric count columns
    for col in CREDIT_COUNT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Clean string columns
    str_cols = [c for c in COLUMN_NAMES if c not in CREDIT_COUNT_COLS]
    for c in str_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()

    df["split"] = split
    return df


def load_all_raw_splits(data_dir: str | Path = "data/raw") -> Dict[str, pd.DataFrame]:
    """Loads all three official splits (train, valid, test) into a dictionary."""
    return {
        "train": load_raw_split("train", data_dir),
        "valid": load_raw_split("valid", data_dir),
        "test": load_raw_split("test", data_dir),
    }


def load_processed_split(
    split: str,
    data_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Loads a preprocessed split from data/processed/ (parquet or csv)."""
    p_path = Path(data_dir)
    parquet_path = p_path / f"liar_{split}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    csv_path = p_path / f"liar_{split}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Processed file not found in {p_path} for split '{split}'")
