"""
Dataset Split & Processing Module for LIAR Benchmark.
Enriches raw splits with canonical 6-way and binary targets while strictly preserving
the official benchmark train, validation, and test splits.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

import pandas as pd

from src.data.label_mapping import map_to_6way_id, map_to_binary
from src.data.load_dataset import load_all_raw_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_and_save_splits(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> Dict[str, pd.DataFrame]:
    """
    Transforms raw splits, attaches 6-way and binary labels, and persists clean
    processed files in parquet and CSV formats.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    splits = load_all_raw_splits(raw_path)
    processed_splits = {}

    for split_name, df in splits.items():
        logger.info(f"Processing '{split_name}' split with {len(df)} records...")
        clean_df = df.copy()

        # Add target variables
        clean_df["label_6way_id"] = clean_df["label"].apply(map_to_6way_id)
        clean_df["label_binary"] = clean_df["label"].apply(map_to_binary)
        clean_df["label_binary_name"] = clean_df["label_binary"].map({1: "Fake", 0: "Real"})

        # Statement length metrics
        clean_df["char_length"] = clean_df["statement"].str.len()
        clean_df["word_count"] = clean_df["statement"].str.split().str.len()

        # Save individual split files
        parquet_file = proc_path / f"liar_{split_name}.parquet"
        csv_file = proc_path / f"liar_{split_name}.csv"

        try:
            clean_df.to_parquet(parquet_file, index=False)
            logger.info(f"Saved: {parquet_file}")
        except Exception as e:
            logger.warning(f"Could not save parquet ({e}), saving CSV...")

        clean_df.to_csv(csv_file, index=False)
        logger.info(f"Saved: {csv_file}")

        processed_splits[split_name] = clean_df

    # Also build a unified dataset with split preservation
    full_df = pd.concat(processed_splits.values(), ignore_index=True)
    full_parquet = proc_path / "liar_full.parquet"
    full_csv = proc_path / "liar_full.csv"

    try:
        full_df.to_parquet(full_parquet, index=False)
        logger.info(f"Saved unified benchmark: {full_parquet}")
    except Exception:
        pass

    full_df.to_csv(full_csv, index=False)
    logger.info(f"Saved unified benchmark: {full_csv}")

    logger.info(
        f"Processing complete. Preserved official splits: "
        f"Train={len(processed_splits['train'])}, "
        f"Valid={len(processed_splits['valid'])}, "
        f"Test={len(processed_splits['test'])}."
    )

    return processed_splits


if __name__ == "__main__":
    raw_p = Path("data/raw")
    proc_p = Path("data/processed")
    if len(sys.argv) > 1:
        raw_p = Path(sys.argv[1])
    if len(sys.argv) > 2:
        proc_p = Path(sys.argv[2])

    process_and_save_splits(raw_p, proc_p)
