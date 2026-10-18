"""
Dataset Download Module for Fake News Detection.
Downloads and extracts the official LIAR benchmark dataset (Wang, ACL 2017).
"""

import io
import logging
import sys
import zipfile
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
FALLBACK_RAW_URLS = {
    "train.tsv": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/train.tsv",
    "valid.tsv": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/valid.tsv",
    "test.tsv": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv",
}


def download_liar_dataset(output_dir: str | Path = "data/raw") -> Path:
    """
    Downloads the LIAR dataset from the primary benchmark archive or fallback mirror.
    Extracts train.tsv, valid.tsv, test.tsv, and README into output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    required_files = ["train.tsv", "valid.tsv", "test.tsv"]
    if all((out_path / f).exists() for f in required_files):
        logger.info(f"LIAR dataset files already present in {out_path.resolve()}")
        return out_path

    # Attempt primary archive download
    try:
        logger.info(f"Downloading official LIAR benchmark archive from {PRIMARY_URL}...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(PRIMARY_URL, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            namelist = z.namelist()
            logger.info(f"Archive contents: {namelist}")
            for filename in namelist:
                extracted_path = z.extract(filename, path=out_path)
                logger.info(f"Extracted: {extracted_path}")

        logger.info("Successfully extracted official LIAR dataset from primary archive.")
        return out_path

    except Exception as e:
        logger.warning(f"Primary download failed ({e}). Attempting fallback GitHub mirrors...")

    # Fallback to direct raw downloads
    for split_file, url in FALLBACK_RAW_URLS.items():
        dest = out_path / split_file
        if dest.exists():
            continue
        logger.info(f"Downloading {split_file} from {url}...")
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        dest.write_bytes(res.content)
        logger.info(f"Saved {dest.name} ({len(res.content)} bytes).")

    return out_path


if __name__ == "__main__":
    target_dir = Path("data/raw")
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    download_liar_dataset(target_dir)
