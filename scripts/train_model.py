"""
Master model training script.
Supports baseline TF-IDF + Logistic Regression training with automatic evaluation and persistence.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.classification.train_baseline import train_baseline_model


def main():
    parser = argparse.ArgumentParser(description="Train fake news classification models.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="baseline",
        choices=["baseline", "transformer"],
        help="Model architecture to train (default: baseline)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to directory containing processed parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/baseline",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum vocabulary features for TF-IDF (default: 10000)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for Logistic Regression (default: 1.0)",
    )

    args = parser.parse_args()

    print("=" * 65)
    print(f"  PHASE 2: TRAINING CLASSIFICATION MODEL [{args.model_type.upper()}]")
    print("=" * 65)

    if args.model_type == "baseline":
        metrics = train_baseline_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_features=args.max_features,
            C=args.C,
        )
        print(f"\nTraining completed successfully! Artifacts stored at: {Path(args.output_dir).resolve()}")
    else:
        print("Transformer training will be configured in subsequent phase.")


if __name__ == "__main__":
    main()
