"""
Baseline Model Training Module.
Builds TF-IDF features and trains a Logistic Regression classifier on the LIAR dataset.
Evaluates accuracy, macro F1, precision, recall, and confusion matrix, then saves artifacts with Joblib.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.classification.evaluate import compute_metrics, format_metrics_table
from src.classification.model_loader import DEFAULT_BASELINE_DIR, save_baseline_artifacts
from src.data.load_dataset import load_processed_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_baseline_pipeline(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 1.0,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """
    Constructs a scikit-learn Pipeline with TfidfVectorizer and LogisticRegression.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        stop_words="english",
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )

    classifier = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])
    return pipeline


def train_baseline_model(
    data_dir: str | Path = "data/processed",
    output_dir: str | Path = DEFAULT_BASELINE_DIR,
    target_column: str = "label_binary",
    max_features: int = 10000,
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Loads processed official splits, trains TF-IDF + Logistic Regression,
    evaluates on validation & test sets, and saves artifacts with Joblib.
    """
    logger.info("Loading processed LIAR splits...")
    train_df = load_processed_split("train", data_dir)
    valid_df = load_processed_split("valid", data_dir)
    test_df = load_processed_split("test", data_dir)

    logger.info(
        f"Splits loaded: Train={len(train_df):,}, Valid={len(valid_df):,}, Test={len(test_df):,}"
    )

    # Prepare statements and targets
    X_train = train_df["statement"].fillna("")
    y_train = train_df[target_column].astype(int)

    X_valid = valid_df["statement"].fillna("")
    y_valid = valid_df[target_column].astype(int)

    X_test = test_df["statement"].fillna("")
    y_test = test_df[target_column].astype(int)

    target_names = ["Real", "Fake"] if target_column == "label_binary" else None

    # Build and fit pipeline
    logger.info("Fitting TF-IDF Vectorizer and training Logistic Regression...")
    pipeline = build_baseline_pipeline(max_features=max_features, C=C)
    pipeline.fit(X_train, y_train)

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    vocab_size = len(vectorizer.vocabulary_)
    logger.info(f"Model trained successfully. TF-IDF Vocabulary Size: {vocab_size:,} terms.")

    # Validation evaluation
    y_valid_pred = pipeline.predict(X_valid)
    y_valid_prob = pipeline.predict_proba(X_valid)
    val_metrics = compute_metrics(y_valid, y_valid_pred, y_valid_prob, target_names=target_names)

    # Test evaluation
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, target_names=target_names)

    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "vocabulary_size": vocab_size,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }

    config = {
        "model_type": "TF-IDF + Logistic Regression",
        "target_column": target_column,
        "max_features": max_features,
        "ngram_range": [1, 2],
        "C": C,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "labels": target_names,
    }

    # Save artifacts with Joblib
    save_baseline_artifacts(
        pipeline=pipeline,
        vectorizer=vectorizer,
        classifier=classifier,
        metrics=all_metrics,
        config=config,
        output_dir=output_dir,
    )

    # Print evaluation summaries
    print("\n" + "=" * 65)
    print("      BASELINE MODEL EVALUATION: VALIDATION SPLIT")
    print("=" * 65)
    print(format_metrics_table(val_metrics, "Validation Performance"))
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"  {val_metrics['confusion_matrix']}")

    print("\n" + "=" * 65)
    print("        BASELINE MODEL EVALUATION: TEST SPLIT")
    print("=" * 65)
    print(format_metrics_table(test_metrics, "Test Performance"))
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"  {test_metrics['confusion_matrix']}")
    print("=" * 65 + "\n")

    return all_metrics


if __name__ == "__main__":
    train_baseline_model()
