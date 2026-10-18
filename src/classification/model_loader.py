"""
Model persistence and loading utilities using Joblib.
Manages saving and loading of the TF-IDF vectorizer and trained Logistic Regression model.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_DIR = Path("artifacts/models/baseline")


def save_baseline_artifacts(
    pipeline: Pipeline,
    vectorizer: BaseEstimator,
    classifier: BaseEstimator,
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    output_dir: str | Path = DEFAULT_BASELINE_DIR,
) -> Path:
    """
    Saves the trained pipeline, individual components, metrics, and config using Joblib and JSON.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save integrated pipeline (enables single-step raw text inference)
    pipeline_file = out_path / "baseline_pipeline.joblib"
    joblib.dump(pipeline, pipeline_file)
    logger.info(f"Saved baseline pipeline: {pipeline_file}")

    # 2. Save individual components
    vec_file = out_path / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vec_file)

    clf_file = out_path / "logistic_regression.joblib"
    joblib.dump(classifier, clf_file)

    # 3. Save evaluation metrics
    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 4. Save metadata config
    if config:
        config_file = out_path / "model_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    logger.info(f"All baseline artifacts successfully saved to {out_path.resolve()}")
    return out_path


def load_baseline_pipeline(model_dir: str | Path = DEFAULT_BASELINE_DIR) -> Pipeline:
    """Loads the integrated TF-IDF + Logistic Regression pipeline from disk."""
    m_path = Path(model_dir)
    pipeline_file = m_path / "baseline_pipeline.joblib"

    if not pipeline_file.exists():
        raise FileNotFoundError(
            f"Baseline pipeline not found at '{pipeline_file}'. Run train_baseline.py first."
        )

    pipeline = joblib.load(pipeline_file)
    return pipeline


def load_baseline_components(
    model_dir: str | Path = DEFAULT_BASELINE_DIR,
) -> Tuple[BaseEstimator, BaseEstimator]:
    """Loads the vectorizer and classifier components separately."""
    m_path = Path(model_dir)
    vec_file = m_path / "tfidf_vectorizer.joblib"
    clf_file = m_path / "logistic_regression.joblib"

    if not vec_file.exists() or not clf_file.exists():
        raise FileNotFoundError(
            f"Model components not found in '{m_path}'. Run train_baseline.py first."
        )

    vectorizer = joblib.load(vec_file)
    classifier = joblib.load(clf_file)
    return vectorizer, classifier
