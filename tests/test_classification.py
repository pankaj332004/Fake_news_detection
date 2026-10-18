"""
Unit tests for Phase 2 Baseline Classification, Evaluation Metrics, and Inference.
"""

from pathlib import Path
import pytest
import numpy as np
from sklearn.pipeline import Pipeline

from src.classification.evaluate import compute_metrics
from src.classification.model_loader import (
    DEFAULT_BASELINE_DIR,
    load_baseline_components,
    load_baseline_pipeline,
)
from src.classification.predict import BaselineClassifier, predict_statement
from src.classification.train_baseline import build_baseline_pipeline


def test_build_pipeline():
    """Verify baseline pipeline structure and components."""
    pipe = build_baseline_pipeline(max_features=500)
    assert isinstance(pipe, Pipeline)
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_compute_metrics():
    """Verify evaluation metric calculations."""
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]
    y_prob = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.4, 0.6],
    ])

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, target_names=["Real", "Fake"])
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
    assert len(metrics["confusion_matrix"]) == 2


def test_model_artifacts_exist():
    """Verify that Joblib artifacts were created and can be loaded."""
    pipeline = load_baseline_pipeline(DEFAULT_BASELINE_DIR)
    assert pipeline is not None

    vec, clf = load_baseline_components(DEFAULT_BASELINE_DIR)
    assert vec is not None
    assert clf is not None


def test_baseline_inference():
    """Verify local CPU inference outputs."""
    clf = BaselineClassifier(DEFAULT_BASELINE_DIR)
    res = clf.predict("The government passed a new healthcare law.")
    assert "label" in res
    assert res["label"] in ["Fake", "Real"]
    assert "prob_fake" in res
    assert "prob_real" in res
    assert round(res["prob_fake"] + res["prob_real"], 2) == 1.0


def test_convenience_predict_statement():
    """Verify predict_statement helper function."""
    out = predict_statement("Breaking: aliens have landed in New York.")
    assert isinstance(out, dict)
    assert out["label_id"] in [0, 1]
    assert 0.0 <= out["confidence"] <= 1.0
