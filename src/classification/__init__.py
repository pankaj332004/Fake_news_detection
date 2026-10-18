"""
Classification package for baseline ML and transformer models.
"""

from src.classification.calibration import calibrate_model, compute_calibration_metrics
from src.classification.evaluate import compute_metrics, format_metrics_table
from src.classification.model_loader import (
    DEFAULT_BASELINE_DIR,
    load_baseline_components,
    load_baseline_pipeline,
    save_baseline_artifacts,
)
from src.classification.predict import BaselineClassifier, predict_statement
from src.classification.train_baseline import build_baseline_pipeline, train_baseline_model

__all__ = [
    "train_baseline_model",
    "build_baseline_pipeline",
    "BaselineClassifier",
    "predict_statement",
    "compute_metrics",
    "format_metrics_table",
    "load_baseline_pipeline",
    "load_baseline_components",
    "save_baseline_artifacts",
    "calibrate_model",
    "compute_calibration_metrics",
    "DEFAULT_BASELINE_DIR",
]
