"""
Probability calibration utilities for classification models.
Ensures veracity probability scores are statistically calibrated for downstream decision policies.
"""

from typing import Any, Dict, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline


def calibrate_model(
    base_pipeline: Pipeline,
    X_val: Any,
    y_val: Any,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """
    Fits a CalibratedClassifierCV wrapper around a pre-fitted pipeline using validation data.
    method can be 'sigmoid' (Platt scaling) or 'isotonic'.
    """
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv="prefit",
    )
    calibrated_clf.fit(X_val, y_val)
    return calibrated_clf


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Brier score and calibration curve bins.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, y_prob))

    # Expected Calibration Error (ECE) approximation
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "prob_true": [round(float(x), 4) for x in prob_true],
        "prob_pred": [round(float(x), 4) for x in prob_pred],
    }
