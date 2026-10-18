"""
Evaluation utilities for fake news classification models.
Calculates Accuracy, Macro F1, Precision, Recall, and Confusion Matrix.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: List[int] | np.ndarray | pd.Series,
    y_pred: List[int] | np.ndarray | pd.Series,
    y_prob: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of evaluation metrics:
    - Accuracy
    - Precision (Macro & Weighted)
    - Recall (Macro & Weighted)
    - F1 Score (Macro & Weighted)
    - Confusion Matrix (as nested list)
    - ROC-AUC (for binary classification if probabilities are provided)
    - Full classification report dictionary
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()

    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm_list,
        "classification_report": report_dict,
    }

    # Binary ROC-AUC calculation
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            if y_prob.ndim == 2:
                prob_positive = y_prob[:, 1]
            else:
                prob_positive = y_prob
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_positive))
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def format_metrics_table(metrics: Dict[str, Any], title: str = "Evaluation Metrics") -> str:
    """Formats metrics as a clean markdown table."""
    lines = [
        f"### {title}",
        "",
        "| Metric | Score |",
        "| :--- | :---: |",
        f"| **Accuracy** | {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) |",
        f"| **Macro F1** | {metrics['f1_macro']:.4f} |",
        f"| **Weighted F1** | {metrics['f1_weighted']:.4f} |",
        f"| **Macro Precision** | {metrics['precision_macro']:.4f} |",
        f"| **Macro Recall** | {metrics['recall_macro']:.4f} |",
    ]
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        lines.append(f"| **ROC-AUC** | {metrics['roc_auc']:.4f} |")
    lines.append("")
    return "\n".join(lines)
