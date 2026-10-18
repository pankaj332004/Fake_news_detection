"""
Inference module for the baseline classifier.
Runs locally on CPU with zero GPU dependencies.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

# Ensure project root is in sys.path
sys_root = Path(__file__).resolve().parent.parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from src.classification.model_loader import DEFAULT_BASELINE_DIR, load_baseline_pipeline

logger = logging.getLogger(__name__)


class BaselineClassifier:
    """
    Lightweight, CPU-only inference wrapper for TF-IDF + Logistic Regression.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_BASELINE_DIR):
        self.model_dir = Path(model_dir)
        self.pipeline = load_baseline_pipeline(self.model_dir)
        self.classes = {0: "Real", 1: "Fake"}

    def predict(
        self, text_or_texts: Union[str, List[str]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generates predictions and veracity probabilities for input statements.
        """
        is_single = isinstance(text_or_texts, str)
        statements = [text_or_texts] if is_single else list(text_or_texts)

        preds = self.pipeline.predict(statements)
        probs = self.pipeline.predict_proba(statements)

        results = []
        for text, pred, prob in zip(statements, preds, probs):
            pred_id = int(pred)
            label = self.classes.get(pred_id, "Unknown")
            prob_real = float(prob[0])
            prob_fake = float(prob[1])
            confidence = prob_fake if pred_id == 1 else prob_real

            results.append({
                "statement": text,
                "label": label,
                "label_id": pred_id,
                "confidence": round(confidence, 4),
                "prob_fake": round(prob_fake, 4),
                "prob_real": round(prob_real, 4),
            })

        return results[0] if is_single else results


def predict_statement(
    statement: str, model_dir: str | Path = DEFAULT_BASELINE_DIR
) -> Dict[str, Any]:
    """Convenience helper to predict a single statement."""
    clf = BaselineClassifier(model_dir)
    return clf.predict(statement)


if __name__ == "__main__":
    test_claim = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "The earth is flat and NASA admitted the moon landings were staged in a studio."
    )
    result = predict_statement(test_claim)
    print("\n--- BASELINE INFERENCE RESULT ---")
    print(f"Statement:  {result['statement']}")
    print(f"Verdict:    {result['label']} (Confidence: {result['confidence']*100:.1f}%)")
    print(f"P(Fake):    {result['prob_fake']:.4f}")
    print(f"P(Real):    {result['prob_real']:.4f}\n")
