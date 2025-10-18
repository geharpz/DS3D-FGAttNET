"""
Metric utilities for violence detection classification tasks.

Provides static methods to compute essential classification metrics robustly, including
precision, recall, F1-score, accuracy, specificity, sensitivity, and AUC, ensuring safety
even when data is sparse or unbalanced.

Author
------
TFM VIU AI Thesis Assistant
"""

from typing import List, Dict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix
)


class MetricUtils:
    """
    Utility class to compute classification metrics for violence detection.

    Methods
    -------
    compute(y_true, y_pred, y_prob)
        Computes precision, recall, F1-score, accuracy, specificity, sensitivity, and ROC AUC.
    """

    @staticmethod
    def compute(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
        """
        Compute precision, recall, F1-score, accuracy, specificity, sensitivity, and ROC AUC.

        Parameters
        ----------
        y_true : List[int]
            Ground truth labels (expected 0 or 1).
        y_pred : List[int]
            Predicted labels (expected 0 or 1).
        y_prob : List[float]
            Predicted probabilities for the positive class (1).

        Returns
        -------
        Dict[str, float]
            Computed metrics.
        """
        if not y_true or not y_pred:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0
            }

        # Validate binary values
        assert all(label in [0, 1] for label in y_true), "y_true contains invalid labels."
        assert all(label in [0, 1] for label in y_pred), "y_pred contains invalid labels."

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else 0.0

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            specificity = 0.0
            sensitivity = 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "specificity": specificity,
            "sensitivity": sensitivity
        }
