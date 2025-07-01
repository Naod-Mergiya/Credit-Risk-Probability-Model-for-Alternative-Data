"""Utility helpers for model training and evaluation."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn import metrics

__all__: Sequence[str] = ["compute_metrics"]

def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, float]:
    """Return common binary-classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred_proba : array-like of shape (n_samples,)
        Predicted probability of the positive class.
    threshold : float, default 0.5
        Cut-off used to convert probabilities to hard class labels.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_true, y_pred_proba),
    }