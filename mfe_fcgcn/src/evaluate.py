"""
MFE-FCGCN — Evaluation Metrics
Paper: Multi-frequency EEG and multi-functional connectivity graph convolutional network
       based detection method of patients with Alzheimer's disease
DOI:   10.1007/s40747-025-01974-x

§4.2 — The paper reports: Accuracy, Sensitivity, Specificity, F1-score, AUC.

All metrics computed using sklearn (matching paper implementation).
Positive class = AD (label=1); negative class = HC (label=0).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class EvalMetrics:
    """
    §4.2 — Metrics reported in the paper's main results table.

    Attributes:
        accuracy:    Fraction of correctly classified samples.
        sensitivity: TP / (TP + FN) — recall for the AD (positive) class.
        specificity: TN / (TN + FP) — recall for the HC (negative) class.
        f1:          F1-score for the AD class (pos_label=1).
        auc:         Area under the ROC curve.
        loss:        Accumulated cross-entropy loss over the epoch / evaluation set.
    """
    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    loss: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: List[float],
    n_correct: int,
    n_total: int,
    total_loss: float,
) -> EvalMetrics:
    """
    §4.2 — Compute all reported evaluation metrics.

    Args:
        y_true:     Ground-truth labels (0 or 1).
        y_pred:     Predicted labels (argmax of softmax output).
        y_scores:   Predicted probability for the positive class (AD).
                    Used for AUC computation.
                    §4.2 — AUC uses the softmax score for class 1.
        n_correct:  Number of correctly classified samples.
        n_total:    Total number of samples in the set.
        total_loss: Accumulated loss sum over the epoch.

    Returns:
        EvalMetrics dataclass.
    """
    # §4.2 — confusion matrix to derive sensitivity & specificity
    # [UNSPECIFIED] Paper does not state what happens if one class is absent;
    # sklearn raises an error — ensure both classes are present in y_true.
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Degenerate fold — only one class present
        sensitivity = 0.0
        specificity = 0.0

    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # §4.2 — AUC computed from continuous softmax score (not hard label)
    auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0

    accuracy = n_correct / n_total if n_total > 0 else 0.0

    return EvalMetrics(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        f1=f1,
        auc=auc,
        loss=total_loss,
    )
