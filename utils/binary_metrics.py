"""Dependency-free binary metrics used to audit saved predictions."""

from __future__ import annotations

from math import isfinite
from typing import Sequence


def _validate(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> None:
    if not y_true or not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("binary metric inputs must be non-empty and aligned")
    if any(value not in (0, 1) for value in y_true):
        raise ValueError("y_true must be binary")
    if any(value not in (0, 1) for value in y_pred):
        raise ValueError("y_pred must be binary")
    if any(not isfinite(value) for value in y_score):
        raise ValueError("y_score contains NaN or infinity")
    if len(set(y_true)) != 2:
        raise ValueError("ROC-AUC/AUPRC require both classes")


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Mann-Whitney ROC-AUC with average ranks for tied scores."""
    ordered = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    positives = sum(y_true)
    negatives = len(y_true) - positives
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Trapezoidal precision-recall AUC matching sklearn's curve convention."""
    ordered = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(y_true)
    true_positive = false_positive = 0
    previous_recall = 0.0
    previous_precision = 1.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        true_positive += sum(label for _, label in group)
        false_positive += len(group) - sum(label for _, label in group)
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * (precision + previous_precision) / 2.0
        previous_recall = recall
        previous_precision = precision
        index = end
    return area


def binary_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]
) -> dict[str, float]:
    _validate(y_true, y_pred, y_score)
    total = len(y_true)
    true_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred == 1)
    true_negative = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred == 0)
    predicted_positive = sum(y_pred)
    actual_positive = sum(y_true)
    return {
        "Accuracy": (true_positive + true_negative) / total,
        "Precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "Recall": true_positive / actual_positive if actual_positive else 0.0,
        "AUC": roc_auc(y_true, y_score),
        "PRC": pr_auc(y_true, y_score),
    }

