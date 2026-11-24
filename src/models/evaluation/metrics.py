"""Classification metrics; rows are truth, columns predictions in label order."""

import math
from numbers import Integral


def confusion_matrix(y_true, y_pred, *, labels):
    truth, predicted, order = list(y_true), list(y_pred), list(labels)
    if not truth or len(truth) != len(predicted):
        raise ValueError("Mismatched or empty predictions")
    if not order or any(not isinstance(v, (str, Integral)) for v in order):
        raise ValueError("Labels must be distinct integers or strings")
    if len(set(order)) != len(order):
        raise ValueError("Labels must be distinct")
    indices = {label: i for i, label in enumerate(order)}
    matrix = [[0] * len(order) for _ in order]
    for actual, prediction in zip(truth, predicted):
        if actual not in indices or prediction not in indices:
            raise ValueError(
                "Truth and predictions must belong to the configured labels"
            )
        matrix[indices[actual]][indices[prediction]] += 1
    return matrix


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def classification_metrics(y_true, y_pred, *, labels):
    labels = [int(v) if isinstance(v, Integral) else v for v in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        per_class.append(
            {
                "label": label,
                "support": support,
                "precision": _ratio(tp, predicted),
                "recall": _ratio(tp, support),
                "f1": _ratio(2 * tp, support + predicted),
            }
        )
    report = {
        "labels": labels,
        "confusion_matrix": matrix,
        "accuracy": sum(matrix[i][i] for i in range(len(labels)))
        / sum(map(sum, matrix)),
        "per_class": per_class,
    }
    for key in ("precision", "recall", "f1"):
        values = [entry[key] for entry in per_class if entry[key] is not None]
        report[f"macro_{key}"] = _ratio(sum(values), len(values))
    report["balanced_accuracy"] = report["macro_recall"]
    return report


def evaluate_binary(y_true, y_pred, scores=None):
    """Legacy clinical binary metric names, backed by the shared matrix."""
    y_true = list(y_true)
    report = classification_metrics(y_true, y_pred, labels=[-1, 1])
    (tn, fp), (fn, tp) = report["confusion_matrix"]
    recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    report.update(
        {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "precision_aki": _ratio(tp, tp + fp),
            "recall_aki_sensitivity": recall,
            "specificity": specificity,
            "f1_aki": _ratio(2 * tp, 2 * tp + fp + fn),
            "negative_predictive_value": _ratio(tn, tn + fn),
            "support_no_aki": tn + fp,
            "support_aki": tp + fn,
            "balanced_accuracy": (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None,
        }
    )
    if scores is not None:
        report["roc_auc"] = auc(y_true, scores)
    return report


def auc(y, scores):
    """Tie-aware binary rank AUC; scores need not be calibrated probabilities."""
    y, scores = list(y), [float(value) for value in scores]
    if not y or len(y) != len(scores):
        raise ValueError("Score length mismatch or empty truth")
    if any(value not in (-1, 1) for value in y):
        raise ValueError("Labels must be -1 or +1")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("Nonfinite scores")
    positives = sum(value == 1 for value in y)
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, y))
    sum_positive_ranks = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        sum_positive_ranks += (
            sum(value == 1 for _, value in ordered[i:j]) * average_rank
        )
        i = j
    return (sum_positive_ranks - positives * (positives + 1) / 2) / (
        positives * negatives
    )
