from math import isfinite, nan
from typing import Dict, Iterable, Optional, Sequence

import torch


def _as_list(values: Iterable) -> list:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).tolist()
    return list(values)


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return nan

    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def detection_metrics(
    labels: Iterable,
    scores: Iterable,
    *,
    predictions: Optional[Iterable] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary malicious-client detection metrics.

    Labels use ``1`` for malicious clients and higher scores must indicate a
    greater anomaly likelihood. AUROC is NaN when only one class is present.
    """

    label_values = [int(value) for value in _as_list(labels)]
    score_values = [float(value) for value in _as_list(scores)]
    if not label_values:
        raise ValueError("at least one detection label is required")
    if len(label_values) != len(score_values):
        raise ValueError("labels and scores must have equal length")
    if any(not isfinite(value) for value in score_values):
        raise ValueError("detection scores must be finite")
    if any(value not in (0, 1) for value in label_values):
        raise ValueError("detection labels must be binary 0/1 values")
    if predictions is None:
        prediction_values = [int(score >= float(threshold)) for score in score_values]
    else:
        prediction_values = [int(value) for value in _as_list(predictions)]
        if len(prediction_values) != len(label_values):
            raise ValueError("predictions and labels must have equal length")
        if any(value not in (0, 1) for value in prediction_values):
            raise ValueError("predictions must be binary 0/1 values")

    pairs = list(zip(label_values, prediction_values))
    tp = sum(label == 1 and prediction == 1 for label, prediction in pairs)
    fp = sum(label == 0 and prediction == 1 for label, prediction in pairs)
    tn = sum(label == 0 and prediction == 0 for label, prediction in pairs)
    fn = sum(label == 1 and prediction == 0 for label, prediction in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auroc": _binary_auroc(label_values, score_values),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
    }
