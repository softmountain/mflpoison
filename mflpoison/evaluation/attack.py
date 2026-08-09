from typing import Dict, Iterable, Sequence

import torch


def federated_attack_metrics(
    clean_truth: torch.Tensor,
    clean_pred: torch.Tensor,
    attack_pred: torch.Tensor,
    attack_targets: torch.Tensor,
) -> Dict[str, float]:
    clean_truth = clean_truth.view(-1)
    clean_pred = clean_pred.view(-1)
    attack_pred = attack_pred.view(-1)
    attack_targets = attack_targets.view(-1)
    if clean_truth.shape != clean_pred.shape:
        raise ValueError("clean predictions and labels must match")
    if attack_pred.shape != attack_targets.shape:
        raise ValueError("attack predictions and targets must match")
    return {
        "clean_accuracy": float((clean_pred == clean_truth).float().mean()),
        "targeted_asr": float((attack_pred == attack_targets).float().mean()),
    }


def _macro_f1_percent(truth: Sequence[int], predictions: Sequence[int]) -> float:
    labels = sorted(set(int(item) for item in truth))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(truth, predictions)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(truth, predictions)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(truth, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores) * 100.0


def targeted_classification_metrics(
    truth: Iterable[int],
    predictions: Iterable[int],
    *,
    victim_class: int,
    goal_class: int,
) -> Dict[str, float]:
    """Compute targeted  source-to-goal metrics from one test prediction set."""

    truth_values = [int(item) for item in truth]
    prediction_values = [int(item) for item in predictions]
    if len(truth_values) != len(prediction_values):
        raise ValueError("truth and predictions must have equal length")
    victim_class = int(victim_class)
    goal_class = int(goal_class)

    source_pairs = [
        (actual, predicted)
        for actual, predicted in zip(truth_values, prediction_values)
        if actual == victim_class
    ]
    non_source_pairs = [
        (actual, predicted)
        for actual, predicted in zip(truth_values, prediction_values)
        if actual != victim_class
    ]
    goal_negative_pairs = [
        (actual, predicted)
        for actual, predicted in zip(truth_values, prediction_values)
        if actual != goal_class
    ]

    metrics: Dict[str, float] = {
        "attack_source_sample_count": float(len(source_pairs)),
        "goal_class_negative_sample_count": float(len(goal_negative_pairs)),
        "non_source_sample_count": float(len(non_source_pairs)),
    }
    if source_pairs:
        attack_success_count = sum(
            predicted == goal_class for _, predicted in source_pairs
        )
        attack_success_rate = attack_success_count / len(source_pairs)
        source_accuracy = sum(
            predicted == victim_class for _, predicted in source_pairs
        ) / len(source_pairs)
        metrics.update(
            {
                "attack_success_count": float(attack_success_count),
                "attack_success_rate": float(attack_success_rate),
                "attack_success_rate_pct": float(attack_success_rate * 100.0),
                "source_class_accuracy": float(source_accuracy * 100.0),
                "source_class_recall": float(source_accuracy * 100.0),
            }
        )
    if goal_negative_pairs:
        metrics["goal_class_false_positive_rate"] = float(
            sum(predicted == goal_class for _, predicted in goal_negative_pairs)
            / len(goal_negative_pairs)
            * 100.0
        )
    if non_source_pairs:
        non_source_truth = [actual for actual, _ in non_source_pairs]
        non_source_predictions = [predicted for _, predicted in non_source_pairs]
        metrics["non_source_accuracy"] = float(
            sum(
                actual == predicted
                for actual, predicted in zip(
                    non_source_truth, non_source_predictions
                )
            )
            / len(non_source_pairs)
            * 100.0
        )
        metrics["non_source_macro_f1"] = _macro_f1_percent(
            non_source_truth, non_source_predictions
        )
    return metrics
