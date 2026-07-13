from typing import Dict

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
