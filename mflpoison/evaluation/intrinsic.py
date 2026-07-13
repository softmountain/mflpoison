from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def kplus1_classification_metrics(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    num_classes: int,
    fake_class: int,
) -> Dict[str, float]:
    probabilities = F.softmax(logits, dim=1)
    target_labels = target_labels.view(-1, 1)
    pred_among_real = logits[:, :num_classes].argmax(dim=1)
    pred_all = logits.argmax(dim=1)
    targets = target_labels.view(-1)
    target_among_real = (pred_among_real == targets).float().mean()
    escape = (pred_all != int(fake_class)).float().mean()
    joint = (pred_all == targets).float().mean()
    return {
        "target_among_real_rate": float(target_among_real.cpu()),
        "discriminator_escape_rate": float(escape.cpu()),
        "joint_target_escape_rate": float(joint.cpu()),
        "target_prob": float(probabilities.gather(1, target_labels).mean().cpu()),
        "fake_prob": float(probabilities[:, int(fake_class)].mean().cpu()),
    }
