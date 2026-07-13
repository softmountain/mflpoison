from typing import Optional

import torch


def balanced_targets(num_samples: int, num_classes: int) -> torch.Tensor:
    if num_samples < 1 or num_classes < 1:
        raise ValueError("num_samples and num_classes must be positive")
    return torch.arange(int(num_samples), dtype=torch.long) % int(num_classes)


def clean_label_labels(target_labels: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(target_labels, dtype=torch.long).view(-1).clone()


def label_flip_labels(
    target_labels: torch.Tensor,
    source_class: Optional[int] = None,
    source_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    targets = torch.as_tensor(target_labels, dtype=torch.long).view(-1)
    if source_labels is not None:
        labels = torch.as_tensor(source_labels, dtype=torch.long).view(-1)
        if labels.shape[0] != targets.shape[0]:
            raise ValueError("source_labels must match target_labels")
        return labels.clone()
    if source_class is None or int(source_class) < 0:
        raise ValueError("label-flip attacks require source_class or source_labels")
    return torch.full_like(targets, int(source_class))
