from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from mflpoison.core.types import SyntheticBatch


class BaseGeneratorBackend(ABC):
    """Inference-facing generator contract independent of GAN trainer state."""

    family = "unknown"
    name = "unknown"

    def __init__(self, checkpoint_path, config, device="cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.device = torch.device(device)

    @property
    def num_classes(self) -> int:
        return int(self.config.num_classes)

    @abstractmethod
    def generate(
        self,
        target_labels: torch.Tensor,
        train_labels: Optional[torch.Tensor] = None,
        source_labels: Optional[torch.Tensor] = None,
        lengths: Optional[Mapping[str, torch.Tensor]] = None,
        batch_size: int = 64,
        seed: Optional[int] = None,
    ) -> SyntheticBatch:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        config = (
            self.config.to_dict()
            if hasattr(self.config, "to_dict")
            else dict(vars(self.config))
        )
        for key, value in tuple(config.items()):
            if isinstance(value, torch.device):
                config[key] = str(value)
        return {
            "generator_family": self.family,
            "generator_variant": self.name,
            "checkpoint": str(self.checkpoint_path),
            "config": config,
        }

    def _validate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
        if labels.numel() == 0:
            raise ValueError("at least one target label is required")
        if int(labels.min()) < 0 or int(labels.max()) >= self.num_classes:
            raise ValueError(
                f"target labels must be in [0, {self.num_classes - 1}]"
            )
        return labels

    @staticmethod
    def _output_labels(
        target_labels: torch.Tensor,
        train_labels: Optional[torch.Tensor],
        source_labels: Optional[torch.Tensor],
    ):
        train = target_labels.clone() if train_labels is None else torch.as_tensor(
            train_labels, dtype=torch.long
        ).view(-1)
        source = None if source_labels is None else torch.as_tensor(
            source_labels, dtype=torch.long
        ).view(-1)
        if train.shape[0] != target_labels.shape[0]:
            raise ValueError("train_labels must match target_labels")
        if source is not None and source.shape[0] != target_labels.shape[0]:
            raise ValueError("source_labels must match target_labels")
        return train, source
