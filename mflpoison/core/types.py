from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch


TensorMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class DatasetSpec:
    """Shape and label contract shared by datasets and generators."""

    name: str
    num_classes: int
    modality_shapes: Mapping[str, tuple]
    split: str = "1"

    def __post_init__(self):
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if not self.modality_shapes:
            raise ValueError("at least one modality is required")
        for name, shape in self.modality_shapes.items():
            if not name or len(shape) != 2 or any(int(value) < 1 for value in shape):
                raise ValueError(
                    "modality_shapes must map names to positive (sequence, feature) pairs"
                )


@dataclass
class MultimodalBatch:
    """Dataset-independent batch with named modalities and sequence lengths."""

    features: Dict[str, torch.Tensor]
    lengths: Dict[str, torch.Tensor]
    labels: torch.Tensor

    def validate(self) -> "MultimodalBatch":
        if not self.features:
            raise ValueError("features cannot be empty")
        batch_size = int(self.labels.shape[0])
        for name, tensor in self.features.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(f"feature batch mismatch for {name}")
            if name not in self.lengths:
                raise ValueError(f"missing lengths for modality {name}")
            if self.lengths[name].shape[0] != batch_size:
                raise ValueError(f"length batch mismatch for {name}")
        return self


@dataclass
class SyntheticBatch:
    """Canonical artifact exchanged between generators and poisoning code.

    ``condition_labels`` describe what the generator was asked to synthesize.
    ``train_labels`` describe labels exposed to the victim training pipeline.
    Keeping them separate prevents clean-label and label-flip experiments from
    silently changing semantics.
    """

    features: Dict[str, torch.Tensor]
    lengths: Dict[str, torch.Tensor]
    condition_labels: torch.Tensor
    train_labels: torch.Tensor
    source_labels: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return int(self.condition_labels.shape[0])

    def validate(self) -> "SyntheticBatch":
        size = self.num_samples
        if self.train_labels.shape[0] != size:
            raise ValueError("condition_labels and train_labels must have equal length")
        if self.source_labels is not None and self.source_labels.shape[0] != size:
            raise ValueError("source_labels must match the synthetic batch size")
        if not self.features:
            raise ValueError("synthetic features cannot be empty")
        for name, tensor in self.features.items():
            if tensor.shape[0] != size:
                raise ValueError(f"feature batch mismatch for {name}")
            if name not in self.lengths:
                raise ValueError(f"missing lengths for modality {name}")
            if self.lengths[name].shape[0] != size:
                raise ValueError(f"length batch mismatch for {name}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "features": dict(self.features),
            "lengths": dict(self.lengths),
            "condition_labels": self.condition_labels,
            "train_labels": self.train_labels,
            "source_labels": self.source_labels,
            "metadata": dict(self.metadata),
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Return the field layout consumed by existing Local scripts."""

        self.validate()
        if "audio" not in self.features or "video" not in self.features:
            raise ValueError("legacy format requires audio and video modalities")
        result = {
            "audio": self.features["audio"],
            "video": self.features["video"],
            "len_a": self.lengths["audio"],
            "len_v": self.lengths["video"],
            "condition_label": self.condition_labels,
            "train_label": self.train_labels,
            "meta": dict(self.metadata),
            "schema_version": 1,
        }
        if self.source_labels is not None:
            result["source_label"] = self.source_labels
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SyntheticBatch":
        if "features" in data:
            return cls(
                features=dict(data["features"]),
                lengths=dict(data["lengths"]),
                condition_labels=data["condition_labels"],
                train_labels=data["train_labels"],
                source_labels=data.get("source_labels"),
                metadata=dict(data.get("metadata", {})),
            ).validate()
        return cls(
            features={"audio": data["audio"], "video": data["video"]},
            lengths={"audio": data["len_a"], "video": data["len_v"]},
            condition_labels=data["condition_label"],
            train_labels=data["train_label"],
            source_labels=data.get("source_label"),
            metadata=dict(data.get("meta", {})),
        ).validate()


@dataclass
class ClientUpdate:
    """One client's model state and provenance before server aggregation."""

    client_id: str
    state: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Dict[str, float] = field(default_factory=dict)
    malicious: bool = False

    def __post_init__(self):
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not self.state:
            raise ValueError("client state cannot be empty")
