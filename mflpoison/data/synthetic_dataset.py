import math
import random
from collections.abc import Mapping
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from mflpoison.core.types import SyntheticBatch


def _first_present(data, names):
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


def _integral_vector(value, name):
    tensor = torch.as_tensor(value).view(-1)
    if tensor.is_floating_point():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor, tensor.round()):
            raise ValueError(f"{name} must contain finite integer values")
    return tensor.to(dtype=torch.long)


def canonical_synthetic_batch(
    batch: Union[SyntheticBatch, Mapping],
) -> SyntheticBatch:
    """Normalize canonical and historical synthetic artifact layouts.

    Legacy scripts used several aliases and sometimes omitted sequence lengths
    or conditioning labels. The compatibility rules live here so attack code
    can use one strict representation everywhere else.
    """

    if isinstance(batch, SyntheticBatch):
        raw = batch
    elif isinstance(batch, Mapping):
        if "features" in batch:
            raw = SyntheticBatch(
                features=dict(batch["features"]),
                lengths=dict(batch["lengths"]),
                condition_labels=batch["condition_labels"],
                train_labels=batch["train_labels"],
                source_labels=batch.get("source_labels"),
                metadata=dict(batch.get("metadata", {})),
            )
        else:
            audio = _first_present(batch, ("audio", "audio_features"))
            video = _first_present(batch, ("video", "video_features"))
            train_labels = _first_present(batch, ("train_label", "labels", "label"))
            if audio is None or video is None or train_labels is None:
                raise ValueError(
                    "legacy synthetic data requires audio, video, and labels"
                )
            audio = torch.as_tensor(audio)
            video = torch.as_tensor(video)
            train_labels = _integral_vector(train_labels, "train labels")
            condition_labels = _first_present(
                batch, ("condition_label", "condition_labels")
            )
            if condition_labels is None:
                condition_labels = train_labels.clone()
            audio_lengths = _first_present(batch, ("len_a", "audio_lengths"))
            video_lengths = _first_present(batch, ("len_v", "video_lengths"))
            if audio_lengths is None:
                audio_lengths = torch.full(
                    (audio.shape[0],), audio.shape[1], dtype=torch.long
                )
            if video_lengths is None:
                video_lengths = torch.full(
                    (video.shape[0],), video.shape[1], dtype=torch.long
                )
            raw = SyntheticBatch(
                features={"audio": audio, "video": video},
                lengths={"audio": audio_lengths, "video": video_lengths},
                condition_labels=condition_labels,
                train_labels=train_labels,
                source_labels=_first_present(
                    batch, ("source_label", "source_labels")
                ),
                metadata=dict(
                    _first_present(batch, ("meta", "metadata")) or {}
                ),
            )
    else:
        raise TypeError("synthetic data must be a SyntheticBatch or mapping")

    features = {name: torch.as_tensor(value) for name, value in raw.features.items()}
    lengths = {
        name: _integral_vector(value, f"{name} lengths")
        for name, value in raw.lengths.items()
    }
    normalized = SyntheticBatch(
        features=features,
        lengths=lengths,
        condition_labels=_integral_vector(
            raw.condition_labels, "condition labels"
        ),
        train_labels=_integral_vector(raw.train_labels, "train labels"),
        source_labels=(
            None
            if raw.source_labels is None
            else _integral_vector(raw.source_labels, "source labels")
        ),
        metadata=dict(raw.metadata),
    ).validate()

    if normalized.num_samples < 1:
        raise ValueError("synthetic batch must contain at least one sample")
    if set(normalized.lengths) != set(normalized.features):
        raise ValueError("lengths must match the synthetic feature modalities")
    for name, tensor in normalized.features.items():
        if tensor.ndim < 2:
            raise ValueError(f"synthetic feature {name} must include a batch axis")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"synthetic feature {name} contains NaN or Inf")
        modality_lengths = normalized.lengths[name]
        if torch.any(modality_lengths < 0):
            raise ValueError(f"{name} lengths cannot be negative")
        if tensor.ndim >= 3 and torch.any(modality_lengths > tensor.shape[1]):
            raise ValueError(f"{name} lengths exceed the feature sequence size")
    return normalized


class SyntheticFeatureDataset(Dataset):
    """Expose a canonical synthetic artifact to legacy FDMM clients."""

    def __init__(self, batch: Union[SyntheticBatch, Mapping]):
        self.batch = canonical_synthetic_batch(batch)
        required = {"audio", "video"}
        if not required.issubset(self.batch.features):
            raise ValueError("FDMM adapter requires audio and video features")

    def __len__(self):
        return self.batch.num_samples

    def __getitem__(self, index):
        return (
            self.batch.features["audio"][index],
            self.batch.features["video"][index],
            int(self.batch.lengths["audio"][index]),
            int(self.batch.lengths["video"][index]),
            self.batch.train_labels[index].long(),
        )


class MixedPoisonDataset(Dataset):
    """Deterministically replace or append an exact poison budget."""

    def __init__(
        self,
        clean_dataset: Dataset,
        poison_dataset: Dataset,
        poison_ratio: Optional[float] = None,
        seed: int = 42,
        length: Optional[int] = None,
        mode: str = "replace",
        poison_count: Optional[int] = None,
    ):
        mode = str(mode).lower()
        if mode not in {"replace", "append"}:
            raise ValueError("mode must be 'replace' or 'append'")
        if len(clean_dataset) < 1 or len(poison_dataset) < 1:
            raise ValueError("clean and poison datasets must be non-empty")
        self.clean_dataset = clean_dataset
        self.poison_dataset = poison_dataset
        self.mode = mode
        self.clean_count = len(clean_dataset) if length is None else int(length)
        if self.clean_count < 1:
            raise ValueError("mixed dataset length must be positive")
        if poison_count is not None and poison_ratio is not None:
            raise ValueError("set poison_count or poison_ratio, not both")
        if poison_count is None:
            if poison_ratio is None or not 0.0 <= float(poison_ratio) <= 1.0:
                raise ValueError("poison_ratio must be in [0, 1]")
            poison_count = int(math.floor(self.clean_count * float(poison_ratio) + 0.5))
        self.poison_count = int(poison_count)
        if self.poison_count < 0:
            raise ValueError("poison_count cannot be negative")
        if self.mode == "replace" and self.poison_count > self.clean_count:
            raise ValueError("replace poison_count cannot exceed clean dataset size")

        if self.mode == "replace":
            self.length = self.clean_count
            indices = list(range(self.length))
            random.Random(int(seed)).shuffle(indices)
            selected = indices[: self.poison_count]
            self.poison_indices = set(selected)
            self._poison_lookup = {
                clean_index: poison_index
                for poison_index, clean_index in enumerate(selected)
            }
        else:
            self.length = self.clean_count + self.poison_count
            self.poison_indices = set(range(self.clean_count, self.length))
            self._poison_lookup = {
                index: index - self.clean_count for index in self.poison_indices
            }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        if index in self.poison_indices:
            return self.poison_dataset[self._poison_lookup[index]]
        return self.clean_dataset[index % len(self.clean_dataset)]
