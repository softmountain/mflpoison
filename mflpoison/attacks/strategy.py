from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Optional, Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from mflpoison.core.types import AttackSpec, SyntheticBatch
from mflpoison.data.synthetic_dataset import (
    MixedPoisonDataset,
    SyntheticFeatureDataset,
    canonical_synthetic_batch,
)


class InjectionMode(str, Enum):
    REPLACE = "replace"
    APPEND = "append"


@dataclass(frozen=True)
class PoisonedDataView:
    dataset: Dataset
    active: bool
    injection_mode: InjectionMode
    clean_sample_count: int
    poison_sample_count: int
    aggregation_sample_count: int
    synthetic: Optional[SyntheticBatch] = None

    @property
    def total_sample_count(self) -> int:
        return len(self.dataset)


@runtime_checkable
class AttackStrategy(Protocol):
    spec: AttackSpec

    def apply(
        self,
        clean_dataset: Dataset,
        generator_backend,
        round_index: int,
        lengths: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> PoisonedDataView:
        ...


class GenerativeFeaturePoisoningStrategy:
    """Generate and inject feature poisons inside one client's data boundary."""

    def __init__(
        self,
        spec: AttackSpec,
        seed: Optional[int] = None,
        generation_batch_size: int = 64,
        backend_factory=None,
    ):
        labels = (
            spec.condition_class,
            spec.assigned_train_label,
            spec.victim_eval_class,
            spec.goal_prediction_class,
        )
        if any(label is None for label in labels):
            raise ValueError(
                "generative feature poisoning requires all four class semantics"
            )
        if int(generation_batch_size) < 1:
            raise ValueError("generation_batch_size must be positive")
        self.spec = spec
        self.seed = int(spec.seed if seed is None else seed)
        self.generation_batch_size = int(generation_batch_size)
        self.backend_factory = backend_factory

    def apply(
        self,
        clean_dataset: Dataset,
        generator_backend,
        round_index: int,
        lengths: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> PoisonedDataView:
        clean_count = len(clean_dataset)
        if clean_count < 1:
            raise ValueError("clean client dataset must be non-empty")
        injection_mode = InjectionMode(self.spec.injection_mode)
        if not self.spec.active(round_index):
            return PoisonedDataView(
                dataset=clean_dataset,
                active=False,
                injection_mode=injection_mode,
                clean_sample_count=clean_count,
                poison_sample_count=0,
                aggregation_sample_count=clean_count,
            )

        poison_count = self._resolve_budget(clean_count)
        if poison_count == 0:
            return PoisonedDataView(
                dataset=clean_dataset,
                active=False,
                injection_mode=injection_mode,
                clean_sample_count=clean_count,
                poison_sample_count=0,
                aggregation_sample_count=clean_count,
            )
        if (
            injection_mode == InjectionMode.REPLACE
            and poison_count > clean_count
        ):
            raise ValueError("replace poison budget exceeds the clean partition")
        condition_labels = torch.full(
            (poison_count,), int(self.spec.condition_class), dtype=torch.long
        )
        train_labels = torch.full(
            (poison_count,), int(self.spec.assigned_train_label), dtype=torch.long
        )
        generation_seed = self.seed + int(round_index)
        backend = self._resolve_backend(generator_backend)
        synthetic = canonical_synthetic_batch(backend.generate(
            target_labels=condition_labels,
            train_labels=train_labels,
            lengths=lengths,
            batch_size=self.generation_batch_size,
            seed=generation_seed,
        ))
        if synthetic.num_samples != poison_count:
            raise ValueError("generator returned a different number of poison samples")
        if not torch.equal(synthetic.condition_labels.cpu(), condition_labels):
            raise ValueError("generator changed the requested condition_class")
        if not torch.equal(synthetic.train_labels.cpu(), train_labels):
            raise ValueError("generator changed the assigned_train_label")

        metadata = dict(synthetic.metadata)
        metadata["attack_semantics"] = {
            "condition_class": int(self.spec.condition_class),
            "assigned_train_label": int(self.spec.assigned_train_label),
            "victim_eval_class": int(self.spec.victim_eval_class),
            "goal_prediction_class": int(self.spec.goal_prediction_class),
            "injection_mode": injection_mode.value,
            "round_index": int(round_index),
        }
        synthetic = SyntheticBatch(
            features=synthetic.features,
            lengths=synthetic.lengths,
            condition_labels=synthetic.condition_labels,
            train_labels=synthetic.train_labels,
            source_labels=synthetic.source_labels,
            metadata=metadata,
        ).validate()
        mixed = MixedPoisonDataset(
            clean_dataset,
            SyntheticFeatureDataset(synthetic),
            seed=generation_seed,
            mode=injection_mode.value,
            poison_count=poison_count,
        )
        return PoisonedDataView(
            dataset=mixed,
            active=True,
            injection_mode=injection_mode,
            clean_sample_count=clean_count,
            poison_sample_count=poison_count,
            aggregation_sample_count=clean_count,
            synthetic=synthetic,
        )

    def prepare_dataloader(
        self,
        clean_bundle,
        backend_or_artifact,
        snapshot=None,
        round_index: int = 0,
        lengths: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        """Return a poisoned loader or a copy of a FedMM client bundle.

        ``snapshot`` is accepted for runner symmetry; generation provenance is
        already enforced by ``GeneratorArtifact`` and its lifecycle.
        """

        del snapshot
        clean_loader = getattr(clean_bundle, "dataloader", clean_bundle)
        if not hasattr(clean_loader, "dataset"):
            raise TypeError("clean_bundle must be a DataLoader or client data bundle")
        view = self.apply(
            clean_loader.dataset,
            backend_or_artifact,
            round_index=round_index,
            lengths=lengths,
        )
        if not view.active:
            return clean_bundle
        num_workers = int(clean_loader.num_workers)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.seed + int(round_index))
        loader = DataLoader(
            view.dataset,
            batch_size=clean_loader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=clean_loader.collate_fn,
            pin_memory=clean_loader.pin_memory,
            drop_last=clean_loader.drop_last,
            worker_init_fn=clean_loader.worker_init_fn,
            persistent_workers=(
                num_workers > 0
                and getattr(clean_loader, "persistent_workers", False)
            ),
            generator=loader_generator,
        )
        if hasattr(clean_bundle, "dataloader"):
            from dataclasses import replace

            return replace(clean_bundle, dataloader=loader)
        return loader

    def _resolve_budget(self, clean_count: int) -> int:
        if self.spec.poison_count is not None:
            return int(self.spec.poison_count)
        ratio = float(self.spec.poison_ratio)
        if ratio == 0.0:
            return 0
        return int(math.floor(clean_count * ratio + 0.5))

    def _resolve_backend(self, backend_or_artifact):
        if hasattr(backend_or_artifact, "generate"):
            return backend_or_artifact
        if self.backend_factory is None:
            raise TypeError(
                "an inference backend is required when no backend_factory is configured"
            )
        backend = self.backend_factory(backend_or_artifact)
        if not hasattr(backend, "generate"):
            raise TypeError("backend_factory must return a generator backend")
        return backend
