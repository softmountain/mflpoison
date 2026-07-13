from typing import Iterable, Mapping, Optional, Protocol, Sequence

import torch

from .types import ClientUpdate, SyntheticBatch


class GeneratorBackend(Protocol):
    name: str
    family: str

    def generate(
        self,
        target_labels: torch.Tensor,
        train_labels: Optional[torch.Tensor] = None,
        source_labels: Optional[torch.Tensor] = None,
        lengths: Optional[Mapping[str, torch.Tensor]] = None,
        batch_size: int = 64,
        seed: Optional[int] = None,
    ) -> SyntheticBatch:
        ...

    def metadata(self) -> Mapping[str, object]:
        ...


class UpdateFilter(Protocol):
    def apply(
        self,
        updates: Sequence[ClientUpdate],
        global_state: Mapping[str, torch.Tensor],
    ) -> Sequence[ClientUpdate]:
        ...


class Aggregator(Protocol):
    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
        global_state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        ...


class FederatedCallback(Protocol):
    def on_round_end(
        self,
        round_index: int,
        global_state: Mapping[str, torch.Tensor],
        updates: Iterable[ClientUpdate],
    ) -> None:
        ...
