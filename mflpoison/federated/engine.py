from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Optional, Sequence

import torch

from mflpoison.core.types import ClientUpdate


@dataclass
class RoundResult:
    round_index: int
    global_state: Mapping[str, torch.Tensor]
    updates: Sequence[ClientUpdate]


class FederatedEngine:
    """Single orchestration loop shared by clean, attack, and defense runs.

    ``client_runner`` is the compatibility boundary to the existing FDMM
    clients.  It receives ``(client_id, global_state, round_index)`` and must
    return a :class:`ClientUpdate`.
    """

    def __init__(
        self,
        client_runner: Callable[[str, Mapping[str, torch.Tensor], int], ClientUpdate],
        aggregator,
        update_filters: Optional[Iterable] = None,
        callbacks: Optional[Iterable] = None,
    ):
        self.client_runner = client_runner
        self.aggregator = aggregator
        self.update_filters = list(update_filters or [])
        self.callbacks = list(callbacks or [])

    def run_round(
        self,
        round_index: int,
        global_state: Mapping[str, torch.Tensor],
        client_ids: Iterable[str],
    ) -> RoundResult:
        updates: List[ClientUpdate] = [
            self.client_runner(str(client_id), global_state, int(round_index))
            for client_id in client_ids
        ]
        if not updates:
            raise ValueError("a federated round requires at least one client")
        filtered = updates
        for update_filter in self.update_filters:
            filtered = list(update_filter.apply(filtered, global_state))
            if not filtered:
                raise RuntimeError("all client updates were rejected by defenses")
        next_state = self.aggregator.aggregate(filtered, global_state)
        for callback in self.callbacks:
            callback.on_round_end(round_index, next_state, filtered)
        return RoundResult(int(round_index), next_state, filtered)
