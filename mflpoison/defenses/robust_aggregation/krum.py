from typing import Mapping, Sequence

import torch

from ..common import flatten_delta, global_model_state, update_delta
from .common import next_model_value, validate_updates


class Krum:
    """Select the update with the smallest neighborhood distance.

    Distances are computed on floating-point model deltas. The Byzantine bound
    ``f`` requires at least ``2f + 3`` participating clients.
    """

    name = "krum"

    def __init__(self, byzantine_clients: int = 1):
        if int(byzantine_clients) < 0:
            raise ValueError("byzantine_clients must be non-negative")
        self.byzantine_clients = int(byzantine_clients)

    def aggregate(
        self,
        updates: Sequence,
        global_state: Mapping[str, torch.Tensor],
    ):
        base = global_model_state(global_state)
        keys = validate_updates(updates, base)
        count = len(updates)
        f = self.byzantine_clients
        if count < 2 * f + 3:
            raise ValueError(
                f"Krum requires at least 2f+3 clients; got n={count}, f={f}"
            )

        vectors = torch.stack(
            [flatten_delta(update, base) for update in updates]
        )
        distances = torch.cdist(vectors, vectors, p=2).square()
        neighbor_count = count - f - 2
        scores = []
        for index in range(count):
            without_self = torch.cat(
                (distances[index, :index], distances[index, index + 1 :])
            )
            scores.append(torch.topk(without_self, neighbor_count, largest=False).values.sum())
        selected = updates[int(torch.argmin(torch.stack(scores)).item())]

        selected_delta = update_delta(selected, base)
        return {
            key: next_model_value(selected_delta[key], base, key)
            for key in keys
        }
