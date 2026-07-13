from typing import Mapping, Sequence

import torch

from mflpoison.core.types import ClientUpdate

from .common import validate_updates


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

    def _flatten_delta(self, update, global_state, keys):
        pieces = []
        for key in keys:
            value = update.state[key]
            if value.is_floating_point() or value.is_complex():
                delta = value - global_state[key].to(value.device)
                if delta.is_complex():
                    delta = torch.view_as_real(delta)
                pieces.append(delta.reshape(-1).to(dtype=torch.float64, device="cpu"))
        if not pieces:
            raise ValueError("Krum requires at least one floating-point parameter")
        return torch.cat(pieces)

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
        global_state: Mapping[str, torch.Tensor],
    ):
        keys = validate_updates(updates)
        count = len(updates)
        f = self.byzantine_clients
        if count < 2 * f + 3:
            raise ValueError(
                f"Krum requires at least 2f+3 clients; got n={count}, f={f}"
            )

        vectors = torch.stack(
            [self._flatten_delta(update, global_state, keys) for update in updates]
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

        return {
            key: (
                selected.state[key].clone()
                if selected.state[key].is_floating_point()
                or selected.state[key].is_complex()
                else global_state[key].clone()
            )
            for key in keys
        }
