from typing import Mapping, Sequence

import torch

from mflpoison.core.types import ClientUpdate

from .common import preserve_nonfloating, validate_updates


class WeightedMean:
    name = "weighted_mean"

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
        global_state: Mapping[str, torch.Tensor],
    ):
        keys = validate_updates(updates)
        total = float(sum(update.num_samples for update in updates))
        result = {}
        for key in keys:
            values = [update.state[key] for update in updates]
            preserved = preserve_nonfloating(values, global_state, key)
            if preserved is not None:
                result[key] = preserved
                continue
            aggregated = torch.zeros_like(values[0])
            for update, value in zip(updates, values):
                aggregated.add_(value, alpha=float(update.num_samples) / total)
            result[key] = aggregated
        return result
