from typing import Mapping, Sequence

import torch

from ..common import global_model_state, update_delta, update_weight
from .common import next_model_value, validate_updates


class WeightedMean:
    name = "weighted_mean"

    def aggregate(
        self,
        updates: Sequence,
        global_state: Mapping[str, torch.Tensor],
    ):
        base = global_model_state(global_state)
        keys = validate_updates(updates, base)
        weights = [update_weight(update) for update in updates]
        total = float(sum(weights))
        result = {}
        for key in keys:
            if not (base[key].is_floating_point() or base[key].is_complex()):
                result[key] = base[key].clone()
                continue
            values = [update_delta(update, base)[key] for update in updates]
            aggregated = torch.zeros_like(base[key])
            for weight, value in zip(weights, values):
                aggregated.add_(value.to(aggregated.device), alpha=weight / total)
            result[key] = next_model_value(aggregated, base, key)
        return result
