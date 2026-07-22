from typing import Mapping, Sequence

import torch

from ..common import global_model_state, update_delta
from .common import next_model_value, validate_updates


class CoordinateMedian:
    name = "coordinate_median"

    def aggregate(
        self,
        updates: Sequence,
        global_state: Mapping[str, torch.Tensor],
    ):
        base = global_model_state(global_state)
        keys = validate_updates(updates, base)
        result = {}
        for key in keys:
            if not (base[key].is_floating_point() or base[key].is_complex()):
                result[key] = base[key].clone()
                continue
            values = [update_delta(update, base)[key].to(base[key].device) for update in updates]
            stacked = torch.stack(values, dim=0)
            if stacked.is_complex():
                aggregate_delta = torch.complex(
                    stacked.real.median(dim=0).values,
                    stacked.imag.median(dim=0).values,
                )
            else:
                aggregate_delta = stacked.median(dim=0).values
            result[key] = next_model_value(aggregate_delta, base, key)
        return result
