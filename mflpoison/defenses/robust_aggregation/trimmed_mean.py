from typing import Mapping, Sequence

import torch

from ..common import global_model_state, update_delta
from .common import next_model_value, validate_updates


class TrimmedMean:
    name = "trimmed_mean"

    def __init__(self, trim_ratio: float = 0.2):
        if not 0.0 <= float(trim_ratio) < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = float(trim_ratio)

    def aggregate(
        self,
        updates: Sequence,
        global_state: Mapping[str, torch.Tensor],
    ):
        base = global_model_state(global_state)
        keys = validate_updates(updates, base)
        trim = int(len(updates) * self.trim_ratio)
        if 2 * trim >= len(updates):
            raise ValueError("trim ratio removes every client update")
        result = {}
        for key in keys:
            if not (base[key].is_floating_point() or base[key].is_complex()):
                result[key] = base[key].clone()
                continue
            values = [update_delta(update, base)[key].to(base[key].device) for update in updates]
            stacked = torch.stack(values, dim=0)
            if stacked.is_complex():
                real = stacked.real.sort(dim=0).values
                imag = stacked.imag.sort(dim=0).values
                selected_real = real[trim : len(updates) - trim] if trim else real
                selected_imag = imag[trim : len(updates) - trim] if trim else imag
                aggregate_delta = torch.complex(selected_real.mean(dim=0), selected_imag.mean(dim=0))
            else:
                sorted_values = stacked.sort(dim=0).values
                selected = sorted_values[trim : len(updates) - trim] if trim else sorted_values
                aggregate_delta = selected.mean(dim=0)
            result[key] = next_model_value(aggregate_delta, base, key)
        return result
