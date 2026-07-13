from typing import Mapping, Sequence

import torch

from mflpoison.core.types import ClientUpdate

from .common import preserve_nonfloating, validate_updates


class TrimmedMean:
    name = "trimmed_mean"

    def __init__(self, trim_ratio: float = 0.2):
        if not 0.0 <= float(trim_ratio) < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = float(trim_ratio)

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
        global_state: Mapping[str, torch.Tensor],
    ):
        keys = validate_updates(updates)
        trim = int(len(updates) * self.trim_ratio)
        if 2 * trim >= len(updates):
            raise ValueError("trim ratio removes every client update")
        result = {}
        for key in keys:
            values = [update.state[key] for update in updates]
            preserved = preserve_nonfloating(values, global_state, key)
            if preserved is not None:
                result[key] = preserved
                continue
            stacked = torch.stack(values, dim=0).sort(dim=0).values
            selected = stacked[trim : len(updates) - trim] if trim else stacked
            result[key] = selected.mean(dim=0)
        return result
