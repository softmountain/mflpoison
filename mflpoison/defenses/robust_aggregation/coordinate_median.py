from typing import Mapping, Sequence

import torch

from mflpoison.core.types import ClientUpdate

from .common import preserve_nonfloating, validate_updates


class CoordinateMedian:
    name = "coordinate_median"

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
        global_state: Mapping[str, torch.Tensor],
    ):
        keys = validate_updates(updates)
        result = {}
        for key in keys:
            values = [update.state[key] for update in updates]
            preserved = preserve_nonfloating(values, global_state, key)
            if preserved is not None:
                result[key] = preserved
                continue
            result[key] = torch.stack(values, dim=0).median(dim=0).values
        return result
