from typing import Mapping, Sequence

import torch

from mflpoison.core.types import ClientUpdate


def validate_updates(updates: Sequence[ClientUpdate]):
    if not updates:
        raise ValueError("at least one client update is required")
    keys = tuple(updates[0].state)
    for update in updates[1:]:
        if tuple(update.state) != keys:
            raise ValueError("all client updates must contain the same ordered keys")
        for key in keys:
            if update.state[key].shape != updates[0].state[key].shape:
                raise ValueError(f"state shape mismatch for {key}")
    return keys


def preserve_nonfloating(values, reference: Mapping[str, torch.Tensor], key: str):
    if values[0].is_floating_point() or values[0].is_complex():
        return None
    return reference[key].clone()
