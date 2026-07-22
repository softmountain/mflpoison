from typing import Any, Mapping, Sequence

import torch

from ..common import global_model_state, update_delta


def validate_updates(updates: Sequence[Any], global_state: Any):
    if not updates:
        raise ValueError("at least one client update is required")
    base = global_model_state(global_state)
    keys = tuple(base)
    expected_keys = set(keys)
    for update in updates:
        delta = update_delta(update, base)
        if set(delta) != expected_keys:
            raise ValueError("all client deltas must match the global model keys")
        for key in keys:
            if delta[key].shape != base[key].shape:
                raise ValueError(f"delta shape mismatch for {key}")
            if delta[key].dtype != base[key].dtype:
                raise ValueError(f"delta dtype mismatch for {key}")
    return keys


def next_model_value(
    aggregate_delta: torch.Tensor,
    reference: Mapping[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    base = reference[key]
    if not (base.is_floating_point() or base.is_complex()):
        return base.clone()
    return base + aggregate_delta.to(base.device)
