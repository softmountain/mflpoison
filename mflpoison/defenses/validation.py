from typing import Any, Mapping, Optional

import torch

from .common import (
    global_model_state,
    global_snapshot_hash,
    update_delta,
    update_weight,
)


class UpdateValidationError(ValueError):
    """Raised when a client update violates the server protocol."""


class UpdateValidator:
    """Validate update provenance and tensor compatibility before detection."""

    def __init__(self, require_base_hash: bool = True):
        self.require_base_hash = bool(require_base_hash)

    def validate(
        self,
        update: Any,
        global_state: Any,
        expected_base_snapshot_hash: Optional[str] = None,
    ) -> Any:
        base = global_model_state(global_state)
        expected_hash = (
            str(expected_base_snapshot_hash)
            if expected_base_snapshot_hash is not None
            else global_snapshot_hash(global_state)
        )
        actual_hash = str(getattr(update, "base_snapshot_hash", "") or "")
        if self.require_base_hash and not actual_hash:
            raise UpdateValidationError("base snapshot hash is required")
        if expected_hash and actual_hash != expected_hash:
            raise UpdateValidationError(
                f"base snapshot hash mismatch: expected {expected_hash}, got {actual_hash or '<missing>'}"
            )
        expected_round = getattr(global_state, "round_index", None)
        if expected_round is not None:
            actual_round = getattr(update, "round_index", None)
            if actual_round is None or int(actual_round) != int(expected_round):
                raise UpdateValidationError(
                    f"round index mismatch: expected {int(expected_round)}, "
                    f"got {actual_round if actual_round is not None else '<missing>'}"
                )
        try:
            delta = update_delta(update, base)
        except (KeyError, TypeError, ValueError) as exc:
            raise UpdateValidationError(str(exc)) from exc

        expected_keys = set(base)
        actual_keys = set(delta)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise UpdateValidationError(
                f"delta keys mismatch; missing={missing}, extra={extra}"
            )
        for key, base_value in base.items():
            value = delta[key]
            if not isinstance(value, torch.Tensor):
                raise UpdateValidationError(f"delta {key} is not a tensor")
            if value.shape != base_value.shape:
                raise UpdateValidationError(
                    f"delta shape mismatch for {key}: expected "
                    f"{tuple(base_value.shape)}, got {tuple(value.shape)}"
                )
            if value.dtype != base_value.dtype:
                raise UpdateValidationError(
                    f"delta dtype mismatch for {key}: expected {base_value.dtype}, got {value.dtype}"
                )
            if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all().item()
            ):
                raise UpdateValidationError(f"delta {key} contains NaN or Inf")
        try:
            update_weight(update)
        except ValueError as exc:
            raise UpdateValidationError(str(exc)) from exc
        return update

    def __call__(
        self,
        update: Any,
        global_state: Any,
        expected_base_snapshot_hash: Optional[str] = None,
    ) -> Any:
        return self.validate(update, global_state, expected_base_snapshot_hash)
