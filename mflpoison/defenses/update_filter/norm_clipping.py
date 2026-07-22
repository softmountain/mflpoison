from typing import Any, Mapping, Optional, Sequence

import torch

from ..common import replace_update, update_delta, update_l2_norm


class NormClipper:
    """Sanitize accepted deltas by clipping their global L2 norm."""

    name = "norm_clipping"

    def __init__(self, max_norm: Optional[float], mad_multiplier: float = 3.5):
        if max_norm is not None and float(max_norm) <= 0:
            raise ValueError("max_norm must be positive")
        if float(mad_multiplier) <= 0:
            raise ValueError("mad_multiplier must be positive")
        self.max_norm = None if max_norm is None else float(max_norm)
        self.mad_multiplier = float(mad_multiplier)

    def resolve_max_norm(self, updates: Sequence[Any], global_state: Any) -> float:
        if self.max_norm is not None:
            return self.max_norm
        if not updates:
            raise ValueError("at least one update is required to estimate a clip norm")
        norms = torch.tensor(
            [update_l2_norm(update, global_state) for update in updates],
            dtype=torch.float64,
        )
        median = norms.median()
        mad = (norms - median).abs().median()
        estimated = float(median + self.mad_multiplier * 1.4826 * mad)
        return max(estimated, torch.finfo(torch.float64).eps)

    def sanitize(
        self,
        update: Any,
        global_state: Any,
        *,
        max_norm: Optional[float] = None,
        aggregation_weight: Optional[float] = None,
    ) -> Any:
        effective_max_norm = (
            float(max_norm)
            if max_norm is not None
            else self.resolve_max_norm([update], global_state)
        )
        delta = update_delta(update, global_state)
        norm = update_l2_norm(update, global_state)
        scale = min(1.0, effective_max_norm / max(norm, 1e-12))
        clipped_delta = {}
        for key, value in delta.items():
            if value.is_floating_point() or value.is_complex():
                clipped_delta[key] = value * scale
            else:
                clipped_delta[key] = torch.zeros_like(value)
        metrics = dict(getattr(update, "metrics", {}) or {})
        metrics["defense_update_norm"] = norm
        metrics["defense_clip_norm"] = effective_max_norm
        metrics["defense_clip_scale"] = scale
        return replace_update(
            update,
            global_state,
            delta=clipped_delta,
            metrics=metrics,
            aggregation_weight=aggregation_weight,
        )

    def apply(
        self,
        updates: Sequence[Any],
        global_state: Mapping[str, torch.Tensor],
    ):
        effective_max_norm = self.resolve_max_norm(updates, global_state)
        return [
            self.sanitize(update, global_state, max_norm=effective_max_norm)
            for update in updates
        ]
