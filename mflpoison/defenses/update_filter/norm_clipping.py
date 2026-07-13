from typing import Mapping, Sequence

import torch

from mflpoison.core.types import ClientUpdate


class NormClipper:
    """Clip each client's floating-point model delta to a global L2 norm."""

    name = "norm_clipping"

    def __init__(self, max_norm: float):
        if float(max_norm) <= 0:
            raise ValueError("max_norm must be positive")
        self.max_norm = float(max_norm)

    def apply(
        self,
        updates: Sequence[ClientUpdate],
        global_state: Mapping[str, torch.Tensor],
    ):
        clipped = []
        for update in updates:
            squared = None
            for key, value in update.state.items():
                if not value.is_floating_point():
                    continue
                delta = value - global_state[key].to(value.device)
                term = delta.double().pow(2).sum()
                squared = term if squared is None else squared + term
            norm = 0.0 if squared is None else float(squared.sqrt().cpu())
            scale = min(1.0, self.max_norm / max(norm, 1e-12))
            state = {}
            for key, value in update.state.items():
                if value.is_floating_point():
                    base = global_state[key].to(value.device)
                    state[key] = base + (value - base) * scale
                else:
                    state[key] = value.clone()
            metrics = dict(update.metrics)
            metrics["defense_update_norm"] = norm
            metrics["defense_clip_scale"] = scale
            clipped.append(
                ClientUpdate(
                    client_id=update.client_id,
                    state=state,
                    num_samples=update.num_samples,
                    metrics=metrics,
                    malicious=update.malicious,
                )
            )
        return clipped
