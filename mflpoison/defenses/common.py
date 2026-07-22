import inspect
from dataclasses import is_dataclass
from typing import Any, Dict, Mapping, Optional

import torch


def global_model_state(global_state: Any) -> Mapping[str, torch.Tensor]:
    """Return the model state from either a snapshot or a legacy mapping."""

    state = getattr(global_state, "state", global_state)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("global_state must be a non-empty tensor mapping or snapshot")
    return state


def global_snapshot_hash(global_state: Any) -> str:
    return str(getattr(global_state, "content_hash", "") or "")


def update_delta(
    update: Any,
    global_state: Any,
) -> Dict[str, torch.Tensor]:
    """Read an explicit delta or derive one from a legacy full model state."""

    base = global_model_state(global_state)
    effective_delta = getattr(update, "effective_delta", None)
    if callable(effective_delta):
        return dict(effective_delta(base))
    explicit = getattr(update, "delta", None)
    if isinstance(explicit, Mapping) and explicit:
        return dict(explicit)

    state = getattr(update, "state", None)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"client {getattr(update, 'client_id', '?')} has no model delta")

    delta = {}
    for key, base_value in base.items():
        if key not in state:
            raise ValueError(f"client update is missing parameter {key}")
        value = state[key]
        if value.is_floating_point() or value.is_complex():
            delta[key] = value - base_value.to(value.device)
        else:
            # Integer buffers are server-owned and are not aggregated.
            delta[key] = torch.zeros_like(base_value, device=value.device)
    return delta


def update_weight(update: Any) -> float:
    """Return the explicit aggregation weight with legacy sample fallbacks."""

    explicit = getattr(update, "aggregation_weight", None)
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    clean_examples = getattr(update, "num_clean_examples", None)
    if clean_examples is None:
        clean_examples = getattr(update, "clean_num_samples", None)
    if clean_examples is not None and int(clean_examples) > 0:
        return float(clean_examples)
    legacy = getattr(update, "num_samples", None)
    if legacy is not None and int(legacy) > 0:
        return float(legacy)
    raise ValueError(
        f"client {getattr(update, 'client_id', '?')} has no positive aggregation weight"
    )


def model_state_from_delta(
    delta: Mapping[str, torch.Tensor],
    global_state: Any,
) -> Dict[str, torch.Tensor]:
    base = global_model_state(global_state)
    state = {}
    for key, base_value in base.items():
        value = delta[key]
        if base_value.is_floating_point() or base_value.is_complex():
            state[key] = base_value + value.to(base_value.device)
        else:
            state[key] = base_value.clone()
    return state


def replace_update(
    update: Any,
    global_state: Any,
    *,
    delta: Optional[Mapping[str, torch.Tensor]] = None,
    metrics: Optional[Mapping[str, float]] = None,
    aggregation_weight: Optional[float] = None,
) -> Any:
    """Copy an update while supporting both old and new ClientUpdate schemas."""

    if not is_dataclass(update):
        raise TypeError("ClientUpdate compatibility requires a dataclass instance")
    parameters = inspect.signature(type(update)).parameters
    copied_delta = None
    if delta is not None:
        copied_delta = {
            key: value.detach().cpu().clone() for key, value in delta.items()
        }
    values: Dict[str, Any] = {
        "client_id": getattr(update, "client_id"),
        "round_index": getattr(update, "round_index", 0),
        "base_snapshot_hash": getattr(update, "base_snapshot_hash", "legacy"),
        "num_clean_examples": getattr(update, "num_clean_examples", None),
        "num_train_examples": getattr(update, "num_train_examples", None),
        "clean_num_samples": getattr(update, "clean_num_samples", None),
        "train_num_samples": getattr(update, "train_num_samples", None),
        "num_samples": getattr(update, "num_samples", None),
        "aggregation_weight": (
            float(aggregation_weight)
            if aggregation_weight is not None
            else getattr(update, "aggregation_weight", None)
        ),
        "metrics": (
            dict(metrics)
            if metrics is not None
            else dict(getattr(update, "metrics", {}) or {})
        ),
        "malicious": bool(getattr(update, "malicious", False)),
        "artifact_ids": list(getattr(update, "artifact_ids", []) or []),
    }
    if "delta" in parameters:
        values["delta"] = (
            copied_delta
            if copied_delta is not None
            else update_delta(update, global_state)
        )
    elif "state" in parameters:
        effective_delta = (
            copied_delta
            if copied_delta is not None
            else update_delta(update, global_state)
        )
        values["state"] = model_state_from_delta(effective_delta, global_state)
    kwargs = {
        name: values[name]
        for name in parameters
        if name in values and values[name] is not None
    }
    return type(update)(**kwargs)


def flatten_delta(
    update: Any,
    global_state: Any,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Flatten floating-point delta tensors in global model key order."""

    base = global_model_state(global_state)
    delta = update_delta(update, base)
    pieces = []
    for key, base_value in base.items():
        if not (base_value.is_floating_point() or base_value.is_complex()):
            continue
        value = delta[key].detach().cpu()
        if value.is_complex():
            value = torch.view_as_real(value)
        pieces.append(value.reshape(-1).to(dtype=dtype))
    if not pieces:
        raise ValueError("model update has no floating-point parameters")
    return torch.cat(pieces)


def update_l2_norm(update: Any, global_state: Any) -> float:
    return float(torch.linalg.vector_norm(flatten_delta(update, global_state)).item())
