"""Runtime state normalization and deterministic seeding helpers."""

import hashlib
import math
import random
from typing import Any, Dict, Mapping

import torch


def cpu_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Keep only finite scalar metrics suitable for hashes and selection."""

    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[str(key)] = number
    return result


def client_round_seed(
    root_seed: int,
    client_id: str,
    round_index: int,
    phase: str,
) -> int:
    identity = f"{int(root_seed)}\0{phase}\0{round_index}\0{client_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") % (2**31)


def seed_runtime(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except ImportError:
        pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_loader(bundle, seed: int) -> None:
    loader = getattr(bundle, "dataloader", bundle)
    if not hasattr(loader, "dataset"):
        return
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if hasattr(loader, "generator"):
        loader.generator = generator
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "generator"):
        sampler.generator = generator
