from pathlib import Path

import torch

from mflpoison.core.types import SyntheticBatch


def save_synthetic(batch: SyntheticBatch, path, legacy: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = batch.to_legacy_dict() if legacy else batch.to_dict()
    torch.save(payload, path)
    return path


def load_synthetic(path, map_location="cpu") -> SyntheticBatch:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError("synthetic artifact must contain a dictionary")
    return SyntheticBatch.from_dict(payload)
