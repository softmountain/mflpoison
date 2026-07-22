from pathlib import Path
from typing import Any, Mapping

import torch

from mflpoison.core.types import GlobalSnapshot, ModelSpec


def save_snapshot(snapshot: GlobalSnapshot, path) -> Path:
    if not isinstance(snapshot, GlobalSnapshot):
        raise TypeError("snapshot must be a GlobalSnapshot")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "state": dict(snapshot.state),
        "round_index": snapshot.round_index,
        "dev_metrics": dict(snapshot.dev_metrics),
        "model_spec": snapshot.model_spec.to_dict(),
        "partition_hash": snapshot.partition_hash,
        "metadata": dict(snapshot.metadata),
        "content_hash": snapshot.content_hash,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_snapshot(path, map_location="cpu") -> GlobalSnapshot:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("snapshot artifact must contain a mapping")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported snapshot schema version")
    return GlobalSnapshot(
        state=dict(payload["state"]),
        round_index=int(payload["round_index"]),
        dev_metrics=dict(payload["dev_metrics"]),
        model_spec=ModelSpec.from_dict(payload["model_spec"]),
        partition_hash=str(payload["partition_hash"]),
        metadata=dict(payload.get("metadata", {})),
        content_hash=str(payload["content_hash"]),
    )
