import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def canonical_json(data: Mapping[str, Any]) -> str:
    """Serialize JSON-compatible metadata deterministically."""

    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def mapping_hash(data: Mapping[str, Any], length: int = 64) -> str:
    if int(length) < 1:
        raise ValueError("hash length must be positive")
    digest = hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
    return digest[: int(length)]


def tensor_map_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor keys, schemas, and values without device-dependent bytes."""

    if not tensors:
        raise ValueError("cannot hash an empty tensor mapping")
    digest = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key]
        if not isinstance(key, str) or not key:
            raise ValueError("tensor mapping keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError("tensor mapping values must be torch.Tensor instances")
        if value.layout != torch.strided:
            raise ValueError("only strided tensors can be hashed")
        normalized = value.detach().cpu().contiguous()
        schema = {
            "key": key,
            "dtype": str(normalized.dtype),
            "shape": list(normalized.shape),
        }
        digest.update(canonical_json(schema).encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalized.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
