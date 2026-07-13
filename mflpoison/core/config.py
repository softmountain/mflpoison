import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(data: Mapping[str, Any], length: int = 12) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:length]


def load_config(path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML config support requires PyYAML; JSON configs work without it"
                ) from exc
            result = yaml.safe_load(handle)
            return result or {}
    raise ValueError(f"unsupported config format: {path.suffix}")
