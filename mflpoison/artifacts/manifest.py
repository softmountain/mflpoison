import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from mflpoison.core.config import config_hash


def _git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    experiment_id: str,
    config: Mapping[str, Any],
    seed: int,
    extra: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "experiment_id": str(experiment_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "config": dict(config),
        "seed": int(seed),
        "git_commit": _git_commit(cwd),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def write_manifest(manifest: Mapping[str, Any], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)
    return path
