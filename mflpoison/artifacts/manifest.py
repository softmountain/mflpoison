import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


def _git_output(args, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_metadata() -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [
                        int(properties.major),
                        int(properties.minor),
                    ],
                }
            )
    driver_versions = None
    try:
        raw_versions = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        driver_versions = [
            item.strip() for item in raw_versions.splitlines() if item.strip()
        ]
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "argv": list(sys.argv),
        "working_directory": str(Path.cwd()),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_devices": devices,
        "nvidia_driver_versions": driver_versions,
    }


def build_manifest(
    experiment_id: str,
    config: Mapping[str, Any],
    seed: int,
    extra: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest = {
        "schema_version": 2,
        "experiment_id": str(experiment_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": dict(config),
        "seed": int(seed),
        "git_commit": _git_output(["rev-parse", "HEAD"], cwd),
        "git_branch": _git_output(["branch", "--show-current"], cwd),
        "runtime": _runtime_metadata(),
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
