import json
import hashlib
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


def _git_bytes(args, cwd: Optional[Path] = None) -> Optional[bytes]:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_tree_hash(cwd: Optional[Path] = None) -> Optional[str]:
    """Hash runtime-source changes without persisting their contents."""

    commit = _git_bytes(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return None
    runtime_paths = (
        "mflpoison",
        "fed_multimodal",
        "scripts",
        "configs",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
    )
    diff = _git_bytes(
        ["diff", "--binary", "--no-ext-diff", "HEAD", "--", *runtime_paths],
        cwd,
    )
    untracked = _git_bytes(
        ["ls-files", "--others", "--exclude-standard", "-z", "--", *runtime_paths],
        cwd,
    )
    root_text = _git_output(["rev-parse", "--show-toplevel"], cwd)
    if diff is None or untracked is None or root_text is None:
        return None
    digest = hashlib.sha256()
    digest.update(commit)
    digest.update(b"\0tracked-diff\0")
    digest.update(diff)
    root = Path(root_text)
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        path = root / relative_path
        if not path.is_file():
            continue
        digest.update(b"\0untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            return None
    return digest.hexdigest()


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
    git_status = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"], cwd
    )
    manifest = {
        "schema_version": 2,
        "experiment_id": str(experiment_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": dict(config),
        "seed": int(seed),
        "git_commit": _git_output(["rev-parse", "HEAD"], cwd),
        "git_branch": _git_output(["branch", "--show-current"], cwd),
        "git_dirty": None if git_status is None else bool(git_status),
        "source_tree_hash": _source_tree_hash(cwd),
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
