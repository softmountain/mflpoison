from dataclasses import dataclass
from pathlib import Path
import os


def _load_system_cfg() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "system.cfg"
    config = {}
    with open(cfg_path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.strip().split("=", 1)
            config[key] = value.replace('"', "")
    return config


_SYSTEM_CFG = _load_system_cfg()
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = Path(_SYSTEM_CFG.get("data_dir", "."))
_DEFAULT_OUTPUT_DIR = Path(_SYSTEM_CFG.get("output_dir", "."))
if _DEFAULT_DATA_DIR == Path("."):
    _DEFAULT_DATA_DIR = _REPO_ROOT / "datasets"
if _DEFAULT_OUTPUT_DIR == Path("."):
    _DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results"


@dataclass(frozen=True)
class DemoConfig:
    dataset: str = "ucf101"
    seed: int = 8
    num_clients: int = 15
    audio_feature_type: str = "mfcc"
    video_feature_type: str = "mobilenet_v2"
    audio_max_len: int = 500
    video_max_len: int = 10
    train_batch_size: int = 16
    eval_batch_size: int = 64
    data_dir: Path = _DEFAULT_DATA_DIR
    output_dir: Path = _DEFAULT_OUTPUT_DIR

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset

    @property
    def demo_root(self) -> Path:
        return self.output_dir / "demo" / self.dataset


def resolve_config(**overrides) -> DemoConfig:
    normalized = {key: value for key, value in overrides.items() if value is not None}
    for key in ["data_dir", "output_dir"]:
        if key in normalized:
            normalized[key] = Path(normalized[key])
    return DemoConfig(**normalized)
