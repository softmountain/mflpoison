from pathlib import Path

from .config import DemoConfig


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def partition_dir(config: DemoConfig, fold_idx: int) -> Path:
    return config.demo_root / "partition" / f"fold{fold_idx}"


def manifest_path(config: DemoConfig, fold_idx: int) -> Path:
    return partition_dir(config, fold_idx) / "manifest.json"


def audio_cache_dir(config: DemoConfig) -> Path:
    return config.demo_root / "feature_cache" / "audio" / config.audio_feature_type


def video_cache_dir(config: DemoConfig) -> Path:
    return config.demo_root / "feature_cache" / "video" / config.video_feature_type


def audio_cache_path(config: DemoConfig) -> Path:
    return audio_cache_dir(config) / "feature.pkl"


def video_cache_path(config: DemoConfig) -> Path:
    return video_cache_dir(config) / "feature.pkl"


def packaged_audio_dir(config: DemoConfig, fold_idx: int) -> Path:
    return config.demo_root / "packaged" / f"fold{fold_idx}" / "audio"


def packaged_video_dir(config: DemoConfig, fold_idx: int) -> Path:
    return config.demo_root / "packaged" / f"fold{fold_idx}" / "video"


def packaged_audio_path(config: DemoConfig, fold_idx: int, client_id: str) -> Path:
    return packaged_audio_dir(config, fold_idx) / f"{client_id}.pkl"


def packaged_video_path(config: DemoConfig, fold_idx: int, client_id: str) -> Path:
    return packaged_video_dir(config, fold_idx) / f"{client_id}.pkl"
