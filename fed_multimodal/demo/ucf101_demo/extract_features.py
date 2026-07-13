import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from fed_multimodal.features.feature_processing.feature_manager import FeatureManager

from .config import resolve_config
from .manifest_io import load_manifest
from .paths import audio_cache_path, video_cache_path, manifest_path


class _ArgsNamespace(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _build_audio_manager(config):
    return FeatureManager(_ArgsNamespace(
        raw_data_dir=str(config.data_dir),
        output_dir=str(config.output_dir),
        dataset=config.dataset,
        feature_type=config.audio_feature_type,
    ))


def _build_video_manager(config):
    return FeatureManager(_ArgsNamespace(
        raw_data_dir=str(config.data_dir),
        output_dir=str(config.output_dir),
        dataset=config.dataset,
        feature_type=config.video_feature_type,
    ))


def collect_manifest_keys(config, folds) -> Set[str]:
    keys = set()
    for fold_idx in folds:
        manifest = load_manifest(manifest_path(config, fold_idx))
        for records in manifest["clients"].values():
            for key, _, _ in records:
                keys.add(key)
        for key, _, _ in manifest["test"]:
            keys.add(key)
    return keys


def build_audio_cache(config, keys: Iterable[str]) -> Dict[str, object]:
    manager = _build_audio_manager(config)
    cache = {}
    for key in sorted(keys):
        label_str, video_name = key.split("/", 1)
        audio_path = config.dataset_dir / "audios" / label_str / f"{video_name}.wav"
        cache[key] = manager.extract_mfcc_features(
            audio_path=str(audio_path),
            label_str=label_str,
            frame_length=40,
            frame_shift=20,
            max_len=config.audio_max_len,
        )
    return cache


def build_video_cache(config, keys: Iterable[str]) -> Dict[str, object]:
    manager = _build_video_manager(config)
    cache = {}
    for key in sorted(keys):
        label_str, video_name = key.split("/", 1)
        cache[key] = manager.extract_frame_features(
            video_id=video_name,
            label_str=label_str,
            max_len=config.video_max_len,
        )
    return cache


def load_cache(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_cache(path: Path, cache: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract demo UCF101 feature caches")
    parser.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--modalities", nargs="*", choices=["audio", "video"], default=["audio", "video"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = resolve_config(data_dir=args.data_dir, output_dir=args.output_dir)
    keys = sorted(collect_manifest_keys(config, args.folds))
    if args.limit > 0:
        keys = keys[:args.limit]

    if "audio" in args.modalities:
        audio_cache = load_cache(audio_cache_path(config)) or {}
        missing_audio_keys = [key for key in keys if key not in audio_cache]
        if missing_audio_keys:
            audio_cache.update(build_audio_cache(config, missing_audio_keys))
        save_cache(audio_cache_path(config), audio_cache)

    if "video" in args.modalities:
        video_cache = load_cache(video_cache_path(config)) or {}
        missing_video_keys = [key for key in keys if key not in video_cache]
        if missing_video_keys:
            video_cache.update(build_video_cache(config, missing_video_keys))
        save_cache(video_cache_path(config), video_cache)
    if "audio" in args.modalities:
        print(f"Saved audio cache to {audio_cache_path(config)}")
    if "video" in args.modalities:
        print(f"Saved video cache to {video_cache_path(config)}")


if __name__ == "__main__":
    main()
