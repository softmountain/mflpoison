import argparse
import pickle
from pathlib import Path

from .config import resolve_config
from .manifest_io import load_manifest
from .paths import (
    audio_cache_path,
    manifest_path,
    packaged_audio_path,
    packaged_video_path,
    video_cache_path,
)


def load_cache(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def package_records(records, cache):
    return [[key, file_path, label, cache[key]] for key, file_path, label in records]


def save_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def package_fold(config, fold_idx: int) -> None:
    manifest = load_manifest(manifest_path(config, fold_idx))
    audio_cache = load_cache(audio_cache_path(config))
    video_cache = load_cache(video_cache_path(config))

    for client_id, records in manifest["clients"].items():
        save_pickle(packaged_audio_path(config, fold_idx, client_id), package_records(records, audio_cache))
        save_pickle(packaged_video_path(config, fold_idx, client_id), package_records(records, video_cache))

    save_pickle(packaged_audio_path(config, fold_idx, "test"), package_records(manifest["test"], audio_cache))
    save_pickle(packaged_video_path(config, fold_idx, "test"), package_records(manifest["test"], video_cache))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package demo UCF101 features per client")
    parser.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = resolve_config(data_dir=args.data_dir, output_dir=args.output_dir)
    for fold_idx in args.folds:
        package_fold(config, fold_idx)
        print(f"Packaged fold{fold_idx} client feature files")


if __name__ == "__main__":
    main()
