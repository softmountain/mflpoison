import argparse
import pickle

from .config import resolve_config
from .manifest_io import load_manifest
from .loader import create_loader
from .paths import audio_cache_path, manifest_path, packaged_audio_path, packaged_video_path, video_cache_path


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def verify_fold(config, fold_idx: int) -> None:
    manifest = load_manifest(manifest_path(config, fold_idx))
    assert manifest["metadata"]["has_dev"] is False
    assert "clients" in manifest
    assert "test" in manifest

    train_keys = set()
    for client_records in manifest["clients"].values():
        for key, _, _ in client_records:
            assert key not in train_keys
            train_keys.add(key)

    test_keys = {key for key, _, _ in manifest["test"]}
    assert train_keys.isdisjoint(test_keys)

    audio_cache = _load_pickle(audio_cache_path(config)) if audio_cache_path(config).exists() else None
    video_cache = _load_pickle(video_cache_path(config)) if video_cache_path(config).exists() else None
    if audio_cache is not None and video_cache is not None:
        sample_client = next(iter(manifest["clients"]))
        audio_records = _load_pickle(packaged_audio_path(config, fold_idx, sample_client))
        video_records = _load_pickle(packaged_video_path(config, fold_idx, sample_client))
        assert len(audio_records) == len(video_records)
        assert all(len(record) == 4 for record in audio_records)
        assert [record[0] for record in audio_records] == [record[0] for record in video_records]

        loader = create_loader(fold_idx, data_dir=config.data_dir, output_dir=config.output_dir)
        batch = next(iter(loader.build_dataloader(sample_client, shuffle=False)))
        assert len(batch) == 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify demo UCF101 outputs")
    parser.add_argument("--folds", type=int, nargs="*", default=[1])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = resolve_config(data_dir=args.data_dir, output_dir=args.output_dir)
    for fold_idx in args.folds:
        verify_fold(config, fold_idx)
        print(f"Verified fold{fold_idx}")


if __name__ == "__main__":
    main()
