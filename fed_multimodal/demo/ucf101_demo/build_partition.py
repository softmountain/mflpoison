import argparse
from typing import Dict, List

from .config import resolve_config
from .manifest_io import save_manifest
from .paths import manifest_path
from .partition_uniform import summarize_partition, uniform_round_robin_partition
from .split_io import build_sample_catalog, filter_available_keys, materialize_records, read_split_keys


def build_fold_manifest(config, fold_idx: int) -> Dict:
    catalog = build_sample_catalog(config)
    train_keys = filter_available_keys(read_split_keys(config, "train", fold_idx), catalog)
    test_keys = filter_available_keys(read_split_keys(config, "test", fold_idx), catalog)
    key_to_label = {key: catalog[key][2] for key in catalog}
    clients = uniform_round_robin_partition(
        keys=train_keys,
        key_to_label=key_to_label,
        num_clients=config.num_clients,
        seed=config.seed + fold_idx,
    )
    summary = summarize_partition(clients, key_to_label)
    return {
        "metadata": {
            "dataset": config.dataset,
            "fold": fold_idx,
            "partition_strategy": "uniform_round_robin",
            "num_clients": config.num_clients,
            "seed": config.seed,
            "has_dev": False,
            "label_space_size": len({record[2] for record in catalog.values()}),
            "train_sample_count": len(train_keys),
            "test_sample_count": len(test_keys),
            **summary,
        },
        "clients": {
            client_id: materialize_records(client_keys, catalog)
            for client_id, client_keys in clients.items()
        },
        "test": materialize_records(test_keys, catalog),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone demo manifests for UCF101")
    parser.add_argument("--num_clients", type=int, default=15)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = resolve_config(
        num_clients=args.num_clients,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    for fold_idx in args.folds:
        manifest = build_fold_manifest(config, fold_idx)
        save_manifest(manifest_path(config, fold_idx), manifest)
        print(f"Saved fold{fold_idx} manifest to {manifest_path(config, fold_idx)}")


if __name__ == "__main__":
    main()
