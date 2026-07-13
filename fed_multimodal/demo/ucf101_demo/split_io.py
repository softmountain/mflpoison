from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from .config import DemoConfig

SampleRecord = List[object]


def list_available_samples(config: DemoConfig) -> Dict[str, str]:
    audio_root = config.dataset_dir / "audios"
    sample_to_audio_path = {}
    for class_dir in sorted(audio_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for audio_path in sorted(class_dir.glob("*.wav")):
            key = f"{class_dir.name}/{audio_path.stem}"
            sample_to_audio_path[key] = str(audio_path)
    return sample_to_audio_path


def build_label_map(sample_keys: Iterable[str]) -> Dict[str, int]:
    class_names = sorted({key.split("/")[0] for key in sample_keys})
    return {class_name: idx for idx, class_name in enumerate(class_names)}


def build_sample_catalog(config: DemoConfig) -> Dict[str, SampleRecord]:
    available = list_available_samples(config)
    label_map = build_label_map(available.keys())
    return {
        key: [key, audio_path, label_map[key.split("/")[0]]]
        for key, audio_path in available.items()
    }


def split_file_path(config: DemoConfig, split_kind: str, fold_idx: int) -> Path:
    suffix = "train" if split_kind == "train" else "val"
    return config.dataset_dir / f"ucf101_{suffix}_split_{fold_idx}_rawframes.txt"


def read_split_keys(config: DemoConfig, split_kind: str, fold_idx: int) -> List[str]:
    split_path = split_file_path(config, split_kind, fold_idx)
    keys = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            keys.append(stripped.split(" ")[0])
    return keys


def filter_available_keys(keys: Iterable[str], catalog: Dict[str, SampleRecord]) -> List[str]:
    return [key for key in keys if key in catalog]


def group_keys_by_label(keys: Iterable[str], catalog: Dict[str, SampleRecord]) -> Dict[int, List[str]]:
    grouped = defaultdict(list)
    for key in keys:
        grouped[catalog[key][2]].append(key)
    return dict(grouped)


def materialize_records(keys: Iterable[str], catalog: Dict[str, SampleRecord]) -> List[SampleRecord]:
    return [list(catalog[key]) for key in keys]
