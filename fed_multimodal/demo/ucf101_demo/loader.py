import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DemoConfig, resolve_config
from .paths import packaged_audio_dir, packaged_audio_path, packaged_video_path


def pad_tensor(vec: torch.Tensor, pad: int) -> torch.Tensor:
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    if pad_size[0] <= 0:
        return vec
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)


def collate_mm_fn_padd(batch):
    max_a_len = max(map(lambda x: x[0].shape[0], batch))
    max_b_len = max(map(lambda x: x[1].shape[0], batch))
    x_a, x_b, len_a, len_b, ys = [], [], [], [], []
    for audio_feat, video_feat, audio_len, video_len, label in batch:
        x_a.append(pad_tensor(audio_feat, pad=max_a_len))
        x_b.append(pad_tensor(video_feat, pad=max_b_len))
        len_a.append(torch.tensor(audio_len))
        len_b.append(torch.tensor(video_len))
        ys.append(label)
    return (
        torch.stack(x_a, dim=0),
        torch.stack(x_b, dim=0),
        torch.stack(len_a, dim=0),
        torch.stack(len_b, dim=0),
        torch.stack(ys, dim=0),
    )


class DemoMultimodalDataset(Dataset):
    def __init__(self, audio_records, video_records, default_audio_shape=np.array([500, 80]), default_video_shape=np.array([10, 1280])):
        self.audio_records = audio_records
        self.video_records = video_records
        self.default_audio_shape = default_audio_shape
        self.default_video_shape = default_video_shape

    def __len__(self):
        return len(self.audio_records)

    def __getitem__(self, idx):
        audio_item = self.audio_records[idx]
        video_item = self.video_records[idx]
        audio_feat = audio_item[-1]
        video_feat = video_item[-1]
        label = torch.tensor(audio_item[-2])
        if audio_feat is not None and len(audio_feat.shape) == 3:
            audio_feat = audio_feat[0]
        if video_feat is not None and len(video_feat.shape) == 3:
            video_feat = video_feat[0]
        audio_tensor = torch.tensor(audio_feat, dtype=torch.float32) if audio_feat is not None else torch.zeros(self.default_audio_shape, dtype=torch.float32)
        video_tensor = torch.tensor(video_feat, dtype=torch.float32) if video_feat is not None else torch.zeros(self.default_video_shape, dtype=torch.float32)
        audio_len = len(audio_tensor) if audio_feat is not None else 0
        video_len = len(video_tensor) if video_feat is not None else 0
        return audio_tensor, video_tensor, audio_len, video_len, label


class DemoUCF101Loader:
    def __init__(self, config: DemoConfig, fold_idx: int):
        self.config = config
        self.fold_idx = fold_idx

    def client_ids(self) -> List[str]:
        client_files = sorted(packaged_audio_dir(self.config, self.fold_idx).glob("*.pkl"))
        return [path.stem for path in client_files if path.stem != "test"]

    def _load_pickle(self, path: Path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def load_client_records(self, client_id: str):
        audio_records = self._load_pickle(packaged_audio_path(self.config, self.fold_idx, client_id))
        video_records = self._load_pickle(packaged_video_path(self.config, self.fold_idx, client_id))
        return audio_records, video_records

    def load_test_records(self):
        return self.load_client_records("test")

    def build_dataloader(self, client_id: str, shuffle: bool = True) -> DataLoader:
        audio_records, video_records = self.load_client_records(client_id)
        dataset = DemoMultimodalDataset(audio_records, video_records)
        batch_size = self.config.train_batch_size if shuffle else self.config.eval_batch_size
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_mm_fn_padd)


def create_loader(fold_idx: int, **overrides) -> DemoUCF101Loader:
    return DemoUCF101Loader(resolve_config(**overrides), fold_idx)
