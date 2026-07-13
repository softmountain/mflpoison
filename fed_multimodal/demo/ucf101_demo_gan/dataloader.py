import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader, Dataset

from fed_multimodal.demo.ucf101_demo.loader import DemoMultimodalDataset, collate_mm_fn_padd
from fed_multimodal.demo.ucf101_demo.paths import packaged_audio_dir, packaged_audio_path, packaged_video_path

from .config import DemoGANConfig, resolve_demo_paths


class DemoGANDataManager:
    def __init__(self, config: DemoGANConfig):
        self.config = config
        self.demo_paths = resolve_demo_paths(config)
        self.audio_root = packaged_audio_dir(self.demo_paths, config.fold_idx)
        self.video_root = self.demo_paths.demo_root / 'packaged' / f'fold{config.fold_idx}' / 'video'
        self.audio_feat_dim = 80
        self.video_feat_dim = 1280
        self.num_classes = 51

    def _load_pickle(self, path: Path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def _client_ids(self) -> List[str]:
        return sorted(path.stem for path in self.audio_root.glob('*.pkl') if path.stem != 'test')

    def _merge_train_records(self):
        train_audio, train_video = [], []
        for client_id in self._client_ids():
            train_audio.extend(self._load_pickle(packaged_audio_path(self.demo_paths, self.config.fold_idx, client_id)))
            train_video.extend(self._load_pickle(packaged_video_path(self.demo_paths, self.config.fold_idx, client_id)))
        return train_audio, train_video

    def _split_train_val(self, audio_records, video_records):
        indices = np.random.default_rng(self.config.seed).permutation(len(audio_records))
        val_size = int(len(indices) * self.config.val_split)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        train_audio = [audio_records[i] for i in train_idx]
        train_video = [video_records[i] for i in train_idx]
        val_audio = [audio_records[i] for i in val_idx]
        val_video = [video_records[i] for i in val_idx]
        return train_audio, train_video, val_audio, val_video

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        full_train_audio, full_train_video = self._merge_train_records()
        train_audio, train_video, val_audio, val_video = self._split_train_val(full_train_audio, full_train_video)
        test_audio = self._load_pickle(packaged_audio_path(self.demo_paths, self.config.fold_idx, 'test'))
        test_video = self._load_pickle(packaged_video_path(self.demo_paths, self.config.fold_idx, 'test'))

        def build(records_a, records_v, shuffle):
            dataset = DemoMultimodalDataset(records_a, records_v)
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=0,
                collate_fn=collate_mm_fn_padd,
                pin_memory=True,
            )

        return {
            'train': build(train_audio, train_video, True),
            'val': build(val_audio, val_video, False),
            'test': build(test_audio, test_video, False),
            'full_train': build(full_train_audio, full_train_video, True),
        }
