import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _require_ucf101(args, owner):
    dataset = getattr(args, "dataset", None)
    if dataset != "ucf101":
        raise ValueError(f"{owner} only supports ucf101, got {dataset!r}")


def _alpha_name(alpha):
    return str(alpha).replace(".", "")


def pad_tensor(vector, padded_length):
    if vector.size(0) == padded_length:
        return vector
    pad_shape = list(vector.shape)
    pad_shape[0] = padded_length - vector.size(0)
    return torch.cat([vector, vector.new_zeros(pad_shape)], dim=0)


def collate_mm_fn_padd(batch):
    max_audio_length = max(item[0].shape[0] for item in batch)
    max_video_length = max(item[1].shape[0] for item in batch)
    audio = torch.stack(
        [pad_tensor(item[0], max_audio_length) for item in batch], dim=0
    )
    video = torch.stack(
        [pad_tensor(item[1], max_video_length) for item in batch], dim=0
    )
    audio_lengths = torch.tensor([item[2] for item in batch])
    video_lengths = torch.tensor([item[3] for item in batch])
    labels = torch.stack([item[4] for item in batch], dim=0)
    return audio, video, audio_lengths, video_lengths, labels


class MMDatasetGenerator(Dataset):
    def __init__(
        self,
        audio,
        video,
        default_audio_shape,
        default_video_shape,
    ):
        if len(audio) != len(video):
            raise ValueError("audio/video data length mismatch")
        self.audio = audio
        self.video = video
        self.default_audio_shape = tuple(int(value) for value in default_audio_shape)
        self.default_video_shape = tuple(int(value) for value in default_video_shape)

    def __len__(self):
        return len(self.audio)

    @staticmethod
    def _to_tensor(feature, default_shape):
        if feature is None:
            return torch.zeros(default_shape, dtype=torch.float32), 0
        tensor = torch.as_tensor(feature)
        if tensor.ndim == 3:
            tensor = tensor[0]
        return tensor, len(tensor)

    def __getitem__(self, item):
        audio, audio_length = self._to_tensor(
            self.audio[item][-1], self.default_audio_shape
        )
        video, video_length = self._to_tensor(
            self.video[item][-1], self.default_video_shape
        )
        label = torch.tensor(self.audio[item][-2])
        return audio, video, audio_length, video_length, label


class DataloadManager:
    """UCF101-only loader for the retained FedMM pickle layout."""

    def __init__(self, args):
        _require_ucf101(args, type(self).__name__)
        self.args = args
        self.get_audio_feat_path()
        self.get_video_feat_path()

    def get_audio_feat_path(self):
        self.audio_feat_path = Path(self.args.data_dir).joinpath(
            "feature", "audio", self.args.audio_feat, "ucf101"
        )
        return self.audio_feat_path

    def get_video_feat_path(self):
        self.video_feat_path = Path(self.args.data_dir).joinpath(
            "feature", "video", self.args.video_feat, "ucf101"
        )
        return self.video_feat_path

    def _partition_dir(self, feature_root, fold_idx):
        return feature_root.joinpath(
            f"alpha{_alpha_name(self.args.alpha)}", f"fold{fold_idx}"
        )

    def get_client_ids(self, fold_idx=1):
        data_path = self._partition_dir(self.video_feat_path, fold_idx)
        self.client_ids = sorted(
            path.stem for path in data_path.iterdir() if path.suffix == ".pkl"
        )

    @staticmethod
    def _load_pickle(path):
        with Path(path).open("rb") as stream:
            return pickle.load(stream)

    def load_audio_feat(self, client_id, fold_idx=1):
        path = self._partition_dir(self.audio_feat_path, fold_idx).joinpath(
            f"{client_id}.pkl"
        )
        return self._load_pickle(path)

    def load_video_feat(self, client_id, fold_idx=1):
        path = self._partition_dir(self.video_feat_path, fold_idx).joinpath(
            f"{client_id}.pkl"
        )
        return self._load_pickle(path)

    def get_client_sim_dict(self, client_id):
        if self.sim_data:
            return self.sim_data[str(client_id)]
        return None

    @staticmethod
    def _apply_simulation(audio, video, simulation):
        if len(simulation) != len(audio) or len(audio) != len(video):
            raise ValueError("simulation/audio/video data length mismatch")
        labeled_audio = []
        labeled_video = []
        for index, simulation_record in enumerate(simulation):
            values = simulation_record[-1]
            if len(values) != 4:
                raise ValueError("simulation record must contain four values")
            audio_missing, video_missing, noisy_label, label_missing = values
            if audio_missing:
                audio[index][-1] = None
            if video_missing:
                video[index][-1] = None
            audio[index][-2] = noisy_label
            video[index][-2] = noisy_label
            if not label_missing:
                labeled_audio.append(audio[index])
                labeled_video.append(video[index])
        return labeled_audio, labeled_video

    def set_dataloader(
        self,
        data_a,
        data_b,
        default_feat_shape_a=np.array([0, 0]),
        default_feat_shape_b=np.array([0, 0]),
        client_sim_dict=None,
        shuffle=False,
    ):
        if len(data_a) != len(data_b):
            raise ValueError("audio/video data length mismatch")
        if client_sim_dict is not None:
            data_a, data_b = self._apply_simulation(
                data_a, data_b, client_sim_dict
            )
        if not data_a:
            return None
        if all(
            audio_record[-1] is None and video_record[-1] is None
            for audio_record, video_record in zip(data_a, data_b)
        ):
            return None
        dataset = MMDatasetGenerator(
            data_a,
            data_b,
            default_feat_shape_a,
            default_feat_shape_b,
        )
        batch_size = (
            int(self.args.batch_size)
            if shuffle
            else int(getattr(self.args, "eval_batch_size", 64))
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=int(getattr(self.args, "num_workers", 0)),
            shuffle=shuffle,
            collate_fn=collate_mm_fn_padd,
        )

    def load_sim_dict(self, fold_idx=1, ext="json"):
        if self.setting_str == "":
            self.sim_data = None
            return
        if ext not in {"json", "pkl"}:
            raise ValueError("simulation extension must be json or pkl")
        data_path = Path(self.args.data_dir).joinpath(
            "simulation_feature",
            "ucf101",
            f"fold{fold_idx}",
            f"{self.setting_str}.{ext}",
        )
        if ext == "pkl":
            self.sim_data = self._load_pickle(data_path)
        else:
            with data_path.open("r") as stream:
                self.sim_data = json.load(stream)

    def get_simulation_setting(self, alpha=None):
        settings = []
        if self.args.missing_modality:
            settings.append(
                "mm" + str(self.args.missing_modailty_rate).replace(".", "")
            )
        if self.args.label_nosiy:
            settings.append(
                "ln" + str(self.args.label_nosiy_level).replace(".", "")
            )
        if self.args.missing_label:
            settings.append(
                "ml" + str(self.args.missing_label_rate).replace(".", "")
            )
        if settings and alpha is not None:
            settings.append("alpha" + _alpha_name(self.args.alpha))
        self.setting_str = "_".join(settings)
