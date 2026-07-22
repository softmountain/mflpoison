import copy
import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from fed_multimodal.constants import constants
from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.model.mm_models import MMActionClassifier
from fed_multimodal.trainers.evaluation import EvalMetric


def _stable_hash(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _item_identity(items: Iterable) -> list:
    result = []
    for index, item in enumerate(items):
        key = item[0] if len(item) > 0 else index
        label = item[-2] if len(item) > 1 else None
        result.append((str(key), int(label) if label is not None else None))
    return result


def _paired_identity(audio: Iterable, video: Iterable) -> list:
    audio_identity = _item_identity(audio)
    video_identity = _item_identity(video)
    for index, (audio_item, video_item) in enumerate(
        zip(audio_identity, video_identity)
    ):
        if audio_item != video_item:
            raise ValueError(
                "audio/video key or label mismatch at partition index "
                + str(index)
            )
    return audio_identity


def _json_identity(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_identity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_identity(item) for item in value]
    return value


@dataclass(frozen=True)
class ClientDataBundle:
    client_id: str
    dataloader: object
    clean_num_samples: int
    partition_hash: str

    @property
    def dataset(self):
        return self.dataloader.dataset


class UCF101FedMMAdapter:
    """Expose the existing FedMM UCF101 partition without repartitioning it.

    The adapter is the only production boundary allowed to understand the
    legacy five-tensor batch and DataloadManager path conventions.
    """

    dataset_name = "ucf101"
    audio_feature = "mfcc"
    video_feature = "mobilenet_v2"

    def __init__(
        self,
        data_dir: str,
        alpha: float = 1.0,
        fold: int = 1,
        batch_size: int = 16,
        hid_size: int = 64,
        attention: bool = False,
        attention_name: str = "base",
        missing_modality: bool = False,
        missing_modality_rate: float = 0.5,
        missing_label: bool = False,
        missing_label_rate: float = 0.5,
        label_noisy: bool = False,
        label_noise_level: float = 0.1,
        audio_seq_len: int = 500,
        video_seq_len: int = 9,
    ):
        self.data_dir = str(data_dir)
        self.alpha = float(alpha)
        self.fold = int(fold)
        self.batch_size = int(batch_size)
        self.hid_size = int(hid_size)
        self.attention = bool(attention)
        self.attention_name = str(attention_name)
        self.audio_seq_len = int(audio_seq_len)
        self.video_seq_len = int(video_seq_len)
        if self.audio_seq_len < 1 or self.video_seq_len < 1:
            raise ValueError("modality sequence lengths must be positive")
        self._args = SimpleNamespace(
            data_dir=self.data_dir,
            dataset=self.dataset_name,
            audio_feat=self.audio_feature,
            video_feat=self.video_feature,
            alpha=self.alpha,
            batch_size=self.batch_size,
            missing_modality=bool(missing_modality),
            missing_modailty_rate=float(missing_modality_rate),
            missing_label=bool(missing_label),
            missing_label_rate=float(missing_label_rate),
            label_nosiy=bool(label_noisy),
            label_nosiy_level=float(label_noise_level),
        )
        self.manager: Optional[DataloadManager] = None
        self.client_data: Dict[str, ClientDataBundle] = {}
        self.eval_loaders: Dict[str, object] = {}
        self.partition_hash = ""

    @property
    def num_classes(self) -> int:
        return int(constants.num_class_dict[self.dataset_name])

    @property
    def modality_shapes(self):
        return {
            "audio": (self.audio_seq_len, constants.feature_len_dict[self.audio_feature]),
            "video": (self.video_seq_len, constants.feature_len_dict[self.video_feature]),
        }

    @property
    def client_ids(self):
        return tuple(sorted(self.client_data))

    def prepare(self) -> "UCF101FedMMAdapter":
        if self.manager is not None:
            return self
        manager = DataloadManager(self._args)
        manager.get_simulation_setting(alpha=self.alpha)
        manager.load_sim_dict(fold_idx=self.fold)
        manager.get_client_ids(fold_idx=self.fold)

        partition_manifest = {}
        for client_id in manager.client_ids:
            audio = manager.load_audio_feat(client_id=client_id, fold_idx=self.fold)
            video = manager.load_video_feat(client_id=client_id, fold_idx=self.fold)
            if len(audio) != len(video):
                raise ValueError(
                    "audio/video partition length mismatch for client " + str(client_id)
                )
            paired_identity = _paired_identity(audio, video)
            manager.get_label_dist(video, client_id)
            is_eval = client_id in ("dev", "test")
            simulation = None if is_eval else manager.get_client_sim_dict(client_id)
            client_manifest = {
                "pairs": paired_identity,
                "simulation": _json_identity(simulation),
                "settings": {
                    "missing_modality": bool(self._args.missing_modality),
                    "missing_modality_rate": float(
                        self._args.missing_modailty_rate
                    ),
                    "missing_label": bool(self._args.missing_label),
                    "missing_label_rate": float(self._args.missing_label_rate),
                    "label_noisy": bool(self._args.label_nosiy),
                    "label_noise_level": float(self._args.label_nosiy_level),
                },
            }
            partition_manifest[str(client_id)] = client_manifest
            loader = manager.set_dataloader(
                copy.deepcopy(audio),
                copy.deepcopy(video),
                client_sim_dict=copy.deepcopy(simulation),
                default_feat_shape_a=np.array(self.modality_shapes["audio"]),
                default_feat_shape_b=np.array(self.modality_shapes["video"]),
                shuffle=not is_eval,
            )
            if loader is None:
                if is_eval:
                    raise ValueError("evaluation partition cannot be empty: " + client_id)
                continue
            bundle = ClientDataBundle(
                client_id=str(client_id),
                dataloader=loader,
                clean_num_samples=len(loader.dataset),
                partition_hash=_stable_hash(client_manifest),
            )
            if is_eval:
                self.eval_loaders[str(client_id)] = loader
            else:
                self.client_data[str(client_id)] = bundle

        if not self.client_data:
            raise ValueError("UCF101 partition contains no train clients")
        if "dev" not in self.eval_loaders or "test" not in self.eval_loaders:
            raise ValueError("UCF101 partition must contain dev and test splits")
        self.partition_hash = _stable_hash(
            {
                "dataset": self.dataset_name,
                "alpha": self.alpha,
                "fold": self.fold,
                "clients": partition_manifest,
            }
        )
        self.manager = manager
        return self

    def get_client(self, client_id: str) -> ClientDataBundle:
        return self.client_data[str(client_id)]

    def build_model(self, state: Optional[Mapping[str, torch.Tensor]] = None):
        model = MMActionClassifier(
            num_classes=self.num_classes,
            audio_input_dim=constants.feature_len_dict[self.audio_feature],
            video_input_dim=constants.feature_len_dict[self.video_feature],
            d_hid=self.hid_size,
            en_att=self.attention,
            att_name=self.attention_name,
        )
        if state is not None:
            model.load_state_dict(dict(state), strict=True)
        return model

    @torch.no_grad()
    def evaluate_state(
        self,
        state: Mapping[str, torch.Tensor],
        split: str,
        device="cpu",
        monitor_labels=None,
    ) -> Dict[str, object]:
        if split not in self.eval_loaders:
            raise KeyError("unknown evaluation split: " + str(split))
        model = self.build_model(state).to(device)
        model.eval()
        criterion = nn.NLLLoss().to(device)
        evaluator = EvalMetric(multilabel=False)
        for audio, video, len_audio, len_video, labels in self.eval_loaders[split]:
            audio = audio.float().to(device)
            video = video.float().to(device)
            len_audio = len_audio.long().to(device)
            len_video = len_video.long().to(device)
            labels = labels.long().to(device)
            logits, _ = model(audio, video, len_audio, len_video)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, labels)
            evaluator.append_classification_results(labels, log_probs, loss)
        return evaluator.classification_detailed_summary(
            monitor_labels=monitor_labels
        )

    def model_metadata(self) -> Dict[str, object]:
        return {
            "name": "MMActionClassifier",
            "num_classes": self.num_classes,
            "audio_input_dim": constants.feature_len_dict[self.audio_feature],
            "video_input_dim": constants.feature_len_dict[self.video_feature],
            "hid_size": self.hid_size,
            "attention": self.attention,
            "attention_name": self.attention_name,
        }

    def dataset_metadata(self) -> Dict[str, object]:
        return {
            "name": self.dataset_name,
            "num_classes": self.num_classes,
            "modality_shapes": self.modality_shapes,
            "alpha": self.alpha,
            "fold": self.fold,
            "partition_hash": self.partition_hash,
            "features": {
                "audio": self.audio_feature,
                "video": self.video_feature,
            },
        }
