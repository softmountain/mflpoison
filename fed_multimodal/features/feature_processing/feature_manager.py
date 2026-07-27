import json
import pickle
from pathlib import Path

import numpy as np
import torch


class FeatureManager:
    """UCF101 MFCC and MobileNetV2 preprocessing compatibility manager."""

    def __init__(self, args):
        dataset = getattr(args, "dataset", None)
        if dataset != "ucf101":
            raise ValueError(
                f"{type(self).__name__} only supports ucf101, got {dataset!r}"
            )
        self.args = args
        if getattr(args, "feature_type", None) is not None:
            self.initialize_feature_module()

    def initialize_feature_module(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.args.feature_type == "mfcc":
            return
        if self.args.feature_type != "mobilenet_v2":
            raise ValueError(
                "UCF101 feature_type must be mfcc or mobilenet_v2, got "
                + repr(self.args.feature_type)
            )

        from torchvision import models, transforms

        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier = self.model.classifier[:-1]
        self.model = self.model.to(self.device)
        self.model.eval()
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract_frame_features(self, video_id, label_str, max_len=-1, split=None):
        from PIL import Image

        video_path = Path(self.args.raw_data_dir).joinpath(
            "ucf101", "rawframes"
        )
        if split is not None:
            video_path = video_path.joinpath(split)
        video_path = video_path.joinpath(label_str, video_id)
        rawframes = sorted(video_path.iterdir())[::25]

        tensors = []
        for rawframe in rawframes:
            with Image.open(rawframe) as image:
                tensors.append(self.img_transform(image.convert("RGB")))
        if not tensors:
            return None
        with torch.no_grad():
            features = self.model(torch.stack(tensors).to(self.device))
        result = features.detach().cpu().numpy()
        return result if max_len == -1 else result[:max_len]

    def extract_mfcc_features(
        self,
        audio_path,
        label_str="",
        frame_length=40,
        frame_shift=20,
        max_len=-1,
        en_znorm=True,
    ):
        del label_str
        import torchaudio

        audio, sample_rate = torchaudio.load(str(audio_path))
        if audio.shape[0] != 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        features = torchaudio.compliance.kaldi.fbank(
            waveform=audio,
            frame_length=frame_length,
            frame_shift=frame_shift,
            num_mel_bins=80,
            window_type="hamming",
        ).detach().cpu().numpy()
        if en_znorm:
            features = (features - np.mean(features, axis=0)) / (
                np.std(features, axis=0) + 1e-5
            )
        return features if max_len == -1 else features[:max_len]

    def fetch_partition(self, fold_idx=1, alpha=0.5, file_ext="json"):
        if file_ext not in {"json", "pkl"}:
            raise ValueError("partition extension must be json or pkl")
        alpha_name = str(alpha).replace(".", "")
        partition_path = Path(self.args.output_dir).joinpath(
            "partition",
            "ucf101",
            f"fold{fold_idx}",
            f"partition_alpha{alpha_name}.{file_ext}",
        )
        if not partition_path.exists():
            raise FileNotFoundError(
                f"no UCF101 partition file exists at {partition_path}"
            )
        if file_ext == "pkl":
            with partition_path.open("rb") as stream:
                return pickle.load(stream)
        with partition_path.open("r") as stream:
            return json.load(stream)
