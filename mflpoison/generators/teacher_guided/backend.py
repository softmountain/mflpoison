import inspect
from typing import Mapping, Optional

import torch

from fed_multimodal.generator.gan_generator import (
    AudioFeatureGenerator,
    FeatureGANConfig,
    VideoFeatureGenerator,
)
from mflpoison.core.types import SyntheticBatch
from mflpoison.generators.base import BaseGeneratorBackend


def _mask_by_length(tensor, lengths):
    steps = torch.arange(tensor.shape[1], device=tensor.device)[None, :]
    mask = steps < lengths[:, None].clamp(0, tensor.shape[1])
    return tensor * mask.unsqueeze(-1).to(tensor.dtype)


def _audio_znorm(tensor, lengths, eps=1e-5):
    steps = torch.arange(tensor.shape[1], device=tensor.device)[None, :]
    mask = (steps < lengths[:, None].clamp(1, tensor.shape[1])).unsqueeze(-1)
    weights = mask.to(tensor.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    mean = (tensor * weights).sum(dim=1) / denominator
    variance = ((tensor - mean[:, None, :]).pow(2) * weights).sum(dim=1)
    variance = variance / denominator
    normalized = (tensor - mean[:, None, :]) / variance.add(eps).sqrt()
    return normalized * weights


class TeacherGuidedBackend(BaseGeneratorBackend):
    family = "teacher_guided"
    name = "teacher_guided"

    def __init__(self, checkpoint_path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved = dict(checkpoint.get("config", {}))
        parameters = inspect.signature(FeatureGANConfig.__init__).parameters
        filtered = {key: value for key, value in saved.items() if key in parameters}
        filtered["device"] = str(device)
        config = FeatureGANConfig(**filtered)
        super().__init__(checkpoint_path, config, device)
        self.audio_generator = AudioFeatureGenerator(config).to(self.device)
        self.video_generator = VideoFeatureGenerator(config).to(self.device)
        self.audio_generator.load_state_dict(checkpoint["audio_generator"])
        self.video_generator.load_state_dict(checkpoint["video_generator"])
        self.audio_generator.eval()
        self.video_generator.eval()

    @torch.no_grad()
    def generate(
        self,
        target_labels: torch.Tensor,
        train_labels: Optional[torch.Tensor] = None,
        source_labels: Optional[torch.Tensor] = None,
        lengths: Optional[Mapping[str, torch.Tensor]] = None,
        batch_size: int = 64,
        seed: Optional[int] = None,
    ) -> SyntheticBatch:
        target_labels = self._validate_labels(target_labels)
        train_labels, source_labels = self._output_labels(
            target_labels, train_labels, source_labels
        )
        count = target_labels.shape[0]
        audio_lengths = torch.full(
            (count,), int(self.config.audio_seq_len), dtype=torch.long
        )
        video_lengths = torch.full(
            (count,), int(self.config.video_seq_len), dtype=torch.long
        )
        if lengths:
            audio_lengths = torch.as_tensor(
                lengths.get("audio", audio_lengths), dtype=torch.long
            ).view(-1)
            video_lengths = torch.as_tensor(
                lengths.get("video", video_lengths), dtype=torch.long
            ).view(-1)
        if audio_lengths.shape[0] != count or video_lengths.shape[0] != count:
            raise ValueError("modality lengths must match target_labels")

        rng = torch.Generator(device=self.device.type)
        if seed is not None:
            rng.manual_seed(int(seed))
        audio_parts, video_parts = [], []
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            labels = target_labels[start:end].to(self.device)
            len_audio = audio_lengths[start:end].to(self.device)
            len_video = video_lengths[start:end].to(self.device)
            noise = torch.randn(
                end - start,
                int(self.config.z_dim),
                device=self.device,
                generator=rng,
            )
            audio = self.audio_generator(noise, labels)
            video = self.video_generator(noise, labels)
            audio = _mask_by_length(_audio_znorm(audio, len_audio), len_audio)
            video = _mask_by_length(video, len_video)
            audio_parts.append(audio.cpu())
            video_parts.append(video.cpu())

        metadata = self.metadata()
        metadata["seed"] = seed
        return SyntheticBatch(
            features={
                "audio": torch.cat(audio_parts, dim=0),
                "video": torch.cat(video_parts, dim=0),
            },
            lengths={"audio": audio_lengths, "video": video_lengths},
            condition_labels=target_labels,
            train_labels=train_labels,
            source_labels=source_labels,
            metadata=metadata,
        ).validate()
