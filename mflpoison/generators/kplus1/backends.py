from typing import Mapping, Optional

import torch

from fed_multimodal.dtm_poison_gan import DTMGANConfig, DTMGenerator
from fed_multimodal.poison_gan import PoisonFeatureGenerator, PoisonGANConfig
from fed_multimodal.temporal_adaptive_gan import (
    TemporalAdaptiveGANConfig,
    TemporalAdaptivePoisonGenerator,
)
from mflpoison.core.types import SyntheticBatch
from mflpoison.generators.base import BaseGeneratorBackend


def _checkpoint_generator_state(checkpoint):
    for key in ("generator_state_dict", "generator_state", "generator"):
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError("checkpoint does not contain generator weights")


class KPlusOneBackend(BaseGeneratorBackend):
    family = "kplus1"
    config_class = None

    def __init__(self, checkpoint_path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict) or "config" not in checkpoint:
            raise ValueError("K+1 checkpoint must contain a config dictionary")
        config = self.config_class.from_dict(checkpoint["config"])
        super().__init__(checkpoint_path, config, device)
        self.generator = self._build_generator(config).to(self.device)
        self.generator.load_state_dict(_checkpoint_generator_state(checkpoint))
        self.generator.eval()

    def _build_generator(self, config):
        raise NotImplementedError

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
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        count = target_labels.shape[0]
        default_audio = torch.full(
            (count,), int(self.config.audio_seq_len), dtype=torch.long
        )
        default_video = torch.full(
            (count,), int(self.config.video_seq_len), dtype=torch.long
        )
        audio_lengths = (
            default_audio
            if lengths is None or "audio" not in lengths
            else torch.as_tensor(lengths["audio"], dtype=torch.long).view(-1)
        )
        video_lengths = (
            default_video
            if lengths is None or "video" not in lengths
            else torch.as_tensor(lengths["video"], dtype=torch.long).view(-1)
        )
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
            audio, video = self.generator(
                noise, labels, len_audio, len_video
            )
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


class LegacyKPlusOneBackend(KPlusOneBackend):
    name = "legacy"
    config_class = PoisonGANConfig

    def _build_generator(self, config):
        return PoisonFeatureGenerator(
            num_classes=config.num_classes,
            audio_seq_len=config.audio_seq_len,
            audio_feat_dim=config.audio_feat_dim,
            video_seq_len=config.video_seq_len,
            video_feat_dim=config.video_feat_dim,
            z_dim=config.z_dim,
            label_emb_dim=config.label_emb_dim,
            hidden_dim=config.hidden_dim,
            audio_out_max=config.audio_out_max,
            video_out_max=config.video_out_max,
            video_scale_max=config.video_scale_max,
        )


class TemporalAdaptiveBackend(KPlusOneBackend):
    name = "temporal_adaptive"
    config_class = TemporalAdaptiveGANConfig

    def _build_generator(self, config):
        return TemporalAdaptivePoisonGenerator(
            num_classes=config.num_classes,
            audio_seq_len=config.audio_seq_len,
            audio_feat_dim=config.audio_feat_dim,
            video_seq_len=config.video_seq_len,
            video_feat_dim=config.video_feat_dim,
            z_dim=config.z_dim,
            label_emb_dim=config.label_emb_dim,
            hidden_dim=config.hidden_dim,
            video_out_max=config.video_out_max,
            video_scale_max=config.video_scale_max,
            frame_noise_dim=config.frame_noise_dim,
            temporal_groups_max=config.temporal_groups_max,
            audio_stats_momentum=config.audio_stats_momentum,
        )


class DTMBackend(KPlusOneBackend):
    name = "dtm"
    config_class = DTMGANConfig

    def _build_generator(self, config):
        return DTMGenerator(config)
