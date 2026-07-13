from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fed_multimodal.demo.ucf101_demo.config import resolve_config as resolve_demo_config


@dataclass(frozen=True)
class DemoGANConfig:
    fold_idx: int = 1
    seed: int = 42
    batch_size: int = 32
    val_split: float = 0.1
    audio_feat: str = 'mfcc'
    video_feat: str = 'mobilenet_v2'
    gan_epochs: int = 200
    gan_lr_g: float = 2e-4
    gan_lr_d: float = 1e-4
    gan_rf_weight: float = 2.0
    gan_aux_weight: float = 1.0
    gan_cls_weight: float = 0.1
    gan_joint_weight: float = 0.05
    gan_fm_weight: float = 0.05
    gan_mom_weight: float = 0.05
    gan_joint_d_steps: int = 3
    gan_joint_lr_mult: float = 2.0
    gan_audio_out_max: float = 1.0
    gan_audio_scale_max: float = 0.3
    gan_audio_bias_max: float = 0.1
    gan_video_out_max: float = 20.0
    gan_video_scale_max: float = 8.0
    warmup_ratio: float = 0.2
    ramp_ratio: float = 0.2
    log_interval: int = 5
    save_interval: int = 0
    output_dir: Optional[Path] = None
    data_dir: Optional[Path] = None


def resolve_gan_config(**overrides) -> DemoGANConfig:
    normalized = {key: value for key, value in overrides.items() if value is not None}
    for key in ['data_dir', 'output_dir']:
        if key in normalized:
            normalized[key] = Path(normalized[key])
    return DemoGANConfig(**normalized)


def resolve_demo_paths(config: DemoGANConfig):
    return resolve_demo_config(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        audio_feature_type=config.audio_feat,
        video_feature_type=config.video_feat,
    )
