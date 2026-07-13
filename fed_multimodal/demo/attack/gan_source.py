from pathlib import Path

import torch

from fed_multimodal.demo.ucf101_demo_gan.train_gan import build_teacher
from fed_multimodal.generator.gan_generator import FeatureGANConfig, MultimodalFeatureGAN


def load_demo_gan(gan_checkpoint: Path, teacher_checkpoint: Path, sample_batch, device):
    teacher_ckpt = torch.load(teacher_checkpoint, map_location=device)
    teacher = build_teacher(teacher_ckpt, device)
    ckpt = torch.load(gan_checkpoint, map_location=device)
    saved_config = ckpt.get('config', {})
    real_a, real_v = sample_batch[0], sample_batch[1]
    config = FeatureGANConfig(
        num_classes=51,
        audio_seq_len=saved_config.get('audio_seq_len', real_a.shape[1]),
        audio_feat_dim=saved_config.get('audio_feat_dim', real_a.shape[2]),
        video_seq_len=saved_config.get('video_seq_len', real_v.shape[1]),
        video_feat_dim=saved_config.get('video_feat_dim', real_v.shape[2]),
        z_dim=saved_config.get('z_dim', 128),
        hidden_dim=saved_config.get('hidden_dim', 256),
        audio_scale_max=saved_config.get('audio_scale_max', 0.3),
        audio_bias_max=saved_config.get('audio_bias_max', 0.1),
        audio_out_max=saved_config.get('audio_out_max', 1.0),
        video_scale_max=saved_config.get('video_scale_max', 8.0),
        video_out_max=saved_config.get('video_out_max', 20.0),
        device=device,
    )
    local_args = type('LocalArgs', (), {'dataset': 'ucf101'})()
    gan = MultimodalFeatureGAN(local_args, teacher, config)
    gan.load_checkpoint(str(gan_checkpoint))
    return gan


def generate_fake_multimodal_batch(gan, labels, len_a, len_v, device):
    z = torch.randn(labels.shape[0], gan.config.z_dim, device=device)
    with torch.no_grad():
        fake_a = gan.audio_generator(z, labels)
        fake_v = gan.video_generator(z, labels)
        len_a_gen = torch.full(
            (labels.shape[0],),
            fake_a.shape[1],
            device=device,
            dtype=len_a.dtype if len_a is not None else torch.long,
        )
        len_v_gen = torch.full(
            (labels.shape[0],),
            fake_v.shape[1],
            device=device,
            dtype=len_v.dtype if len_v is not None else torch.long,
        )
        fake_a = gan._apply_per_sample_znorm(fake_a, len_a_gen)
        fake_a = gan._mask_by_len(fake_a, len_a_gen)
        fake_v = gan._mask_by_len(fake_v, len_v_gen)
    return fake_a, fake_v
