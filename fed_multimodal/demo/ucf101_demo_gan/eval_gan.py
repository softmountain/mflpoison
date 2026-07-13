#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from fed_multimodal.generator.eval_gan_quality import masked_mean, masked_mean_std, visualize_tsne
from fed_multimodal.generator.gan_generator import FeatureGANConfig, MultimodalFeatureGAN
from fed_multimodal.model.mm_models import MMActionClassifier

from .config import resolve_demo_paths, resolve_gan_config
from .dataloader import DemoGANDataManager

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate demo GAN quality')
    parser.add_argument('--fold_idx', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str, default=None)
    parser.add_argument('--num_batches', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_tsne', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_root', type=str, default=None)
    return parser.parse_args()


def build_teacher(teacher_ckpt, device):
    saved_args = teacher_ckpt.get('args', {})
    model = MMActionClassifier(
        num_classes=51,
        audio_input_dim=80,
        video_input_dim=1280,
        d_hid=saved_args.get('hid_size', 128),
        en_att=saved_args.get('att', True),
        att_name=saved_args.get('att_name', 'fuse_base')
    ).to(device)
    model.load_state_dict(teacher_ckpt['model_state_dict'])
    model.eval()
    return model


def stats_basic(x: np.ndarray):
    return {
        'mean': float(x.mean()),
        'std': float(x.std()),
        'min': float(x.min()),
        'max': float(x.max()),
    }


def avg_pairwise_l2(x: np.ndarray):
    n = x.shape[0]
    if n < 2:
        return None
    x = x.astype(np.float32)
    g = np.sum(x * x, axis=1, keepdims=True)
    dist2 = g + g.T - 2.0 * (x @ x.T)
    dist2 = np.maximum(dist2, 0.0)
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(dist2[iu] + 1e-8).mean())


def class_diversity_ratio(real_feats: np.ndarray, fake_feats: np.ndarray, labels: np.ndarray, max_per_class: int = 200):
    ratios = {}
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if idx.shape[0] < 3:
            continue
        if idx.shape[0] > max_per_class:
            idx = np.random.choice(idx, max_per_class, replace=False)
        r = avg_pairwise_l2(real_feats[idx])
        f = avg_pairwise_l2(fake_feats[idx])
        if r is None or r < 1e-8:
            continue
        ratios[int(cls)] = f / r
    if not ratios:
        return {'mean_ratio': None, 'per_class': {}, 'n_classes': 0}
    return {'mean_ratio': float(np.mean(list(ratios.values()))), 'per_class': ratios, 'n_classes': len(ratios)}


def plot_distribution_stats(real_data: np.ndarray, fake_data: np.ndarray, save_path: str, title_prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax1, ax2, ax3 = axes
    ax1.hist(real_data.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax1.hist(fake_data.flatten(), bins=50, alpha=0.5, label='Fake', density=True)
    ax1.set_title(f'{title_prefix} Value Distribution')
    ax1.legend()
    real_dim_means = real_data.mean(axis=0)
    fake_dim_means = fake_data.mean(axis=0)
    dims = np.arange(len(real_dim_means))
    ax2.bar(dims, real_dim_means, alpha=0.5, label='Real', width=0.8)
    ax2.bar(dims, fake_dim_means, alpha=0.5, label='Fake', width=0.8)
    ax2.set_title(f'{title_prefix} Per-Dim Mean')
    real_dim_stds = real_data.std(axis=0)
    fake_dim_stds = fake_data.std(axis=0)
    ax3.bar(dims, real_dim_stds, alpha=0.5, label='Real', width=0.8)
    ax3.bar(dims, fake_dim_stds, alpha=0.5, label='Fake', width=0.8)
    ax3.set_title(f'{title_prefix} Per-Dim Std')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _sanitize_config(cfg: dict):
    cleaned = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gan_cfg = resolve_gan_config(fold_idx=args.fold_idx, data_dir=args.data_dir, output_dir=args.output_root)
    demo_paths = resolve_demo_paths(gan_cfg)
    teacher_path = Path(args.teacher_checkpoint) if args.teacher_checkpoint else demo_paths.demo_root / 'training' / f'fold{args.fold_idx}_best_model.pt'
    teacher_ckpt = torch.load(teacher_path, map_location=device)
    teacher = build_teacher(teacher_ckpt, device)

    dm = DemoGANDataManager(gan_cfg)
    dls = dm.get_dataloaders()
    dataloader = dls['test']

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_config = ckpt.get('config', {})
    sample = next(iter(dataloader))
    real_a, real_v = sample[0], sample[1]
    config = FeatureGANConfig(
        num_classes=51,
        audio_seq_len=real_a.shape[1],
        audio_feat_dim=real_a.shape[2],
        video_seq_len=real_v.shape[1],
        video_feat_dim=real_v.shape[2],
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
    gan.load_checkpoint(args.checkpoint)

    total = 0
    correct_ff = correct_rr = correct_rf = correct_fr = 0
    real_a_list = []
    fake_a_list = []
    real_v_list = []
    fake_v_list = []
    labels_list = []
    joint_gap_curve = []
    joint_real_vals = []
    joint_fake_vals = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_batches:
                break
            real_a, real_v, len_a, len_v, y = batch
            real_a = real_a.to(device)
            real_v = real_v.to(device)
            len_a = len_a.to(device)
            len_v = len_v.to(device)
            labels = y.to(device)
            z = torch.randn(labels.shape[0], gan.config.z_dim, device=device)
            fake_a = gan.audio_generator(z, labels)
            fake_v = gan.video_generator(z, labels)
            fake_a = gan._apply_per_sample_znorm(fake_a, len_a)
            fake_a = gan._mask_by_len(fake_a, len_a)
            fake_v = gan._mask_by_len(fake_v, len_v)
            la = torch.clamp(len_a, min=1, max=fake_a.shape[1]).long()
            lv = torch.clamp(len_v, min=1, max=fake_v.shape[1]).long()
            preds_ff, _ = teacher(fake_a, fake_v, la, lv)
            preds_rr, _ = teacher(real_a, real_v, la, lv)
            preds_rf, _ = teacher(real_a, fake_v, la, lv)
            preds_fr, _ = teacher(fake_a, real_v, la, lv)
            correct_ff += (preds_ff.argmax(1) == labels).sum().item()
            correct_rr += (preds_rr.argmax(1) == labels).sum().item()
            correct_rf += (preds_rf.argmax(1) == labels).sum().item()
            correct_fr += (preds_fr.argmax(1) == labels).sum().item()
            total += labels.shape[0]

            real_a_feat = masked_mean_std(real_a, len_a)
            fake_a_feat = masked_mean_std(fake_a, len_a)
            real_v_feat = masked_mean(real_v, len_v)
            fake_v_feat = masked_mean(fake_v, len_v)
            real_a_list.append(real_a_feat)
            fake_a_list.append(fake_a_feat)
            real_v_list.append(real_v_feat)
            fake_v_list.append(fake_v_feat)
            labels_list.append(labels.cpu().numpy())

            real_a_pool = torch.tensor(real_a_feat[:, :config.audio_feat_dim], device=device, dtype=torch.float32)
            fake_a_pool = torch.tensor(fake_a_feat[:, :config.audio_feat_dim], device=device, dtype=torch.float32)
            real_v_pool = torch.tensor(real_v_feat, device=device, dtype=torch.float32)
            fake_v_pool = torch.tensor(fake_v_feat, device=device, dtype=torch.float32)
            j_real = gan.joint_discriminator(real_a_pool, real_v_pool)
            j_fake = gan.joint_discriminator(fake_a_pool, fake_v_pool)
            joint_real_vals.append(j_real)
            joint_fake_vals.append(j_fake)
            joint_gap_curve.append((j_real.mean() - j_fake.mean()).item())

    real_audio = np.concatenate(real_a_list, axis=0)
    fake_audio = np.concatenate(fake_a_list, axis=0)
    real_video = np.concatenate(real_v_list, axis=0)
    fake_video = np.concatenate(fake_v_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    teacher_acc = {
        'fake_fake': correct_ff / total if total else 0.0,
        'real_real': correct_rr / total if total else 0.0,
        'real_fake': correct_rf / total if total else 0.0,
        'fake_real': correct_fr / total if total else 0.0,
        'teacher_gap': (correct_ff - correct_rr) / total if total else 0.0,
    }
    diversity = {
        'audio': class_diversity_ratio(real_audio, fake_audio, labels_all),
        'video': class_diversity_ratio(real_video, fake_video, labels_all),
    }
    joint_real = torch.cat(joint_real_vals, dim=0).mean().item() if joint_real_vals else 0.0
    joint_fake = torch.cat(joint_fake_vals, dim=0).mean().item() if joint_fake_vals else 0.0

    out_dir = Path(args.output_dir) if args.output_dir else demo_paths.demo_root / 'gan_analysis' / Path(args.checkpoint).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_distribution_stats(real_audio, fake_audio, str(out_dir / 'dist_audio.png'), 'Audio')
    plot_distribution_stats(real_video, fake_video, str(out_dir / 'dist_video.png'), 'Video')

    if not args.no_tsne:
        feats = {
            'real_audio': real_audio,
            'fake_audio': fake_audio,
            'real_video': real_video,
            'fake_video': fake_video,
            'labels': labels_all,
        }
        visualize_tsne(feats, str(out_dir / 'tsne_audio.png'), modality='audio')
        visualize_tsne(feats, str(out_dir / 'tsne_video.png'), modality='video')

    if joint_gap_curve:
        plt.figure(figsize=(6, 4))
        plt.plot(joint_gap_curve, linewidth=1.5)
        plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        plt.title('Joint Gap Curve (per batch)')
        plt.xlabel('Batch')
        plt.ylabel('Real - Fake logit')
        plt.tight_layout()
        plt.savefig(out_dir / 'joint_gap_curve.png', dpi=150)
        plt.close()

    collapse_flags = {
        'audio_mean_collapse': bool(abs(float(fake_audio[:, :config.audio_feat_dim].mean())) < 1e-4 and float(fake_audio[:, :config.audio_feat_dim].std()) < 1e-3),
        'video_low_variance': bool(float(fake_video.std()) < 0.05),
    }
    results = {
        'n_samples': int(total),
        'teacher_acc': teacher_acc,
        'audio': {'real': stats_basic(real_audio), 'fake': stats_basic(fake_audio)},
        'video': {'real': stats_basic(real_video), 'fake': stats_basic(fake_video)},
        'joint': {
            'real_logit_mean': joint_real,
            'fake_logit_mean': joint_fake,
            'gap': joint_real - joint_fake,
            'gap_curve': joint_gap_curve,
        },
        'diversity_ratio': diversity,
        'collapse_flags': collapse_flags,
        'meta': {
            'checkpoint': args.checkpoint,
            'teacher_checkpoint': str(teacher_path),
            'fold_idx': args.fold_idx,
            'num_batches': args.num_batches,
            'seed': args.seed,
            'gan_config': _sanitize_config(saved_config),
        },
    }
    with open(out_dir / 'analysis_results.json', 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    print(f'Saved GAN analysis to {out_dir}')


if __name__ == '__main__':
    main()
