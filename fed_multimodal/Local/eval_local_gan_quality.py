#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local GAN 质量评估脚本（主干精简版）

仅保留核心指标：
1) Teacher 在 fake 上的准确率（语义一致性）
2) Real/Fake 边缘分布统计（mean/std/min/max）
3) Joint Critic 的 real/fake logit gap（模态对齐）
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from model.mm_models import MMActionClassifier
from Local.dataloader import UCF101LocalDataManager
from generator.gan_generator import MultimodalFeatureGAN, FeatureGANConfig
from generator.eval_gan_quality import masked_mean, masked_mean_std, visualize_tsne

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
    parser = argparse.ArgumentParser(description='Local GAN Core Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--data_dir', type=str, default=str(Path(__file__).parents[1] / 'results'))
    parser.add_argument('--dataset_dir', type=str, default=str(Path(__file__).parents[1] / 'datasets' / 'ucf101'))
    parser.add_argument('--audio_feat', type=str, default='mfcc')
    parser.add_argument('--video_feat', type=str, default='mobilenet_v2')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_batches', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='results/gan_analysis')
    parser.add_argument('--use_train', action='store_true', help='Use train set instead of test set')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_tsne', action='store_true', help='Skip t-SNE visualization')
    parser.add_argument('--no_extra_metrics', action='store_true', help='Skip FID/MMD/domain classifier metrics')
    parser.add_argument('--no_domain_clf', action='store_true', help='Skip domain classifier metric only')
    parser.add_argument('--max_metric_samples', type=int, default=2000,
                        help='Max samples per split for FID/MMD/domain metrics')

    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--att', action='store_true')
    parser.add_argument('--att_name', type=str, default='base')

    return parser.parse_args()


def masked_mean_torch(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.shape
    if lengths is None:
        return x.mean(dim=1)
    lengths = torch.clamp(lengths.long(), min=1, max=T)
    idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
    mask = (idx < lengths[:, None]).float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom


def masked_mean_std_torch(x: torch.Tensor, lengths: torch.Tensor):
    B, T, D = x.shape
    if lengths is None:
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        return mean, std
    lengths = torch.clamp(lengths.long(), min=1, max=T)
    idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
    mask = (idx < lengths[:, None]).float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    mean = (x * mask).sum(dim=1) / denom
    var = ((x - mean[:, None, :]) ** 2 * mask).sum(dim=1) / denom
    std = torch.sqrt(var + 1e-8)
    return mean, std


def stats_basic(x: np.ndarray):
    return {
        'mean': float(x.mean()),
        'std': float(x.std()),
        'min': float(x.min()),
        'max': float(x.max()),
    }


def _covariance(x: np.ndarray):
    if x.shape[0] <= 1:
        return None
    x = x.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    return cov


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-6):
    mat = (mat + mat.T) * 0.5
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(real: np.ndarray, fake: np.ndarray):
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return None
    real = real.astype(np.float64)
    fake = fake.astype(np.float64)
    mu_r = real.mean(axis=0)
    mu_f = fake.mean(axis=0)
    cov_r = _covariance(real)
    cov_f = _covariance(fake)
    if cov_r is None or cov_f is None:
        return None
    dim = cov_r.shape[0]
    cov_r = cov_r + np.eye(dim) * 1e-6
    cov_f = cov_f + np.eye(dim) * 1e-6
    cov_r_sqrt = _sqrtm_psd(cov_r)
    cov_prod = cov_r_sqrt @ cov_f @ cov_r_sqrt
    cov_prod = (cov_prod + cov_prod.T) * 0.5
    cov_mean = _sqrtm_psd(cov_prod)
    diff = mu_r - mu_f
    fid = float(diff @ diff + np.trace(cov_r + cov_f - 2.0 * cov_mean))
    if fid < 0:
        fid = 0.0
    return fid


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_norm = (x * x).sum(axis=1, keepdims=True)
    y_norm = (y * y).sum(axis=1, keepdims=True)
    dist2 = x_norm + y_norm.T - 2.0 * (x @ y.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-dist2 / (2.0 * sigma ** 2))


def _median_sigma(x: np.ndarray, seed: int = 42, max_pairs: int = 2000):
    if x.shape[0] < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=min(x.shape[0], 512), replace=False)
    sample = x[idx]
    dist2 = np.sum(sample ** 2, axis=1, keepdims=True) + np.sum(sample ** 2, axis=1) - 2.0 * (sample @ sample.T)
    dist2 = dist2[np.triu_indices(dist2.shape[0], k=1)]
    if dist2.size == 0:
        return None
    if dist2.size > max_pairs:
        dist2 = rng.choice(dist2, size=max_pairs, replace=False)
    med = np.median(dist2)
    if med <= 0:
        return None
    return float(np.sqrt(0.5 * med))


def mmd_rbf(real: np.ndarray, fake: np.ndarray, seed: int = 42):
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return None
    sigma = _median_sigma(np.vstack([real, fake]), seed=seed)
    if sigma is None:
        return None
    k_xx = _rbf_kernel(real, real, sigma)
    k_yy = _rbf_kernel(fake, fake, sigma)
    k_xy = _rbf_kernel(real, fake, sigma)
    mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return max(mmd2, 0.0)


def domain_classifier_metrics(real: np.ndarray, fake: np.ndarray, seed: int = 42):
    if real.shape[0] < 10 or fake.shape[0] < 10:
        return None
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
    except Exception:
        return None
    x = np.vstack([real, fake]).astype(np.float64)
    y = np.concatenate([np.zeros(real.shape[0]), np.ones(fake.shape[0])])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=200, solver='liblinear')
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)[:, 1]
    acc = float(accuracy_score(y_test, pred))
    try:
        auc = float(roc_auc_score(y_test, prob))
    except ValueError:
        auc = None
    return {
        'acc': acc,
        'auc': auc,
        'n_train': int(x_train.shape[0]),
        'n_test': int(x_test.shape[0]),
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
    mean_ratio = float(np.mean(list(ratios.values())))
    return {'mean_ratio': mean_ratio, 'per_class': ratios, 'n_classes': len(ratios)}


def plot_distribution_stats(real_data: np.ndarray, fake_data: np.ndarray, save_path: str, title_prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Value Distribution (All Features)
    ax1 = axes[0]
    ax1.hist(real_data.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax1.hist(fake_data.flatten(), bins=50, alpha=0.5, label='Fake', density=True)
    ax1.set_title(f'{title_prefix} Value Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()

    # 2) Per-Dimension Mean
    ax2 = axes[1]
    real_dim_means = real_data.mean(axis=0)
    fake_dim_means = fake_data.mean(axis=0)
    dims = np.arange(len(real_dim_means))
    ax2.bar(dims, real_dim_means, alpha=0.5, label='Real', width=0.8)
    ax2.bar(dims, fake_dim_means, alpha=0.5, label='Fake', width=0.8)
    ax2.set_title(f'{title_prefix} Per-Dimension Mean')
    ax2.set_xlabel('Feature Dimension')
    ax2.set_ylabel('Mean')
    ax2.legend()

    # 3) Per-Dimension Std
    ax3 = axes[2]
    real_dim_stds = real_data.std(axis=0)
    fake_dim_stds = fake_data.std(axis=0)
    ax3.bar(dims, real_dim_stds, alpha=0.5, label='Real', width=0.8)
    ax3.bar(dims, fake_dim_stds, alpha=0.5, label='Fake', width=0.8)
    ax3.set_title(f'{title_prefix} Per-Dimension Std')
    ax3.set_xlabel('Feature Dimension')
    ax3.set_ylabel('Std')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def sanitize_config(cfg: dict):
    cleaned = {}
    for k, v in cfg.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            cleaned[k] = v
        elif isinstance(v, (list, tuple)):
            cleaned[k] = list(v)
        elif isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        else:
            cleaned[k] = str(v)
    return cleaned


def evaluate_core(gan, teacher, dataloader, num_batches, device):
    gan.audio_generator.eval()
    gan.video_generator.eval()
    gan.joint_discriminator.eval()
    teacher.eval()

    total = 0
    correct = 0
    correct_rr = 0
    correct_rf = 0
    correct_fr = 0

    real_a_mean_list, real_a_std_list = [], []
    fake_a_mean_list, fake_a_std_list = [], []
    real_v_list, fake_v_list = [], []
    labels_list = []
    joint_real_list, joint_fake_list = [], []
    joint_gap_curve = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            real_a, real_v, len_a, len_v, y = batch
            real_a = real_a.to(device)
            real_v = real_v.to(device)
            len_a = len_a.to(device)
            len_v = len_v.to(device)
            labels = y.to(device)

            # 生成 fake
            z = torch.randn(labels.shape[0], gan.config.z_dim, device=device)
            fake_a = gan.audio_generator(z, labels)
            fake_v = gan.video_generator(z, labels)

            # 对 fake audio 做 per-sample Z-norm + mask，保持与训练一致
            fake_a = gan._apply_per_sample_znorm(fake_a, len_a)
            fake_a = gan._mask_by_len(fake_a, len_a)
            fake_v = gan._mask_by_len(fake_v, len_v)

            # Teacher Acc (四种组合)
            la = torch.clamp(len_a, min=1, max=fake_a.shape[1]).long()
            lv = torch.clamp(len_v, min=1, max=fake_v.shape[1]).long()
            preds_ff, _ = teacher(fake_a, fake_v, la, lv)
            preds_rr, _ = teacher(real_a, real_v, la, lv)
            preds_rf, _ = teacher(real_a, fake_v, la, lv)
            preds_fr, _ = teacher(fake_a, real_v, la, lv)

            correct += (preds_ff.argmax(1) == labels).sum().item()
            correct_rr += (preds_rr.argmax(1) == labels).sum().item()
            correct_rf += (preds_rf.argmax(1) == labels).sum().item()
            correct_fr += (preds_fr.argmax(1) == labels).sum().item()
            total += labels.shape[0]

            # 分布统计（audio: mean+std, video: mean）
            real_a_mean, real_a_std = masked_mean_std_torch(real_a, len_a)
            fake_a_mean, fake_a_std = masked_mean_std_torch(fake_a, len_a)
            real_a_pool = masked_mean_torch(real_a, len_a)
            fake_a_pool = masked_mean_torch(fake_a, len_a)
            real_v_pool = masked_mean_torch(real_v, len_v)
            fake_v_pool = masked_mean_torch(fake_v, len_v)

            real_a_mean_list.append(real_a_mean)
            real_a_std_list.append(real_a_std)
            fake_a_mean_list.append(fake_a_mean)
            fake_a_std_list.append(fake_a_std)
            real_v_list.append(real_v_pool)
            fake_v_list.append(fake_v_pool)
            labels_list.append(labels)

            # Joint gap
            joint_real_val = gan.joint_discriminator(real_a_pool, real_v_pool)
            joint_fake_val = gan.joint_discriminator(fake_a_pool, fake_v_pool)
            joint_real_list.append(joint_real_val)
            joint_fake_list.append(joint_fake_val)
            joint_gap_curve.append((joint_real_val.mean() - joint_fake_val.mean()).item())

    teacher_acc = correct / total if total else 0.0
    teacher_rr = correct_rr / total if total else 0.0
    teacher_rf = correct_rf / total if total else 0.0
    teacher_fr = correct_fr / total if total else 0.0

    real_a_mean = torch.cat(real_a_mean_list, dim=0).cpu().numpy()
    real_a_std = torch.cat(real_a_std_list, dim=0).cpu().numpy()
    fake_a_mean = torch.cat(fake_a_mean_list, dim=0).cpu().numpy()
    fake_a_std = torch.cat(fake_a_std_list, dim=0).cpu().numpy()
    real_v = torch.cat(real_v_list, dim=0).cpu().numpy()
    fake_v = torch.cat(fake_v_list, dim=0).cpu().numpy()
    labels_all = torch.cat(labels_list, dim=0).cpu().numpy()

    audio_real = {
        'mean': stats_basic(real_a_mean),
        'std': stats_basic(real_a_std),
    }
    audio_fake = {
        'mean': stats_basic(fake_a_mean),
        'std': stats_basic(fake_a_std),
    }
    video_real = stats_basic(real_v)
    video_fake = stats_basic(fake_v)

    # 类内多样性比（fake/real），用于检测 mode collapse
    real_a_feat = np.concatenate([real_a_mean, real_a_std], axis=1)
    fake_a_feat = np.concatenate([fake_a_mean, fake_a_std], axis=1)
    div_audio = class_diversity_ratio(real_a_feat, fake_a_feat, labels_all)
    div_video = class_diversity_ratio(real_v, fake_v, labels_all)

    joint_real = torch.cat(joint_real_list, dim=0).mean().item() if joint_real_list else 0.0
    joint_fake = torch.cat(joint_fake_list, dim=0).mean().item() if joint_fake_list else 0.0

    return {
        'n_samples': total,
        'teacher_acc': {
            'fake_fake': teacher_acc,
            'real_real': teacher_rr,
            'real_fake': teacher_rf,
            'fake_real': teacher_fr,
        },
        'audio': {
            'real': audio_real,
            'fake': audio_fake,
        },
        'video': {
            'real': video_real,
            'fake': video_fake,
        },
        'joint': {
            'real_logit_mean': joint_real,
            'fake_logit_mean': joint_fake,
            'gap': joint_real - joint_fake,
            'gap_curve': joint_gap_curve,
        },
        'diversity_ratio': {
            'audio': div_audio,
            'video': div_video,
        },
        # keep raw pooled features for plotting
        '_plot_cache': {
            'audio_real': real_a_feat,
            'audio_fake': fake_a_feat,
            'video_real': real_v,
            'video_fake': fake_v,
        },
    }


def compute_similarity_metrics(plot_cache, seed: int, max_samples: int, run_domain_clf: bool):
    rng = np.random.default_rng(seed)
    metrics = {}

    def _subset(arr, idx):
        if idx is None:
            return arr
        return arr[idx]

    n_real = min(plot_cache['audio_real'].shape[0], plot_cache['video_real'].shape[0])
    n_fake = min(plot_cache['audio_fake'].shape[0], plot_cache['video_fake'].shape[0])
    real_idx = None
    fake_idx = None
    if n_real > max_samples:
        real_idx = rng.choice(n_real, size=max_samples, replace=False)
    if n_fake > max_samples:
        fake_idx = rng.choice(n_fake, size=max_samples, replace=False)

    audio_real = _subset(plot_cache['audio_real'][:n_real], real_idx)
    video_real = _subset(plot_cache['video_real'][:n_real], real_idx)
    audio_fake = _subset(plot_cache['audio_fake'][:n_fake], fake_idx)
    video_fake = _subset(plot_cache['video_fake'][:n_fake], fake_idx)
    joint_real = np.concatenate([audio_real, video_real], axis=1)
    joint_fake = np.concatenate([audio_fake, video_fake], axis=1)

    metrics['fid'] = {
        'audio': frechet_distance(audio_real, audio_fake),
        'video': frechet_distance(video_real, video_fake),
        'joint': frechet_distance(joint_real, joint_fake),
    }
    metrics['mmd_rbf'] = {
        'audio': mmd_rbf(audio_real, audio_fake, seed=seed),
        'video': mmd_rbf(video_real, video_fake, seed=seed),
        'joint': mmd_rbf(joint_real, joint_fake, seed=seed),
    }
    if run_domain_clf:
        metrics['domain_clf'] = {
            'audio': domain_classifier_metrics(audio_real, audio_fake, seed=seed),
            'video': domain_classifier_metrics(video_real, video_fake, seed=seed),
            'joint': domain_classifier_metrics(joint_real, joint_fake, seed=seed),
        }
    metrics['n_samples'] = {
        'audio': int(audio_real.shape[0]),
        'video': int(video_real.shape[0]),
        'joint': int(joint_real.shape[0]),
    }
    return metrics


def collect_features_for_tsne(gan, dataloader, device, num_batches):
    gan.audio_generator.eval()
    gan.video_generator.eval()

    real_audio_list, fake_audio_list = [], []
    real_video_list, fake_video_list = [], []
    labels_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            real_a, real_v, len_a, len_v, y = batch
            real_a = real_a.to(device)
            real_v = real_v.to(device)
            len_a = len_a.to(device)
            len_v = len_v.to(device)
            y = y.to(device)

            z = torch.randn(y.shape[0], gan.config.z_dim, device=device)
            fake_a = gan.audio_generator(z, y)
            fake_v = gan.video_generator(z, y)

            fake_a = gan._apply_per_sample_znorm(fake_a, len_a)
            fake_a = gan._mask_by_len(fake_a, len_a)
            fake_v = gan._mask_by_len(fake_v, len_v)

            real_audio_list.append(masked_mean_std(real_a, len_a))
            fake_audio_list.append(masked_mean_std(fake_a, len_a))
            real_video_list.append(masked_mean(real_v, len_v))
            fake_video_list.append(masked_mean(fake_v, len_v))
            labels_list.append(y.cpu().numpy())

    return {
        'real_audio': np.concatenate(real_audio_list, axis=0),
        'fake_audio': np.concatenate(fake_audio_list, axis=0),
        'real_video': np.concatenate(real_video_list, axis=0),
        'fake_video': np.concatenate(fake_video_list, axis=0),
        'labels': np.concatenate(labels_list, axis=0),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("=" * 60)
    logging.info("Local GAN Core Evaluation")
    logging.info("=" * 60)
    logging.info(f"Device: {device}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Teacher: {args.model_path}")

    dm = UCF101LocalDataManager(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        audio_feat=args.audio_feat,
        video_feat=args.video_feat,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dls = dm.get_dataloaders()
    dataloader = dls['full_train'] if args.use_train else dls['test']

    # Teacher
    checkpoint = torch.load(args.model_path, map_location=device)
    saved_args = checkpoint.get('args', {})
    hid_size = saved_args.get('hid_size', args.hid_size) if isinstance(saved_args, dict) else getattr(saved_args, 'hid_size', args.hid_size)
    en_att = saved_args.get('att', args.att) if isinstance(saved_args, dict) else getattr(saved_args, 'att', args.att)
    att_name = saved_args.get('att_name', args.att_name) if isinstance(saved_args, dict) else getattr(saved_args, 'att_name', args.att_name)

    teacher = MMActionClassifier(
        num_classes=dm.num_classes,
        audio_input_dim=dm.audio_feat_dim,
        video_input_dim=dm.video_feat_dim,
        d_hid=hid_size,
        en_att=en_att,
        att_name=att_name
    ).to(device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()

    # GAN config from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_config = ckpt.get('config', {})
    sample = next(iter(dataloader))
    real_a, real_v = sample[0], sample[1]

    config = FeatureGANConfig(
        num_classes=dm.num_classes,
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
        device=device
    )

    gan_args = type('LocalArgs', (), {'dataset': 'ucf101'})()
    gan = MultimodalFeatureGAN(args=gan_args, global_model=teacher, config=config)
    gan.load_checkpoint(args.checkpoint)

    results = evaluate_core(gan, teacher, dataloader, args.num_batches, device)

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_tsne:
        logging.info("Running t-SNE visualization...")
        feats = collect_features_for_tsne(gan, dataloader, device, args.num_batches)
        visualize_tsne(feats, os.path.join(args.output_dir, 'tsne_audio.png'), modality='audio')
        visualize_tsne(feats, os.path.join(args.output_dir, 'tsne_video.png'), modality='video')

    # Distribution plots (Value Distribution, Per-Dim Mean/Std)
    plot_cache = results.pop('_plot_cache', None)
    if plot_cache:
        if not args.no_extra_metrics:
            metrics = compute_similarity_metrics(
                plot_cache,
                seed=args.seed,
                max_samples=args.max_metric_samples,
                run_domain_clf=not args.no_domain_clf,
            )
            results['similarity_metrics'] = metrics
        plot_distribution_stats(
            plot_cache['audio_real'],
            plot_cache['audio_fake'],
            os.path.join(args.output_dir, 'dist_audio.png'),
            title_prefix='Audio',
        )
        plot_distribution_stats(
            plot_cache['video_real'],
            plot_cache['video_fake'],
            os.path.join(args.output_dir, 'dist_video.png'),
            title_prefix='Video',
        )

    # Joint gap 曲线
    gap_curve = results.get('joint', {}).get('gap_curve', [])
    if gap_curve:
        plt.figure(figsize=(6, 4))
        plt.plot(gap_curve, linewidth=1.5)
        plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        plt.title('Joint Gap Curve (per batch)')
        plt.xlabel('Batch')
        plt.ylabel('Real - Fake logit')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'joint_gap_curve.png'), dpi=150)
        plt.close()

    # attach config/params to results.json
    results['meta'] = {
        'checkpoint': args.checkpoint,
        'model_path': args.model_path,
        'num_batches': args.num_batches,
        'use_train': args.use_train,
        'batch_size': args.batch_size,
        'audio_feat': args.audio_feat,
        'video_feat': args.video_feat,
        'seed': args.seed,
        'gan_config': sanitize_config(saved_config),
    }

    out_path = os.path.join(args.output_dir, 'analysis_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("\n" + "=" * 60)
    logging.info("KEY METRICS SUMMARY")
    logging.info("=" * 60)
    ta = results['teacher_acc']
    logging.info(
        "Teacher Acc: "
        f"real+real={ta['real_real']:.4f}, "
        f"fake+fake={ta['fake_fake']:.4f}, "
        f"real+fake={ta['real_fake']:.4f}, "
        f"fake+real={ta['fake_real']:.4f}"
    )
    logging.info(
        "Audio mean (per-dim time-mean): "
        f"real mu={results['audio']['real']['mean']['mean']:.4f}, "
        f"fake mu={results['audio']['fake']['mean']['mean']:.4f}"
    )
    logging.info(
        "Audio std  (per-dim time-std): "
        f"real mu={results['audio']['real']['std']['mean']:.4f}, "
        f"fake mu={results['audio']['fake']['std']['mean']:.4f}"
    )
    logging.info(
        "Video pooled mean/std: "
        f"real mu={results['video']['real']['mean']:.4f}, sigma={results['video']['real']['std']:.4f} | "
        f"fake mu={results['video']['fake']['mean']:.4f}, sigma={results['video']['fake']['std']:.4f}"
    )
    logging.info(
        f"Joint Gap: {results['joint']['gap']:.4f} "
        f"(real={results['joint']['real_logit_mean']:.3f}, fake={results['joint']['fake_logit_mean']:.3f})"
    )
    dr = results['diversity_ratio']
    if dr['video']['mean_ratio'] is not None:
        logging.info(f"Video diversity ratio (fake/real): {dr['video']['mean_ratio']:.3f}")
    if dr['audio']['mean_ratio'] is not None:
        logging.info(f"Audio diversity ratio (fake/real): {dr['audio']['mean_ratio']:.3f}")
    sim = results.get('similarity_metrics')
    if sim:
        def _fmt(val, prec):
            if val is None:
                return "n/a"
            return f"{val:.{prec}f}"
        fid = sim.get('fid', {})
        mmd = sim.get('mmd_rbf', {})
        logging.info(
            "FID (audio/video/joint): "
            f"{_fmt(fid.get('audio'), 4)} / {_fmt(fid.get('video'), 4)} / {_fmt(fid.get('joint'), 4)}"
        )
        logging.info(
            "MMD (audio/video/joint): "
            f"{_fmt(mmd.get('audio'), 6)} / {_fmt(mmd.get('video'), 6)} / {_fmt(mmd.get('joint'), 6)}"
        )
        domain = sim.get('domain_clf', {})
        if domain:
            da = domain.get('audio', {})
            dv = domain.get('video', {})
            dj = domain.get('joint', {})
            if da:
                logging.info(
                    "Domain Clf Acc (audio/video/joint): "
                    f"{_fmt(da.get('acc'), 3)} / {_fmt(dv.get('acc'), 3)} / {_fmt(dj.get('acc'), 3)}"
                )
    logging.info(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
