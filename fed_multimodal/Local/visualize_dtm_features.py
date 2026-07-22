#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTM-GAN 生成数据可视化：t-SNE + 分布对比（audio / video 两模态）

复用 generator/eval_gan_quality.py 的 masked_mean / masked_mean_std 工具函数
将变长时序特征压成定长向量，再与真实 UCF101 特征做 t-SNE 与逐维均值/标准差对比。

与原 eval_gan_quality.py 的区别：real / fake 来自不同数据集、标签独立，
因此可视化函数接受 real_labels / fake_labels 分别传入（原函数假设同 batch 同标签）。

Usage:
    PYTHONPATH=/home/xp/fedpoi python visualize_dtm_features.py \
        --synthetic_data .../dtm_final_dtm_final_train5100.pt \
        --dataset_dir /home/xp/fedpoigan/fed_multimodal/datasets/ucf101 \
        --max_per_class 50 --output_dir .../dtm_vis_final50
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 服务器端非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parents[1]))

from Local.dataloader import UCF101LocalDataManager


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
    """对时序特征按有效长度取均值（向量化，忽略 padding）。返回 (B, D)。"""
    B, T, D = x.shape
    lengths = torch.clamp(lengths.long(), min=1, max=T)
    idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
    mask = (idx < lengths[:, None]).float().unsqueeze(-1)  # (B, T, 1)
    denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
    return ((x * mask).sum(dim=1) / denom).cpu().numpy()


def masked_mean_std(x: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
    """对时序特征按有效长度取 (mean, std) 拼接（向量化）。audio 经 per-sample Z-norm 后
    mean≈0 信息量低，用 (mean, std) 拼接保留更多分布信息。返回 (B, 2*D)。"""
    B, T, D = x.shape
    lengths = torch.clamp(lengths.long(), min=1, max=T)
    idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
    mask = (idx < lengths[:, None]).float().unsqueeze(-1)  # (B, T, 1)
    denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
    mean = (x * mask).sum(dim=1) / denom  # (B, D)
    var = (((x - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / denom
    return torch.cat([mean, var.sqrt()], dim=1).cpu().numpy()

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)


def _to_numpy(x):
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def sample_per_class(audio, video, len_a, len_v, labels, max_per_class, seed=42):
    """每类均匀采样 max_per_class 个样本，保证 real/fake 各类等量可比。"""
    if max_per_class is None or max_per_class <= 0:
        return audio, video, len_a, len_v, labels
    rng = np.random.default_rng(seed)
    labels_np = _to_numpy(labels)
    keep = []
    for c in np.unique(labels_np):
        idx = np.where(labels_np == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.extend(idx.tolist())
    keep = sorted(keep)
    idx_t = torch.as_tensor(keep, dtype=torch.long)
    return audio[idx_t], video[idx_t], len_a[idx_t], len_v[idx_t], labels[idx_t]


def load_fake(pt_path, max_per_class):
    d = torch.load(pt_path, map_location='cpu')
    audio = d.get('audio_features', d.get('audio')).float()
    video = d.get('video_features', d.get('video')).float()
    len_a = d.get('audio_lengths', d.get('len_a'))
    len_v = d.get('video_lengths', d.get('len_v'))
    labels = d.get('labels', d.get('train_label')).long()
    T_a = audio.size(1)
    T_v = video.size(1)
    if len_a is None:
        len_a = torch.full((len(labels),), T_a, dtype=torch.long)
    if len_v is None:
        len_v = torch.full((len(labels),), T_v, dtype=torch.long)
    len_a = len_a.long().clamp(max=T_a)
    len_v = len_v.long().clamp(max=T_v)
    logging.info(f'fake: {len(labels)} samples, audio {tuple(audio.shape)}, video {tuple(video.shape)}')
    return sample_per_class(audio, video, len_a, len_v, labels, max_per_class)


def _sample_indices_per_class(labels, max_per_class, seed=42):
    """每类均匀采样 max_per_class 个索引，返回排序后的索引数组。"""
    if max_per_class is None or max_per_class <= 0:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.extend(idx.tolist())
    return np.array(sorted(keep))


def load_real(dm, max_per_class, split='train'):
    """直接从 dataset 的原始 numpy 特征构建，绕过 __getitem__/collate 的逐样本
    tensor 转换与 padding（迭代 dataloader 在 4893 样本上过慢）。先按类采样索引，
    只构建采样后样本的定长 tensor。"""
    loaders = dm.get_dataloaders(val_split=0.0)
    dataset = loaders['full_train' if split == 'train' else 'test'].dataset
    n = len(dataset)
    labels_all = np.array([int(dataset.audio_data[i][-2]) for i in range(n)])
    keep = _sample_indices_per_class(labels_all, max_per_class)
    m = len(keep)
    T_a, D_a = dm.audio_seq_len, dm.audio_feat_dim
    T_v, D_v = dm.video_seq_len, dm.video_feat_dim
    audio = torch.zeros(m, T_a, D_a, dtype=torch.float32)
    video = torch.zeros(m, T_v, D_v, dtype=torch.float32)
    len_a = torch.zeros(m, dtype=torch.long)
    len_v = torch.zeros(m, dtype=torch.long)
    for j, i in enumerate(keep):
        a = np.asarray(dataset.audio_data[i][-1], dtype=np.float32)
        v = np.asarray(dataset.video_data[i][-1], dtype=np.float32)
        if a.ndim == 3:
            a = a[0]
        if v.ndim == 3:
            v = v[0]
        la = min(a.shape[0], T_a)
        lv = min(v.shape[0], T_v)
        audio[j, :la] = torch.from_numpy(a[:la])
        video[j, :lv] = torch.from_numpy(v[:lv])
        len_a[j] = la
        len_v[j] = lv
    labels = torch.from_numpy(labels_all[keep]).long()
    logging.info(f'real({split}): {m}/{n} samples (per-class<={max_per_class}), '
                 f'audio {tuple(audio.shape)}, video {tuple(video.shape)}')
    return audio, video, len_a, len_v, labels


def to_feature_dict(real, fake):
    """时序特征 → 定长向量。audio 用 (mean,std) 拼接，video 用 mean。"""
    ra, rv, rla, rlv, ry = real
    fa, fv, fla, flv, fy = fake
    feats = {
        'real_audio': masked_mean_std(ra, rla),
        'fake_audio': masked_mean_std(fa, fla),
        'real_video': masked_mean(rv, rlv),
        'fake_video': masked_mean(fv, flv),
        'real_labels': _to_numpy(ry),
        'fake_labels': _to_numpy(fy),
    }
    for k, v in feats.items():
        logging.info(f'  {k}: shape {v.shape}')
    return feats


def visualize_tsne_dtm(features, save_path, modality, max_classes=10):
    real = features[f'real_{modality}']
    fake = features[f'fake_{modality}']
    real_labels = features['real_labels']
    fake_labels = features['fake_labels']

    combined = np.vstack([real, fake])
    combined_labels = np.concatenate([real_labels, fake_labels])
    domain = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])

    logging.info(f't-SNE [{modality}]: {combined.shape[0]} samples, fitting (CPU)...')
    # 高维数据先 PCA 降到 50 维（sklearn 推荐 t-SNE 预处理），显著提速且去噪
    if combined.shape[1] > 50:
        from sklearn.decomposition import PCA
        n_comp = min(50, combined.shape[0] - 1, combined.shape[1])
        combined = PCA(n_components=n_comp, random_state=42).fit_transform(combined)
        logging.info(f'  PCA预处理 -> {combined.shape[1]} dims')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500,
                init='pca', learning_rate='auto')
    emb = tsne.fit_transform(combined)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1) Real vs Fake 域分布
    ax1 = axes[0, 0]
    colors = ['steelblue' if d == 0 else 'crimson' for d in domain]
    ax1.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.45, s=18)
    ax1.set_title(f't-SNE: Real (Blue) vs DTM-Fake (Red) — {modality.upper()}', fontsize=12)
    ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')
    ax1.legend(handles=[Patch(facecolor='steelblue', label='Real'),
                        Patch(facecolor='crimson', label='DTM-Fake')])

    # 2) 全部样本按类别染色
    ax2 = axes[0, 1]
    mask = combined_labels < max_classes
    sc = ax2.scatter(emb[mask, 0], emb[mask, 1], c=combined_labels[mask],
                     cmap='tab10', alpha=0.6, s=20)
    ax2.set_title(f'All Samples by Class (0–{max_classes - 1})', fontsize=12)
    ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=ax2, label='Class')

    # 3) Real 按类别
    ax3 = axes[1, 0]
    real_emb = emb[:len(real)]
    rmask = real_labels < max_classes
    sc3 = ax3.scatter(real_emb[rmask, 0], real_emb[rmask, 1], c=real_labels[rmask],
                      cmap='tab10', alpha=0.7, s=25)
    ax3.set_title('Real Features by Class', fontsize=12)
    ax3.set_xlabel('t-SNE 1'); ax3.set_ylabel('t-SNE 2')
    plt.colorbar(sc3, ax=ax3, label='Class')

    # 4) Fake 按类别
    ax4 = axes[1, 1]
    fake_emb = emb[len(real):]
    fmask = fake_labels < max_classes
    sc4 = ax4.scatter(fake_emb[fmask, 0], fake_emb[fmask, 1], c=fake_labels[fmask],
                      cmap='tab10', alpha=0.7, s=25)
    ax4.set_title('DTM-Fake Features by Class', fontsize=12)
    ax4.set_xlabel('t-SNE 1'); ax4.set_ylabel('t-SNE 2')
    plt.colorbar(sc4, ax=ax4, label='Class')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'saved -> {save_path}')


def distribution_dtm(features, save_path, modality, max_dims=80):
    """数值分布直方图 + 逐维 mean/std 对比。维度过多时均匀采样到 max_dims。"""
    real = features[f'real_{modality}']
    fake = features[f'fake_{modality}']

    real_mean = np.mean(real); fake_mean = np.mean(fake)
    real_std = np.std(real); fake_std = np.std(fake)
    logging.info(f'[{modality}] real: mean={real_mean:.4f} std={real_std:.4f} '
                 f'range=[{np.min(real):.3f}, {np.max(real):.3f}]')
    logging.info(f'[{modality}] fake: mean={fake_mean:.4f} std={fake_std:.4f} '
                 f'range=[{np.min(fake):.3f}, {np.max(fake):.3f}]')

    D = real.shape[1]
    if D > max_dims:
        sel = np.linspace(0, D - 1, max_dims).astype(int)
        dim_label = f'dim (sampled {max_dims}/{D})'
    else:
        sel = np.arange(D)
        dim_label = 'Feature Dimension'
    rm = np.mean(real, axis=0)[sel]; fm = np.mean(fake, axis=0)[sel]
    rs = np.std(real, axis=0)[sel]; fs = np.std(fake, axis=0)[sel]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax1 = axes[0]
    ax1.hist(real.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax1.hist(fake.flatten(), bins=50, alpha=0.5, label='DTM-Fake', density=True)
    ax1.set_title(f'{modality.upper()} Value Distribution')
    ax1.set_xlabel('Value'); ax1.set_ylabel('Density'); ax1.legend()

    ax2 = axes[1]
    x = np.arange(len(sel))
    ax2.bar(x - 0.2, rm, width=0.4, alpha=0.6, label='Real')
    ax2.bar(x + 0.2, fm, width=0.4, alpha=0.6, label='DTM-Fake')
    ax2.set_title(f'{modality.upper()} Per-Dim Mean')
    ax2.set_xlabel(dim_label); ax2.set_ylabel('Mean'); ax2.legend()

    ax3 = axes[2]
    ax3.bar(x - 0.2, rs, width=0.4, alpha=0.6, label='Real')
    ax3.bar(x + 0.2, fs, width=0.4, alpha=0.6, label='DTM-Fake')
    ax3.set_title(f'{modality.upper()} Per-Dim Std')
    ax3.set_xlabel(dim_label); ax3.set_ylabel('Std'); ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'saved -> {save_path}')


def parse_args():
    p = argparse.ArgumentParser(description='DTM-GAN 生成数据 t-SNE + 分布可视化')
    p.add_argument('--synthetic_data', required=True, help='DTM 合成特征 .pt')
    p.add_argument('--data_dir', default=str(Path(__file__).parents[1] / 'results'))
    p.add_argument('--dataset_dir', default='/home/xp/fedpoigan/fed_multimodal/datasets/ucf101')
    p.add_argument('--real_split', default='train', choices=['train', 'test'])
    p.add_argument('--max_per_class', type=int, default=50, help='每类采样数（控制 t-SNE 规模）')
    p.add_argument('--audio_feat', default='mfcc')
    p.add_argument('--video_feat', default='mobilenet_v2')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--split_idx', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--max_classes', type=int, default=10, help='t-SNE 类别染色显示前 N 类')
    p.add_argument('--output_dir', default='fed_multimodal/Local/results/dtm_vis')
    return p.parse_args()


def main():
    args = parse_args()
    dm = UCF101LocalDataManager(
        data_dir=args.data_dir, dataset_dir=args.dataset_dir,
        audio_feat=args.audio_feat, video_feat=args.video_feat,
        batch_size=args.batch_size, split_idx=args.split_idx,
        num_workers=args.num_workers,
    )
    real = load_real(dm, args.max_per_class, split=args.real_split)
    fake = load_fake(args.synthetic_data, args.max_per_class)
    features = to_feature_dict(real, fake)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for modality in ['audio', 'video']:
        visualize_tsne_dtm(
            features, out / f'tsne_{modality}.png', modality,
            max_classes=args.max_classes,
        )
        distribution_dtm(features, out / f'dist_{modality}.png', modality)
    logging.info(f'all done -> {out}')


if __name__ == '__main__':
    main()
