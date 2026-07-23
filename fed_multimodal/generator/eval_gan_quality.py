#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAN 特征质量评估脚本

本脚本提供全面的分析工具来评估 GAN 生成特征的质量：

1. t-SNE 可视化：真实与生成特征分布对比
2. 距离指标：中心点之间的余弦相似度和 L2 距离
3. 机器学习效能测试：在生成数据上训练，在真实数据上测试
4. 统计分布分析

用法：
    python eval_gan_quality.py --checkpoint path/to/checkpoint.pt --data_dir path/to/data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 服务器端非交互式后端
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import argparse
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm

# 添加父目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parents[2]))

from fed_multimodal.constants import constants
from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.model.mm_models import MMActionClassifier
from fed_multimodal.generator.gan_generator import MultimodalFeatureGAN, FeatureGANConfig

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
    """
    对时序特征按有效长度取均值（忽略 padding 部分）
    
    Args:
        x: (B, T, D) 时序特征
        lengths: (B,) 每个样本的有效长度
    
    Returns:
        (B, D) numpy array，每个样本的有效部分均值
    """
    B, T, D = x.shape
    result = []
    for i in range(B):
        valid_len = min(int(lengths[i].item()), T)
        if valid_len > 0:
            result.append(x[i, :valid_len, :].mean(dim=0).cpu().numpy())
        else:
            result.append(x[i].mean(dim=0).cpu().numpy())  # fallback
    return np.stack(result, axis=0)


def masked_mean_std(x: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
    """
    对时序特征按有效长度取 (mean, std) 拼接表示
    
    当 audio 做了 per-sample Z-score 后，mean ≈ 0 信息量很低，
    但 std 仍有信息。用 (mean, std) 拼接可以保留更多分布信息。
    
    Args:
        x: (B, T, D) 时序特征
        lengths: (B,) 每个样本的有效长度
    
    Returns:
        (B, 2*D) numpy array，每个样本的有效部分 [mean, std] 拼接
    """
    B, T, D = x.shape
    result = []
    for i in range(B):
        valid_len = min(int(lengths[i].item()), T)
        if valid_len > 1:
            valid_part = x[i, :valid_len, :]
            mean_vec = valid_part.mean(dim=0).cpu().numpy()
            std_vec = valid_part.std(dim=0).cpu().numpy()
        elif valid_len == 1:
            mean_vec = x[i, 0, :].cpu().numpy()
            std_vec = np.zeros(D)
        else:
            mean_vec = x[i].mean(dim=0).cpu().numpy()
            std_vec = x[i].std(dim=0).cpu().numpy()
        result.append(np.concatenate([mean_vec, std_vec]))
    return np.stack(result, axis=0)


def masked_mean_torch(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Torch 版本的 masked mean，用于需要梯度的场景
    
    Args:
        x: (B, T, D) 时序特征
        lengths: (B,) 每个样本的有效长度
    
    Returns:
        (B, D) torch.Tensor
    """
    B, T, D = x.shape
    lengths = torch.clamp(lengths.long(), min=1, max=T)
    idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
    mask = (idx < lengths[:, None]).float().unsqueeze(-1)  # (B, T, 1)
    
    denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
    return (x * mask).sum(dim=1) / denom


def collect_features(generator, dataloader, device, num_batches=10, use_same_labels=True):
    """
    Collect real and generated features for analysis
    
    Args:
        generator: GAN instance (GANgeneratorer or MultimodalFeatureGAN)
        dataloader: DataLoader with real data
        device: torch device
        num_batches: number of batches to collect
        use_same_labels: if True, generate fake features with same labels as real
    
    Returns:
        dict with real_audio, fake_audio, real_video, fake_video, labels
    """
    generator.audio_generator.eval()
    generator.video_generator.eval()
    
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
            batch_size = real_a.shape[0]
            
            # Generate fake features
            z = torch.randn(batch_size, generator.config.z_dim, device=device)
            gen_labels = y if use_same_labels else torch.randint(
                0, generator.config.num_classes, (batch_size,), device=device
            )
            
            fake_a = generator.audio_generator(z, gen_labels)
            fake_v = generator.video_generator(z, gen_labels)
            
            # 对生成特征应用 mask（模拟真实数据的 padding）
            fake_a = generator._mask_by_len(fake_a, len_a)
            fake_v = generator._mask_by_len(fake_v, len_v)
            
            # 对 audio 使用 (mean, std) 拼接表示，因为 per-sample Z-norm 后 mean ≈ 0
            # 对 video 仍用 mean（video 特征没有做 per-sample Z-norm）
            real_audio_list.append(masked_mean_std(real_a, len_a))
            fake_audio_list.append(masked_mean_std(fake_a, len_a))
            real_video_list.append(masked_mean(real_v, len_v))
            fake_video_list.append(masked_mean(fake_v, len_v))
            labels_list.append(y.cpu().numpy())
    
    generator.audio_generator.train()
    generator.video_generator.train()
    
    return {
        'real_audio': np.concatenate(real_audio_list, axis=0),
        'fake_audio': np.concatenate(fake_audio_list, axis=0),
        'real_video': np.concatenate(real_video_list, axis=0),
        'fake_video': np.concatenate(fake_video_list, axis=0),
        'labels': np.concatenate(labels_list, axis=0)
    }


def visualize_tsne(features_dict, save_path, modality='audio', max_classes=10):
    """
    Create t-SNE visualization comparing real vs fake features
    
    Args:
        features_dict: dict from collect_features()
        save_path: path to save the figure
        modality: 'audio' or 'video'
        max_classes: max number of classes to visualize
    """
    logging.info(f"Running t-SNE visualization for {modality}...")
    
    real_key = f'real_{modality}'
    fake_key = f'fake_{modality}'
    
    real_data = features_dict[real_key]
    fake_data = features_dict[fake_key]
    labels = features_dict['labels']
    
    # Combine data
    combined_data = np.vstack([real_data, fake_data])
    combined_labels = np.concatenate([labels, labels])
    domain_labels = np.concatenate([
        np.zeros(len(real_data)),  # 0 = Real
        np.ones(len(fake_data))    # 1 = Fake
    ])
    
    # Apply t-SNE
    logging.info("Fitting t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings = tsne.fit_transform(combined_data)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Real vs Fake distribution
    ax1 = axes[0, 0]
    colors = ['blue' if d == 0 else 'red' for d in domain_labels]
    ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.5, s=20)
    ax1.set_title(f't-SNE: Real (Blue) vs Fake (Red) - {modality.upper()}', fontsize=12)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Real'),
                       Patch(facecolor='red', label='Fake')]
    ax1.legend(handles=legend_elements)
    
    # Plot 2: Class distribution (all data)
    ax2 = axes[0, 1]
    mask = combined_labels < max_classes
    scatter = ax2.scatter(
        embeddings[mask, 0], embeddings[mask, 1],
        c=combined_labels[mask], cmap='tab10', alpha=0.6, s=20
    )
    ax2.set_title(f'Class Distribution (Classes 0-{max_classes-1})', fontsize=12)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax2, label='Class')
    
    # Plot 3: Real only with classes
    ax3 = axes[1, 0]
    real_embeddings = embeddings[:len(real_data)]
    real_mask = labels < max_classes
    scatter3 = ax3.scatter(
        real_embeddings[real_mask, 0], real_embeddings[real_mask, 1],
        c=labels[real_mask], cmap='tab10', alpha=0.7, s=25
    )
    ax3.set_title('Real Features by Class', fontsize=12)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    plt.colorbar(scatter3, ax=ax3, label='Class')
    
    # Plot 4: Fake only with classes
    ax4 = axes[1, 1]
    fake_embeddings = embeddings[len(real_data):]
    scatter4 = ax4.scatter(
        fake_embeddings[real_mask, 0], fake_embeddings[real_mask, 1],
        c=labels[real_mask], cmap='tab10', alpha=0.7, s=25
    )
    ax4.set_title('Fake Features by Class', fontsize=12)
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    plt.colorbar(scatter4, ax=ax4, label='Class')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"t-SNE visualization saved to: {save_path}")


def compute_distance_metrics(features_dict, modality='audio', num_classes=10):
    """
    Compute distance metrics between real and fake feature distributions
    
    Args:
        features_dict: dict from collect_features()
        modality: 'audio' or 'video'
        num_classes: number of classes to analyze
    
    Returns:
        dict with per-class metrics
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Distance Metrics Analysis ({modality.upper()})")
    logging.info(f"{'='*60}")
    
    real_data = features_dict[f'real_{modality}']
    fake_data = features_dict[f'fake_{modality}']
    labels = features_dict['labels']
    
    metrics = {}
    
    for cls in range(num_classes):
        mask = labels == cls
        if np.sum(mask) < 2:
            continue
        
        real_cls = real_data[mask]
        fake_cls = fake_data[mask]
        
        # Compute centroids
        real_center = np.mean(real_cls, axis=0).reshape(1, -1)
        fake_center = np.mean(fake_cls, axis=0).reshape(1, -1)
        
        # Cosine similarity
        cos_sim = cosine_similarity(real_center, fake_center)[0][0]
        
        # L2 distance
        l2_dist = np.linalg.norm(real_center - fake_center)
        
        # Intra-class variance
        real_var = np.mean(np.var(real_cls, axis=0))
        fake_var = np.mean(np.var(fake_cls, axis=0))
        
        metrics[cls] = {
            'cosine_similarity': cos_sim,
            'l2_distance': l2_dist,
            'real_variance': real_var,
            'fake_variance': fake_var,
            'n_samples': np.sum(mask)
        }
        
        logging.info(
            f"Class {cls:2d}: CosSim={cos_sim:.4f}, L2={l2_dist:.4f}, "
            f"RealVar={real_var:.4f}, FakeVar={fake_var:.4f}, N={np.sum(mask)}"
        )
    
    # Overall statistics
    avg_cos = np.mean([m['cosine_similarity'] for m in metrics.values()])
    avg_l2 = np.mean([m['l2_distance'] for m in metrics.values()])
    
    logging.info(f"\n--- Overall ---")
    logging.info(f"Average Cosine Similarity: {avg_cos:.4f} (closer to 1.0 = more similar)")
    logging.info(f"Average L2 Distance: {avg_l2:.4f} (lower = more similar)")
    
    return metrics


def compute_distribution_statistics(features_dict, modality='audio', save_path=None):
    """
    Compute and visualize statistical distribution comparisons
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Statistical Distribution Analysis ({modality.upper()})")
    logging.info(f"{'='*60}")
    
    real_data = features_dict[f'real_{modality}']
    fake_data = features_dict[f'fake_{modality}']
    
    # Global statistics
    real_mean = np.mean(real_data)
    fake_mean = np.mean(fake_data)
    real_std = np.std(real_data)
    fake_std = np.std(fake_data)
    real_min, real_max = np.min(real_data), np.max(real_data)
    fake_min, fake_max = np.min(fake_data), np.max(fake_data)
    
    logging.info(f"Real: mean={real_mean:.4f}, std={real_std:.4f}, range=[{real_min:.4f}, {real_max:.4f}]")
    logging.info(f"Fake: mean={fake_mean:.4f}, std={fake_std:.4f}, range=[{fake_min:.4f}, {fake_max:.4f}]")
    
    if save_path:
        # Create distribution comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histogram of all values
        ax1 = axes[0]
        ax1.hist(real_data.flatten(), bins=50, alpha=0.5, label='Real', density=True)
        ax1.hist(fake_data.flatten(), bins=50, alpha=0.5, label='Fake', density=True)
        ax1.set_title('Value Distribution (All Features)')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Per-dimension mean comparison
        ax2 = axes[1]
        real_dim_means = np.mean(real_data, axis=0)
        fake_dim_means = np.mean(fake_data, axis=0)
        dims = range(len(real_dim_means))
        ax2.bar(dims, real_dim_means, alpha=0.5, label='Real', width=0.8)
        ax2.bar(dims, fake_dim_means, alpha=0.5, label='Fake', width=0.8)
        ax2.set_title('Per-Dimension Mean')
        ax2.set_xlabel('Feature Dimension')
        ax2.set_ylabel('Mean Value')
        ax2.legend()
        
        # Per-dimension std comparison
        ax3 = axes[2]
        real_dim_stds = np.std(real_data, axis=0)
        fake_dim_stds = np.std(fake_data, axis=0)
        ax3.bar(dims, real_dim_stds, alpha=0.5, label='Real', width=0.8)
        ax3.bar(dims, fake_dim_stds, alpha=0.5, label='Fake', width=0.8)
        ax3.set_title('Per-Dimension Std')
        ax3.set_xlabel('Feature Dimension')
        ax3.set_ylabel('Std Value')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Distribution plot saved to: {save_path}")


def ml_efficacy_test(generator, dataloader, device, num_train_batches=20, num_test_batches=10):
    """
    Machine Learning Efficacy Test:
    Train a simple classifier on fake data, test on real data
    
    This tests if the generated features contain meaningful semantic information.
    """
    logging.info(f"\n{'='*60}")
    logging.info("Machine Learning Efficacy Test")
    logging.info("(Train on Fake, Test on Real)")
    logging.info(f"{'='*60}")
    
    import torch.nn as nn
    
    generator.audio_generator.eval()
    generator.video_generator.eval()
    
    # Simple classifier
    classifier = nn.Sequential(
        nn.Linear(generator.config.audio_feat_dim + generator.config.video_feat_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, generator.config.num_classes)
    ).to(device)
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training on fake data
    logging.info("Training classifier on generated (fake) features...")
    classifier.train()
    
    for epoch in range(10):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            if i >= num_train_batches:
                break
            
            _, _, len_a, len_v, y = batch
            y = y.to(device)
            len_a = len_a.to(device)
            len_v = len_v.to(device)
            batch_size = y.shape[0]
            
            # Generate fake features
            z = torch.randn(batch_size, generator.config.z_dim, device=device)
            with torch.no_grad():
                fake_a = generator.audio_generator(z, y)
                fake_v = generator.video_generator(z, y)
                # 对生成特征应用 mask（模拟真实数据的 padding）
                fake_a = generator._mask_by_len(fake_a, len_a)
                fake_v = generator._mask_by_len(fake_v, len_v)
            
            # 使用 masked pooling 避免 padding 干扰
            fake_a_pool = masked_mean_torch(fake_a, len_a)
            fake_v_pool = masked_mean_torch(fake_v, len_v)
            fake_feat = torch.cat([fake_a_pool, fake_v_pool], dim=1)
            
            # Train step
            optimizer.zero_grad()
            pred = classifier(fake_feat)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Testing on real data
    logging.info("Testing classifier on real features...")
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_test_batches:
                break
            
            real_a, real_v, len_a, len_v, y = batch
            real_a = real_a.to(device)
            real_v = real_v.to(device)
            len_a = len_a.to(device)
            len_v = len_v.to(device)
            y = y.to(device)
            
            # 使用 masked pooling 避免 padding 干扰
            real_a_pool = masked_mean_torch(real_a, len_a)
            real_v_pool = masked_mean_torch(real_v, len_v)
            real_feat = torch.cat([real_a_pool, real_v_pool], dim=1)
            
            pred = classifier(real_feat)
            pred_labels = pred.argmax(dim=1)
            
            correct += (pred_labels == y).sum().item()
            total += y.shape[0]
    
    accuracy = correct / total if total > 0 else 0
    random_baseline = 1.0 / generator.config.num_classes
    
    logging.info(f"\n--- ML Efficacy Results ---")
    logging.info(f"Accuracy (Train on Fake, Test on Real): {accuracy:.4f}")
    logging.info(f"Random Baseline: {random_baseline:.4f}")
    logging.info(f"Relative Improvement: {accuracy / random_baseline:.2f}x")
    
    if accuracy > random_baseline * 2:
        logging.info("✓ Generated features contain meaningful semantic information!")
    else:
        logging.info("✗ Generated features may not capture semantic information well.")
    
    generator.audio_generator.train()
    generator.video_generator.train()
    
    return accuracy


def analyze_feature_quality(
    generator, 
    dataloader, 
    device, 
    num_batches=10, 
    save_dir="./gan_analysis",
    run_ml_test=True,
    exp_name=None
):
    """
    Comprehensive GAN feature quality analysis
    
    Args:
        generator: GAN instance (GANgeneratorer or MultimodalFeatureGAN)
        dataloader: DataLoader with real data
        device: torch device
        num_batches: number of batches for analysis
        save_dir: directory to save analysis results
        run_ml_test: whether to run ML efficacy test
        exp_name: experiment name suffix for output files (e.g., 'OUTFIX' -> 'tsne_audio_OUTFIX.png')
    
    Returns:
        dict with all analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建文件名后缀
    name_suffix = f'_{exp_name}' if exp_name else ''
    
    logging.info("\n" + "="*60)
    logging.info("GAN Feature Quality Analysis")
    if exp_name:
        logging.info(f"Experiment: {exp_name}")
    logging.info("="*60)
    
    # 1. Collect features
    logging.info("\nStep 1: Collecting real and generated features...")
    features = collect_features(generator, dataloader, device, num_batches)
    logging.info(f"Collected {len(features['labels'])} samples")
    
    results = {'n_samples': len(features['labels']), 'exp_name': exp_name}
    
    # 2. t-SNE Visualization
    logging.info("\nStep 2: t-SNE Visualization...")
    visualize_tsne(
        features, 
        os.path.join(save_dir, f'tsne_audio{name_suffix}.png'),
        modality='audio'
    )
    visualize_tsne(
        features,
        os.path.join(save_dir, f'tsne_video{name_suffix}.png'),
        modality='video'
    )
    
    # 3. Distance Metrics
    logging.info("\nStep 3: Distance Metrics...")
    results['audio_metrics'] = compute_distance_metrics(features, modality='audio')
    results['video_metrics'] = compute_distance_metrics(features, modality='video')
    
    # 4. Distribution Statistics
    logging.info("\nStep 4: Distribution Statistics...")
    compute_distribution_statistics(
        features, 
        modality='audio',
        save_path=os.path.join(save_dir, f'dist_audio{name_suffix}.png')
    )
    compute_distribution_statistics(
        features,
        modality='video', 
        save_path=os.path.join(save_dir, f'dist_video{name_suffix}.png')
    )
    
    # 5. ML Efficacy Test
    if run_ml_test:
        logging.info("\nStep 5: ML Efficacy Test...")
        results['ml_efficacy'] = ml_efficacy_test(generator, dataloader, device)
    
    logging.info("\n" + "="*60)
    logging.info("Analysis Complete!")
    logging.info(f"Results saved to: {save_dir}")
    logging.info("="*60)
    
    return results


def main():
    """Main function for standalone evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate GAN Feature Quality')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to GAN checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory')
    parser.add_argument('--dataset', type=str, default='ucf101')
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--audio_feat', type=str, default='mfcc')
    parser.add_argument('--video_feat', type=str, default='mobilenet_v2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_batches', type=int, default=10)
    parser.add_argument('--output_dir', type=str,
                        default='artifacts/legacy_evaluation/teacher_guided')
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--att', action='store_true')
    parser.add_argument('--att_name', type=str, default='base')
    
    # Add missing arguments for DataloadManager
    parser.add_argument('--missing_modality', type=bool, default=False)
    parser.add_argument('--missing_modailty_rate', type=float, default=0.5)
    parser.add_argument('--missing_label', type=bool, default=False)
    parser.add_argument('--missing_label_rate', type=float, default=0.5)
    parser.add_argument('--label_nosiy', type=bool, default=False)
    parser.add_argument('--label_nosiy_level', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Set data_dir if not provided
    if args.data_dir is None:
        path_conf = {}
        cfg_path = Path(__file__).parents[1].joinpath('system.cfg')
        with open(str(cfg_path)) as f:
            for line in f:
                key, val = line.strip().split('=')
                path_conf[key] = val.replace("\"", "")
        args.data_dir = path_conf.get('output_dir', '.')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data
    dm = DataloadManager(args)
    dm.get_simulation_setting(alpha=args.alpha)
    dm.load_sim_dict(fold_idx=1)
    dm.get_client_ids(fold_idx=1)
    
    # Get first client's dataloader
    client_id = [cid for cid in dm.client_ids if cid not in ['dev', 'test']][0]
    audio_dict = dm.load_audio_feat(client_id=client_id, fold_idx=1)
    video_dict = dm.load_video_feat(client_id=client_id, fold_idx=1)
    
    dataloader = dm.set_dataloader(
        audio_dict, video_dict,
        default_feat_shape_a=np.array([500, constants.feature_len_dict["mfcc"]]),
        default_feat_shape_b=np.array([9, constants.feature_len_dict["mobilenet_v2"]]),
        shuffle=False
    )
    
    # Create global model
    global_model = MMActionClassifier(
        num_classes=constants.num_class_dict[args.dataset],
        audio_input_dim=constants.feature_len_dict[args.audio_feat],
        video_input_dim=constants.feature_len_dict[args.video_feat],
        d_hid=args.hid_size,
        en_att=args.att,
        att_name=args.att_name
    ).to(device)
    
    # Create GAN config and generator
    config = FeatureGANConfig(
        num_classes=constants.num_class_dict[args.dataset],
        audio_feat_dim=constants.feature_len_dict[args.audio_feat],
        video_feat_dim=constants.feature_len_dict[args.video_feat],
        video_seq_len=9,
        device=device
    )
    
    generator = MultimodalFeatureGAN(args, global_model, config)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from: {args.checkpoint}")
    generator.load_checkpoint(args.checkpoint)
    
    # Run analysis
    results = analyze_feature_quality(
        generator, dataloader, device,
        num_batches=args.num_batches,
        save_dir=args.output_dir
    )
    
    # Save results
    import json
    with open(os.path.join(args.output_dir, 'analysis_results.json'), 'w') as f:
        # Convert numpy types to python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    logging.info(f"Results saved to {args.output_dir}/analysis_results.json")


if __name__ == '__main__':
    main()
