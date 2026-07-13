#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio 特征诊断脚本 - 快速定位 GAN Audio 崩溃的根因

检查项:
A) 每维均值分布：找出接近 1 的 binary/gate 维度
B) 生成器输出层激活/归一化配置
C) 是否需要分段建模

运行: python diagnose_audio_features.py
"""

import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).parents[1]))
from Local.dataloader import UCF101LocalDataManager, collate_mm_fn_padd
from generator.gan_generator import FeatureGANConfig, AudioFeatureGenerator

def analyze_audio_feature_distribution(data_dir, dataset_dir):
    """
    分析真实 audio 特征的分布特性
    """
    print("="*80)
    print("A) 分析 Audio 特征每维均值分布")
    print("="*80)
    
    # 加载数据
    dm = UCF101LocalDataManager(
        data_dir=data_dir,
        dataset_dir=dataset_dir,
        batch_size=64,
        num_workers=0  # 避免多进程问题
    )
    
    dataloaders = dm.get_dataloaders(val_split=0.0)  # 不分 val，用完整 train
    train_loader = dataloaders['train']
    
    # 收集所有 audio 特征
    all_audio = []
    all_lengths = []
    
    for batch in train_loader:
        x_a, x_v, len_a, len_v, y = batch
        all_audio.append(x_a.numpy())
        all_lengths.append(len_a.numpy())
    
    all_audio = np.concatenate(all_audio, axis=0)  # [N, T, F]
    all_lengths = np.concatenate(all_lengths, axis=0)  # [N]
    
    print(f"\n[基本信息]")
    print(f"  样本数: {all_audio.shape[0]}")
    print(f"  序列长度 T: {all_audio.shape[1]}")
    print(f"  特征维度 F: {all_audio.shape[2]}")
    print(f"  有效长度范围: [{all_lengths.min()}, {all_lengths.max()}]")
    
    # ===== 1. 计算每维度的统计量 =====
    print(f"\n[每维度统计 - 原始数据]")
    
    # 使用 masked mean
    N, T, F = all_audio.shape
    flat_audio = all_audio.reshape(-1, F)  # [N*T, F]
    
    per_dim_mean = flat_audio.mean(axis=0)  # [F]
    per_dim_std = flat_audio.std(axis=0)    # [F]
    per_dim_min = flat_audio.min(axis=0)    # [F]
    per_dim_max = flat_audio.max(axis=0)    # [F]
    
    print(f"\n  全局均值范围: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")
    print(f"  全局标准差范围: [{per_dim_std.min():.4f}, {per_dim_std.max():.4f}]")
    print(f"  全局最小值范围: [{per_dim_min.min():.4f}, {per_dim_min.max():.4f}]")
    print(f"  全局最大值范围: [{per_dim_max.min():.4f}, {per_dim_max.max():.4f}]")
    
    # ===== 2. 找出 binary/gate 维度 (均值接近 0 或 1) =====
    print(f"\n[检测 Binary/Gate 维度]")
    
    # 检查均值接近 1 的维度
    near_one_mask = per_dim_mean > 0.8
    near_one_dims = np.where(near_one_mask)[0]
    
    # 检查均值接近 0 的维度
    near_zero_mask = np.abs(per_dim_mean) < 0.1
    near_zero_dims = np.where(near_zero_mask)[0]
    
    # 检查标准差很小的维度 (几乎是常量)
    low_std_mask = per_dim_std < 0.1
    low_std_dims = np.where(low_std_mask)[0]
    
    print(f"\n  均值 > 0.8 的维度数: {len(near_one_dims)}")
    if len(near_one_dims) > 0:
        print(f"    维度索引: {near_one_dims[:20]}{'...' if len(near_one_dims) > 20 else ''}")
        print(f"    对应均值: {per_dim_mean[near_one_dims][:20]}")
    
    print(f"\n  均值接近 0 的维度数: {len(near_zero_dims)}")
    if len(near_zero_dims) > 0:
        print(f"    维度索引: {near_zero_dims[:20]}{'...' if len(near_zero_dims) > 20 else ''}")
        print(f"    对应均值: {per_dim_mean[near_zero_dims][:20]}")
    
    print(f"\n  标准差 < 0.1 的维度数 (近常量): {len(low_std_dims)}")
    if len(low_std_dims) > 0:
        print(f"    维度索引: {low_std_dims[:20]}{'...' if len(low_std_dims) > 20 else ''}")
        print(f"    对应均值: {per_dim_mean[low_std_dims][:20]}")
        print(f"    对应std:  {per_dim_std[low_std_dims][:20]}")
    
    # ===== 3. 检查 bimodal 分布 (可能是 binary flag) =====
    print(f"\n[检测 Bimodal 分布]")
    bimodal_dims = []
    for d in range(F):
        vals = flat_audio[:, d]
        # 简单检测：看数据是否主要集中在两个峰
        hist, bin_edges = np.histogram(vals, bins=50)
        # 找到最大的两个峰
        top2_bins = np.argsort(hist)[-2:]
        if hist[top2_bins].sum() / hist.sum() > 0.6:  # 两个峰占 60% 以上
            # 检查两个峰是否分布在两端
            if np.abs(top2_bins[0] - top2_bins[1]) > 30:  # 峰距离较远
                bimodal_dims.append(d)
    
    print(f"  检测到 bimodal 分布的维度数: {len(bimodal_dims)}")
    if len(bimodal_dims) > 0:
        print(f"    维度索引: {bimodal_dims[:20]}{'...' if len(bimodal_dims) > 20 else ''}")
    
    # ===== 4. 详细维度分析表 =====
    print(f"\n[每维度详细统计 (前40维)]")
    print("-"*90)
    print(f"{'Dim':<6}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}{'Type':>15}")
    print("-"*90)
    
    for d in range(min(40, F)):
        mean_val = per_dim_mean[d]
        std_val = per_dim_std[d]
        min_val = per_dim_min[d]
        max_val = per_dim_max[d]
        
        # 判断类型
        if std_val < 0.1:
            dtype = "CONSTANT"
        elif mean_val > 0.8 and max_val <= 1.1:
            dtype = "BINARY(~1)"
        elif np.abs(mean_val) < 0.1 and min_val >= -0.1:
            dtype = "BINARY(~0)"
        elif d in bimodal_dims:
            dtype = "BIMODAL"
        else:
            dtype = "CONTINUOUS"
        
        print(f"{d:<6}{mean_val:>10.4f}{std_val:>10.4f}{min_val:>10.4f}{max_val:>10.4f}{dtype:>15}")
    
    print("-"*90)
    
    # ===== 5. 按样本的统计 (Z-norm 后) =====
    print(f"\n[Per-Sample Z-norm 后的统计]")
    
    # 模拟 per-sample Z-norm
    sample_means = all_audio.mean(axis=1, keepdims=True)  # [N, 1, F]
    sample_stds = all_audio.std(axis=1, keepdims=True) + 1e-5  # [N, 1, F]
    normed_audio = (all_audio - sample_means) / sample_stds
    
    normed_flat = normed_audio.reshape(-1, F)
    normed_per_dim_mean = normed_flat.mean(axis=0)
    normed_per_dim_std = normed_flat.std(axis=0)
    
    print(f"  归一化后均值范围: [{normed_per_dim_mean.min():.4f}, {normed_per_dim_mean.max():.4f}]")
    print(f"  归一化后标准差范围: [{normed_per_dim_std.min():.4f}, {normed_per_dim_std.max():.4f}]")
    
    # Masked mean pooling 后的统计
    print(f"\n[Masked Mean Pooling 后的统计]")
    pooled_feats = []
    for i in range(N):
        valid_len = min(int(all_lengths[i]), T)
        if valid_len > 0:
            pooled = normed_audio[i, :valid_len, :].mean(axis=0)
        else:
            pooled = normed_audio[i].mean(axis=0)
        pooled_feats.append(pooled)
    pooled_feats = np.stack(pooled_feats, axis=0)  # [N, F]
    
    pooled_mean = pooled_feats.mean(axis=0)
    pooled_std = pooled_feats.std(axis=0)
    
    print(f"  Pooled 均值范围: [{pooled_mean.min():.4f}, {pooled_mean.max():.4f}]")
    print(f"  Pooled 标准差范围: [{pooled_std.min():.4f}, {pooled_std.max():.4f}]")
    print(f"  Pooled 均值 > 0.5 的维度数: {(pooled_mean > 0.5).sum()}")
    print(f"  Pooled 均值 < -0.5 的维度数: {(pooled_mean < -0.5).sum()}")
    
    return {
        'per_dim_mean': per_dim_mean,
        'per_dim_std': per_dim_std,
        'near_one_dims': near_one_dims,
        'near_zero_dims': near_zero_dims,
        'low_std_dims': low_std_dims,
        'bimodal_dims': bimodal_dims,
        'pooled_mean': pooled_mean,
        'pooled_std': pooled_std
    }


def analyze_generator_architecture():
    """
    B) 检查生成器输出层的激活/归一化配置
    """
    print("\n" + "="*80)
    print("B) 检查 Audio Generator 输出层配置")
    print("="*80)
    
    # 创建一个示例 config
    config = FeatureGANConfig(
        audio_seq_len=500,
        audio_feat_dim=80,
    )
    
    gen = AudioFeatureGenerator(config)
    
    print(f"\n[AudioFeatureGenerator 结构]")
    print(f"  最后的 conv 层: ConvTranspose1d -> 线性输出 (无激活)")
    print(f"  后处理: scale * x + bias, 然后 clamp")
    
    print(f"\n[输出约束参数]")
    print(f"  audio_scale_max: {config.audio_scale_max} (sigmoid 后的最大 scale)")
    print(f"  audio_bias_max: {config.audio_bias_max} (tanh 后的最大 bias)")
    print(f"  audio_out_max: {config.audio_out_max} (clamp 范围)")
    
    print(f"\n[初始化参数值]")
    print(f"  scale_logit 初值: {gen.scale_logit.item():.2f}")
    print(f"  对应 scale 初值: {torch.sigmoid(gen.scale_logit).item() * config.audio_scale_max:.4f}")
    print(f"  bias_param 初值: {gen.bias_param.item():.2f}")
    print(f"  对应 bias 初值: {torch.tanh(gen.bias_param).item() * config.audio_bias_max:.4f}")
    
    # 测试输出范围
    print(f"\n[输出范围测试]")
    z = torch.randn(4, config.z_dim)
    labels = torch.randint(0, config.num_classes, (4,))
    
    with torch.no_grad():
        out = gen(z, labels)
    
    print(f"  输出 shape: {out.shape}")
    print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"  输出均值: {out.mean().item():.4f}")
    print(f"  输出标准差: {out.std().item():.4f}")
    
    # 关键问题分析
    print(f"\n[潜在问题分析]")
    print(f"""
  ⚠️  当前配置:
      - 输出经过 scale * x + bias，再 clamp 到 [-{config.audio_out_max}, {config.audio_out_max}]
      - 初始 scale ≈ 0.04, bias ≈ 0，输出接近 0
      
  🔴 问题: 如果真实特征中存在某些维度的均值接近 1 (binary/gate)，
           而生成器输出被 clamp 在 [-1, 1] 且初始接近 0，
           则生成器很难学到这些 "高均值" 维度。
           
  🔴 问题: 如果真实特征经过了 per-sample Z-norm，均值约 0，
           但 G 的 clamp 范围可能限制了输出方差。
""")


def suggest_segmentation_strategy(stats):
    """
    C) 分析是否需要分段建模
    """
    print("\n" + "="*80)
    print("C) 分段建模建议")
    print("="*80)
    
    near_one = stats['near_one_dims']
    low_std = stats['low_std_dims']
    bimodal = stats['bimodal_dims']
    
    print(f"\n[维度分类汇总]")
    print(f"  BINARY/GATE 维度 (均值~1): {len(near_one)}")
    print(f"  CONSTANT 维度 (std<0.1): {len(low_std)}")
    print(f"  BIMODAL 维度: {len(bimodal)}")
    print(f"  其他 (CONTINUOUS): {80 - len(set(list(near_one) + list(low_std) + bimodal))}")
    
    # 分析重叠
    discrete_set = set(list(near_one) + list(low_std) + bimodal)
    continuous_set = set(range(80)) - discrete_set
    
    print(f"\n[分段建议]")
    if len(discrete_set) > 5:
        print(f"  ✅ 建议分段建模:")
        print(f"     - Discrete/Gate 维度 ({len(discrete_set)}个): 使用 BCE / Gumbel-Sigmoid")
        print(f"     - Continuous 维度 ({len(continuous_set)}个): 使用 L2 / Cosine / WGAN-GP")
        print(f"\n  Discrete 维度索引: {sorted(discrete_set)}")
    else:
        print(f"  ❌ 不需要分段建模，离散维度较少 ({len(discrete_set)}个)")
    
    print(f"\n[具体实施方案]")
    print("""
  如果确认存在 discrete/gate 段：
  
  1. 在 Generator 中:
     - continuous 部分: 保持线性输出
     - discrete 部分: 单独分支 + Sigmoid/Gumbel-Sigmoid
  
  2. 在 Loss 中:
     - continuous 部分: MSE/L1/Cosine + Critic
     - discrete 部分: BCE + 可选 straight-through
  
  3. 在 Discriminator 中:
     - 分别处理两部分，或者忽略 discrete 部分
""")


def main():
    # 路径配置
    base_path = Path(__file__).parents[1]
    data_dir = base_path / 'results'
    dataset_dir = base_path / 'datasets' / 'ucf101'
    
    print("="*80)
    print("Audio 特征诊断 - 定位 GAN 崩溃根因")
    print("="*80)
    
    # A) 分析特征分布
    stats = analyze_audio_feature_distribution(str(data_dir), str(dataset_dir))
    
    # B) 检查生成器配置
    analyze_generator_architecture()
    
    # C) 分段建模建议
    suggest_segmentation_strategy(stats)
    
    print("\n" + "="*80)
    print("诊断完成!")
    print("="*80)


if __name__ == '__main__':
    main()
