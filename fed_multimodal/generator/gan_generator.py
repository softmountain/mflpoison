#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习多模态特征 GAN (Final Robust Version)

架构特点：
1. Independent-ACGAN: A/V 独立判别器 (Logits 输出, 去 BN, 加 SN)
2. Joint Critic: 联合判别器 (去 Dropout, 加 SN, Logits 输出)
3. Audio Generator: 无 Tanh 线性直出 (Soft Init -2.0)
4. Teacher: Eval 模式 + 长度 Clamp (min=1, None-safe) + cuDNN RNN Backward Fix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging

class FeatureGANConfig:
    """多模态特征 GAN 配置类 (Base Template 参数)"""
    def __init__(
        self,
        z_dim: int = 128,
        num_classes: int = 51,
        audio_seq_len: int = 500,
        audio_feat_dim: int = 80,
        video_seq_len: int = 9,
        video_feat_dim: int = 1280,
        hidden_dim: int = 256,
        # 学习率
        lr_g: float = 2e-4,
        lr_d: float = 1e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        # 损失权重 (Base Template)
        rf_weight: float = 2.0,      
        aux_weight: float = 1.0,     # Base Template 值
        cls_weight: float = 0.1,     # Teacher 权重
        joint_weight: float = 0.05,   # Joint 权重 (建议 0.05-0.1，过大会拖坏 audio)
        fm_weight: float = 0.0,      # 默认关闭
        mom_weight: float = 0.0,     # 默认关闭
        # [B] Joint Critic 增强训练
        joint_d_steps: int = 3,      # Joint Critic 每 G step 训练次数
        joint_lr_mult: float = 2.0,  # Joint Critic 学习率倍数
        # [C] Audio 边缘分布正则
        audio_std_target: float = 0.05,  # 目标 std（与 real 对齐）
        audio_std_weight: float = 0.0,   # std 正则权重（0=关闭）
        # 标签平滑 (Base Template: no smooth, 内部用 0.9/0.1)
        real_label_smoothing: float = 0.9,
        fake_label_smoothing: float = 0.1,
        # 噪声 (Base Template: off)
        noise_std: float = 0.0,
        # 梯度惩罚 (Base Template: off)
        use_gradient_penalty: bool = False,
        gp_weight: float = 5.0,
        # 输出尺度约束 (Base Template)
        audio_scale_max: float = 0.3,    
        audio_bias_max: float = 0.1,     
        audio_out_max: float = 1.0,     
        video_scale_max: float = 8.0,    
        video_out_max: float = 20.0,     
        device: str = None
    ):
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.audio_seq_len = audio_seq_len
        self.audio_feat_dim = audio_feat_dim
        self.video_seq_len = video_seq_len
        self.video_feat_dim = video_feat_dim
        self.hidden_dim = hidden_dim
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.beta2 = beta2
        self.rf_weight = rf_weight
        self.aux_weight = aux_weight
        self.cls_weight = cls_weight
        self.joint_weight = joint_weight
        self.fm_weight = fm_weight
        self.mom_weight = mom_weight
        self.joint_d_steps = joint_d_steps
        self.joint_lr_mult = joint_lr_mult
        self.audio_std_target = audio_std_target
        self.audio_std_weight = audio_std_weight
        self.real_label_smoothing = real_label_smoothing
        self.fake_label_smoothing = fake_label_smoothing
        self.noise_std = noise_std
        self.use_gradient_penalty = use_gradient_penalty
        self.gp_weight = gp_weight
        self.audio_scale_max = audio_scale_max
        self.audio_bias_max = audio_bias_max
        self.audio_out_max = audio_out_max
        self.video_scale_max = video_scale_max
        self.video_out_max = video_out_max
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def masked_mean_torch(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Torch 版本的 masked mean pooling"""
    B, T, D = x.shape
    if lengths is None:
        return x.mean(dim=1)
    
    # 确保长度至少为1，避免除零
    lengths = torch.clamp(lengths.long(), min=1, max=T)
    
    idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
    mask = (idx < lengths[:, None]).float().unsqueeze(-1)  # (B, T, 1)
    
    denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
    return (x * mask).sum(dim=1) / denom


class JointCritic(nn.Module):
    """
    两塔投影联合判别器
    
    解决维度不平衡问题（audio 80维 vs video 1280维）：
    1. 先对各模态做 LayerNorm 归一化
    2. 各自投影到同维度空间 (z_dim)
    3. 再拼接做 real/fake 判别
    
    这样 audio/video 对 logits 的贡献方差接近，不会被 video 主导。
    
    输入: Pooling 后的 Audio + Video 特征
    输出: Logits (无 Sigmoid)
    """
    def __init__(self, audio_dim, video_dim, hidden_dim=256, proj_dim=128):
        super(JointCritic, self).__init__()
        
        # Audio 塔: LayerNorm -> 投影到 proj_dim
        self.audio_tower = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Video 塔: LayerNorm -> 投影到 proj_dim
        self.video_tower = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # 融合头: 拼接后判别 (使用 SN)
        self.head = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(2 * proj_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
        )

    def forward(self, audio_feat, video_feat):
        # 各自投影到同维度
        a_proj = self.audio_tower(audio_feat)  # [B, proj_dim]
        v_proj = self.video_tower(video_feat)  # [B, proj_dim]
        # 拼接后判别
        x = torch.cat([a_proj, v_proj], dim=1)  # [B, 2*proj_dim]
        return self.head(x)


class AudioFeatureGenerator(nn.Module):
    def __init__(self, config: FeatureGANConfig):
        super(AudioFeatureGenerator, self).__init__()
        self.config = config
        self.label_emb = nn.Embedding(config.num_classes, config.num_classes)
        self.init_len = config.audio_seq_len // 8
        # [Fix] 移除 BN，改用 LayerNorm 避免小 batch 不稳定
        self.fc = nn.Sequential(
            nn.Linear(config.z_dim + config.num_classes, self.init_len * config.hidden_dim),
            nn.LayerNorm(self.init_len * config.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # [Fix] 移除 BN 和 Dropout，使用 GroupNorm
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(config.hidden_dim, config.hidden_dim // 2, 4, 2, 1),
            nn.GroupNorm(8, config.hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(config.hidden_dim // 2, config.hidden_dim // 4, 4, 2, 1),
            nn.GroupNorm(8, config.hidden_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            # [Fix] 最后一层纯线性，移除 Tanh
            nn.ConvTranspose1d(config.hidden_dim // 4, config.audio_feat_dim, 4, 2, 1)
        )
        self.final_resize = nn.AdaptiveAvgPool1d(config.audio_seq_len)
        
        # [Patch 3] Soft Initialization: -2.0 对应 sigmoid ≈ 0.12, scale ≈ 0.6
        # 避免开局梯度过硬
        self.scale_logit = nn.Parameter(torch.tensor(-2.0))
        self.bias_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, z, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([z, label_emb], dim=1)
        x = self.fc(x)
        x = x.view(x.shape[0], self.config.hidden_dim, self.init_len)
        x = self.conv_blocks(x)
        x = self.final_resize(x)
        
        # [Fix] 线性缩放 + Clamp
        scale = torch.sigmoid(self.scale_logit) * self.config.audio_scale_max
        bias = torch.tanh(self.bias_param) * self.config.audio_bias_max
        x = x * scale + bias
        x = x.clamp(min=-self.config.audio_out_max, max=self.config.audio_out_max)
        return x.permute(0, 2, 1)


class VideoFeatureGenerator(nn.Module):
    def __init__(self, config: FeatureGANConfig):
        super(VideoFeatureGenerator, self).__init__()
        self.config = config
        self.label_emb = nn.Embedding(config.num_classes, config.num_classes)
        # [Fix] 移除 BN 和 Dropout，改用 LayerNorm
        self.fc = nn.Sequential(
            nn.Linear(config.z_dim + config.num_classes, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, config.video_seq_len * config.video_feat_dim),
            nn.ReLU() # 视频特征非负
        )
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, z, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([z, label_emb], dim=1)
        x = self.fc(x)
        x = x.view(x.shape[0], self.config.video_seq_len, self.config.video_feat_dim)
        
        scale = F.softplus(self.log_scale).clamp(max=self.config.video_scale_max)
        x = x * scale + self.output_bias
        x = torch.clamp(x, min=0.0, max=self.config.video_out_max)
        return x


class AudioFeatureDiscriminator(nn.Module):
    def __init__(self, config: FeatureGANConfig, use_spectral_norm: bool = True):
        super(AudioFeatureDiscriminator, self).__init__()
        self.config = config
        def maybe_sn(module):
            return nn.utils.spectral_norm(module) if use_spectral_norm else module
        
        # [Fix] 移除 BatchNorm 和 Dropout，依赖 Spectral Norm
        self.conv = nn.Sequential(
            maybe_sn(nn.Conv1d(config.audio_feat_dim, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv1d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv1d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        # [Fix] Removed Sigmoid -> Logits
        self.adv_layer = nn.Sequential(maybe_sn(nn.Linear(256, 1))) 
        self.aux_layer = maybe_sn(nn.Linear(256, config.num_classes))
        
    def forward(self, features, return_aux=False, add_noise=True, return_feat=False):
        if add_noise and self.training and self.config.noise_std > 0:
            features = features + torch.randn_like(features) * self.config.noise_std
        x = features.permute(0, 2, 1)
        x = self.conv(x)
        feat = x.view(x.shape[0], -1)
        validity = self.adv_layer(feat)
        ret = [validity]
        if return_aux: ret.append(self.aux_layer(feat))
        if return_feat: ret.append(feat)
        return tuple(ret) if len(ret) > 1 else ret[0]


class VideoFeatureDiscriminator(nn.Module):
    def __init__(self, config: FeatureGANConfig, use_spectral_norm: bool = True):
        super(VideoFeatureDiscriminator, self).__init__()
        self.config = config
        def maybe_sn(module):
            return nn.utils.spectral_norm(module) if use_spectral_norm else module
        
        # [Fix] 移除 BatchNorm 和 Dropout，依赖 Spectral Norm
        self.conv = nn.Sequential(
            maybe_sn(nn.Conv1d(config.video_feat_dim, 512, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv1d(512, 256, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        # [Fix] Removed Sigmoid -> Logits
        self.adv_layer = nn.Sequential(maybe_sn(nn.Linear(256, 1)))
        self.aux_layer = maybe_sn(nn.Linear(256, config.num_classes))
        
    def forward(self, features, return_aux=False, add_noise=True, return_feat=False):
        if add_noise and self.training and self.config.noise_std > 0:
            features = features + torch.randn_like(features) * self.config.noise_std
        x = features.permute(0, 2, 1)
        x = self.conv(x)
        feat = x.view(x.shape[0], -1)
        validity = self.adv_layer(feat)
        ret = [validity]
        if return_aux: ret.append(self.aux_layer(feat))
        if return_feat: ret.append(feat)
        return tuple(ret) if len(ret) > 1 else ret[0]


class MultimodalFeatureGAN:
    def __init__(
        self,
        args,
        global_model,
        config: FeatureGANConfig,
        use_spectral_norm: bool = True # Default True
    ):
        self.args = args
        self.config = config
        self.device = config.device
        
        # Generators
        self.audio_generator = AudioFeatureGenerator(config).to(self.device)
        self.video_generator = VideoFeatureGenerator(config).to(self.device)
         
        # Discriminators (SN on by default)
        self.audio_discriminator = AudioFeatureDiscriminator(config, use_spectral_norm).to(self.device)
        self.video_discriminator = VideoFeatureDiscriminator(config, use_spectral_norm).to(self.device)
        self.joint_discriminator = JointCritic(config.audio_feat_dim, config.video_feat_dim).to(self.device)
        
        # Teacher Model (Eval Mode & Freeze)
        self.classifier = copy.deepcopy(global_model).to(self.device)
        self.classifier.eval() # 关键
        for param in self.classifier.parameters():
            param.requires_grad = False
        
        # [Fix] flatten_parameters 必须在 .to(device) 之后调用才生效
        for m in self.classifier.modules():
            if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
                m.flatten_parameters()
        
        # Optimizers
        self.opt_g_audio = torch.optim.Adam(self.audio_generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
        self.opt_g_video = torch.optim.Adam(self.video_generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
        
        self.opt_d_audio = torch.optim.Adam(self.audio_discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
        self.opt_d_video = torch.optim.Adam(self.video_discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
        # [B] Joint Critic 使用独立的放大学习率（R0-2 证明这能让 Joint 保持正向 Gap）
        self.opt_d_joint = torch.optim.Adam(self.joint_discriminator.parameters(), lr=config.lr_d * config.joint_lr_mult, betas=(config.beta1, config.beta2))
        
        # [Fix] 使用 BCEWithLogitsLoss
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()
        
        self.train_history = {
            'g_loss': [], 'd_loss': [], 'g_aux_loss': [], 'd_acc': []
        }
    
    def _get_smooth_labels(self, batch_size, real=True):
        val = self.config.real_label_smoothing if real else self.config.fake_label_smoothing
        return torch.full((batch_size, 1), val, device=self.device)
    
    def _mask_by_len(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if lengths is None: return x
        lengths = lengths.to(x.device).long()
        B, T, _ = x.shape
        lengths = torch.clamp(lengths, min=0, max=T)
        idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
        mask = (idx < lengths[:, None]).unsqueeze(-1).float()
        return x * mask
    
    def _apply_per_sample_znorm(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        对 fake audio 应用 per-sample Z-norm（与 real 的预处理对齐）
        
        真实 audio 在预处理时做了 per-sample、per-feature-dim 的 Z-score：
            features = (features - mean(axis=0)) / (std(axis=0) + eps)
        这导致 real 的 masked_mean pooling 后会塌缩到接近 0。
        
        为了让 fake 和 real 在同一表征空间对齐，需要对 fake 也做同样的约束。
        
        Args:
            x: [B, T, F] 生成的 audio 特征
            lengths: [B] 有效长度
        
        Returns:
            [B, T, F] 归一化后的特征（padding 部分为 0）
        """
        if lengths is None:
            # 没有 mask，直接对整个序列做 Z-norm (axis=1 即时间维度)
            mean = x.mean(dim=1, keepdim=True)  # [B, 1, F]
            std = x.std(dim=1, keepdim=True)    # [B, 1, F]
            return (x - mean) / (std + 1e-5)
        
        lengths = lengths.to(x.device).long()
        B, T, F = x.shape
        lengths = torch.clamp(lengths, min=1, max=T)
        
        # 构建 mask: [B, T, 1]
        idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
        mask = (idx < lengths[:, None]).unsqueeze(-1).float()  # [B, T, 1]
        
        # 计算有效部分的 mean 和 var (沿时间维度，对每个特征维度独立)
        den = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
        
        # mean: [B, 1, F]
        mean = (x * mask).sum(dim=1, keepdim=True) / den
        
        # var: [B, 1, F]
        var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / den
        
        # 归一化
        x_normed = (x - mean) / (var.sqrt() + 1e-5)
        
        # padding 部分强制为 0（与 real 保持一致）
        x_normed = x_normed * mask
        
        return x_normed
    
    def train_step_multimodal(self, real_audio, real_video, real_labels, len_a=None, len_v=None):
        # [Fix] 自动搬运所有输入到 device，避免调用方遗漏
        real_audio = real_audio.to(self.device)
        real_video = real_video.to(self.device)
        batch_size = real_audio.shape[0]
        # [Patch 2] Ensure labels are on device
        real_labels = real_labels.to(self.device)
        if len_a is not None: len_a = len_a.to(self.device)
        if len_v is not None: len_v = len_v.to(self.device)
        
        valid = self._get_smooth_labels(batch_size, real=True)
        fake = self._get_smooth_labels(batch_size, real=False)
        
        # ================= Train Generators =================
        # [Fix] G-step 时冻结 D 参数（比 eval() 更稳定，尤其对 FM/MoM）
        for p in self.audio_discriminator.parameters():
            p.requires_grad_(False)
        for p in self.video_discriminator.parameters():
            p.requires_grad_(False)
        for p in self.joint_discriminator.parameters():
            p.requires_grad_(False)
        
        self.opt_g_audio.zero_grad()
        self.opt_g_video.zero_grad()
        
        z = torch.randn(batch_size, self.config.z_dim, device=self.device)
        gen_labels = real_labels
        
        # 1. Generate & Mask
        gen_audio = self.audio_generator(z, gen_labels)
        gen_video = self.video_generator(z, gen_labels)
        
        # [关键修复] 对 fake audio 做 per-sample Z-norm，与 real 预处理对齐
        # 真实 audio 在特征提取时做了 per-sample Z-score，导致 pooled mean ≈ 0
        # 如果不对 fake 做同样约束，fake 在 pooled 空间会看起来"到处飞"
        gen_audio = self._apply_per_sample_znorm(gen_audio, len_a)
        
        # Mask padding 部分
        if len_a is not None: gen_audio = self._mask_by_len(gen_audio, len_a)
        if len_v is not None: gen_video = self._mask_by_len(gen_video, len_v)
        
        # 2. Independent Discriminator Losses (Output Logits)
        a_validity_logit, a_aux = self.audio_discriminator(gen_audio, return_aux=True, add_noise=False)
        v_validity_logit, v_aux = self.video_discriminator(gen_video, return_aux=True, add_noise=False)
        
        g_loss_adv = (self.adversarial_loss(a_validity_logit, valid) + self.adversarial_loss(v_validity_logit, valid)) / 2
        g_loss_aux_d = (self.auxiliary_loss(a_aux, gen_labels) + self.auxiliary_loss(v_aux, gen_labels)) / 2
        
        # 3. Joint Critic Loss (Output Logits)
        g_a_pool = masked_mean_torch(gen_audio, len_a)
        g_v_pool = masked_mean_torch(gen_video, len_v)
        joint_validity_logit = self.joint_discriminator(g_a_pool, g_v_pool)
        g_loss_joint = self.adversarial_loss(joint_validity_logit, valid)
        
        # 3.5 Feature Matching & Moment Matching Loss
        g_loss_fm = torch.tensor(0.0, device=self.device)
        g_loss_mom = torch.tensor(0.0, device=self.device)
        
        if self.config.fm_weight > 0 or self.config.mom_weight > 0:
            # 获取 D 的中间特征
            _, _, a_real_feat = self.audio_discriminator(real_audio, return_aux=True, add_noise=False, return_feat=True)
            _, _, v_real_feat = self.video_discriminator(real_video, return_aux=True, add_noise=False, return_feat=True)
            _, _, a_fake_feat = self.audio_discriminator(gen_audio, return_aux=True, add_noise=False, return_feat=True)
            _, _, v_fake_feat = self.video_discriminator(gen_video, return_aux=True, add_noise=False, return_feat=True)
            
            if self.config.fm_weight > 0:
                # Feature Matching: 对齐 D 中间特征的均值
                g_loss_fm = (
                    F.mse_loss(a_fake_feat.mean(dim=0), a_real_feat.mean(dim=0).detach()) +
                    F.mse_loss(v_fake_feat.mean(dim=0), v_real_feat.mean(dim=0).detach())
                ) / 2
            
            if self.config.mom_weight > 0:
                # Moment Matching: 对齐每个维度的 mean 和 std
                a_real_mean, a_real_std = a_real_feat.mean(dim=0).detach(), a_real_feat.std(dim=0).detach()
                a_fake_mean, a_fake_std = a_fake_feat.mean(dim=0), a_fake_feat.std(dim=0)
                v_real_mean, v_real_std = v_real_feat.mean(dim=0).detach(), v_real_feat.std(dim=0).detach()
                v_fake_mean, v_fake_std = v_fake_feat.mean(dim=0), v_fake_feat.std(dim=0)
                
                g_loss_mom = (
                    F.mse_loss(a_fake_mean, a_real_mean) + F.mse_loss(a_fake_std, a_real_std) +
                    F.mse_loss(v_fake_mean, v_real_mean) + F.mse_loss(v_fake_std, v_real_std)
                ) / 4
        
        # 3.6 [C] Audio std 正则化损失
        # 目的：约束 fake audio 的边缘分布 std 接近目标（real 约 0.03~0.05）
        g_loss_audio_std = torch.tensor(0.0, device=self.device)
        if self.config.audio_std_weight > 0:
            # 计算 fake audio pooled 后的 std (across batch)
            fake_audio_pooled_std = g_a_pool.std(dim=0).mean()  # 所有维度的 std 均值
            g_loss_audio_std = ((fake_audio_pooled_std - self.config.audio_std_target) ** 2)
        
        # 4. Teacher Loss
        g_loss_cls = torch.tensor(0.0, device=self.device)
        teacher_used = False # [FLAG] 标记是否启用了 teacher
        
        # [Fix] 使用 1e-6 阈值避免浮点精度问题导致的意外触发
        if self.config.cls_weight > 1e-6:
            # [FIX] 切到训练模式以满足 cuDNN RNN backward 要求
            teacher_used = True
            self._teacher_train_for_backward()
            
            # [Patch 1] Teacher Input Safety
            B_gen = gen_audio.shape[0]
            if len_a is None:
                la_cls = torch.full((B_gen,), gen_audio.shape[1], device=self.device, dtype=torch.long)
            else:
                la_cls = torch.clamp(len_a, min=1, max=gen_audio.shape[1]).long()
                
            if len_v is None:
                lv_cls = torch.full((B_gen,), gen_video.shape[1], device=self.device, dtype=torch.long)
            else:
                lv_cls = torch.clamp(len_v, min=1, max=gen_video.shape[1]).long()
                
            preds, _ = self.classifier(gen_audio, gen_video, la_cls, lv_cls)
            g_loss_cls = self.auxiliary_loss(preds, gen_labels)
            
        # Total G Loss
        g_loss = (
            self.config.rf_weight * g_loss_adv +
            self.config.aux_weight * g_loss_aux_d +
            self.config.cls_weight * g_loss_cls +
            self.config.joint_weight * g_loss_joint +
            self.config.fm_weight * g_loss_fm +
            self.config.mom_weight * g_loss_mom +
            self.config.audio_std_weight * g_loss_audio_std  # [C] audio std 正则化
        )
        
        g_loss.backward()
        
        # [FIX] 如果使用了 teacher，反传后立即切回 eval 模式
        if teacher_used:
            self._teacher_eval_for_inference()
            
        self.opt_g_audio.step()
        self.opt_g_video.step()
        
        # [Fix] 解冻 D 参数准备 D-step
        for p in self.audio_discriminator.parameters():
            p.requires_grad_(True)
        for p in self.video_discriminator.parameters():
            p.requires_grad_(True)
        for p in self.joint_discriminator.parameters():
            p.requires_grad_(True)

        # ================= Train Discriminators =================
        self.opt_d_audio.zero_grad()
        self.opt_d_video.zero_grad()
        
        # 1. Independent Ds
        a_real_val, a_real_aux = self.audio_discriminator(real_audio, return_aux=True)
        v_real_val, v_real_aux = self.video_discriminator(real_video, return_aux=True)
        
        a_fake_val, a_fake_aux = self.audio_discriminator(gen_audio.detach(), return_aux=True)
        v_fake_val, v_fake_aux = self.video_discriminator(gen_video.detach(), return_aux=True)
        
        d_loss_adv = (
            self.adversarial_loss(a_real_val, valid) + self.adversarial_loss(v_real_val, valid) +
            self.adversarial_loss(a_fake_val, fake) + self.adversarial_loss(v_fake_val, fake)
        ) / 4
        
        d_loss_aux = (
            self.auxiliary_loss(a_real_aux, real_labels) + self.auxiliary_loss(v_real_aux, real_labels) +
            self.auxiliary_loss(a_fake_aux, gen_labels) + self.auxiliary_loss(v_fake_aux, gen_labels)
        ) / 4
        
        # 2. Joint Critic (pooled features)
        r_a_pool = masked_mean_torch(real_audio, len_a)
        r_v_pool = masked_mean_torch(real_video, len_v)
        
        # [B] Multi-step Joint D training (R0-2 证明这能让 Joint 保持正向 Gap)
        # 第一步与 Independent Ds 一起训练，后续步独立训练 Joint
        for joint_step in range(self.config.joint_d_steps):
            self.opt_d_joint.zero_grad()
            
            real_joint_val = self.joint_discriminator(r_a_pool, r_v_pool)
            fake_joint_val = self.joint_discriminator(g_a_pool.detach(), g_v_pool.detach())
            
            d_loss_joint = (
                self.adversarial_loss(real_joint_val, valid) +
                self.adversarial_loss(fake_joint_val, fake)
            ) / 2
            
            # 第一步：Joint 与 Independent Ds 一起反向传播
            if joint_step == 0:
                d_loss = (
                    self.config.rf_weight * d_loss_adv +
                    self.config.aux_weight * d_loss_aux +
                    self.config.joint_weight * d_loss_joint
                )
                d_loss.backward()
                self.opt_d_audio.step()
                self.opt_d_video.step()
                self.opt_d_joint.step()
            else:
                # 后续步：只训练 Joint（Independent Ds 已经更新完）
                (self.config.joint_weight * d_loss_joint).backward()
                self.opt_d_joint.step()
        
        # Metrics (Sigmoid applied here for accuracy calculation only)
        with torch.no_grad():
            # 1. D(real), D(fake) 的 logit 均值/方差（sigmoid 前）
            d_real_logit = torch.cat([a_real_val, v_real_val], dim=0)
            d_fake_logit = torch.cat([a_fake_val, v_fake_val], dim=0)
            d_real_logit_mean = d_real_logit.mean().item()
            d_real_logit_std = d_real_logit.std().item()
            d_fake_logit_mean = d_fake_logit.mean().item()
            d_fake_logit_std = d_fake_logit.std().item()
            
            # 2. D(real), D(fake) 的 sigmoid 均值
            d_real_sigmoid_mean = torch.sigmoid(d_real_logit).mean().item()
            d_fake_sigmoid_mean = torch.sigmoid(d_fake_logit).mean().item()
            
            # 3. D_acc_real, D_acc_fake 分开计算
            d_acc_real = ((torch.sigmoid(a_real_val) > 0.5).float().mean() + 
                          (torch.sigmoid(v_real_val) > 0.5).float().mean()) / 2
            d_acc_fake = ((torch.sigmoid(a_fake_val) < 0.5).float().mean() + 
                          (torch.sigmoid(v_fake_val) < 0.5).float().mean()) / 2
            d_acc = (d_acc_real + d_acc_fake) / 2
                 
        return {
            'g_loss': g_loss.item(), 'd_loss': d_loss.item(),
            'g_aux_loss': g_loss_cls.item(), 'd_acc': d_acc.item(),
            # 新增可解释 D 指标
            'd_real_logit_mean': d_real_logit_mean,
            'd_real_logit_std': d_real_logit_std,
            'd_fake_logit_mean': d_fake_logit_mean,
            'd_fake_logit_std': d_fake_logit_std,
            'd_real_sigmoid': d_real_sigmoid_mean,
            'd_fake_sigmoid': d_fake_sigmoid_mean,
            'd_acc_real': d_acc_real.item(),
            'd_acc_fake': d_acc_fake.item(),
            # D_loss 各分量 (已乘权重)
            'd_loss_rf': (self.config.rf_weight * d_loss_adv).item(),
            'd_loss_aux': (self.config.aux_weight * d_loss_aux).item(),
            'd_loss_joint': (self.config.joint_weight * d_loss_joint).item(),
            # G_loss 各分量 (已乘权重)
            'g_loss_rf': (self.config.rf_weight * g_loss_adv).item(),
            'g_loss_aux': (self.config.aux_weight * g_loss_aux_d).item(),
            'g_loss_cls': (self.config.cls_weight * g_loss_cls).item(),
            'g_loss_joint': (self.config.joint_weight * g_loss_joint).item(),
            'g_loss_fm': (self.config.fm_weight * g_loss_fm).item(),
            'g_loss_mom': (self.config.mom_weight * g_loss_mom).item(),
            'g_loss_audio_std': (self.config.audio_std_weight * g_loss_audio_std).item(),  # [C] audio std loss
        }

    def evaluate_quality(self, dataloader, num_batches=10):
        """评估生成质量，返回 (teacher_acc, detailed_metrics_dict)
        
        新增指标：
        - audio/video fake 的 mean/std/range（边缘分布监控）
        - joint critic 的 real/fake logit 差（joint 是否在做事）
        """
        self.audio_generator.eval()
        self.video_generator.eval()
        self.audio_discriminator.eval()
        self.video_discriminator.eval()
        self.joint_discriminator.eval()
        
        correct = 0; total = 0
        # 用于收集 D 的各项统计
        d_real_logits = []
        d_fake_logits = []
        
        # 用于收集边缘分布统计
        fake_audio_all = []
        fake_video_all = []
        real_audio_all = []
        real_video_all = []
        
        # 用于收集 Joint Critic 统计
        joint_real_logits = []
        joint_fake_logits = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches: break
                real_a, real_v, len_a, len_v, y = batch
                real_a = real_a.to(self.device); real_v = real_v.to(self.device)
                len_a = len_a.to(self.device); len_v = len_v.to(self.device)
                labels = y.to(self.device)
                
                z = torch.randn(labels.shape[0], self.config.z_dim, device=self.device)
                gen_a = self.audio_generator(z, labels)
                gen_v = self.video_generator(z, labels)
                
                gen_a = self._mask_by_len(gen_a, len_a)
                gen_v = self._mask_by_len(gen_v, len_v)
                
                # [Patch 1] Teacher Input Safety for Evaluation
                B_eval = gen_a.shape[0]
                if len_a is None:
                    la_c = torch.full((B_eval,), gen_a.shape[1], device=self.device, dtype=torch.long)
                else:
                    la_c = torch.clamp(len_a, min=1, max=gen_a.shape[1]).long()
                    
                if len_v is None:
                    lv_c = torch.full((B_eval,), gen_v.shape[1], device=self.device, dtype=torch.long)
                else:
                    lv_c = torch.clamp(len_v, min=1, max=gen_v.shape[1]).long()

                preds, _ = self.classifier(gen_a, gen_v, la_c, lv_c)
                correct += (preds.argmax(1) == labels).sum().item()
                total += labels.shape[0]
                
                # 收集 D 对 real 和 fake 的 logit
                a_real_val = self.audio_discriminator(real_a, add_noise=False)
                v_real_val = self.video_discriminator(real_v, add_noise=False)
                a_fake_val = self.audio_discriminator(gen_a, add_noise=False)
                v_fake_val = self.video_discriminator(gen_v, add_noise=False)
                
                d_real_logits.append(torch.cat([a_real_val, v_real_val], dim=0))
                d_fake_logits.append(torch.cat([a_fake_val, v_fake_val], dim=0))
                
                # 收集边缘分布数据 (pooled)
                fake_audio_all.append(masked_mean_torch(gen_a, len_a))
                fake_video_all.append(masked_mean_torch(gen_v, len_v))
                real_audio_all.append(masked_mean_torch(real_a, len_a))
                real_video_all.append(masked_mean_torch(real_v, len_v))
                
                # 收集 Joint Critic 的 logit
                r_a_pool = masked_mean_torch(real_a, len_a)
                r_v_pool = masked_mean_torch(real_v, len_v)
                g_a_pool = masked_mean_torch(gen_a, len_a)
                g_v_pool = masked_mean_torch(gen_v, len_v)
                
                joint_real_logits.append(self.joint_discriminator(r_a_pool, r_v_pool))
                joint_fake_logits.append(self.joint_discriminator(g_a_pool, g_v_pool))
        
        self.audio_generator.train()
        self.video_generator.train()
        self.audio_discriminator.train()
        self.video_discriminator.train()
        self.joint_discriminator.train()
        
        teacher_acc = correct / total if total else 0
        
        d_metrics = {}
        
        # 计算 D 的可解释指标
        if d_real_logits and d_fake_logits:
            all_real_logits = torch.cat(d_real_logits, dim=0)
            all_fake_logits = torch.cat(d_fake_logits, dim=0)
            
            d_metrics.update({
                'd_real_logit_mean': all_real_logits.mean().item(),
                'd_real_logit_std': all_real_logits.std().item(),
                'd_fake_logit_mean': all_fake_logits.mean().item(),
                'd_fake_logit_std': all_fake_logits.std().item(),
                'd_real_sigmoid': torch.sigmoid(all_real_logits).mean().item(),
                'd_fake_sigmoid': torch.sigmoid(all_fake_logits).mean().item(),
                'd_acc_real': (torch.sigmoid(all_real_logits) > 0.5).float().mean().item(),
                'd_acc_fake': (torch.sigmoid(all_fake_logits) < 0.5).float().mean().item(),
            })
        
        # 计算边缘分布指标
        if fake_audio_all and fake_video_all:
            fake_a = torch.cat(fake_audio_all, dim=0)  # [N, D]
            fake_v = torch.cat(fake_video_all, dim=0)
            real_a = torch.cat(real_audio_all, dim=0)
            real_v = torch.cat(real_video_all, dim=0)
            
            d_metrics.update({
                # Audio fake 边缘分布
                'fake_audio_mean': fake_a.mean().item(),
                'fake_audio_std': fake_a.std().item(),
                'fake_audio_min': fake_a.min().item(),
                'fake_audio_max': fake_a.max().item(),
                # Audio real 边缘分布 (参考)
                'real_audio_mean': real_a.mean().item(),
                'real_audio_std': real_a.std().item(),
                # Video fake 边缘分布
                'fake_video_mean': fake_v.mean().item(),
                'fake_video_std': fake_v.std().item(),
                'fake_video_min': fake_v.min().item(),
                'fake_video_max': fake_v.max().item(),
                # Video real 边缘分布 (参考)
                'real_video_mean': real_v.mean().item(),
                'real_video_std': real_v.std().item(),
            })
        
        # 计算 Joint Critic 指标
        if joint_real_logits and joint_fake_logits:
            all_joint_real = torch.cat(joint_real_logits, dim=0)
            all_joint_fake = torch.cat(joint_fake_logits, dim=0)
            
            joint_real_mean = all_joint_real.mean().item()
            joint_fake_mean = all_joint_fake.mean().item()
            
            d_metrics.update({
                'joint_real_logit_mean': joint_real_mean,
                'joint_fake_logit_mean': joint_fake_mean,
                'joint_logit_gap': joint_real_mean - joint_fake_mean,  # 关键指标：看 joint 是否在区分
                'joint_real_sigmoid': torch.sigmoid(all_joint_real).mean().item(),
                'joint_fake_sigmoid': torch.sigmoid(all_joint_fake).mean().item(),
            })
        
        return teacher_acc, d_metrics

    def evaluate_oracle_joint_score(self, dataloader, train_steps=100, log_interval=20):
        """
        [A] Oracle Joint Score: 冻结 G，让 Joint Critic 独立训练到饱和，
        返回最终 Gap（Real_logit - Fake_logit）
        
        这个分数衡量的是："在 G 固定的情况下，D_joint 最多能区分多好"。
        如果 Oracle Gap 小，说明 fake 确实接近 real；如果大，说明 G 生成质量差。
        
        Args:
            dataloader: 评估数据加载器
            train_steps: Joint Critic 独立训练的步数
            log_interval: 日志间隔
            
        Returns:
            dict: {
                'oracle_joint_gap': float,      # 最终 Gap (越小越好)
                'oracle_joint_real_logit': float,
                'oracle_joint_fake_logit': float,
                'train_history': list[dict],    # 训练过程中的 Gap 变化
            }
        """
        import copy
        
        # 保存原始 joint_discriminator 状态，评估完后恢复
        original_joint_state = copy.deepcopy(self.joint_discriminator.state_dict())
        
        # 冻结 G
        self.audio_generator.eval()
        self.video_generator.eval()
        for p in self.audio_generator.parameters():
            p.requires_grad_(False)
        for p in self.video_generator.parameters():
            p.requires_grad_(False)
            
        # 确保 Joint D 可训练
        self.joint_discriminator.train()
        for p in self.joint_discriminator.parameters():
            p.requires_grad_(True)
        
        # 使用增强的学习率
        opt_joint = torch.optim.Adam(
            self.joint_discriminator.parameters(),
            lr=self.config.lr_d * self.config.joint_lr_mult,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        train_history = []
        data_iter = iter(dataloader)
        
        for step in range(train_steps):
            # 获取 batch，如果用完就重置
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            real_a, real_v, len_a, len_v, y = batch
            real_a = real_a.to(self.device)
            real_v = real_v.to(self.device)
            len_a = len_a.to(self.device)
            len_v = len_v.to(self.device)
            labels = y.to(self.device)
            
            batch_size = labels.shape[0]
            valid = self._get_smooth_labels(batch_size, real=True)
            fake = self._get_smooth_labels(batch_size, real=False)
            
            # 生成 fake（G 已冻结）
            with torch.no_grad():
                z = torch.randn(batch_size, self.config.z_dim, device=self.device)
                gen_a = self.audio_generator(z, labels)
                gen_v = self.video_generator(z, labels)
                gen_a = self._apply_per_sample_znorm(gen_a, len_a)
                gen_a = self._mask_by_len(gen_a, len_a)
                gen_v = self._mask_by_len(gen_v, len_v)
            
            # Pooled features
            r_a_pool = masked_mean_torch(real_a, len_a)
            r_v_pool = masked_mean_torch(real_v, len_v)
            g_a_pool = masked_mean_torch(gen_a, len_a)
            g_v_pool = masked_mean_torch(gen_v, len_v)
            
            # 训练 Joint
            opt_joint.zero_grad()
            real_joint_val = self.joint_discriminator(r_a_pool, r_v_pool)
            fake_joint_val = self.joint_discriminator(g_a_pool, g_v_pool)
            
            d_loss_joint = (
                self.adversarial_loss(real_joint_val, valid) +
                self.adversarial_loss(fake_joint_val, fake)
            ) / 2
            d_loss_joint.backward()
            opt_joint.step()
            
            # 记录
            if (step + 1) % log_interval == 0 or step == 0:
                with torch.no_grad():
                    gap = real_joint_val.mean().item() - fake_joint_val.mean().item()
                    train_history.append({
                        'step': step + 1,
                        'loss': d_loss_joint.item(),
                        'gap': gap,
                        'real_logit': real_joint_val.mean().item(),
                        'fake_logit': fake_joint_val.mean().item(),
                    })
        
        # 最终评估：在整个 dataloader 上测一次
        self.joint_discriminator.eval()
        all_real_logits = []
        all_fake_logits = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:  # 用 10 个 batch 评估
                    break
                real_a, real_v, len_a, len_v, y = batch
                real_a = real_a.to(self.device)
                real_v = real_v.to(self.device)
                len_a = len_a.to(self.device)
                len_v = len_v.to(self.device)
                labels = y.to(self.device)
                
                z = torch.randn(labels.shape[0], self.config.z_dim, device=self.device)
                gen_a = self.audio_generator(z, labels)
                gen_v = self.video_generator(z, labels)
                gen_a = self._apply_per_sample_znorm(gen_a, len_a)
                gen_a = self._mask_by_len(gen_a, len_a)
                gen_v = self._mask_by_len(gen_v, len_v)
                
                r_a_pool = masked_mean_torch(real_a, len_a)
                r_v_pool = masked_mean_torch(real_v, len_v)
                g_a_pool = masked_mean_torch(gen_a, len_a)
                g_v_pool = masked_mean_torch(gen_v, len_v)
                
                all_real_logits.append(self.joint_discriminator(r_a_pool, r_v_pool))
                all_fake_logits.append(self.joint_discriminator(g_a_pool, g_v_pool))
        
        final_real = torch.cat(all_real_logits, dim=0).mean().item()
        final_fake = torch.cat(all_fake_logits, dim=0).mean().item()
        final_gap = final_real - final_fake
        
        # 恢复原始状态
        self.joint_discriminator.load_state_dict(original_joint_state)
        
        # 解冻 G
        self.audio_generator.train()
        self.video_generator.train()
        for p in self.audio_generator.parameters():
            p.requires_grad_(True)
        for p in self.video_generator.parameters():
            p.requires_grad_(True)
        self.joint_discriminator.train()
        
        return {
            'oracle_joint_gap': final_gap,
            'oracle_joint_real_logit': final_real,
            'oracle_joint_fake_logit': final_fake,
            'train_history': train_history,
        }

    def save_checkpoint(self, path, epoch=None):
        """保存完整 checkpoint，支持 resume 训练"""
        ckpt = {
            'epoch': epoch,
            'config': self.config.__dict__,
            # Generators
            'audio_generator': self.audio_generator.state_dict(),
            'video_generator': self.video_generator.state_dict(),
            # Discriminators (可选，用于 resume)
            'audio_discriminator': self.audio_discriminator.state_dict(),
            'video_discriminator': self.video_discriminator.state_dict(),
            'joint_discriminator': self.joint_discriminator.state_dict(),
            # Optimizers (可选，用于 resume)
            'opt_g_audio': self.opt_g_audio.state_dict(),
            'opt_g_video': self.opt_g_video.state_dict(),
            'opt_d_audio': self.opt_d_audio.state_dict(),
            'opt_d_video': self.opt_d_video.state_dict(),
            'opt_d_joint': self.opt_d_joint.state_dict(),
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path, strict=True, load_discriminators=False, load_optimizers=False):
        """加载 checkpoint，支持完整恢复或仅加载 Generator"""
        ckpt = torch.load(path, map_location=self.device)
        
        # 1) 先恢复 config（关键：确保评估/继续训练时参数一致）
        if 'config' in ckpt:
            saved_config = ckpt['config']
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # 同步更新 generator 的 config 引用
            self.audio_generator.config = self.config
            self.video_generator.config = self.config
            
            logging.info(f"Restored config from checkpoint: "
                        f"audio_out_max={self.config.audio_out_max}, "
                        f"audio_scale_max={self.config.audio_scale_max}, "
                        f"video_out_max={self.config.video_out_max}")
        
        # 2) 加载 Generator 权重
        self.audio_generator.load_state_dict(ckpt['audio_generator'], strict=strict)
        self.video_generator.load_state_dict(ckpt['video_generator'], strict=strict)
        
        # 3) 可选：加载 Discriminator 权重（用于 resume 训练）
        if load_discriminators:
            if 'audio_discriminator' in ckpt:
                self.audio_discriminator.load_state_dict(ckpt['audio_discriminator'], strict=strict)
            if 'video_discriminator' in ckpt:
                self.video_discriminator.load_state_dict(ckpt['video_discriminator'], strict=strict)
            if 'joint_discriminator' in ckpt:
                self.joint_discriminator.load_state_dict(ckpt['joint_discriminator'], strict=strict)
            logging.info("Loaded discriminator weights")
        
        # 4) 可选：加载 Optimizer 状态（用于 resume 训练）
        if load_optimizers:
            if 'opt_g_audio' in ckpt:
                self.opt_g_audio.load_state_dict(ckpt['opt_g_audio'])
            if 'opt_g_video' in ckpt:
                self.opt_g_video.load_state_dict(ckpt['opt_g_video'])
            if 'opt_d_audio' in ckpt:
                self.opt_d_audio.load_state_dict(ckpt['opt_d_audio'])
            if 'opt_d_video' in ckpt:
                self.opt_d_video.load_state_dict(ckpt['opt_d_video'])
            if 'opt_d_joint' in ckpt:
                self.opt_d_joint.load_state_dict(ckpt['opt_d_joint'])
            logging.info("Loaded optimizer states")
        
        return ckpt.get('epoch', None)
    
    def _teacher_train_for_backward(self):
        """
        cuDNN RNN backward 要求 training=True。
        但我们又不想要 Dropout/BN 噪声，所以：
        - 整体 train() 以满足 RNN backward
        - 单独把 Dropout/BN 设为 eval() 来禁用噪声/冻结统计
        """
        self.classifier.train()

        for m in self.classifier.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                m.eval()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
        
        # [Fix] flatten_parameters 已移到 init，这里不再重复调用

    def _teacher_eval_for_inference(self):
        self.classifier.eval()