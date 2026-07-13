#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCF101 GAN 训练脚本（主干精简版）

仅保留核心结构：
- 双生成器（音频/视频）+ 双判别器 + Joint Critic
- 核心损失：RF 对抗 + Aux 分类 + Teacher 语义 + Joint 对齐
- 轻量 warmup/ramp 以稳定训练

其余复杂正则（FM/MoM/GP/Noise/Oracle）全部移除。
"""

import sys
import random
import logging
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from model.mm_models import MMActionClassifier
from Local.dataloader import UCF101LocalDataManager
from generator.gan_generator import MultimodalFeatureGAN, FeatureGANConfig

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train core multimodal GAN (UCF101)')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default='BASE')
    parser.add_argument('--seed', type=int, default=42)

    # 训练超参数
    parser.add_argument('--gan_epochs', type=int, default=200)
    parser.add_argument('--gan_lr_g', type=float, default=2e-4)
    parser.add_argument('--gan_lr_d', type=float, default=1e-4)

    # 核心损失权重
    parser.add_argument('--gan_rf_weight', type=float, default=2.0)
    parser.add_argument('--gan_aux_weight', type=float, default=1.0)
    parser.add_argument('--gan_cls_weight', type=float, default=0.1)
    parser.add_argument('--gan_joint_weight', type=float, default=0.05)
    parser.add_argument('--gan_fm_weight', type=float, default=0.05)
    parser.add_argument('--gan_mom_weight', type=float, default=0.05)

    # Joint Critic 强化训练（默认开启）
    parser.add_argument('--gan_joint_d_steps', type=int, default=3)
    parser.add_argument('--gan_joint_lr_mult', type=float, default=2.0)

    # 输出约束（用于对齐真实特征范围）
    parser.add_argument('--gan_audio_out_max', type=float, default=1.0)
    parser.add_argument('--gan_audio_scale_max', type=float, default=0.3)
    parser.add_argument('--gan_audio_bias_max', type=float, default=0.1)
    parser.add_argument('--gan_video_out_max', type=float, default=20.0)
    parser.add_argument('--gan_video_scale_max', type=float, default=8.0)

    # 训练调度与保存
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=0)  # 0=只保存最终
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    parser.add_argument('--ramp_ratio', type=float, default=0.2)

    # 数据
    parser.add_argument('--data_dir', type=str, default=str(Path(__file__).parents[1] / 'results'))
    parser.add_argument('--dataset_dir', type=str, default=str(Path(__file__).parents[1] / 'datasets' / 'ucf101'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--audio_feat', type=str, default='mfcc')
    parser.add_argument('--video_feat', type=str, default='mobilenet_v2')

    # Teacher 模型结构（与训练一致）
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--att', action='store_true')
    parser.add_argument('--att_name', type=str, default='base')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class LocalArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_config(args, dataloader, num_classes, device):
    sample = next(iter(dataloader))
    real_a, real_v = sample[0], sample[1]

    return FeatureGANConfig(
        num_classes=num_classes,
        audio_seq_len=real_a.shape[1],
        audio_feat_dim=real_a.shape[2],
        video_seq_len=real_v.shape[1],
        video_feat_dim=real_v.shape[2],
        lr_g=args.gan_lr_g,
        lr_d=args.gan_lr_d,
        rf_weight=args.gan_rf_weight,
        aux_weight=args.gan_aux_weight,
        cls_weight=args.gan_cls_weight,
        joint_weight=args.gan_joint_weight,
        # 固定平滑值（简化接口）
        real_label_smoothing=0.9,
        fake_label_smoothing=0.1,
        # 防塌缩项（可选）
        fm_weight=args.gan_fm_weight,
        mom_weight=args.gan_mom_weight,
        # Joint 强化
        joint_d_steps=args.gan_joint_d_steps,
        joint_lr_mult=args.gan_joint_lr_mult,
        audio_std_weight=0.0,
        noise_std=0.0,
        use_gradient_penalty=False,
        # 输出范围约束
        audio_out_max=args.gan_audio_out_max,
        audio_scale_max=args.gan_audio_scale_max,
        audio_bias_max=args.gan_audio_bias_max,
        video_out_max=args.gan_video_out_max,
        video_scale_max=args.gan_video_scale_max,
        device=device
    )


def train_gan(args, model, dataloader, device, num_classes):
    logging.info(f"Starting GAN Training: exp_name={args.exp_name}")

    config = build_config(args, dataloader, num_classes, device)
    gan_args = LocalArgs(dataset='ucf101')
    gan = MultimodalFeatureGAN(gan_args, model, config)

    warmup_epochs = int(args.gan_epochs * args.warmup_ratio)
    ramp_epochs = int(args.gan_epochs * args.ramp_ratio)
    target_cls = args.gan_cls_weight
    target_joint = args.gan_joint_weight

    output_dir = Path(__file__).parent / 'results' / 'local_gan'
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.gan_epochs):
        if epoch < warmup_epochs:
            gan.config.cls_weight = 0.0
            gan.config.joint_weight = 0.0
            phase = "WARMUP"
        else:
            ramp_steps = max(1, ramp_epochs)
            progress = min(1.0, (epoch - warmup_epochs) / ramp_steps)
            gan.config.cls_weight = target_cls * progress
            gan.config.joint_weight = target_joint * progress
            phase = f"RAMP ({progress*100:.0f}%)"

        epoch_metrics = {}
        for batch in dataloader:
            x_a, x_v, len_a, len_v, y = batch
            metrics = gan.train_step_multimodal(x_a, x_v, y, len_a, len_v)
            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

        if (epoch + 1) % args.log_interval == 0:
            avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
            logging.info(
                f"[{phase}] Epoch {epoch+1}: "
                f"G={avg.get('g_loss', 0):.4f}, D={avg.get('d_loss', 0):.4f}, "
                f"D_acc={avg.get('d_acc', 0):.3f}, "
                f"ClsW={gan.config.cls_weight:.3f}, JointW={gan.config.joint_weight:.3f}"
            )
            if 'd_real_logit_mean' in avg and 'd_fake_logit_mean' in avg:
                logging.info(
                    "  [D-Stats] "
                    f"Logit real={avg.get('d_real_logit_mean', 0):.2f}±{avg.get('d_real_logit_std', 0):.2f}, "
                    f"fake={avg.get('d_fake_logit_mean', 0):.2f}±{avg.get('d_fake_logit_std', 0):.2f} | "
                    f"Sigmoid real={avg.get('d_real_sigmoid', 0):.3f}, "
                    f"fake={avg.get('d_fake_sigmoid', 0):.3f} | "
                    f"Acc real={avg.get('d_acc_real', 0):.3f}, "
                    f"fake={avg.get('d_acc_fake', 0):.3f}"
                )

        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0 and (epoch + 1) < args.gan_epochs:
            mid_ckpt = output_dir / f'ckpt_{epoch+1}_{args.exp_name}.pt'
            gan.save_checkpoint(str(mid_ckpt), epoch=epoch+1)
            logging.info(f"Saved intermediate checkpoint: {mid_ckpt.name}")

    final_ckpt = output_dir / f'ckpt_{args.gan_epochs}_{args.exp_name}.pt'
    gan.save_checkpoint(str(final_ckpt), epoch=args.gan_epochs)
    logging.info(f"Training complete. Final checkpoint: {final_ckpt}")
    return final_ckpt


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dm = UCF101LocalDataManager(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        audio_feat=args.audio_feat,
        video_feat=args.video_feat,
        batch_size=args.batch_size
    )
    dataloaders = dm.get_dataloaders()

    checkpoint = torch.load(args.model_path, map_location=device)
    saved_args = checkpoint.get('args', {})

    model = MMActionClassifier(
        num_classes=dm.num_classes,
        audio_input_dim=dm.audio_feat_dim,
        video_input_dim=dm.video_feat_dim,
        d_hid=saved_args.get('hid_size', args.hid_size),
        en_att=saved_args.get('att', args.att),
        att_name=saved_args.get('att_name', args.att_name)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    train_gan(args, model, dataloaders['full_train'], device, dm.num_classes)


if __name__ == '__main__':
    main()
