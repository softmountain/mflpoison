#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from fed_multimodal.generator.gan_generator import MultimodalFeatureGAN, FeatureGANConfig
from fed_multimodal.model.mm_models import MMActionClassifier

from .config import resolve_demo_paths, resolve_gan_config
from .dataloader import DemoGANDataManager


def setup_logging(log_dir: Path, exp_name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'{exp_name}.train_gan.log'
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
    logging.basicConfig(
        format='%(asctime)s %(levelname)-3s ==> %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True,
    )
    return log_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train demo multimodal GAN from demo global model')
    parser.add_argument('--fold_idx', type=int, default=1)
    parser.add_argument('--teacher_checkpoint', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='0309BASE')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--gan_epochs', type=int, default=200)
    parser.add_argument('--gan_lr_g', type=float, default=2e-4)
    parser.add_argument('--gan_lr_d', type=float, default=1e-4)
    parser.add_argument('--gan_rf_weight', type=float, default=2.0)
    parser.add_argument('--gan_aux_weight', type=float, default=1.0)
    parser.add_argument('--gan_cls_weight', type=float, default=0.1)
    parser.add_argument('--gan_joint_weight', type=float, default=0.05)
    parser.add_argument('--gan_fm_weight', type=float, default=0.05)
    parser.add_argument('--gan_mom_weight', type=float, default=0.05)
    parser.add_argument('--gan_joint_d_steps', type=int, default=3)
    parser.add_argument('--gan_joint_lr_mult', type=float, default=2.0)
    parser.add_argument('--gan_audio_out_max', type=float, default=1.0)
    parser.add_argument('--gan_audio_scale_max', type=float, default=0.3)
    parser.add_argument('--gan_audio_bias_max', type=float, default=0.1)
    parser.add_argument('--gan_video_out_max', type=float, default=20.0)
    parser.add_argument('--gan_video_scale_max', type=float, default=8.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    parser.add_argument('--ramp_ratio', type=float, default=0.2)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_teacher(teacher_ckpt: dict, device):
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


def build_config(args, dataloader, device):
    sample = next(iter(dataloader))
    real_a, real_v = sample[0], sample[1]
    return FeatureGANConfig(
        num_classes=51,
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
        fm_weight=args.gan_fm_weight,
        mom_weight=args.gan_mom_weight,
        joint_d_steps=args.gan_joint_d_steps,
        joint_lr_mult=args.gan_joint_lr_mult,
        audio_std_weight=0.0,
        noise_std=0.0,
        use_gradient_penalty=False,
        audio_out_max=args.gan_audio_out_max,
        audio_scale_max=args.gan_audio_scale_max,
        audio_bias_max=args.gan_audio_bias_max,
        video_out_max=args.gan_video_out_max,
        video_scale_max=args.gan_video_scale_max,
        device=device,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gan_config = resolve_gan_config(
        fold_idx=args.fold_idx,
        seed=args.seed,
        batch_size=args.batch_size,
        val_split=args.val_split,
        gan_epochs=args.gan_epochs,
        gan_lr_g=args.gan_lr_g,
        gan_lr_d=args.gan_lr_d,
        gan_rf_weight=args.gan_rf_weight,
        gan_aux_weight=args.gan_aux_weight,
        gan_cls_weight=args.gan_cls_weight,
        gan_joint_weight=args.gan_joint_weight,
        gan_fm_weight=args.gan_fm_weight,
        gan_mom_weight=args.gan_mom_weight,
        gan_joint_d_steps=args.gan_joint_d_steps,
        gan_joint_lr_mult=args.gan_joint_lr_mult,
        gan_audio_out_max=args.gan_audio_out_max,
        gan_audio_scale_max=args.gan_audio_scale_max,
        gan_audio_bias_max=args.gan_audio_bias_max,
        gan_video_out_max=args.gan_video_out_max,
        gan_video_scale_max=args.gan_video_scale_max,
        warmup_ratio=args.warmup_ratio,
        ramp_ratio=args.ramp_ratio,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    demo_paths = resolve_demo_paths(gan_config)
    log_file = setup_logging(demo_paths.demo_root / 'gan' / 'logs', args.exp_name)
    logging.info('GAN log file: %s', log_file)
    teacher_path = Path(args.teacher_checkpoint) if args.teacher_checkpoint else demo_paths.demo_root / 'training' / f'fold{args.fold_idx}_best_model.pt'
    teacher_ckpt = torch.load(teacher_path, map_location=device)
    teacher = build_teacher(teacher_ckpt, device)

    dm = DemoGANDataManager(gan_config)
    dls = dm.get_dataloaders()
    config = build_config(args, dls['full_train'], device)
    local_args = type('LocalArgs', (), {'dataset': 'ucf101'})()
    gan = MultimodalFeatureGAN(local_args, teacher, config)

    output_dir = demo_paths.demo_root / 'gan' / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_epochs = int(args.gan_epochs * args.warmup_ratio)
    ramp_epochs = int(args.gan_epochs * args.ramp_ratio)
    target_cls = args.gan_cls_weight
    target_joint = args.gan_joint_weight

    for epoch in range(args.gan_epochs):
        if epoch < warmup_epochs:
            gan.config.cls_weight = 0.0
            gan.config.joint_weight = 0.0
            phase = 'WARMUP'
        else:
            ramp_steps = max(1, ramp_epochs)
            progress = min(1.0, (epoch - warmup_epochs) / ramp_steps)
            gan.config.cls_weight = target_cls * progress
            gan.config.joint_weight = target_joint * progress
            phase = f'RAMP ({progress*100:.0f}%)'

        epoch_metrics = {}
        for batch in dls['full_train']:
            x_a, x_v, len_a, len_v, y = batch
            metrics = gan.train_step_multimodal(x_a, x_v, y, len_a, len_v)
            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

        if (epoch + 1) % args.log_interval == 0:
            avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
            logging.info(
                f'[{phase}] Epoch {epoch+1}: G={avg.get("g_loss", 0):.4f}, D={avg.get("d_loss", 0):.4f}, D_acc={avg.get("d_acc", 0):.3f}, ClsW={gan.config.cls_weight:.3f}, JointW={gan.config.joint_weight:.3f}'
            )

        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0 and (epoch + 1) < args.gan_epochs:
            gan.save_checkpoint(str(output_dir / f'ckpt_{epoch+1}_{args.exp_name}.pt'), epoch=epoch+1)

    final_ckpt = output_dir / f'ckpt_{args.gan_epochs}_{args.exp_name}.pt'
    gan.save_checkpoint(str(final_ckpt), epoch=args.gan_epochs)
    print(f'Saved GAN checkpoint to {final_ckpt}')


if __name__ == '__main__':
    main()
