#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or fine-tune the UCF101 classifier on GAN-generated fake features.

Supports label-flip attack by flipping labels for a target class while keeping
generated features conditioned on the original label.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from model.mm_models import MMActionClassifier
from Local.dataloader import UCF101LocalDataManager
from generator.gan_generator import MultimodalFeatureGAN, FeatureGANConfig


def set_seed(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train classifier with fake GAN features')

    # Data args
    parser.add_argument('--data_dir', type=str,
                        default=str(Path(__file__).parents[1] / 'results'))
    parser.add_argument('--dataset_dir', type=str,
                        default=str(Path(__file__).parents[1] / 'datasets' / 'ucf101'))
    parser.add_argument('--audio_feat', type=str, default='mfcc')
    parser.add_argument('--video_feat', type=str, default='mobilenet_v2')
    parser.add_argument('--split_idx', type=int, default=1)
    parser.add_argument('--val_split', type=float, default=0.1)

    # GAN/model args
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--model_path', type=str,
                        default=str(Path(__file__).parent / 'results' / 'local_training' / 'best_model.pt'))
    parser.add_argument('--init_from_model', action='store_true',
                        help='Initialize classifier weights from model_path (for attack/fine-tune)')
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--att', action='store_true')
    parser.add_argument('--att_name', type=str, default='base')

    # Training args
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=5)

    # Attack args
    parser.add_argument('--attack_src_label', type=int, default=-1,
                        help='Label id to flip from (set >=0 to enable)')
    parser.add_argument('--attack_dst_label', type=int, default=-1,
                        help='Label id to flip to')
    parser.add_argument('--attack_prob', type=float, default=0.0,
                        help='Flip probability for attack samples')
    parser.add_argument('--attack_seed', type=int, default=42)

    # Output args
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_model', action='store_true', default=True)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_baseline', action='store_true',
                        help='Evaluate model before training (useful for attacks)')

    return parser.parse_args()


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'train.log'
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
    logging.basicConfig(
        format='%(asctime)s %(levelname)-3s ==> %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return log_file


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


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    acc = (preds == labels).mean() * 100

    recalls = []
    per_class = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_acc = (preds[mask] == c).mean()
            per_class[c] = class_acc * 100
            recalls.append(class_acc)
    uar = np.mean(recalls) * 100 if recalls else 0.0

    from sklearn.metrics import f1_score
    f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100

    return acc, uar, f1, per_class


def apply_label_flip(labels: torch.Tensor, src: int, dst: int, prob: float, rng: np.random.Generator):
    if prob <= 0 or src < 0 or dst < 0:
        return labels, 0
    mask = labels == src
    if not mask.any():
        return labels, 0
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    flip_mask = rng.random(idx.shape[0]) < prob
    if not flip_mask.any():
        return labels, 0
    labels_flipped = labels.clone()
    flip_idx = idx[torch.from_numpy(flip_mask).to(idx.device)]
    labels_flipped[flip_idx] = dst
    return labels_flipped, int(flip_idx.numel())


def generate_fake_batch(gan, labels, len_a, len_v, device):
    z = torch.randn(labels.shape[0], gan.config.z_dim, device=device)
    with torch.no_grad():
        fake_a = gan.audio_generator(z, labels)
        fake_v = gan.video_generator(z, labels)
        fake_a = gan._apply_per_sample_znorm(fake_a, len_a)
        fake_a = gan._mask_by_len(fake_a, len_a)
        fake_v = gan._mask_by_len(fake_v, len_v)
    return fake_a, fake_v


def train_epoch_fake(model, gan, dataloader, criterion, optimizer, device, attack_cfg):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    flipped = 0

    attack_rng = attack_cfg['rng']
    src = attack_cfg['src']
    dst = attack_cfg['dst']
    prob = attack_cfg['prob']

    for batch in dataloader:
        _, _, len_a, len_v, labels = batch
        labels = labels.to(device)
        len_a = len_a.to(device)
        len_v = len_v.to(device)

        labels_train, flip_count = apply_label_flip(labels, src, dst, prob, attack_rng)
        flipped += flip_count

        fake_a, fake_v = generate_fake_batch(gan, labels, len_a, len_v, device)

        optimizer.zero_grad()
        outputs, _ = model(fake_a, fake_v, len_a, len_v)
        loss = criterion(outputs, labels_train)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.shape[0]
        preds = outputs.argmax(dim=1)
        correct += (preds == labels_train).sum().item()
        total += labels.shape[0]

    avg_loss = total_loss / total if total else 0.0
    acc = (correct / total * 100.0) if total else 0.0
    return avg_loss, acc, flipped, total


def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x_audio, x_video, len_a, len_v, labels = batch
            x_audio = x_audio.to(device)
            x_video = x_video.to(device)
            len_a = len_a.to(device)
            len_v = len_v.to(device)
            labels = labels.to(device)

            outputs, _ = model(x_audio, x_video, len_a, len_v)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(all_labels)
    acc, uar, f1, per_class = compute_metrics(all_preds, all_labels, num_classes)
    return avg_loss, acc, uar, f1, per_class


def build_classifier(dm, model_args):
    return MMActionClassifier(
        num_classes=dm.num_classes,
        audio_input_dim=dm.audio_feat_dim,
        video_input_dim=dm.video_feat_dim,
        d_hid=model_args['hid_size'],
        en_att=model_args['att'],
        att_name=model_args['att_name']
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    attack_enabled = args.attack_prob > 0 and args.attack_src_label >= 0 and args.attack_dst_label >= 0
    default_root = 'fake_attack' if attack_enabled else 'fake_training'
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        exp_name = args.exp_name or default_root
        stamp = datetime.now().strftime('%y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / 'results' / default_root / f'{stamp}_{exp_name}'
    log_file = setup_logging(output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)
    logging.info("Output dir: %s", output_dir)
    logging.info("Log file: %s", log_file)

    dm = UCF101LocalDataManager(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        audio_feat=args.audio_feat,
        video_feat=args.video_feat,
        split_idx=args.split_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dataloaders = dm.get_dataloaders(val_split=args.val_split)

    # Load model config from existing checkpoint if available
    model_args = {'hid_size': args.hid_size, 'att': args.att, 'att_name': args.att_name}
    base_ckpt = None
    if args.model_path and Path(args.model_path).exists():
        base_ckpt = torch.load(args.model_path, map_location=device)
        saved_args = base_ckpt.get('args', {})
        if isinstance(saved_args, dict):
            model_args['hid_size'] = saved_args.get('hid_size', model_args['hid_size'])
            model_args['att'] = saved_args.get('att', model_args['att'])
            model_args['att_name'] = saved_args.get('att_name', model_args['att_name'])

    # Build teacher model for GAN init
    teacher = build_classifier(dm, model_args).to(device)
    if base_ckpt is not None and 'model_state_dict' in base_ckpt:
        teacher.load_state_dict(base_ckpt['model_state_dict'])
    teacher.eval()

    # Build GAN config from checkpoint
    ckpt = torch.load(args.gan_checkpoint, map_location=device)
    saved_config = ckpt.get('config', {})
    sample = next(iter(dataloaders['train']))
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
    gan.load_checkpoint(args.gan_checkpoint)
    gan.audio_generator.eval()
    gan.video_generator.eval()

    # Classifier to train
    model = build_classifier(dm, model_args).to(device)
    if args.init_from_model and base_ckpt is not None and 'model_state_dict' in base_ckpt:
        model.load_state_dict(base_ckpt['model_state_dict'])

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model params: %d total, %d trainable", total_params, trainable_params)

    # Optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [], 'train_flip': [], 'train_samples': [],
        'val_loss': [], 'val_acc': [], 'val_uar': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_uar': [], 'test_f1': [],
        'lr': []
    }

    baseline = None
    if args.eval_baseline or (attack_enabled and args.init_from_model):
        val_metrics = evaluate(model, dataloaders['val'], criterion, device, dm.num_classes)
        test_metrics = evaluate(model, dataloaders['test'], criterion, device, dm.num_classes)
        baseline = {
            'val': {'loss': val_metrics[0], 'acc': val_metrics[1], 'uar': val_metrics[2], 'f1': val_metrics[3],
                    'per_class_acc': val_metrics[4]},
            'test': {'loss': test_metrics[0], 'acc': test_metrics[1], 'uar': test_metrics[2], 'f1': test_metrics[3],
                     'per_class_acc': test_metrics[4]},
        }
        logging.info("Baseline Val Acc: %.2f%% | Test Acc: %.2f%%", val_metrics[1], test_metrics[1])

    attack_cfg = {
        'src': args.attack_src_label,
        'dst': args.attack_dst_label,
        'prob': args.attack_prob,
        'rng': np.random.default_rng(args.attack_seed),
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_test_acc = 0.0

    logging.info("=" * 70)
    logging.info("Training with fake features")
    logging.info("GAN checkpoint: %s", args.gan_checkpoint)
    logging.info("Init from model: %s", args.init_from_model)
    if attack_enabled:
        logging.info("Attack enabled: %d -> %d (p=%.2f)", args.attack_src_label, args.attack_dst_label, args.attack_prob)
    logging.info("=" * 70)

    for epoch in range(args.num_epochs):
        train_loss, train_acc, flipped, total_samples = train_epoch_fake(
            model, gan, dataloaders['train'], criterion, optimizer, device, attack_cfg
        )

        val_loss, val_acc, val_uar, val_f1, val_per_class = evaluate(
            model, dataloaders['val'], criterion, device, dm.num_classes
        )
        test_loss, test_acc, test_uar, test_f1, test_per_class = evaluate(
            model, dataloaders['test'], criterion, device, dm.num_classes
        )

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_flip'].append(flipped)
        history['train_samples'].append(total_samples)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_uar'].append(val_uar)
        history['val_f1'].append(val_f1)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_uar'].append(test_uar)
        history['test_f1'].append(test_f1)
        history['lr'].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch + 1
            if args.save_model:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'args': vars(args),
                }, output_dir / 'best_model.pt')

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            logging.info(
                "Epoch %d/%d (LR %.6f) | Train Loss %.4f Acc %.2f%% | "
                "Val Acc %.2f%% | Test Acc %.2f%% | Flipped %d/%d",
                epoch + 1, args.num_epochs, current_lr, train_loss, train_acc,
                val_acc, test_acc, flipped, total_samples
            )

    if args.save_model:
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'args': vars(args),
        }, output_dir / 'final_model.pt')

    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'fake_attack' if attack_enabled else 'fake_training',
        'gan_checkpoint': args.gan_checkpoint,
        'gan_config': sanitize_config(saved_config),
        'model_path': args.model_path,
        'attack': {
            'enabled': attack_enabled,
            'src_label': args.attack_src_label,
            'dst_label': args.attack_dst_label,
            'prob': args.attack_prob,
            'seed': args.attack_seed,
            'last_epoch_flip_ratio': (
                history['train_flip'][-1] / max(history['train_samples'][-1], 1)
                if history['train_flip'] else 0.0
            ),
        },
        'baseline': baseline,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'final_test_acc': test_acc,
        'final_test_uar': test_uar,
        'final_test_f1': test_f1,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'val_per_class_acc': val_per_class,
        'test_per_class_acc': test_per_class,
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info("Training complete.")
    logging.info("Best Val Acc: %.2f%% (Epoch %d), Best Test Acc: %.2f%%", best_val_acc, best_epoch, best_test_acc)
    logging.info("Final Test Acc: %.2f%%", test_acc)
    logging.info("Outputs saved to: %s", output_dir)


if __name__ == '__main__':
    main()
