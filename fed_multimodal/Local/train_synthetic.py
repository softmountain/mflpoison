#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用合成数据训练模型，在真实测试集上评估质量

Usage:
    python train_synthetic.py --synthetic_data path/to/synthetic.pt --num_epochs 100
"""

import os
import sys
import json
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from model.mm_models import MMActionClassifier
from Local.dataloader import UCF101LocalDataManager

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_seed(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class SyntheticDataset(Dataset):
    """Dataset wrapper for synthetic features"""
    def __init__(self, audio_features, video_features, labels, audio_lengths=None, video_lengths=None):
        self.audio = audio_features
        self.video = video_features
        self.labels = labels
        self.audio_lengths = audio_lengths if audio_lengths is not None else torch.full((len(labels),), audio_features.size(1), dtype=torch.long)
        self.video_lengths = video_lengths if video_lengths is not None else torch.full((len(labels),), video_features.size(1), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.audio[idx],
            self.video[idx],
            self.audio_lengths[idx],
            self.video_lengths[idx],
            self.labels[idx]
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Train on Synthetic Data')

    # Data arguments
    parser.add_argument('--synthetic_data', type=str, required=True,
                        help='Path to synthetic data .pt file')
    parser.add_argument('--dataset_dir', type=str,
                        default=str(Path(__file__).parents[1] / 'datasets' / 'ucf101'),
                        help='Path to UCF101 dataset directory')
    parser.add_argument('--split_idx', type=int, default=1,
                        help='UCF101 split index (1, 2, or 3)')

    # Model arguments
    parser.add_argument('--hid_size', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--att', action='store_true',
                        help='Use attention mechanism')
    parser.add_argument('--att_name', type=str, default='base',
                        help='Attention type')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of the real training split used for model selection')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'],
                        help='Learning rate scheduler')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (epochs)')
    parser.add_argument('--output_dir', type=str,
                        default='fed_multimodal/Local/results/synthetic_training',
                        help='Output directory')

    args = parser.parse_args()
    if not 0.0 < args.val_split < 1.0:
        parser.error('--val_split must be between 0 and 1')
    return args


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """Compute accuracy, UAR, and F1 score"""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    # Accuracy
    acc = (preds == labels).mean() * 100

    # UAR (Unweighted Average Recall)
    recalls = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            recalls.append((preds[mask] == c).mean())
    uar = np.mean(recalls) * 100 if recalls else 0.0

    # F1 (macro)
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100

    return acc, uar, f1


def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        x_audio, x_video, len_a, len_v, labels = batch
        x_audio = x_audio.to(device)
        x_video = x_video.to(device)
        len_a = len_a.to(device)
        len_v = len_v.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(x_audio, x_video, len_a, len_v)
        # Model returns (logits, embeddings)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, uar, f1 = compute_metrics(all_preds, all_labels, num_classes)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, acc, uar, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        x_audio, x_video, len_a, len_v, labels = batch
        x_audio = x_audio.to(device)
        x_video = x_video.to(device)
        len_a = len_a.to(device)
        len_v = len_v.to(device)
        labels = labels.to(device)

        logits = model(x_audio, x_video, len_a, len_v)
        # Model returns (logits, embeddings)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, uar, f1 = compute_metrics(all_preds, all_labels, num_classes)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, acc, uar, f1


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load synthetic data
    logging.info(f"Loading synthetic data from {args.synthetic_data}")
    synth_data = torch.load(args.synthetic_data)

    # Prefer the canonical refactored schema while keeping legacy files readable.
    if 'features' in synth_data:
        from mflpoison.core.types import SyntheticBatch

        canonical = SyntheticBatch.from_dict(synth_data)
        audio_syn = canonical.features['audio']
        video_syn = canonical.features['video']
        labels_syn = canonical.train_labels
        audio_len_syn = canonical.lengths['audio']
        video_len_syn = canonical.lengths['video']
    else:
        audio_syn = synth_data.get('audio_features', synth_data.get('audio'))
        video_syn = synth_data.get('video_features', synth_data.get('video'))
        labels_syn = synth_data.get('labels', synth_data.get('train_label'))
        audio_len_syn = synth_data.get('audio_lengths', synth_data.get('len_a', None))
        video_len_syn = synth_data.get('video_lengths', synth_data.get('len_v', None))

    if audio_syn is None or video_syn is None or labels_syn is None:
        raise ValueError('Synthetic file is missing audio, video, or training labels')

    logging.info(f"Loaded {len(labels_syn)} synthetic samples")
    logging.info(f"  Audio shape: {audio_syn.shape}")
    logging.info(f"  Video shape: {video_syn.shape}")
    logging.info(f"  Num classes: {labels_syn.max().item() + 1}")

    # Create synthetic dataset and dataloader
    synth_dataset = SyntheticDataset(audio_syn, video_syn, labels_syn, audio_len_syn, video_len_syn)
    synth_loader = DataLoader(
        synth_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Load real test data
    logging.info("Loading real test data")
    data_dir = str(Path(__file__).parents[1] / 'results')
    dm = UCF101LocalDataManager(
        data_dir=data_dir,
        dataset_dir=args.dataset_dir,
        audio_feat='mfcc',
        video_feat='mobilenet_v2',
        batch_size=args.batch_size,
        split_idx=args.split_idx,
        num_workers=args.num_workers
    )
    dataloaders = dm.get_dataloaders(val_split=args.val_split)
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # Get dimensions from real data
    audio_dim = dm.audio_feat_dim
    video_dim = dm.video_feat_dim
    num_classes = dm.num_classes

    logging.info(f"Real data - audio_dim: {audio_dim}, video_dim: {video_dim}, num_classes: {num_classes}")

    # Initialize model (same architecture as federated learning)
    model = MMActionClassifier(
        num_classes=num_classes,
        audio_input_dim=audio_dim,
        video_input_dim=video_dim,
        d_hid=args.hid_size,
        en_att=args.att,
        att_name=args.att_name
    ).to(device)

    logging.info(f"Model: {model}")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer and criterion
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    # Setup scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    # Training loop
    logging.info(f"Starting training for {args.num_epochs} epochs")

    best_val_acc = float('-inf')
    best_state = None
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_uar': [],
        'val_f1': []
    }

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_acc, train_uar, train_f1 = train_epoch(
            model, synth_loader, criterion, optimizer, device, num_classes
        )

        # Select the checkpoint on held-out real training data, never on test data.
        val_loss, val_acc, val_uar, val_f1 = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_uar'].append(val_uar)
        history['val_f1'].append(val_f1)

        # Log
        if epoch % args.log_interval == 0 or epoch == 1:
            logging.info(
                f"Epoch {epoch:3d}/{args.num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% "
                f"UAR: {val_uar:.2f}% F1: {val_f1:.2f}%"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    # Evaluate the test set exactly once using the validation-selected checkpoint.
    if best_state is None:
        raise RuntimeError('Training completed without producing a checkpoint')
    model.load_state_dict(best_state)
    test_loss, test_acc, test_uar, test_f1 = evaluate(
        model, test_loader, criterion, device, num_classes
    )

    logging.info("=" * 80)
    logging.info("Training completed!")
    logging.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logging.info(f"Test Accuracy: {test_acc:.2f}%")
    logging.info(f"Test UAR: {test_uar:.2f}%")
    logging.info(f"Test F1: {test_f1:.2f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'args': vars(args),
        'selection_metric': 'validation_accuracy',
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_uar': test_uar,
        'test_f1': test_f1,
        'history': history
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    synth_name = Path(args.synthetic_data).stem
    results_file = output_dir / f'results_{synth_name}_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
