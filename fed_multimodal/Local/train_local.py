#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCF101 Local Training Script

This script performs centralized (non-federated) training on the UCF101 dataset
to establish the upper bound performance of the multimodal classifier.

Features:
- Uses official UCF101 train/test splits
- Full training data access (no federation)
- Same model architecture as federated experiments
- Saves model for GAN training

Usage:
    python train_local.py --num_epochs 100 --batch_size 32 --learning_rate 0.001
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from constants import constants
from model.mm_models import MMActionClassifier
from Local.dataloader import UCF101LocalDataManager

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='UCF101 Local Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default=str(Path(__file__).parents[1] / 'results'),
                        help='Path to feature data directory')
    parser.add_argument('--dataset_dir', type=str,
                        default=str(Path(__file__).parents[1] / 'datasets' / 'ucf101'),
                        help='Path to UCF101 dataset directory')
    parser.add_argument('--audio_feat', type=str, default='mfcc',
                        help='Audio feature type')
    parser.add_argument('--video_feat', type=str, default='mobilenet_v2',
                        help='Video feature type')
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
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (epochs)')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='Save trained model')
    
    args = parser.parse_args()
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
        
        # Forward pass - model returns (preds, embeddings)
        outputs, _ = model(x_audio, x_video, len_a, len_v)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds)
        all_labels.append(labels)
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(all_labels)
    acc, uar, f1 = compute_metrics(all_preds, all_labels, num_classes)
    
    return avg_loss, acc, uar, f1


def evaluate(model, dataloader, criterion, device, num_classes):
    """Evaluate model on validation/test data"""
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
            
            # Model returns (preds, embeddings)
            outputs, _ = model(x_audio, x_video, len_a, len_v)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(all_labels)
    acc, uar, f1 = compute_metrics(all_preds, all_labels, num_classes)
    
    return avg_loss, acc, uar, f1


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'results' / 'local_training'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data Manager
    logging.info("Loading data...")
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
    
    # Model
    logging.info("Initializing model...")
    model = MMActionClassifier(
        num_classes=dm.num_classes,
        audio_input_dim=dm.audio_feat_dim,
        video_input_dim=dm.video_feat_dim,
        d_hid=args.hid_size,
        en_att=args.att,
        att_name=args.att_name
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss function - use CrossEntropyLoss since model outputs logits
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=0.9, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-6
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_uar': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_uar': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_uar': [], 'test_f1': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    
    # Training loop
    logging.info("=" * 70)
    logging.info("Starting Local Training")
    logging.info("=" * 70)
    logging.info(f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}")
    logging.info(f"Optimizer: {args.optimizer}, Scheduler: {args.scheduler}")
    logging.info("=" * 70)
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_acc, train_uar, train_f1 = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, dm.num_classes
        )
        
        # Validate
        val_loss, val_acc, val_uar, val_f1 = evaluate(
            model, dataloaders['val'], criterion, device, dm.num_classes
        )
        
        # Test
        test_loss, test_acc, test_uar, test_f1 = evaluate(
            model, dataloaders['test'], criterion, device, dm.num_classes
        )
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_uar'].append(train_uar)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_uar'].append(val_uar)
        history['val_f1'].append(val_f1)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_uar'].append(test_uar)
        history['test_f1'].append(test_f1)
        history['lr'].append(current_lr)
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch + 1
            
            # Save best model
            if args.save_model:
                model_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'args': vars(args)
                }, model_path)
        
        # Log progress
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            logging.info(f"\nEpoch {epoch+1}/{args.num_epochs} (LR: {current_lr:.6f})")
            logging.info(f"  [Train] Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, UAR: {train_uar:.2f}%, F1: {train_f1:.2f}%")
            logging.info(f"  [Val]   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, UAR: {val_uar:.2f}%, F1: {val_f1:.2f}%")
            logging.info(f"  [Test]  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, UAR: {test_uar:.2f}%, F1: {test_f1:.2f}%")
            logging.info(f"  [Best]  Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch}), Test Acc: {best_test_acc:.2f}%")
    
    # Final results
    logging.info("\n" + "=" * 70)
    logging.info("FINAL RESULTS")
    logging.info("=" * 70)
    logging.info(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    logging.info(f"Corresponding Test Accuracy: {best_test_acc:.2f}%")
    logging.info(f"Final Test Accuracy: {test_acc:.2f}%")
    logging.info(f"Final Test UAR: {test_uar:.2f}%")
    logging.info(f"Final Test F1: {test_f1:.2f}%")
    logging.info("=" * 70)
    
    # Save final model
    if args.save_model:
        final_model_path = output_dir / 'final_model.pt'
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'args': vars(args)
        }, final_model_path)
        logging.info(f"Final model saved to: {final_model_path}")
    
    # Save history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logging.info(f"Training history saved to: {history_path}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'final_test_acc': test_acc,
        'final_test_uar': test_uar,
        'final_test_f1': test_f1,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
