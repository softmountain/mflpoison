#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCF101 多模态特征 GAN 训练脚本

本脚本训练 GAN 生成与真实数据分布相似的合成多模态特征（音频和视频）。
主要特性：

1. 分离学习率（判别器比生成器慢）
2. 标签平滑
3. 噪声注入
4. 更好的损失权重平衡

用法：
    python train_gan_generator.py --num_epochs 200 --gan_start_epoch 50 --gan_epochs 100
"""

import torch
import random
import numpy as np
import torch.nn as nn
import argparse
import logging
import copy
import time
import sys
import os
import json

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.constants import constants
from fed_multimodal.trainers.server_trainer import Server
from fed_multimodal.model.mm_models import MMActionClassifier
from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from fed_multimodal.generator.gan_generator import MultimodalFeatureGAN, FeatureGANConfig
from fed_multimodal.generator.eval_gan_quality import analyze_feature_quality

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")
    
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('output'))
    
    parser = argparse.ArgumentParser(description='UCF101 Multimodal Feature GAN Training')
    
    # Data arguments
    parser.add_argument('--data_dir', default=path_conf['output_dir'], type=str)
    parser.add_argument('--audio_feat', default='mfcc', type=str)
    parser.add_argument('--video_feat', default='mobilenet_v2', type=str)
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--alpha', type=float, default=5.0, help='Dirichlet alpha (5.0 -> alpha50)')
    
    # Model arguments
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--att', action='store_true')
    parser.add_argument('--att_name', type=str, default='base')
    
    # Federated learning arguments
    parser.add_argument('--fed_alg', default='fed_avg', type=str)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--global_learning_rate', type=float, default=0.05)
    parser.add_argument('--sample_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--mu', type=float, default=0.001)
    parser.add_argument('--modality', default='multimodal', type=str)
    
    # FL model save/load arguments
    parser.add_argument('--fl_model_path', type=str, default=None,
                        help='Path to pretrained FL model. If provided, skip FL training and load directly')
    parser.add_argument('--save_fl_model', action='store_true', default=True,
                        help='Save FL model after training')
    parser.add_argument('--skip_fl_training', action='store_true',
                        help='Skip FL training (requires --fl_model_path)')
    
    # Improved GAN arguments
    parser.add_argument('--gan_start_epoch', type=int, default=50)
    parser.add_argument('--gan_epochs', type=int, default=100)
    parser.add_argument('--gan_z_dim', type=int, default=128)
    parser.add_argument('--gan_hidden_dim', type=int, default=256)
    
    # Improved training parameters
    parser.add_argument('--gan_lr_g', type=float, default=2e-4,
                        help='Generator learning rate')
    parser.add_argument('--gan_lr_d', type=float, default=1e-4,
                        help='Discriminator learning rate (should be < lr_g)')
    parser.add_argument('--gan_rf_weight', type=float, default=2.0,
                        help='Real/fake loss weight (higher = focus on realism)')
    parser.add_argument('--gan_aux_weight', type=float, default=1.0,
                        help='Auxiliary classification loss weight')
    parser.add_argument('--gan_real_smooth', type=float, default=0.9,
                        help='Label smoothing for real samples')
    parser.add_argument('--gan_fake_smooth', type=float, default=0.1,
                        help='Label smoothing for fake samples')
    parser.add_argument('--gan_noise_std', type=float, default=0.1,
                        help='Noise std for discriminator input')
    parser.add_argument('--gan_use_gp', action='store_true',
                        help='Use gradient penalty')
    parser.add_argument('--gan_gp_weight', type=float, default=10.0)
    
    parser.add_argument('--gan_eval_interval', type=int, default=10)
    parser.add_argument('--gan_save_interval', type=int, default=20)
    parser.add_argument('--run_analysis', action='store_true',
                        help='Run quality analysis after training')
    
    # Simulation arguments
    parser.add_argument('--missing_modality', type=bool, default=False)
    parser.add_argument('--missing_modailty_rate', type=float, default=0.5)
    parser.add_argument('--missing_label', type=bool, default=False)
    parser.add_argument('--missing_label_rate', type=float, default=0.5)
    parser.add_argument('--label_nosiy', type=bool, default=False)
    parser.add_argument('--label_nosiy_level', type=float, default=0.1)
    
    args = parser.parse_args()
    return args


def train_federated(args, server, dataloader_dict, client_ids, num_epochs, device):
    """Train federated model"""
    logging.info(f"Starting FL training for {num_epochs} epochs...")
    
    criterion = nn.NLLLoss().to(device)
    
    for epoch in range(num_epochs):
        server.initialize_epoch_updates(epoch)
        
        # Track training metrics for this epoch
        epoch_train_loss = []
        epoch_train_acc = []
        
        for idx in server.clients_list[epoch]:
            client_id = client_ids[idx]
            dataloader = dataloader_dict[client_id]
            
            if dataloader is None:
                continue
            
            client = ClientFedAvg(
                args, device, criterion, dataloader,
                model=copy.deepcopy(server.global_model),
                label_dict={},
                num_class=constants.num_class_dict[args.dataset]
            )
            
            client.update_weights()
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result['sample'],
                client.result
            )
            
            # Collect training metrics
            if 'loss' in client.result:
                epoch_train_loss.append(client.result['loss'])
            if 'acc' in client.result:
                epoch_train_acc.append(client.result['acc'])
            
            del client
        
        if len(server.num_samples_list) == 0:
            continue
        
        server.average_weights()
        
        # Log detailed metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Calculate average training metrics
            avg_train_loss = np.mean(epoch_train_loss) if epoch_train_loss else 0.0
            avg_train_acc = np.mean(epoch_train_acc) if epoch_train_acc else 0.0
            
            logging.info(f"\n{'='*70}")
            logging.info(f"FL Epoch {epoch+1}/{num_epochs}")
            logging.info(f"{'='*70}")
            
            # Training metrics
            logging.info(f"[Train] Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
            
            # Dev metrics
            with torch.no_grad():
                server.inference(dataloader_dict['dev'])
                dev_result = server.result.copy()
                dev_metrics = f"[Dev]   Loss: {dev_result.get('loss', 0):.4f}, Acc: {dev_result.get('acc', 0):.4f}"
                if 'uar' in dev_result:
                    dev_metrics += f", UAR: {dev_result['uar']:.4f}"
                if 'f1' in dev_result:
                    dev_metrics += f", F1: {dev_result['f1']:.4f}"
                logging.info(dev_metrics)
            
            # Test metrics
            with torch.no_grad():
                server.inference(dataloader_dict['test'])
                test_result = server.result.copy()
                test_metrics = f"[Test]  Loss: {test_result.get('loss', 0):.4f}, Acc: {test_result.get('acc', 0):.4f}"
                if 'uar' in test_result:
                    test_metrics += f", UAR: {test_result['uar']:.4f}"
                if 'f1' in test_result:
                    test_metrics += f", F1: {test_result['f1']:.4f}"
                logging.info(test_metrics)
            
            logging.info(f"{'='*70}\n")
    
    return server


def train_improved_gan(args, global_model, dataloader_dict, client_ids, device):
    """Train GAN to generate synthetic multimodal features"""
    logging.info("Starting Feature GAN training...")
    logging.info(f"  G LR: {args.gan_lr_g}, D LR: {args.gan_lr_d}")
    logging.info(f"  RF Weight: {args.gan_rf_weight}, Aux Weight: {args.gan_aux_weight}")
    logging.info(f"  Label Smoothing: real={args.gan_real_smooth}, fake={args.gan_fake_smooth}")
    logging.info(f"  Noise Std: {args.gan_noise_std}")
    
    # Create config
    config = FeatureGANConfig(
        z_dim=args.gan_z_dim,
        num_classes=constants.num_class_dict[args.dataset],
        audio_seq_len=500,
        audio_feat_dim=constants.feature_len_dict[args.audio_feat],
        video_seq_len=9,
        video_feat_dim=constants.feature_len_dict[args.video_feat],
        hidden_dim=args.gan_hidden_dim,
        lr_g=args.gan_lr_g,
        lr_d=args.gan_lr_d,
        rf_weight=args.gan_rf_weight,
        aux_weight=args.gan_aux_weight,
        real_label_smoothing=args.gan_real_smooth,
        fake_label_smoothing=args.gan_fake_smooth,
        noise_std=args.gan_noise_std,
        use_gradient_penalty=args.gan_use_gp,
        gp_weight=args.gan_gp_weight,
        device=device
    )
    
    # Initialize feature generator
    generator = MultimodalFeatureGAN(
        args=args,
        global_model=global_model,
        config=config
    )
    
    # Use first client's data for training
    train_client_id = client_ids[0]
    train_dataloader = dataloader_dict[train_client_id]
    
    # Training history
    history = {
        'g_loss': [], 'd_loss': [], 'g_aux_loss': [],
        'd_acc': [], 'd_real_acc': [], 'd_fake_acc': [],
        'gen_quality': [], 'd_catches_fake': []
    }
    
    # Training loop
    for epoch in range(args.gan_epochs):
        epoch_metrics = {k: [] for k in ['g_loss', 'd_loss', 'g_aux_loss', 
                                          'd_acc', 'd_real_acc', 'd_fake_acc', 'cls_acc']}
        
        for batch_idx, batch in enumerate(train_dataloader):
            x_audio, x_video, len_a, len_v, labels = batch
            x_audio = x_audio.to(device)
            x_video = x_video.to(device)
            labels = labels.to(device)
            
            metrics = generator.train_step_multimodal(x_audio, x_video, labels)
            
            for k, v in metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k].append(v)
        
        # Record averages
        for k in ['g_loss', 'd_loss', 'g_aux_loss', 'd_acc', 'd_real_acc', 'd_fake_acc']:
            if epoch_metrics[k]:
                history[k].append(np.mean(epoch_metrics[k]))
        
        # Evaluate
        if (epoch + 1) % args.gan_eval_interval == 0:
            gen_quality, d_catches_fake = generator.evaluate_quality(train_dataloader, num_batches=5)
            history['gen_quality'].append(gen_quality)
            history['d_catches_fake'].append(d_catches_fake)
            
            logging.info(
                f"GAN Epoch [{epoch+1}/{args.gan_epochs}] "
                f"G_Loss: {history['g_loss'][-1]:.4f}, D_Loss: {history['d_loss'][-1]:.4f}, "
                f"D_Acc: {history['d_acc'][-1]:.4f} (Real: {history['d_real_acc'][-1]:.4f}, "
                f"Fake: {history['d_fake_acc'][-1]:.4f}), "
                f"Gen_Quality: {gen_quality:.4f}, D_Catches_Fake: {d_catches_fake:.4f}"
            )
        else:
            if (epoch + 1) % 5 == 0:
                logging.info(
                    f"GAN Epoch [{epoch+1}/{args.gan_epochs}] "
                    f"G_Loss: {history['g_loss'][-1]:.4f}, D_Loss: {history['d_loss'][-1]:.4f}, "
                    f"D_Acc: {history['d_acc'][-1]:.4f}"
                )
        
        # Save checkpoint
        if (epoch + 1) % args.gan_save_interval == 0:
            save_dir = Path(os.path.realpath(__file__)).parents[2].joinpath(
                'result', 'gan_generator', args.dataset
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = save_dir / f'gan_checkpoint_epoch{epoch+1}.pt'
            generator.save_checkpoint(str(checkpoint_path))
            logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    return generator, history, config


def main():
    args = parse_args()
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")
    
    # Data manager
    dm = DataloadManager(args)
    dm.get_simulation_setting(alpha=args.alpha)
    
    fold_idx = 1
    
    dm.load_sim_dict(fold_idx=fold_idx)
    dm.get_client_ids(fold_idx=fold_idx)
    
    dataloader_dict = dict()
    logging.info('Loading Data...')
    
    for client_id in tqdm(dm.client_ids):
        audio_dict = dm.load_audio_feat(client_id=client_id, fold_idx=fold_idx)
        video_dict = dm.load_video_feat(client_id=client_id, fold_idx=fold_idx)
        dm.get_label_dist(video_dict, client_id)
        
        shuffle = False if client_id in ['dev', 'test'] else True
        client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
        
        dataloader_dict[client_id] = dm.set_dataloader(
            audio_dict, video_dict,
            client_sim_dict=client_sim_dict,
            default_feat_shape_a=np.array([500, constants.feature_len_dict["mfcc"]]),
            default_feat_shape_b=np.array([9, constants.feature_len_dict["mobilenet_v2"]]),
            shuffle=shuffle
        )
    
    client_ids = [cid for cid in dm.client_ids if cid not in ['dev', 'test']]
    num_clients = len(client_ids)
    
    set_seed(8)
    
    # Initialize global model
    criterion = nn.NLLLoss().to(device)
    global_model = MMActionClassifier(
        num_classes=constants.num_class_dict[args.dataset],
        audio_input_dim=constants.feature_len_dict[args.audio_feat],
        video_input_dim=constants.feature_len_dict[args.video_feat],
        d_hid=args.hid_size,
        en_att=args.att,
        att_name=args.att_name
    ).to(device)
    
    # Initialize server
    server = Server(
        args, global_model, device=device,
        criterion=criterion, client_ids=client_ids
    )
    server.initialize_log(fold_idx)
    server.sample_clients(num_clients, sample_rate=args.sample_rate)
    
    # Define save directory for FL model
    fl_save_dir = Path(os.path.realpath(__file__)).parents[2].joinpath(
        'result', 'fl_models', args.dataset
    )
    fl_save_dir.mkdir(parents=True, exist_ok=True)
    fl_model_save_path = fl_save_dir / f'fl_model_ep{args.num_epochs}_sr{args.sample_rate}_lr{args.learning_rate}.pt'
    
    # Phase 1: Federated Learning Training (or load pretrained model)
    if args.fl_model_path and os.path.exists(args.fl_model_path):
        # Load pretrained FL model
        logging.info("=" * 60)
        logging.info("PHASE 1: Loading Pretrained FL Model")
        logging.info("=" * 60)
        logging.info(f"Loading FL model from: {args.fl_model_path}")
        
        checkpoint = torch.load(args.fl_model_path, map_location=device)
        global_model.load_state_dict(checkpoint['model_state_dict'])
        server.global_model = global_model
        final_fl_acc = checkpoint.get('test_acc', 0.0)
        
        logging.info(f"Loaded FL model. Saved accuracy: {final_fl_acc:.4f}")
        
        # Verify loaded model
        with torch.no_grad():
            server.inference(dataloader_dict['test'])
            verified_acc = server.result['acc']
            logging.info(f"Verified test accuracy: {verified_acc:.4f}")
        final_fl_acc = verified_acc
        
    elif args.skip_fl_training:
        logging.info("=" * 60)
        logging.info("ERROR: --skip_fl_training requires --fl_model_path")
        logging.info("=" * 60)
        logging.info("Please provide a pretrained FL model path or remove --skip_fl_training")
        return
        
    else:
        # Train FL model
        logging.info("=" * 60)
        logging.info("PHASE 1: Complete Federated Learning Training")
        logging.info("=" * 60)
        
        server = train_federated(
            args, server, dataloader_dict, client_ids,
            num_epochs=args.num_epochs, device=device
        )
        
        with torch.no_grad():
            server.inference(dataloader_dict['test'])
            final_fl_acc = server.result['acc']
            logging.info(f"FL Training complete. Final accuracy: {final_fl_acc:.4f}")
        
        # Save FL model
        if args.save_fl_model:
            fl_checkpoint = {
                'model_state_dict': server.global_model.state_dict(),
                'test_acc': final_fl_acc,
                'num_epochs': args.num_epochs,
                'sample_rate': args.sample_rate,
                'learning_rate': args.learning_rate,
                'num_clients': num_clients,
                'alpha': args.alpha,
                'hid_size': args.hid_size
            }
            torch.save(fl_checkpoint, fl_model_save_path)
            logging.info(f"FL model saved to: {fl_model_save_path}")
    
    # Phase 2: Feature GAN training (after FL is complete)
    logging.info("=" * 60)
    logging.info("PHASE 2: Feature GAN Training")
    logging.info("=" * 60)
    logging.info("Training GAN on the fully trained global model...")
    
    generator, gan_history, config = train_improved_gan(
        args, server.global_model, dataloader_dict, client_ids, device
    )
    
    # Phase 3: Quality Analysis
    logging.info("=" * 60)
    logging.info("PHASE 3: Quality Analysis")
    logging.info("=" * 60)
    
    save_dir = Path(os.path.realpath(__file__)).parents[2].joinpath(
        'result', 'gan_generator', args.dataset
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.run_analysis:
        analysis_results = analyze_feature_quality(
            generator, dataloader_dict[client_ids[0]], device,
            num_batches=10,
            save_dir=str(save_dir / 'analysis')
        )
    else:
        # Quick evaluation
        gen_quality, d_catches_fake = generator.evaluate_quality(
            dataloader_dict[client_ids[0]], num_batches=10
        )
        analysis_results = {
            'gen_quality': gen_quality,
            'd_catches_fake': d_catches_fake
        }
        logging.info(f"Generation Quality: {gen_quality:.4f}")
        logging.info(f"D Catches Fake Rate: {d_catches_fake:.4f}")
    
    # Final results
    logging.info("=" * 60)
    logging.info("FINAL RESULTS")
    logging.info("=" * 60)
    logging.info(f"Final FL accuracy (epoch {args.num_epochs}): {final_fl_acc:.4f}")
    logging.info(f"Generation Quality: {analysis_results.get('gen_quality', 'N/A')}")
    logging.info(f"D Catches Fake Rate: {analysis_results.get('d_catches_fake', 'N/A')}")
    
    # Compare with baseline (if D catches fake ~50%, features are more realistic)
    d_catches = analysis_results.get('d_catches_fake', 1.0)
    if d_catches < 0.6:
        logging.info("✓ Good: D struggles to distinguish fake features (more realistic)")
    elif d_catches < 0.8:
        logging.info("○ Moderate: D can somewhat distinguish fake features")
    else:
        logging.info("✗ Poor: D easily catches fake features (adversarial samples)")
    
    # Save results
    results = {
        'final_fl_acc': final_fl_acc,
        'gan_history': {k: [float(v) for v in vals] for k, vals in gan_history.items()},
        'config': {
            'lr_g': config.lr_g,
            'lr_d': config.lr_d,
            'rf_weight': config.rf_weight,
            'aux_weight': config.aux_weight,
            'real_label_smoothing': config.real_label_smoothing,
            'fake_label_smoothing': config.fake_label_smoothing,
            'noise_std': config.noise_std
        },
        'analysis': {k: float(v) if isinstance(v, (float, np.floating)) else v 
                     for k, v in analysis_results.items() if not isinstance(v, dict)}
    }
    
    with open(save_dir / 'gan_generator_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {save_dir / 'gan_generator_results.json'}")
    
    generator.save_checkpoint(str(save_dir / 'gan_final_checkpoint.pt'))
    logging.info(f"Final checkpoint saved to {save_dir / 'gan_final_checkpoint.pt'}")


if __name__ == '__main__':
    main()
