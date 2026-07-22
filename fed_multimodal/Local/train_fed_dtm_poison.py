#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习 + DTM-GAN 特征中毒攻击（路径 A：离线特征注入，label_flip）

用 UCF101LocalDataManager 加载完整特征，Dirichlet 划分客户端，
恶意客户端在本地数据混入 DTM-GAN label_flip 投毒特征（生成 target 类特征、
标签标 source 类），fed_avg 联邦训练后在真实测试集评估：
  - overall_acc   总体任务精度
  - main_acc      主任务精度（排除 target 类）
  - asr           中毒任务成功率（target 类测试样本被误判为 source 类的比例）

Usage:
    PYTHONPATH=/home/xp/fedpoi python train_fed_dtm_poison.py \
        --poison_path .../dtm_final50_labelflip_s0_t1.pt \
        --num_clients 10 --malicious_ratio 0.4 --poison_ratio 0.2 \
        --source_class 0 --target_class 1 --num_epochs 30 --device cpu
"""

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

sys_path = Path(__file__).parents[1]
import sys
sys.path.insert(0, str(sys_path))

from model.mm_models import MMActionClassifier
from Local.dataloader import UCF101LocalDataManager, collate_mm_fn_padd
from trainers.fed_avg_trainer import ClientFedAvg

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)


class PoisonFeatureDataset(Dataset):
    """DTM-GAN 投毒特征 → 5 元组，兼容 collate_mm_fn_padd。"""

    def __init__(self, audio, video, len_a, len_v, labels):
        self.audio = audio.float()
        self.video = video.float()
        self.len_a = len_a.long()
        self.len_v = len_v.long()
        self.labels = labels.long()

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return (
            self.audio[idx], self.video[idx],
            self.len_a[idx], self.len_v[idx], self.labels[idx],
        )


def dirichlet_partition(labels, num_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    client_indices = [[] for _ in range(num_clients)]
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        props = rng.dirichlet(alpha * np.ones(num_clients))
        cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
        for cid, part in enumerate(np.split(c_idx, cuts)):
            client_indices[cid].extend(part.tolist())
    return [np.array(idxs, dtype=np.int64) for idxs in client_indices]


def parse_args():
    p = argparse.ArgumentParser(description='FedAvg + DTM-GAN label-flip poisoning')
    # data
    p.add_argument('--data_dir', default=str(Path(__file__).parents[1] / 'results'))
    p.add_argument('--dataset_dir', default='/home/xp/fedpoigan/fed_multimodal/datasets/ucf101')
    p.add_argument('--poison_path', default='', help='DTM-GAN label_flip 投毒特征 .pt；空=无攻击基线')
    # attack
    p.add_argument('--malicious_ratio', type=float, default=0.4, help='恶意客户端比例')
    p.add_argument('--poison_ratio', type=float, default=0.2, help='恶意客户端本地数据中投毒样本占比')
    p.add_argument('--source_class', type=int, default=0, help='投毒标签翻到的类（被误判去向）')
    p.add_argument('--target_class', type=int, default=1, help='被攻击的类（生成其特征）')
    # federation
    p.add_argument('--num_clients', type=int, default=10)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--sample_rate', type=float, default=1.0)
    p.add_argument('--num_epochs', type=int, default=30)
    p.add_argument('--local_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=0.05)
    # ClientFedAvg args
    p.add_argument('--fed_alg', default='fed_avg')
    p.add_argument('--modality', default='multimodal')
    p.add_argument('--dataset', default='ucf101')
    p.add_argument('--mu', type=float, default=0.0)
    # model
    p.add_argument('--hid_size', type=int, default=64)
    p.add_argument('--att', action='store_true')
    p.add_argument('--att_name', default='base')
    # misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--log_interval', type=int, default=5)
    p.add_argument('--output_dir', default='fed_multimodal/Local/results/fed_dtm_poison')
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, source_class, target_class):
    model.eval()
    preds_all, labels_all = [], []
    for x_a, x_b, l_a, l_b, y in loader:
        x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
        l_a, l_b = l_a.to(device), l_b.to(device)
        logits, _ = model(x_a.float(), x_b.float(), l_a, l_b)
        preds_all.append(logits.argmax(dim=1).cpu())
        labels_all.append(y.cpu())
    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()

    overall_acc = (preds == labels).mean() * 100

    non_target = labels != target_class
    main_acc = (preds[non_target] == labels[non_target]).mean() * 100 if non_target.any() else 0.0

    target_mask = labels == target_class
    if target_mask.any():
        asr = (preds[target_mask] == source_class).mean() * 100
    else:
        asr = 0.0

    recalls = []
    for c in np.unique(labels):
        m = labels == c
        if m.sum() > 0:
            recalls.append((preds[m] == c).mean())
    uar = np.mean(recalls) * 100 if recalls else 0.0
    return overall_acc, main_acc, asr, uar


def fed_avg_aggregate(model, updates):
    total = sum(n for _, n in updates)
    avg = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    for sd, n in updates:
        for k in avg:
            avg[k] += sd[k].float() * (n / total)
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in avg.items()})
    return model


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    logging.info(f'device={device} | attack: malicious={args.malicious_ratio:.0%} '
                 f'poison={args.poison_ratio:.0%} src={args.source_class} tgt={args.target_class}')

    # 1. 真实特征
    dm = UCF101LocalDataManager(
        data_dir=args.data_dir, dataset_dir=args.dataset_dir,
        audio_feat='mfcc', video_feat='mobilenet_v2',
        batch_size=args.batch_size, split_idx=1, num_workers=args.num_workers,
    )
    loaders = dm.get_dataloaders(val_split=0.0)
    train_dataset = loaders['full_train'].dataset
    test_loader = loaders['test']
    num_classes = dm.num_classes
    logging.info(f'train={len(train_dataset)} test={len(test_loader.dataset)} classes={num_classes}')

    # 2. Dirichlet 划分
    labels = [train_dataset.audio_data[i][-2] for i in range(len(train_dataset))]
    client_indices = dirichlet_partition(labels, args.num_clients, args.alpha, args.seed)
    client_sizes = [len(idxs) for idxs in client_indices]
    logging.info(f'client sizes: {client_sizes}')
    client_datasets = [Subset(train_dataset, idxs) for idxs in client_indices]

    # 3. 选恶意客户端 + 注入
    num_malicious = int(round(args.malicious_ratio * args.num_clients))
    malicious_ids = list(range(num_malicious))
    rng = np.random.default_rng(args.seed)
    if args.poison_path and num_malicious > 0:
        poison = torch.load(args.poison_path, map_location='cpu')
        poison_full = PoisonFeatureDataset(
            poison['audio'], poison['video'], poison['len_a'], poison['len_v'],
            poison['train_label'],
        )
        logging.info(f'poison pool: {len(poison_full)} samples '
                     f'(label={int(poison["train_label"][0])}, condition={int(poison["condition_label"][0])})')
        for cid in malicious_ids:
            n_poison = int(args.poison_ratio * client_sizes[cid])
            n_poison = min(n_poison, len(poison_full))
            pick = rng.choice(len(poison_full), size=n_poison, replace=False)
            client_datasets[cid] = ConcatDataset([client_datasets[cid], Subset(poison_full, pick)])
            logging.info(f'  [malicious] client {cid}: +{n_poison} poison ({args.poison_ratio:.0%} of {client_sizes[cid]})')
    else:
        logging.info('NO poisoning (clean baseline)')

    client_loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers, collate_fn=collate_mm_fn_padd)
        for ds in client_datasets
    ]

    # 4. 模型
    global_model = MMActionClassifier(
        num_classes=num_classes, audio_input_dim=dm.audio_feat_dim,
        video_input_dim=dm.video_feat_dim, d_hid=args.hid_size,
        en_att=args.att, att_name=args.att_name,
    ).to(device)
    criterion = nn.NLLLoss().to(device)

    # 5. 联邦训练
    num_active = max(1, int(args.sample_rate * args.num_clients))
    rng = np.random.default_rng(args.seed + 1)
    history = []
    for epoch in range(1, args.num_epochs + 1):
        sampled = rng.choice(args.num_clients, size=num_active, replace=False)
        updates = []
        train_accs = []
        for cid in sampled:
            client = ClientFedAvg(
                args, device, criterion, client_loaders[cid],
                model=copy.deepcopy(global_model), num_class=num_classes,
            )
            client.update_weights()
            updates.append((client.get_parameters(), client.result['sample']))
            train_accs.append(client.result.get('acc', 0.0))
            del client
        global_model = fed_avg_aggregate(global_model, updates)
        overall_acc, main_acc, asr, uar = evaluate(
            global_model, test_loader, device, args.source_class, args.target_class
        )
        history.append({
            'epoch': epoch, 'train_acc': float(np.mean(train_accs)),
            'overall_acc': overall_acc, 'main_acc': main_acc, 'asr': asr, 'uar': uar,
        })
        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.num_epochs:
            logging.info(
                f'epoch {epoch:3d}/{args.num_epochs} | train={np.mean(train_accs):.2f}% | '
                f'overall={overall_acc:.2f}% main={main_acc:.2f}% ASR={asr:.2f}% uar={uar:.2f}%'
            )

    # 6. 保存
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{"attack" if args.poison_path else "baseline"}_m{args.malicious_ratio}_p{args.poison_ratio}_s{args.source_class}_t{args.target_class}'
    result = {
        'args': vars(args), 'malicious_ids': malicious_ids, 'history': history,
        'final_overall_acc': history[-1]['overall_acc'],
        'final_main_acc': history[-1]['main_acc'],
        'final_asr': history[-1]['asr'],
        'best_overall_acc': max(h['overall_acc'] for h in history),
    }
    fpath = out / f'result_{tag}.json'
    with open(fpath, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f'saved → {fpath}')
    logging.info(f'FINAL overall={history[-1]["overall_acc"]:.2f}% '
                 f'main={history[-1]["main_acc"]:.2f}% ASR={history[-1]["asr"]:.2f}%')


if __name__ == '__main__':
    main()
