"""从已提取的 feature.pkl 直接生成 N 客户端联邦划分。

原始 data_partition.py + extract_* 管线依赖原始视频/音频文件与 rawframes split，
本机只有提取好的 feature.pkl（pickle 格式 {key: feature}）+ ucfTrainTestlist 标准 split，
因此这里直接从特征池切分，输出 FL train.py 期望的 client pkl。

输出: {feature_root}/feature/{audio|video}/{feat}/ucf101/alpha{alpha_str}/fold{fold}/{client_id}.pkl
每个 pkl 是 list of [key, file_path, label, feature]（与 extract_audio_feature.py 一致）。
client_id ∈ {"0".."num_clients-1", "dev", "test"}。
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def load_feature_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_class_index(video_dict: dict) -> dict:
    """类名→索引，字母序，与 Local/dataloader.py 及 partition_manager 一致。"""
    classes = sorted({key.split("/")[0] for key in video_dict.keys()})
    return {name: idx for idx, name in enumerate(classes)}


def load_split(split_file: Path, class_to_idx: dict):
    """读 trainlist/testlist，返回 (keys, labels)，仅保留特征池中存在的类。"""
    keys, labels = [], []
    skipped = 0
    with open(split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            video_path = parts[0].replace(".avi", "")  # ClassName/v_xxx_gNN_cNN
            class_name = video_path.split("/")[0]
            if class_name in class_to_idx:
                keys.append(video_path)
                labels.append(class_to_idx[class_name])
            else:
                skipped += 1
    if skipped:
        print(f"  [split {split_file.name}] skipped {skipped} samples (class not in feature pool)")
    return keys, labels


def split_train_dev(train_keys, train_labels, seed: int = 8, dev_ratio: float = 0.2):
    """与 partition_manager.split_train_dev 一致：shuffle 后取前 dev_ratio 作 dev。"""
    arr = np.arange(len(train_keys))
    np.random.seed(seed)
    np.random.shuffle(arr)
    dev_len = int(len(arr) * dev_ratio)
    dev_idx = arr[:dev_len]
    train_idx = arr[dev_len:]
    tr_keys = [train_keys[i] for i in train_idx]
    tr_labels = [train_labels[i] for i in train_idx]
    dev_keys = [train_keys[i] for i in dev_idx]
    dev_labels = [train_labels[i] for i in dev_idx]
    return (tr_keys, tr_labels), (dev_keys, dev_labels)


def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int = 8, min_sample_size: int = 5):
    """复用 partition_manager.direchlet_partition 逻辑（label-based Dirichlet，带 balance）。"""
    K = len(np.unique(labels))
    N = len(labels)
    labels = np.array(labels)
    np.random.seed(seed)
    min_size = 0
    while min_size < min_sample_size:
        file_idx_clients = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, file_idx_clients)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            file_idx_clients = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(file_idx_clients, np.split(idx_k, proportions))
            ]
            min_size = min(len(idx_j) for idx_j in file_idx_clients)
    return file_idx_clients


def assemble_client_pkl(keys, labels, audio_dict, video_dict):
    """组装成 [key, file_path, label, feature]，audio 与 video 按相同 key 对齐。"""
    audio_items, video_items, missing = [], [], 0
    for key, label in zip(keys, labels):
        if key in audio_dict and key in video_dict:
            audio_items.append([key, key, int(label), audio_dict[key]])
            video_items.append([key, key, int(label), video_dict[key]])
        else:
            missing += 1
    if missing:
        print(f"    warning: {missing}/{len(keys)} keys missing in feature pool")
    return audio_items, video_items


def main():
    parser = argparse.ArgumentParser()
    here = Path(os.path.realpath(__file__))
    feat_root_default = str(here.parents[3] / "results")
    dataset_dir_default = str(here.parents[3] / "datasets" / "ucf101")
    parser.add_argument("--feature_root", default=feat_root_default, help="results 目录（含 feature/）")
    parser.add_argument("--dataset_dir", default=dataset_dir_default, help="ucf101 目录（含 ucfTrainTestlist/）")
    parser.add_argument("--audio_feat", default="mfcc")
    parser.add_argument("--video_feat", default="mobilenet_v2")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    audio_pkl = feature_root / "feature" / "audio" / args.audio_feat / "ucf101" / "feature.pkl"
    video_pkl = feature_root / "feature" / "video" / args.video_feat / "ucf101" / "feature.pkl"
    print(f"Loading {audio_pkl}")
    audio_dict = load_feature_pkl(audio_pkl)
    print(f"Loading {video_pkl}")
    video_dict = load_feature_pkl(video_pkl)
    print(f"  audio keys: {len(audio_dict)}, video keys: {len(video_dict)}")

    class_to_idx = build_class_index(video_dict)
    num_classes = len(class_to_idx)
    print(f"  classes: {num_classes}")

    split_dir = Path(args.dataset_dir) / "ucfTrainTestlist"
    train_keys, train_labels = load_split(split_dir / f"trainlist0{args.fold}.txt", class_to_idx)
    test_keys, test_labels = load_split(split_dir / f"testlist0{args.fold}.txt", class_to_idx)
    (tr_keys, tr_labels), (dev_keys, dev_labels) = split_train_dev(train_keys, train_labels, seed=args.seed)
    print(f"  train: {len(tr_keys)}, dev: {len(dev_keys)}, test: {len(test_keys)}")

    client_indices = dirichlet_partition(tr_labels, args.num_clients, args.alpha, seed=args.seed)
    alpha_str = str(args.alpha).replace(".", "")

    audio_out_dir = feature_root / "feature" / "audio" / args.audio_feat / "ucf101" / f"alpha{alpha_str}" / f"fold{args.fold}"
    video_out_dir = feature_root / "feature" / "video" / args.video_feat / "ucf101" / f"alpha{alpha_str}" / f"fold{args.fold}"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)

    total_train = 0
    for cid in range(args.num_clients):
        idxs = client_indices[cid]
        keys = [tr_keys[i] for i in idxs]
        labels = [tr_labels[i] for i in idxs]
        audio_items, video_items = assemble_client_pkl(keys, labels, audio_dict, video_dict)
        with open(audio_out_dir / f"{cid}.pkl", "wb") as f:
            pickle.dump(audio_items, f)
        with open(video_out_dir / f"{cid}.pkl", "wb") as f:
            pickle.dump(video_items, f)
        total_train += len(audio_items)
        print(f"  client {cid}: {len(audio_items)} samples")

    # dev & test
    for split_name, keys, labels in [("dev", dev_keys, dev_labels), ("test", test_keys, test_labels)]:
        audio_items, video_items = assemble_client_pkl(keys, labels, audio_dict, video_dict)
        with open(audio_out_dir / f"{split_name}.pkl", "wb") as f:
            pickle.dump(audio_items, f)
        with open(video_out_dir / f"{split_name}.pkl", "wb") as f:
            pickle.dump(video_items, f)
        print(f"  {split_name}: {len(audio_items)} samples")

    print(f"Done. total train across {args.num_clients} clients: {total_train}")
    print(f"Output: {audio_out_dir}")


if __name__ == "__main__":
    main()
