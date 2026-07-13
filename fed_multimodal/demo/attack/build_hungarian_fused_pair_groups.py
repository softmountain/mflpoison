import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from fed_multimodal.demo.ucf101_demo.config import resolve_config as resolve_demo_config
from fed_multimodal.demo.ucf101_demo.loader import create_loader

from .poisoned_eval import save_json
from .visual_groups import label_id_to_name, sorted_label_names

DEFAULT_OUTPUT_DIR = Path('/home/xp/fed-multimodal/fed_multimodal/demo/results/ucf101/attack/hungarian_pair_distance_analysis')
DEFAULT_GROUP_SPEC_PATH = Path('/home/xp/fed-multimodal/fed_multimodal/demo/attack/ucf101_hungarian_fused_pairs.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Build UCF101 paired groups from fused train-set cosine distances')
    parser.add_argument('--fold_idx', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--analysis_output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--group_spec_output_path', type=str, default=str(DEFAULT_GROUP_SPEC_PATH))
    parser.add_argument('--anchor_unpaired_label', type=str, default='BandMarching')
    return parser.parse_args()


def _masked_mean(features: torch.Tensor, valid_len: int):
    valid_len = int(valid_len)
    if valid_len <= 0:
        return np.zeros(int(features.shape[-1]), dtype=np.float64)
    return features[:valid_len].detach().cpu().numpy().astype(np.float64).mean(axis=0)


def _l2_normalize(matrix: np.ndarray):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _cosine_distance_matrix(features_by_class: dict, class_ids: list):
    centroids = []
    support = {}
    for class_id in class_ids:
        values = features_by_class.get(int(class_id), [])
        support[str(class_id)] = len(values)
        centroids.append(np.stack(values, axis=0).mean(axis=0))
    centroid_matrix = _l2_normalize(np.stack(centroids, axis=0))
    return np.clip(1.0 - centroid_matrix @ centroid_matrix.T, 0.0, 2.0), support


def _save_matrix_csv(matrix: np.ndarray, labels: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['source_target'] + labels)
        for idx, label in enumerate(labels):
            writer.writerow([label] + [float(value) for value in matrix[idx]])


def _plot_matrix(matrix: np.ndarray, labels: list, path: Path):
    size = max(9, min(22, 0.32 * len(labels)))
    plt.figure(figsize=(size, size))
    plt.imshow(matrix, cmap='viridis', vmin=0.0, vmax=min(2.0, max(0.6, float(np.percentile(matrix, 95)))))
    plt.colorbar(label='Cosine distance')
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.title('UCF101 train fused cosine distance matrix')
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _minimum_weight_pairs(distance_matrix: np.ndarray):
    n = int(distance_matrix.shape[0])
    large = 1e6
    if n % 2 == 0:
        cost = np.full((n, n), large, dtype=np.float64)
        for left in range(0, n, 2):
            for class_i in range(n):
                cost[left, class_i] = 0.0
                for class_j in range(n):
                    if class_i != class_j:
                        cost[left + 1, class_j] = min(cost[left + 1, class_j], distance_matrix[class_i, class_j])
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
        pairs = []
        for left in range(0, n, 2):
            source = assignment[left]
            target = assignment[left + 1]
            pairs.append((source, target, float(distance_matrix[source, target])))
        return pairs, None

    best_pairs = None
    best_unpaired = None
    best_total = None
    for unpaired in range(n):
        active = [idx for idx in range(n) if idx != unpaired]
        submatrix = distance_matrix[np.ix_(active, active)]
        sub_pairs, _ = _minimum_weight_pairs(submatrix)
        pairs = [(active[source], active[target], float(distance_matrix[active[source], active[target]])) for source, target, _ in sub_pairs]
        total = sum(distance for _, _, distance in pairs)
        if best_total is None or total < best_total:
            best_total = total
            best_pairs = pairs
            best_unpaired = unpaired
    return best_pairs, best_unpaired


def _build_group_spec(pairs: list, class_ids: list, id_to_name: dict, unpaired_class_id: int = None):
    groups = {}
    for pair_idx, (source_id, target_id, distance) in enumerate(pairs):
        source_id = int(source_id)
        target_id = int(target_id)
        source_name = id_to_name[source_id]
        target_name = id_to_name[target_id]
        group_id = f'fused_pair_{pair_idx:02d}_{source_name}_to_{target_name}'
        groups[group_id] = {
            'display_name_zh': f'融合特征近邻配对 {pair_idx:02d}',
            'display_name_en': f'Fused nearest pair {pair_idx:02d}',
            'attack_enabled': True,
            'attack_source': source_name,
            'attack_target': target_name,
            'fused_cosine_distance': distance,
            'reason_zh': f'由训练集融合特征余弦距离的匈牙利最小权匹配得到；仅攻击 {source_name}，目标标签为 {target_name}。',
            'reason_en': f'Built by minimum-cost Hungarian matching on train-set fused cosine distances; only {source_name} is attacked and shifted to {target_name}.',
            'labels': [source_name, target_name],
        }
    if unpaired_class_id is not None:
        label_name = id_to_name[int(unpaired_class_id)]
        groups[f'unpaired_{label_name}'] = {
            'display_name_zh': '未配对类别',
            'display_name_en': 'Unpaired class',
            'attack_enabled': False,
            'reason_zh': f'UCF101 demo 当前有奇数个类别，该类别 {label_name} 不参与二元配对攻击。',
            'reason_en': f'The UCF101 demo has an odd number of classes, so {label_name} is excluded from paired attack groups.',
            'labels': [label_name],
        }
    return {
        'version': 'hungarian_fused_pair_v1',
        'dataset': 'ucf101_demo',
        'description': 'Class pairs built from train-set fused cosine distances using Hungarian minimum-cost matching. Each group attacks only one source class and leaves the paired target class clean.',
        'groups': groups,
    }


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_output_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    group_spec_path = Path(args.group_spec_output_path)

    demo_cfg = resolve_demo_config(data_dir=args.data_dir, output_dir=args.output_dir, train_batch_size=args.batch_size)
    loader = create_loader(args.fold_idx, data_dir=demo_cfg.data_dir, output_dir=demo_cfg.output_dir, train_batch_size=args.batch_size)
    class_names = sorted_label_names()
    id_to_name = label_id_to_name()
    class_ids = list(range(len(class_names)))
    fused_features = {int(class_id): [] for class_id in class_ids}
    total_train_samples = 0

    for client_id in loader.client_ids():
        dataloader = loader.build_dataloader(client_id, shuffle=False)
        for x_a, x_b, l_a, l_b, y in dataloader:
            total_train_samples += int(y.shape[0])
            for idx, label in enumerate(y.tolist()):
                audio_vec = _masked_mean(x_a[idx], int(l_a[idx].item()))
                video_vec = _masked_mean(x_b[idx], int(l_b[idx].item()))
                fused_vec = np.concatenate([_l2_normalize(audio_vec.reshape(1, -1))[0], _l2_normalize(video_vec.reshape(1, -1))[0]])
                fused_features[int(label)].append(fused_vec)

    distance_matrix, support = _cosine_distance_matrix(fused_features, class_ids)
    np.fill_diagonal(distance_matrix, 1e6)
    anchor_unpaired_id = class_names.index(args.anchor_unpaired_label)
    anchor_order = [class_id for class_id in class_ids if class_id != anchor_unpaired_id]
    active_matrix = distance_matrix[np.ix_(anchor_order, anchor_order)]
    pairs, unpaired_idx = _minimum_weight_pairs(active_matrix)
    pairs = [(anchor_order[source], anchor_order[target], float(distance_matrix[anchor_order[source], anchor_order[target]])) for source, target, _ in pairs]
    unpaired_class_id = anchor_unpaired_id if unpaired_idx is None else anchor_order[unpaired_idx]
    distance_matrix_for_output = distance_matrix.copy()
    np.fill_diagonal(distance_matrix_for_output, 0.0)
    labels = [id_to_name[class_id] for class_id in class_ids]
    _save_matrix_csv(distance_matrix_for_output, labels, analysis_dir / 'all_class_fused_cosine_distance_matrix.csv')
    _plot_matrix(distance_matrix_for_output, labels, analysis_dir / 'all_class_fused_cosine_distance_matrix.png')

    group_spec = _build_group_spec(pairs, class_ids, id_to_name, unpaired_class_id=unpaired_class_id)
    save_json(group_spec, group_spec_path)

    pair_rows = []
    for pair_idx, (source_id, target_id, distance) in enumerate(pairs):
        source_id = int(source_id)
        target_id = int(target_id)
        pair_rows.append({
            'pair_idx': pair_idx,
            'source_id': source_id,
            'source_label': id_to_name[source_id],
            'target_id': target_id,
            'target_label': id_to_name[target_id],
            'fused_cosine_distance': distance,
        })
    with open(analysis_dir / 'hungarian_source_target_pairs.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['pair_idx', 'source_id', 'source_label', 'target_id', 'target_label', 'fused_cosine_distance'])
        writer.writeheader()
        writer.writerows(pair_rows)

    distances = np.array([row['fused_cosine_distance'] for row in pair_rows], dtype=np.float64)
    summary = {
        'fold_idx': int(args.fold_idx),
        'total_train_samples': int(total_train_samples),
        'class_count': int(len(class_ids)),
        'pair_count': int(len(pair_rows)),
        'attacked_label_count': int(len(pair_rows)),
        'unpaired_label': id_to_name[int(unpaired_class_id)] if unpaired_class_id is not None else None,
        'class_support': support,
        'distance_summary': {
            'mean': float(distances.mean()),
            'median': float(np.median(distances)),
            'min': float(distances.min()),
            'max': float(distances.max()),
        },
        'outputs': {
            'group_spec': str(group_spec_path),
            'pairs_csv': str(analysis_dir / 'hungarian_source_target_pairs.csv'),
            'fused_distance_matrix': str(analysis_dir / 'all_class_fused_cosine_distance_matrix.csv'),
        },
    }
    save_json(summary, analysis_dir / 'summary.json')
    print(f'Saved Hungarian fused pair analysis to {analysis_dir}')
    print(f'Saved group spec to {group_spec_path}')


if __name__ == '__main__':
    main()
