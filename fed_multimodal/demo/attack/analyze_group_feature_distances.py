import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fed_multimodal.demo.ucf101_demo.config import resolve_config as resolve_demo_config
from fed_multimodal.demo.ucf101_demo.loader import create_loader

from .poisoned_eval import save_json
from .visual_groups import (
    attacked_label_ids_from_groups,
    build_label_to_attack_group,
    build_same_group_candidate_map,
    label_id_to_name,
    load_attack_group_spec,
    sorted_label_names,
)

DEFAULT_OUTPUT_DIR = Path('/home/xp/fed-multimodal/fed_multimodal/demo/results/ucf101/attack/group_distance_analysis')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze train-set source-target feature distances for UCF101 demo groups')
    parser.add_argument('--fold_idx', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--attack_group_spec_path', type=str, default=None)
    parser.add_argument('--analysis_output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _masked_mean(features: torch.Tensor, valid_len: int):
    valid_len = int(valid_len)
    if valid_len <= 0:
        return np.zeros(int(features.shape[-1]), dtype=np.float64)
    return features[:valid_len].detach().cpu().numpy().astype(np.float64).mean(axis=0)


def _l2_normalize(matrix: np.ndarray):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _cosine_distance_matrix(class_features: dict, class_ids: list):
    centroids = []
    support = {}
    for class_id in class_ids:
        values = class_features.get(int(class_id), [])
        support[str(class_id)] = len(values)
        if values:
            centroids.append(np.stack(values, axis=0).mean(axis=0))
        else:
            first_dim = len(next(iter(class_features.values()))[0])
            centroids.append(np.zeros(first_dim, dtype=np.float64))
    centroid_matrix = _l2_normalize(np.stack(centroids, axis=0))
    similarity = centroid_matrix @ centroid_matrix.T
    return np.clip(1.0 - similarity, 0.0, 2.0), support


def _save_matrix_csv(matrix: np.ndarray, class_ids: list, class_names: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [class_names[class_id] for class_id in class_ids]
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['source_target'] + headers)
        for row_idx, class_id in enumerate(class_ids):
            writer.writerow([class_names[class_id]] + [float(value) for value in matrix[row_idx]])


def _plot_matrix(matrix: np.ndarray, labels: list, title: str, path: Path):
    size = max(8, min(22, 0.32 * len(labels)))
    plt.figure(figsize=(size, size))
    plt.imshow(matrix, cmap='viridis', vmin=0.0, vmax=min(2.0, max(0.6, float(np.percentile(matrix, 95)))))
    plt.colorbar(label='Cosine distance')
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _build_pair_rows(matrix: np.ndarray, class_ids: list, class_names: list, candidate_map: dict, label_to_group_id: dict):
    rows = []
    index_by_id = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    for source_id in class_ids:
        for target_id in candidate_map.get(int(source_id), []):
            rows.append({
                'source_id': int(source_id),
                'source_label': class_names[int(source_id)],
                'target_id': int(target_id),
                'target_label': class_names[int(target_id)],
                'group_id': label_to_group_id[int(source_id)],
                'cosine_distance': float(matrix[index_by_id[int(source_id)], index_by_id[int(target_id)]]),
            })
    return rows


def _save_pair_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['source_id', 'source_label', 'target_id', 'target_label', 'group_id', 'cosine_distance'])
        writer.writeheader()
        writer.writerows(rows)


def _summarize_group_distances(pair_rows_by_modality: dict):
    grouped = {}
    for modality, rows in pair_rows_by_modality.items():
        for row in rows:
            group_id = row['group_id']
            grouped.setdefault(group_id, {}).setdefault(modality, []).append(float(row['cosine_distance']))
    summary = {}
    for group_id, values_by_modality in grouped.items():
        summary[group_id] = {}
        for modality, values in values_by_modality.items():
            arr = np.array(values, dtype=np.float64)
            summary[group_id][modality] = {
                'pair_count': int(len(arr)),
                'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'min': float(arr.min()),
                'max': float(arr.max()),
            }
    return summary


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_output_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    demo_cfg = resolve_demo_config(data_dir=args.data_dir, output_dir=args.output_dir, train_batch_size=args.batch_size)
    loader = create_loader(args.fold_idx, data_dir=demo_cfg.data_dir, output_dir=demo_cfg.output_dir, train_batch_size=args.batch_size)
    group_spec = load_attack_group_spec(args.attack_group_spec_path)
    candidate_map = build_same_group_candidate_map(group_spec, attack_enabled_only=True)
    attacked_label_ids = attacked_label_ids_from_groups(group_spec)
    label_to_group_id = build_label_to_attack_group(group_spec)
    id_to_name = label_id_to_name()
    class_names = sorted_label_names()

    audio_features = {int(label_id): [] for label_id in attacked_label_ids}
    video_features = {int(label_id): [] for label_id in attacked_label_ids}
    fused_features = {int(label_id): [] for label_id in attacked_label_ids}
    total_train_samples = 0

    for client_id in loader.client_ids():
        dataloader = loader.build_dataloader(client_id, shuffle=False)
        for x_a, x_b, l_a, l_b, y in dataloader:
            total_train_samples += int(y.shape[0])
            for idx, label in enumerate(y.tolist()):
                label = int(label)
                if label not in audio_features:
                    continue
                audio_vec = _masked_mean(x_a[idx], int(l_a[idx].item()))
                video_vec = _masked_mean(x_b[idx], int(l_b[idx].item()))
                audio_features[label].append(audio_vec)
                video_features[label].append(video_vec)
                fused_features[label].append(np.concatenate([_l2_normalize(audio_vec.reshape(1, -1))[0], _l2_normalize(video_vec.reshape(1, -1))[0]]))

    class_ids = sorted(attacked_label_ids)
    modality_features = {
        'audio': audio_features,
        'video': video_features,
        'fused': fused_features,
    }
    matrices = {}
    pair_rows_by_modality = {}
    supports = {}
    short_labels = [id_to_name[int(class_id)] for class_id in class_ids]

    for modality, features in modality_features.items():
        matrix, support = _cosine_distance_matrix(features, class_ids)
        matrices[modality] = matrix
        supports[modality] = support
        _save_matrix_csv(matrix, class_ids, class_names, analysis_dir / f'{modality}_cosine_distance_matrix.csv')
        _plot_matrix(matrix, short_labels, f'UCF101 train {modality} cosine distance matrix', analysis_dir / f'{modality}_cosine_distance_matrix.png')
        rows = _build_pair_rows(matrix, class_ids, class_names, candidate_map, label_to_group_id)
        pair_rows_by_modality[modality] = rows
        _save_pair_csv(rows, analysis_dir / f'{modality}_source_target_pair_distances.csv')

    summary = {
        'fold_idx': int(args.fold_idx),
        'total_train_samples': int(total_train_samples),
        'attacked_label_count': int(len(attacked_label_ids)),
        'class_support': supports['audio'],
        'group_distance_summary': _summarize_group_distances(pair_rows_by_modality),
        'outputs': {
            'video_cosine_distance_matrix': str(analysis_dir / 'video_cosine_distance_matrix.csv'),
            'audio_cosine_distance_matrix': str(analysis_dir / 'audio_cosine_distance_matrix.csv'),
            'fused_feature_distance_matrix': str(analysis_dir / 'fused_cosine_distance_matrix.csv'),
        },
    }
    save_json(summary, analysis_dir / 'summary.json')
    print(f'Saved source-target distance analysis to {analysis_dir}')


if __name__ == '__main__':
    main()
