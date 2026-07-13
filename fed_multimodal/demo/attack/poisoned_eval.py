import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OTHER_GROUP_ID = -1
OTHER_GROUP_NAME = 'OTHER_GROUP'


def build_confusion_outputs(truth: Iterable[int], pred: Iterable[int], num_classes: int) -> Dict[str, List[List[float]]]:
    truth_arr = np.asarray(list(truth), dtype=np.int64)
    pred_arr = np.asarray(list(pred), dtype=np.int64)
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(truth_arr, pred_arr):
        if 0 <= y_true < num_classes and 0 <= y_pred < num_classes:
            counts[y_true, y_pred] += 1
    support = counts.sum(axis=1)
    normalized = np.zeros_like(counts, dtype=np.float64)
    nonzero = support > 0
    normalized[nonzero] = counts[nonzero] / support[nonzero, None] * 100.0
    return {
        'counts': counts.tolist(),
        'row_normalized': np.round(normalized, 4).tolist(),
        'label_support': support.tolist(),
    }


def build_attacked_label_confusion_outputs(
    truth: Iterable[int],
    pred: Iterable[int],
    attacked_ids: Iterable[int],
    num_classes: int,
) -> Dict[str, List[List[float]]]:
    truth_arr = np.asarray(list(truth), dtype=np.int64)
    pred_arr = np.asarray(list(pred), dtype=np.int64)
    attacked_labels = [int(x) for x in attacked_ids]
    row_index = {label_id: idx for idx, label_id in enumerate(attacked_labels)}
    counts = np.zeros((len(attacked_labels), num_classes), dtype=np.int64)
    for y_true, y_pred in zip(truth_arr, pred_arr):
        if y_true in row_index and 0 <= y_pred < num_classes:
            counts[row_index[int(y_true)], int(y_pred)] += 1
    support = counts.sum(axis=1)
    normalized = np.zeros_like(counts, dtype=np.float64)
    nonzero = support > 0
    normalized[nonzero] = counts[nonzero] / support[nonzero, None] * 100.0
    return {
        'row_labels': attacked_labels,
        'col_labels': list(range(num_classes)),
        'counts': counts.tolist(),
        'row_normalized': np.round(normalized, 4).tolist(),
        'row_support': support.tolist(),
    }


def build_group_confusion_outputs(
    truth: Iterable[int],
    pred: Iterable[int],
    group_spec: Dict,
    group_to_label_ids: Dict[str, List[int]],
    class_names: List[str],
) -> Dict[str, Dict]:
    truth_arr = np.asarray(list(truth), dtype=np.int64)
    pred_arr = np.asarray(list(pred), dtype=np.int64)
    outputs = {}
    for group_id, label_ids in group_to_label_ids.items():
        label_ids = [int(label_id) for label_id in label_ids]
        if not label_ids:
            continue
        col_ids = label_ids + [OTHER_GROUP_ID]
        row_index = {label_id: idx for idx, label_id in enumerate(label_ids)}
        col_index = {label_id: idx for idx, label_id in enumerate(col_ids)}
        counts = np.zeros((len(label_ids), len(col_ids)), dtype=np.int64)
        for y_true, y_pred in zip(truth_arr, pred_arr):
            y_true = int(y_true)
            y_pred = int(y_pred)
            if y_true not in row_index:
                continue
            col_label = y_pred if y_pred in col_index else OTHER_GROUP_ID
            counts[row_index[y_true], col_index[col_label]] += 1
        support = counts.sum(axis=1)
        normalized = np.zeros_like(counts, dtype=np.float64)
        nonzero = support > 0
        normalized[nonzero] = counts[nonzero] / support[nonzero, None] * 100.0
        group = group_spec['groups'][group_id]
        outputs[group_id] = {
            'group_id': group_id,
            'display_name_zh': group.get('display_name_zh', group_id),
            'display_name_en': group.get('display_name_en', group_id),
            'attack_enabled': bool(group.get('attack_enabled', False)),
            'reason_zh': group.get('reason_zh', ''),
            'reason_en': group.get('reason_en', ''),
            'row_label_ids': label_ids,
            'row_label_names': [class_names[label_id] for label_id in label_ids],
            'col_label_ids': col_ids,
            'col_label_names': [class_names[label_id] if label_id != OTHER_GROUP_ID else OTHER_GROUP_NAME for label_id in col_ids],
            'counts': counts.tolist(),
            'row_normalized': np.round(normalized, 4).tolist(),
            'row_support': support.tolist(),
        }
    return outputs


def compute_main_task_success_excluding_attacked(truth: Iterable[int], pred: Iterable[int], attacked_ids: Iterable[int]) -> float:
    truth_arr = np.asarray(list(truth), dtype=np.int64)
    pred_arr = np.asarray(list(pred), dtype=np.int64)
    attacked = np.asarray(sorted(set(int(x) for x in attacked_ids)), dtype=np.int64)
    if truth_arr.size == 0:
        return 0.0
    mask = ~np.isin(truth_arr, attacked)
    if not np.any(mask):
        return 0.0
    return float(np.mean(pred_arr[mask] == truth_arr[mask]) * 100.0)


def compute_attack_success_metrics(
    truth: Iterable[int],
    pred: Iterable[int],
    attacked_ids: Iterable[int],
    target_map: Dict[int, int],
) -> Dict[str, float]:
    truth_arr = np.asarray(list(truth), dtype=np.int64)
    pred_arr = np.asarray(list(pred), dtype=np.int64)
    attacked = np.asarray(sorted(set(int(x) for x in attacked_ids)), dtype=np.int64)
    mask = np.isin(truth_arr, attacked)
    support = int(mask.sum())
    if support == 0:
        return {
            'attacked_support': 0,
            'attack_success_wrong_rate': 0.0,
            'attack_success_target_hit_rate': 0.0,
        }
    attacked_truth = truth_arr[mask]
    attacked_pred = pred_arr[mask]
    target_hits = np.array([
        attacked_pred[idx] == int(target_map.get(int(label), int(label)))
        for idx, label in enumerate(attacked_truth)
    ], dtype=bool)
    return {
        'attacked_support': support,
        'attack_success_wrong_rate': float(np.mean(attacked_pred != attacked_truth) * 100.0),
        'attack_success_target_hit_rate': float(np.mean(target_hits) * 100.0),
    }


def compute_same_group_attack_success_metrics(
    truth: Iterable[int],
    pred: Iterable[int],
    attacked_ids: Iterable[int],
    label_to_group_id: Dict[int, str],
    group_to_label_ids: Dict[str, List[int]],
    class_names: List[str],
) -> Dict:
    truth_arr = np.asarray(list(truth), dtype=np.int64)
    pred_arr = np.asarray(list(pred), dtype=np.int64)
    attacked_set = {int(x) for x in attacked_ids}
    mask = np.isin(truth_arr, list(attacked_set)) if attacked_set else np.zeros_like(truth_arr, dtype=bool)
    attacked_truth = truth_arr[mask]
    attacked_pred = pred_arr[mask]
    support = int(mask.sum())
    per_group = {}
    per_label = {}

    def _empty_metrics(local_support: int = 0):
        return {
            'support': int(local_support),
            'same_group_other_hit_count': 0,
            'same_group_other_hit_rate': 0.0,
            'self_correct_count': 0,
            'self_correct_rate': 0.0,
            'outside_group_hit_count': 0,
            'outside_group_hit_rate': 0.0,
        }

    if support == 0:
        return {'same_group_attacked_support': 0, **_empty_metrics(0), 'per_group': per_group, 'per_attacked_label': per_label}

    same_group_other = []
    self_correct = []
    outside_group = []
    for y_true, y_pred in zip(attacked_truth, attacked_pred):
        true_group = label_to_group_id.get(int(y_true))
        pred_group = label_to_group_id.get(int(y_pred))
        is_self = int(y_pred) == int(y_true)
        is_same_group_other = (not is_self) and true_group is not None and pred_group == true_group
        same_group_other.append(is_same_group_other)
        self_correct.append(is_self)
        outside_group.append((not is_self) and not is_same_group_other)

    def _summarize(local_truth, local_pred):
        local_support = len(local_truth)
        if local_support == 0:
            return _empty_metrics(0)
        local_same = []
        local_self = []
        local_outside = []
        for y_true, y_pred in zip(local_truth, local_pred):
            true_group = label_to_group_id.get(int(y_true))
            pred_group = label_to_group_id.get(int(y_pred))
            is_self = int(y_pred) == int(y_true)
            is_same = (not is_self) and true_group is not None and pred_group == true_group
            local_same.append(is_same)
            local_self.append(is_self)
            local_outside.append((not is_self) and not is_same)
        return {
            'support': int(local_support),
            'same_group_other_hit_count': int(np.sum(local_same)),
            'same_group_other_hit_rate': float(np.mean(local_same) * 100.0),
            'self_correct_count': int(np.sum(local_self)),
            'self_correct_rate': float(np.mean(local_self) * 100.0),
            'outside_group_hit_count': int(np.sum(local_outside)),
            'outside_group_hit_rate': float(np.mean(local_outside) * 100.0),
        }

    for group_id, label_ids in group_to_label_ids.items():
        group_attacked = sorted(set(int(label_id) for label_id in label_ids) & attacked_set)
        group_mask = np.isin(attacked_truth, group_attacked) if group_attacked else np.zeros_like(attacked_truth, dtype=bool)
        metrics = _summarize(attacked_truth[group_mask], attacked_pred[group_mask])
        if metrics['support'] > 0:
            per_group[group_id] = metrics

    for label_id in sorted(attacked_set):
        label_mask = attacked_truth == int(label_id)
        metrics = _summarize(attacked_truth[label_mask], attacked_pred[label_mask])
        if metrics['support'] > 0:
            metrics['label_id'] = int(label_id)
            metrics['label_name'] = class_names[int(label_id)]
            metrics['group_id'] = label_to_group_id.get(int(label_id), '')
            per_label[str(label_id)] = metrics

    return {
        'same_group_attacked_support': support,
        'same_group_other_hit_count': int(np.sum(same_group_other)),
        'same_group_other_hit_rate': float(np.mean(same_group_other) * 100.0),
        'self_correct_count': int(np.sum(self_correct)),
        'self_correct_rate': float(np.mean(self_correct) * 100.0),
        'outside_group_hit_count': int(np.sum(outside_group)),
        'outside_group_hit_rate': float(np.mean(outside_group) * 100.0),
        'per_group': per_group,
        'per_attacked_label': per_label,
    }


def compute_non_attacked_impact(clean_non_attacked_acc: float, mode_non_attacked_acc: float) -> float:
    return float(clean_non_attacked_acc - mode_non_attacked_acc)


def _truncate_label_names(class_names: List[str], max_len: int = 16) -> List[str]:
    return [name if len(name) <= max_len else f'{name[:max_len-1]}…' for name in class_names]


def plot_confusion_heatmap(
    conf_matrix: Iterable[Iterable[float]],
    class_names: List[str],
    title: str,
    save_path: Path,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    matrix = np.asarray(conf_matrix, dtype=np.float64)
    labels = _truncate_label_names(class_names)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_rectangular_confusion_heatmap(
    conf_matrix: Iterable[Iterable[float]],
    row_names: List[str],
    col_names: List[str],
    title: str,
    save_path: Path,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    matrix = np.asarray(conf_matrix, dtype=np.float64)
    row_labels = _truncate_label_names(row_names)
    col_labels = _truncate_label_names(col_names, max_len=12)
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.8 + 2), max(5, len(row_labels) * 0.45 + 2)))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_difference_heatmap(
    conf_attack: Iterable[Iterable[float]],
    conf_clean: Iterable[Iterable[float]],
    class_names: List[str],
    title: str,
    save_path: Path,
) -> None:
    attack_arr = np.asarray(conf_attack, dtype=np.float64)
    clean_arr = np.asarray(conf_clean, dtype=np.float64)
    delta = attack_arr - clean_arr
    vmax = float(np.abs(delta).max()) if delta.size else 1.0
    plot_confusion_heatmap(
        delta,
        class_names,
        title,
        save_path,
        cmap='coolwarm',
        vmin=-vmax,
        vmax=vmax,
    )


def plot_rectangular_confusion_difference_heatmap(
    conf_attack: Iterable[Iterable[float]],
    conf_clean: Iterable[Iterable[float]],
    row_names: List[str],
    col_names: List[str],
    title: str,
    save_path: Path,
) -> None:
    attack_arr = np.asarray(conf_attack, dtype=np.float64)
    clean_arr = np.asarray(conf_clean, dtype=np.float64)
    delta = attack_arr - clean_arr
    vmax = float(np.abs(delta).max()) if delta.size else 1.0
    plot_rectangular_confusion_heatmap(
        delta,
        row_names,
        col_names,
        title,
        save_path,
        cmap='coolwarm',
        vmin=-vmax,
        vmax=vmax,
    )


def plot_group_confusion_heatmaps(group_confusions: Dict[str, Dict], title_prefix: str, save_dir: Path, cmap: str = 'magma') -> None:
    for group_id, group in group_confusions.items():
        plot_rectangular_confusion_heatmap(
            group['row_normalized'],
            group['row_label_names'],
            group['col_label_names'],
            f'{title_prefix} - {group["display_name_en"]}',
            save_dir / f'{group_id}.png',
            cmap=cmap,
            vmin=0.0,
            vmax=100.0,
        )


def plot_group_confusion_difference_heatmaps(group_attack: Dict[str, Dict], group_clean: Dict[str, Dict], title_prefix: str, save_dir: Path) -> None:
    for group_id, group in group_attack.items():
        if group_id not in group_clean:
            continue
        plot_rectangular_confusion_difference_heatmap(
            group['row_normalized'],
            group_clean[group_id]['row_normalized'],
            group['row_label_names'],
            group['col_label_names'],
            f'{title_prefix} - {group["display_name_en"]}',
            save_dir / f'{group_id}.png',
        )


def save_prediction_table(
    truth: Iterable[int],
    pred: Iterable[int],
    class_names: List[str],
    attacked_ids: Iterable[int],
    same_group_target_map: Dict[int, int],
    cross_group_target_map: Dict[int, int],
    save_path: Path,
    label_to_group_id: Optional[Dict[int, str]] = None,
    mode_display_name: Optional[Dict[str, str]] = None,
) -> None:
    attacked_set = {int(x) for x in attacked_ids}
    label_to_group_id = label_to_group_id or {}
    rows = []
    truth_arr = list(int(x) for x in truth)
    pred_arr = list(int(x) for x in pred)
    for idx, (truth_id, pred_id) in enumerate(zip(truth_arr, pred_arr)):
        same_target_id = same_group_target_map.get(truth_id)
        cross_target_id = cross_group_target_map.get(truth_id)
        truth_group = label_to_group_id.get(truth_id, '')
        pred_group = label_to_group_id.get(pred_id, '')
        rows.append({
            'index': idx,
            'truth_id': truth_id,
            'truth_name': class_names[truth_id],
            'pred_id': pred_id,
            'pred_name': class_names[pred_id],
            'truth_group_id': truth_group,
            'pred_group_id': pred_group,
            'is_same_group_other_prediction': bool(truth_group and truth_group == pred_group and truth_id != pred_id),
            'is_attacked_label': truth_id in attacked_set,
            'same_group_target_id': '' if same_target_id is None else int(same_target_id),
            'same_group_target_name': '' if same_target_id is None else class_names[int(same_target_id)],
            'cross_group_target_id': '' if cross_target_id is None else int(cross_target_id),
            'cross_group_target_name': '' if cross_target_id is None else class_names[int(cross_target_id)],
            'mode_display_name_zh': '' if mode_display_name is None else mode_display_name.get('zh', ''),
            'mode_display_name_en': '' if mode_display_name is None else mode_display_name.get('en', ''),
        })
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def save_json(payload, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_comparison_table(rows: List[Dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(save_path, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['mode'])
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if isinstance(row[key], (dict, list)):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with open(save_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})
