import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _safe_rate(count: int, total: int):
    if total == 0:
        return 0.0
    return float(count) * 100.0 / float(total)


def _prediction_distribution(rows, support: int, top_k: int):
    counts = Counter((int(row['pred_id']), row['pred_name']) for row in rows)
    return [
        {
            'pred_id': int(pred_id),
            'pred_name': pred_name,
            'count': int(count),
            'rate': _safe_rate(int(count), support),
        }
        for (pred_id, pred_name), count in counts.most_common(top_k)
    ]


def _class_rows(predictions):
    grouped = {}
    for row in predictions:
        grouped.setdefault(int(row['truth_id']), []).append(row)
    return grouped


def _target_info(rows):
    for row in rows:
        target_id = row.get('same_group_target_id', '')
        target_name = row.get('same_group_target_name', '')
        if target_id != '' and target_name != '':
            return int(target_id), str(target_name)
    return None, ''


def _bool_value(value):
    return bool(value) if isinstance(value, bool) else str(value).lower() == 'true'


def analyze_per_class_changes(clean_predictions, attack_predictions, top_k: int = 5):
    clean_by_class = _class_rows(clean_predictions)
    attack_by_class = _class_rows(attack_predictions)
    class_ids = sorted(set(clean_by_class.keys()) | set(attack_by_class.keys()))
    rows = []
    details = {}

    for class_id in class_ids:
        clean_rows = sorted(clean_by_class.get(class_id, []), key=lambda row: int(row['index']))
        attack_rows = sorted(attack_by_class.get(class_id, []), key=lambda row: int(row['index']))
        if len(clean_rows) != len(attack_rows):
            raise ValueError(f'Prediction count mismatch for class {class_id}: clean={len(clean_rows)} attack={len(attack_rows)}')
        support = len(clean_rows)
        if support == 0:
            continue
        truth_name = clean_rows[0]['truth_name'] if clean_rows else attack_rows[0]['truth_name']
        truth_group_id = clean_rows[0].get('truth_group_id', '') if clean_rows else attack_rows[0].get('truth_group_id', '')
        target_id, target_name = _target_info(clean_rows + attack_rows)
        is_attacked = any(_bool_value(row.get('is_attacked_label', False)) for row in clean_rows + attack_rows)

        clean_correct = sum(int(row['pred_id']) == class_id for row in clean_rows)
        attack_correct = sum(int(row['pred_id']) == class_id for row in attack_rows)
        clean_wrong = support - clean_correct
        attack_wrong = support - attack_correct
        clean_same_group_other = sum(_bool_value(row.get('is_same_group_other_prediction', False)) for row in clean_rows)
        attack_same_group_other = sum(_bool_value(row.get('is_same_group_other_prediction', False)) for row in attack_rows)
        clean_target_hit = sum(int(row['pred_id']) == target_id for row in clean_rows) if target_id is not None else 0
        attack_target_hit = sum(int(row['pred_id']) == target_id for row in attack_rows) if target_id is not None else 0

        clean_index_map = {int(row['index']): row for row in clean_rows}
        attack_index_map = {int(row['index']): row for row in attack_rows}
        clean_correct_to_attack_wrong = 0
        clean_wrong_to_attack_correct = 0
        clean_not_target_to_attack_target = 0
        prediction_changed = 0
        for sample_idx, clean_row in clean_index_map.items():
            attack_row = attack_index_map[sample_idx]
            clean_pred = int(clean_row['pred_id'])
            attack_pred = int(attack_row['pred_id'])
            if clean_pred != attack_pred:
                prediction_changed += 1
            if clean_pred == class_id and attack_pred != class_id:
                clean_correct_to_attack_wrong += 1
            if clean_pred != class_id and attack_pred == class_id:
                clean_wrong_to_attack_correct += 1
            if target_id is not None and clean_pred != target_id and attack_pred == target_id:
                clean_not_target_to_attack_target += 1

        row = {
            'label_id': int(class_id),
            'label_name': truth_name,
            'group_id': truth_group_id,
            'is_attacked_label': bool(is_attacked),
            'target_id': '' if target_id is None else int(target_id),
            'target_name': target_name,
            'support': int(support),
            'clean_acc': _safe_rate(clean_correct, support),
            'attack_acc': _safe_rate(attack_correct, support),
            'acc_delta': _safe_rate(attack_correct, support) - _safe_rate(clean_correct, support),
            'clean_wrong_rate': _safe_rate(clean_wrong, support),
            'attack_wrong_rate': _safe_rate(attack_wrong, support),
            'wrong_rate_delta': _safe_rate(attack_wrong, support) - _safe_rate(clean_wrong, support),
            'clean_target_hit_rate': _safe_rate(clean_target_hit, support),
            'attack_target_hit_rate': _safe_rate(attack_target_hit, support),
            'target_hit_rate_delta': _safe_rate(attack_target_hit, support) - _safe_rate(clean_target_hit, support),
            'clean_same_group_other_rate': _safe_rate(clean_same_group_other, support),
            'attack_same_group_other_rate': _safe_rate(attack_same_group_other, support),
            'same_group_other_rate_delta': _safe_rate(attack_same_group_other, support) - _safe_rate(clean_same_group_other, support),
            'prediction_changed_rate': _safe_rate(prediction_changed, support),
            'clean_correct_to_attack_wrong_rate': _safe_rate(clean_correct_to_attack_wrong, support),
            'clean_wrong_to_attack_correct_rate': _safe_rate(clean_wrong_to_attack_correct, support),
            'clean_not_target_to_attack_target_rate': _safe_rate(clean_not_target_to_attack_target, support),
        }
        rows.append(row)
        details[str(class_id)] = {
            **row,
            'clean_top_predictions': _prediction_distribution(clean_rows, support, top_k),
            'attack_top_predictions': _prediction_distribution(attack_rows, support, top_k),
        }

    attacked_rows = [row for row in rows if row['is_attacked_label']]
    non_attacked_rows = [row for row in rows if not row['is_attacked_label']]
    summary = {
        'class_count': int(len(rows)),
        'attacked_class_count': int(len(attacked_rows)),
        'non_attacked_class_count': int(len(non_attacked_rows)),
        'total_support': int(sum(row['support'] for row in rows)),
        'attacked_support': int(sum(row['support'] for row in attacked_rows)),
        'non_attacked_support': int(sum(row['support'] for row in non_attacked_rows)),
        'mean_acc_delta_attacked': float(sum(row['acc_delta'] for row in attacked_rows) / len(attacked_rows)) if attacked_rows else 0.0,
        'mean_target_hit_rate_delta_attacked': float(sum(row['target_hit_rate_delta'] for row in attacked_rows) / len(attacked_rows)) if attacked_rows else 0.0,
        'mean_same_group_other_rate_delta_attacked': float(sum(row['same_group_other_rate_delta'] for row in attacked_rows) / len(attacked_rows)) if attacked_rows else 0.0,
        'mean_acc_delta_non_attacked': float(sum(row['acc_delta'] for row in non_attacked_rows) / len(non_attacked_rows)) if non_attacked_rows else 0.0,
        'most_target_shifted_attacked_classes': sorted(attacked_rows, key=lambda row: row['target_hit_rate_delta'], reverse=True)[:10],
        'most_accuracy_damaged_classes': sorted(rows, key=lambda row: row['acc_delta'])[:10],
        'most_accuracy_improved_classes': sorted(rows, key=lambda row: row['acc_delta'], reverse=True)[:10],
    }
    return summary, rows, details


def save_per_class_analysis(summary: dict, rows: list, details: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json({'summary': summary, 'per_class': details}, output_dir / 'per_class_attack_change_summary.json')
    fieldnames = [
        'label_id',
        'label_name',
        'group_id',
        'is_attacked_label',
        'target_id',
        'target_name',
        'support',
        'clean_acc',
        'attack_acc',
        'acc_delta',
        'clean_wrong_rate',
        'attack_wrong_rate',
        'wrong_rate_delta',
        'clean_target_hit_rate',
        'attack_target_hit_rate',
        'target_hit_rate_delta',
        'clean_same_group_other_rate',
        'attack_same_group_other_rate',
        'same_group_other_rate_delta',
        'prediction_changed_rate',
        'clean_correct_to_attack_wrong_rate',
        'clean_wrong_to_attack_correct_rate',
        'clean_not_target_to_attack_target_rate',
    ]
    with open(output_dir / 'per_class_attack_change_table.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    attacked_rows = [row for row in rows if row['is_attacked_label']]
    with open(output_dir / 'attacked_class_target_shift_rank.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(attacked_rows, key=lambda row: row['target_hit_rate_delta'], reverse=True))


def analyze_experiment_per_class_changes(experiment_dir: Path, clean_mode: str = 'clean', attack_mode: str = 'paired_real_label_shift', output_subdir: str = 'per_class_analysis', top_k: int = 5):
    experiment_dir = Path(experiment_dir)
    clean_predictions = _load_json(experiment_dir / 'modes' / clean_mode / 'test_predictions.json')
    attack_predictions = _load_json(experiment_dir / 'modes' / attack_mode / 'test_predictions.json')
    summary, rows, details = analyze_per_class_changes(clean_predictions, attack_predictions, top_k=top_k)
    output_dir = experiment_dir / output_subdir
    save_per_class_analysis(summary, rows, details, output_dir)
    return {
        'output_dir': str(output_dir),
        'summary_json': str(output_dir / 'per_class_attack_change_summary.json'),
        'table_csv': str(output_dir / 'per_class_attack_change_table.csv'),
        'target_shift_rank_csv': str(output_dir / 'attacked_class_target_shift_rank.csv'),
        'summary': summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze per-class prediction changes after attack')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--clean_mode', type=str, default='clean')
    parser.add_argument('--attack_mode', type=str, default='paired_real_label_shift')
    parser.add_argument('--output_subdir', type=str, default='per_class_analysis')
    parser.add_argument('--top_k', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_experiment_per_class_changes(
        Path(args.experiment_dir),
        clean_mode=args.clean_mode,
        attack_mode=args.attack_mode,
        output_subdir=args.output_subdir,
        top_k=args.top_k,
    )
    print(f'Saved per-class attack-change analysis to {result["output_dir"]}')


if __name__ == '__main__':
    main()
