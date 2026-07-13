import argparse
import copy
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from fed_multimodal.constants import constants
from fed_multimodal.demo.ucf101_demo.config import resolve_config as resolve_demo_config
from fed_multimodal.demo.ucf101_demo.loader import create_loader
from fed_multimodal.model.mm_models import MMActionClassifier
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from fed_multimodal.trainers.server_trainer import Server

from .config import resolve_robustness_config
from .poisoned_eval import (
    build_confusion_outputs,
    build_group_confusion_outputs,
    compute_attack_success_metrics,
    compute_main_task_success_excluding_attacked,
    compute_non_attacked_impact,
    compute_same_group_attack_success_metrics,
    plot_confusion_difference_heatmap,
    plot_confusion_heatmap,
    plot_group_confusion_difference_heatmaps,
    plot_group_confusion_heatmaps,
    save_comparison_table,
    save_json,
    save_prediction_table,
)
from .analyze_per_class_attack_changes import analyze_experiment_per_class_changes
from .real_label_shift_client import ClientRealSameGroupLabelShift
from .visual_groups import build_group_to_label_ids, build_label_to_attack_group, label_report, load_attack_group_spec, label_name_to_id, sorted_label_names

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

MODE_DISPLAY_NAMES = {
    'clean': {'zh': '干净基线', 'en': 'Clean baseline'},
    'paired_real_label_shift': {
        'zh': '融合近邻单向真实特征标签攻击',
        'en': 'Fused-nearest one-way real-feature label-shift attack',
    },
    'delayed_paired_real_label_shift': {
        'zh': '收敛后融合近邻单向真实特征标签攻击',
        'en': 'Post-convergence fused-nearest one-way real-feature label-shift attack',
    },
}
DEFAULT_ATTACK_OUTPUT_ROOT = Path('/home/xp/fed-multimodal/fed_multimodal/demo/results/ucf101/attack')
DEFAULT_PAIR_GROUP_SPEC = Path('/home/xp/fed-multimodal/fed_multimodal/demo/attack/ucf101_hungarian_fused_pairs.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Paired fused-nearest real-feature label-shift experiment for demo federated learning')
    parser.add_argument('--exp_name', type=str, default='ucf101_hungarian_pair_real_label_shift_100r')
    parser.add_argument('--fold_idx', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ucf101')
    parser.add_argument('--modality', type=str, default='multimodal')
    parser.add_argument('--audio_feat', type=str, default='mfcc')
    parser.add_argument('--video_feat', type=str, default='mobilenet_v2')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--clients_per_round', type=int, default=5)
    parser.add_argument('--malicious_clients', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--global_learning_rate', type=float, default=0.01)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--hid_size', type=int, default=128)
    parser.add_argument('--att', action='store_true', default=True)
    parser.add_argument('--att_name', type=str, default='fuse_base')
    parser.add_argument('--fed_alg', type=str, default='fed_avg')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--sample_rate', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--missing_modality', type=bool, default=False)
    parser.add_argument('--missing_label', type=bool, default=False)
    parser.add_argument('--label_nosiy', type=bool, default=False)
    parser.add_argument('--missing_modailty_rate', type=float, default=0.5)
    parser.add_argument('--missing_label_rate', type=float, default=0.5)
    parser.add_argument('--label_nosiy_level', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--attack_output_dir', type=str, default=None)
    parser.add_argument('--attack_group_spec_path', type=str, default=str(DEFAULT_PAIR_GROUP_SPEC))
    parser.add_argument('--validate_groups_only', action='store_true', default=False)
    parser.add_argument('--save_predictions', action='store_true', default=True)
    parser.add_argument('--save_heatmaps', action='store_true', default=True)
    parser.add_argument('--modes', type=str, nargs='*', default=['clean', 'paired_real_label_shift'], choices=['clean', 'paired_real_label_shift', 'delayed_paired_real_label_shift'])
    parser.add_argument('--attack_start_round', type=int, default=80)
    parser.add_argument('--clean_reference_dir', type=str, default=None)
    parser.add_argument('--skip_plots', action='store_true', default=False)
    parser.add_argument('--skip_per_class_analysis', action='store_true', default=False)
    return parser.parse_args()


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def plot_curves(metrics, save_path: Path):
    plt.figure(figsize=(8, 5))
    for mode, values in metrics.items():
        display = MODE_DISPLAY_NAMES.get(mode, {'en': mode})['en']
        plt.plot(values['rounds'], values['test_acc'], marker='o', label=display)
    plt.xlabel('Round')
    plt.ylabel('Global Test Accuracy')
    plt.title('Paired Real-Feature Label-Shift Evaluation Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _build_pair_attack_maps(group_spec):
    name_to_id = label_name_to_id()
    attacked_label_ids = []
    target_map = {}
    candidate_map = {}
    for group in group_spec['groups'].values():
        if not group.get('attack_enabled', False):
            continue
        source_name = group['attack_source']
        target_name = group['attack_target']
        source_id = int(name_to_id[source_name])
        target_id = int(name_to_id[target_name])
        attacked_label_ids.append(source_id)
        target_map[source_id] = target_id
        candidate_map[source_id] = [target_id]
    return sorted(attacked_label_ids), target_map, candidate_map


def _build_fallback_cross_group_target_map(attacked_label_ids, label_to_group_id, group_to_label_ids):
    target_map = {}
    for label_id in attacked_label_ids:
        source_group = label_to_group_id[int(label_id)]
        for group_id, label_ids in group_to_label_ids.items():
            if group_id != source_group and label_ids:
                target_map[int(label_id)] = int(label_ids[0])
                break
    return target_map


def _build_poisoned_metrics(mode, detailed_result, attacked_label_ids, target_map, label_to_group_id, group_to_label_ids, class_names, clean_reference=None):
    truth = detailed_result['truth']
    pred = detailed_result['pred']
    main_task = compute_main_task_success_excluding_attacked(truth, pred, attacked_label_ids)
    attack_metrics = compute_attack_success_metrics(truth, pred, attacked_label_ids, target_map)
    group_metrics = compute_same_group_attack_success_metrics(truth, pred, attacked_label_ids, label_to_group_id, group_to_label_ids, class_names)
    display_name = MODE_DISPLAY_NAMES.get(mode, {'zh': mode, 'en': mode})
    poisoned_metrics = {
        'mode': mode,
        'mode_display_name_zh': display_name['zh'],
        'mode_display_name_en': display_name['en'],
        'main_task_success_excluding_attacked': main_task,
        **attack_metrics,
        **group_metrics,
    }
    if clean_reference is not None:
        clean_attack_metrics = compute_attack_success_metrics(clean_reference['truth'], clean_reference['pred'], attacked_label_ids, target_map)
        clean_group_metrics = compute_same_group_attack_success_metrics(clean_reference['truth'], clean_reference['pred'], attacked_label_ids, label_to_group_id, group_to_label_ids, class_names)
        poisoned_metrics['attack_success_wrong_rate_increase_vs_clean'] = float(attack_metrics['attack_success_wrong_rate'] - clean_attack_metrics['attack_success_wrong_rate'])
        poisoned_metrics['attack_success_target_hit_rate_increase_vs_clean'] = float(attack_metrics['attack_success_target_hit_rate'] - clean_attack_metrics['attack_success_target_hit_rate'])
        poisoned_metrics['same_group_other_hit_rate_increase_vs_clean'] = float(group_metrics['same_group_other_hit_rate'] - clean_group_metrics['same_group_other_hit_rate'])
        poisoned_metrics['impact_on_non_attacked_classes'] = compute_non_attacked_impact(clean_reference['main_task_success_excluding_attacked'], main_task)
    else:
        poisoned_metrics['attack_success_wrong_rate_increase_vs_clean'] = 0.0
        poisoned_metrics['attack_success_target_hit_rate_increase_vs_clean'] = 0.0
        poisoned_metrics['same_group_other_hit_rate_increase_vs_clean'] = 0.0
        poisoned_metrics['impact_on_non_attacked_classes'] = 0.0
    return poisoned_metrics


def _save_mode_outputs(mode, output_dir, detailed_result, poisoned_metrics, class_names, attacked_label_ids, target_map, cross_group_target_map, group_spec, group_to_label_ids, label_to_group_id, client_sampling, save_predictions):
    mode_dir = output_dir / 'modes' / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    confusion_outputs = build_confusion_outputs(detailed_result['truth'], detailed_result['pred'], len(class_names))
    group_confusions = build_group_confusion_outputs(detailed_result['truth'], detailed_result['pred'], group_spec, group_to_label_ids, class_names)
    save_json(confusion_outputs['counts'], mode_dir / 'confusion_counts.json')
    save_json(confusion_outputs['row_normalized'], mode_dir / 'confusion_row_normalized.json')
    save_json(poisoned_metrics, mode_dir / 'poisoned_metrics.json')
    save_json(group_confusions, mode_dir / 'group_confusions.json')
    save_json({
        'per_group': poisoned_metrics.get('per_group', {}),
        'per_attacked_label': poisoned_metrics.get('per_attacked_label', {}),
        'same_group_other_hit_rate': poisoned_metrics.get('same_group_other_hit_rate', 0.0),
        'same_group_other_hit_rate_increase_vs_clean': poisoned_metrics.get('same_group_other_hit_rate_increase_vs_clean', 0.0),
    }, mode_dir / 'group_metrics.json')
    save_json(client_sampling, mode_dir / 'client_sampling.json')
    if save_predictions:
        save_prediction_table(
            detailed_result['truth'],
            detailed_result['pred'],
            class_names,
            attacked_label_ids,
            target_map,
            cross_group_target_map,
            mode_dir / 'test_predictions.json',
            label_to_group_id=label_to_group_id,
            mode_display_name=MODE_DISPLAY_NAMES.get(mode, {'zh': mode, 'en': mode}),
        )
    return {'full': confusion_outputs, 'groups': group_confusions}


def _load_clean_reference(clean_reference_dir: Path):
    summary_path = clean_reference_dir / 'poisoned_eval_summary.json'
    mode_predictions_path = clean_reference_dir / 'modes' / 'clean' / 'test_predictions.json'
    if not summary_path.exists() or not mode_predictions_path.exists():
        raise FileNotFoundError(f'Clean reference not found in {clean_reference_dir}')
    with open(summary_path, 'r', encoding='utf-8') as handle:
        summary = json.load(handle)
    with open(mode_predictions_path, 'r', encoding='utf-8') as handle:
        predictions = json.load(handle)
    return {
        'truth': [int(row['truth_id']) for row in predictions],
        'pred': [int(row['pred_id']) for row in predictions],
        'main_task_success_excluding_attacked': float(summary['clean']['main_task_success_excluding_attacked']),
    }


def _save_heatmaps(output_dir, confusion_by_mode, class_names, modes):
    heatmap_dir = output_dir / 'heatmaps'
    if 'clean' in confusion_by_mode:
        plot_confusion_heatmap(confusion_by_mode['clean']['full']['row_normalized'], class_names, 'Clean baseline confusion matrix (row normalized, %)', heatmap_dir / 'confusion_clean_normalized.png', cmap='magma', vmin=0.0, vmax=100.0)
        plot_group_confusion_heatmaps(confusion_by_mode['clean']['groups'], 'Clean baseline', heatmap_dir / 'groups' / 'clean', cmap='magma')
    for mode in modes:
        if mode == 'clean':
            continue
        display = MODE_DISPLAY_NAMES.get(mode, {'en': mode})['en']
        plot_confusion_heatmap(confusion_by_mode[mode]['full']['row_normalized'], class_names, f'{display} confusion matrix (row normalized, %)', heatmap_dir / f'confusion_{mode}_normalized.png', cmap='magma', vmin=0.0, vmax=100.0)
        plot_group_confusion_heatmaps(confusion_by_mode[mode]['groups'], display, heatmap_dir / 'groups' / mode, cmap='magma')
        if 'clean' in confusion_by_mode:
            plot_confusion_difference_heatmap(confusion_by_mode[mode]['full']['row_normalized'], confusion_by_mode['clean']['full']['row_normalized'], class_names, f'{display} minus clean confusion matrix (row normalized, %)', heatmap_dir / f'confusion_{mode}_minus_clean.png')
            plot_group_confusion_difference_heatmaps(confusion_by_mode[mode]['groups'], confusion_by_mode['clean']['groups'], f'{display} minus clean', heatmap_dir / 'groups' / f'{mode}_minus_clean')


def _choose_malicious_clients(client_ids, args):
    if int(args.malicious_clients) > len(client_ids):
        raise ValueError(f'--malicious_clients={args.malicious_clients} exceeds total clients={len(client_ids)}')
    rng = np.random.default_rng(int(args.seed) + 7919)
    return sorted([client_ids[idx] for idx in rng.choice(len(client_ids), size=int(args.malicious_clients), replace=False)])


def run_mode(mode, args, attacked_label_ids, target_map, candidate_map, class_names):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    demo_cfg = resolve_demo_config(data_dir=args.data_dir, output_dir=args.output_dir, train_batch_size=args.batch_size)
    args.data_dir = str(demo_cfg.output_dir)
    criterion = nn.NLLLoss().to(device)
    loader = create_loader(args.fold_idx, data_dir=demo_cfg.data_dir, output_dir=demo_cfg.output_dir, train_batch_size=args.batch_size)
    client_ids = loader.client_ids()
    if int(args.clients_per_round) > len(client_ids):
        raise ValueError(f'--clients_per_round={args.clients_per_round} exceeds total clients={len(client_ids)}')
    dataloader_dict = {client_id: loader.build_dataloader(client_id, shuffle=True) for client_id in client_ids}
    dataloader_dict['test'] = loader.build_dataloader('test', shuffle=False)

    set_seed(args.seed)
    global_model = MMActionClassifier(
        num_classes=constants.num_class_dict['ucf101'],
        audio_input_dim=constants.feature_len_dict['mfcc'],
        video_input_dim=constants.feature_len_dict['mobilenet_v2'],
        d_hid=args.hid_size,
        en_att=args.att,
        att_name=args.att_name,
    ).to(device)

    server = Server(args, global_model, device=device, criterion=criterion, client_ids=client_ids)
    server.initialize_log(args.fold_idx)
    server.get_num_params()

    malicious_client_ids = [] if mode == 'clean' else _choose_malicious_clients(client_ids, args)
    malicious_client_set = set(malicious_client_ids)
    delayed_mode = mode == 'delayed_paired_real_label_shift'
    attack_start_round = int(getattr(args, 'attack_start_round', 1))
    client_sampling = {
        'mode': mode,
        'mode_display_name_zh': MODE_DISPLAY_NAMES.get(mode, {'zh': mode})['zh'],
        'mode_display_name_en': MODE_DISPLAY_NAMES.get(mode, {'en': mode})['en'],
        'num_total_clients': len(client_ids),
        'clients_per_round': int(args.clients_per_round),
        'num_malicious_clients': len(malicious_client_ids),
        'malicious_client_ids': malicious_client_ids,
        'attack_start_round': attack_start_round if delayed_mode else 1,
        'rounds': [],
    }

    rounds, test_acc = [], []
    final_result = None
    for epoch in range(int(args.num_epochs)):
        server.initialize_epoch_updates(epoch)
        rng = np.random.default_rng(args.seed + epoch)
        selected_clients = [client_ids[idx] for idx in rng.choice(len(client_ids), size=int(args.clients_per_round), replace=False)]
        attack_active = mode == 'paired_real_label_shift' or (delayed_mode and epoch + 1 >= attack_start_round)
        malicious_selected = [client_id for client_id in selected_clients if attack_active and client_id in malicious_client_set]
        client_sampling['rounds'].append({'round': epoch + 1, 'attack_active': bool(attack_active), 'selected_client_ids': selected_clients, 'malicious_selected_client_ids': malicious_selected})

        for client_id in selected_clients:
            if attack_active and client_id in malicious_client_set:
                client = ClientRealSameGroupLabelShift(args, device, criterion, dataloader_dict[client_id], copy.deepcopy(server.global_model), attack_label_ids=attacked_label_ids, same_group_target_map=target_map, same_group_candidate_map=candidate_map, client_id=client_id, round_idx=epoch)
            else:
                client = ClientFedAvg(args, device, criterion, dataloader_dict[client_id], model=copy.deepcopy(server.global_model), label_dict=None, num_class=51)
            client.update_weights()
            server.save_train_updates(copy.deepcopy(client.get_parameters()), client.result['sample'], client.result)
            del client

        if len(server.num_samples_list) == 0:
            continue
        server.average_weights()
        if (epoch + 1) % args.eval_interval == 0:
            with torch.no_grad():
                server.inference(dataloader_dict['test'])
                rounds.append(epoch + 1)
                test_acc.append(float(server.result.get('acc', 0.0)))
                final_result = dict(server.result)
                logging.info('Mode=%s Round=%d TestAcc=%.2f', mode, epoch + 1, test_acc[-1])

    if final_result is None:
        with torch.no_grad():
            server.inference(dataloader_dict['test'])
            final_result = dict(server.result)

    detailed_result = server.eval.classification_detailed_summary(monitor_labels=attacked_label_ids)
    final_result.update({
        'truth': detailed_result['truth'],
        'pred': detailed_result['pred'],
        'confusion_count': detailed_result['confusion_count'],
        'confusion_row_normalized': detailed_result['confusion_row_normalized'],
        'label_support': detailed_result['label_support'],
        'class_names': class_names,
    })
    return {'rounds': rounds, 'test_acc': test_acc, 'final_result': final_result, 'client_sampling': client_sampling}


def _resolve_output_dir(cfg, args):
    root = Path(args.attack_output_dir) if args.attack_output_dir else (cfg.attack_output_dir or DEFAULT_ATTACK_OUTPUT_ROOT)
    return root / cfg.exp_name


def main():
    args = parse_args()
    cfg = resolve_robustness_config(
        exp_name=args.exp_name,
        fold_idx=args.fold_idx,
        num_epochs=args.num_epochs,
        eval_interval=args.eval_interval,
        clients_per_round=args.clients_per_round,
        malicious_clients=args.malicious_clients,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        global_learning_rate=args.global_learning_rate,
        local_epochs=args.local_epochs,
        hid_size=args.hid_size,
        att=args.att,
        att_name=args.att_name,
        fed_alg=args.fed_alg,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        attack_output_dir=args.attack_output_dir,
        attack_group_spec_path=args.attack_group_spec_path,
    )
    output_dir = _resolve_output_dir(cfg, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted_label_names()
    group_spec = load_attack_group_spec(cfg.attack_group_spec_path)
    group_to_label_ids = build_group_to_label_ids(group_spec, attack_enabled_only=False)
    label_to_group_id = build_label_to_attack_group(group_spec)
    attacked_label_ids, target_map, candidate_map = _build_pair_attack_maps(group_spec)
    cross_group_target_map = _build_fallback_cross_group_target_map(attacked_label_ids, label_to_group_id, group_to_label_ids)
    attacked_report = label_report(
        attacked_label_ids=attacked_label_ids,
        same_group_target_map=target_map,
        cross_group_target_map=cross_group_target_map,
        spec_version='hungarian_fused_pair_v1',
        group_spec=group_spec,
        same_group_candidate_map=candidate_map,
    )
    save_json(attacked_report, output_dir / 'attacked_label_report.json')
    if args.validate_groups_only:
        print(f'Validated paired attack group spec: {len(group_spec["groups"])} groups, {len(attacked_label_ids)} attacked labels')
        return

    metrics = {}
    detailed_outputs = {}
    clean_reference = _load_clean_reference(Path(args.clean_reference_dir)) if args.clean_reference_dir else None
    comparison_rows = []
    confusion_by_mode = {}
    selected_modes = list(dict.fromkeys(args.modes))

    for mode in selected_modes:
        mode_result = run_mode(mode, args, attacked_label_ids, target_map, candidate_map, class_names)
        metrics[mode] = {
            'mode_display_name_zh': MODE_DISPLAY_NAMES.get(mode, {'zh': mode})['zh'],
            'mode_display_name_en': MODE_DISPLAY_NAMES.get(mode, {'en': mode})['en'],
            'rounds': mode_result['rounds'],
            'test_acc': mode_result['test_acc'],
        }
        detailed_result = mode_result['final_result']
        poisoned_metrics = _build_poisoned_metrics(mode, detailed_result, attacked_label_ids, target_map, label_to_group_id, group_to_label_ids, class_names, clean_reference=clean_reference)
        poisoned_metrics['test_acc'] = float(detailed_result.get('acc', 0.0))
        poisoned_metrics['monitored_label_acc'] = detailed_result.get('monitored_label_acc', {})
        confusion_outputs = _save_mode_outputs(mode, output_dir, detailed_result, poisoned_metrics, class_names, attacked_label_ids, target_map, cross_group_target_map, group_spec, group_to_label_ids, label_to_group_id, mode_result['client_sampling'], save_predictions=args.save_predictions)
        confusion_by_mode[mode] = confusion_outputs
        detailed_outputs[mode] = poisoned_metrics
        comparison_rows.append(poisoned_metrics)
        if mode == 'clean':
            clean_reference = {
                'truth': detailed_result['truth'],
                'pred': detailed_result['pred'],
                'main_task_success_excluding_attacked': poisoned_metrics['main_task_success_excluding_attacked'],
            }

    save_json(metrics, output_dir / 'metrics.json')
    save_json(detailed_outputs, output_dir / 'poisoned_eval_summary.json')
    save_comparison_table(comparison_rows, output_dir / 'comparison_table.csv')
    if not args.skip_plots:
        plot_curves(metrics, output_dir / 'accuracy_curves.png')
    if args.save_heatmaps:
        _save_heatmaps(output_dir, confusion_by_mode, class_names, selected_modes)
    attack_modes_for_analysis = [mode for mode in selected_modes if mode != 'clean']
    if not args.skip_per_class_analysis and 'clean' in selected_modes and attack_modes_for_analysis and args.save_predictions:
        per_class_results = {}
        for attack_mode in attack_modes_for_analysis:
            output_subdir = 'per_class_analysis' if len(attack_modes_for_analysis) == 1 else f'per_class_analysis_{attack_mode}'
            per_class_results[attack_mode] = analyze_experiment_per_class_changes(output_dir, attack_mode=attack_mode, output_subdir=output_subdir)
        save_json(per_class_results if len(per_class_results) > 1 else next(iter(per_class_results.values())), output_dir / 'per_class_analysis.json')
    print(f'Saved paired real-label-shift metrics to {output_dir / "metrics.json"}')


if __name__ == '__main__':
    main()
