import argparse
import copy
import csv
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

from .analyze_per_class_attack_changes import analyze_experiment_per_class_changes
from .config import resolve_robustness_config
from .gan_source import generate_fake_multimodal_batch, load_demo_gan
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
from .real_label_shift_client import ClientRealSameGroupLabelShift
from .visual_groups import label_id_to_name, sorted_label_names

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

MODE_DISPLAY_NAMES = {
    'clean': {'zh': '干净基线', 'en': 'Clean baseline'},
    'adaptive_gan_selected_real_label_shift': {
        'zh': 'GAN辅助自适应真实特征标签攻击',
        'en': 'GAN-assisted adaptive real-feature label-shift attack',
    },
    'adaptive_local_selected_real_label_shift': {
        'zh': '本地数据自适应真实特征标签攻击',
        'en': 'Local-data-selected adaptive real-feature label-shift attack',
    },
}
DEFAULT_ATTACK_OUTPUT_ROOT = Path('/home/xp/fed-multimodal/fed_multimodal/demo/results/ucf101/attack')
DEFAULT_GAN_CHECKPOINT = '/home/xp/fed-multimodal/fed_multimodal/results/demo/ucf101/gan/checkpoints/ckpt_200_0309BASE_STRONGT.pt'
DEFAULT_TEACHER_CHECKPOINT = '/home/xp/fed-multimodal/fed_multimodal/results/demo/ucf101/training/fold1_fed_avg_sr02_ep200_lr005_glr001_hid128_best_model.pt'


def parse_args():
    parser = argparse.ArgumentParser(description='GAN-assisted adaptive delayed label-shift experiment for demo federated learning')
    parser.add_argument('--exp_name', type=str, default='ucf101_adaptive_gan_selected_label_shift_100r')
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
    parser.add_argument('--selection_round', type=int, default=80)
    parser.add_argument('--gan_samples_per_class', type=int, default=64)
    parser.add_argument('--min_local_support', type=int, default=8)
    parser.add_argument('--gan_checkpoint', type=str, default=DEFAULT_GAN_CHECKPOINT)
    parser.add_argument('--teacher_checkpoint', type=str, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument('--save_predictions', action='store_true', default=True)
    parser.add_argument('--save_heatmaps', action='store_true', default=True)
    parser.add_argument('--modes', type=str, nargs='*', default=['clean', 'adaptive_gan_selected_real_label_shift'], choices=['clean', 'adaptive_gan_selected_real_label_shift', 'adaptive_local_selected_real_label_shift'])
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


def _l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _tensor_embeddings(model, x_a, x_b, l_a, l_b):
    model.eval()
    logits, embeddings = model(x_a, x_b, l_a, l_b)
    return embeddings.detach().cpu().numpy(), torch.argmax(logits, dim=1).detach().cpu().numpy()


def _choose_malicious_clients(client_ids, args):
    if int(args.malicious_clients) > len(client_ids):
        raise ValueError(f'--malicious_clients={args.malicious_clients} exceeds total clients={len(client_ids)}')
    rng = np.random.default_rng(int(args.seed) + 7919)
    return sorted([client_ids[idx] for idx in rng.choice(len(client_ids), size=int(args.malicious_clients), replace=False)])


def _collect_local_centroids(model, dataloader_dict, malicious_client_ids, device, min_support):
    values = {}
    predictions = {}
    with torch.no_grad():
        for client_id in malicious_client_ids:
            for x_a, x_b, l_a, l_b, y in dataloader_dict[client_id]:
                x_a = x_a.float().to(device)
                x_b = x_b.float().to(device)
                l_a = l_a.to(device)
                l_b = l_b.to(device)
                embeddings, preds = _tensor_embeddings(model, x_a, x_b, l_a, l_b)
                for idx, label in enumerate(y.tolist()):
                    label = int(label)
                    values.setdefault(label, []).append(embeddings[idx])
                    predictions.setdefault(label, []).append(int(preds[idx]))
    centroids = {}
    support = {}
    pred_counts = {}
    for label, rows in values.items():
        support[str(label)] = int(len(rows))
        if len(rows) >= min_support:
            centroids[int(label)] = np.stack(rows, axis=0).mean(axis=0)
        counts = {}
        for pred in predictions.get(label, []):
            counts[str(pred)] = counts.get(str(pred), 0) + 1
        pred_counts[str(label)] = counts
    return centroids, support, pred_counts


def _collect_gan_centroids(model, gan, sample_batch, samples_per_class, class_count, device):
    real_a, real_v, len_a, len_v = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3]
    len_a_value = int(real_a.shape[1])
    len_v_value = int(real_v.shape[1])
    centroids = {}
    with torch.no_grad():
        for class_id in range(class_count):
            labels = torch.full((samples_per_class,), int(class_id), device=device, dtype=torch.long)
            fake_len_a = torch.full((samples_per_class,), len_a_value, device=device, dtype=len_a.dtype)
            fake_len_v = torch.full((samples_per_class,), len_v_value, device=device, dtype=len_v.dtype)
            fake_a, fake_v = generate_fake_multimodal_batch(gan, labels, fake_len_a, fake_len_v, device)
            embeddings, _ = _tensor_embeddings(model, fake_a.float(), fake_v.float(), fake_len_a, fake_len_v)
            centroids[int(class_id)] = embeddings.mean(axis=0)
    return centroids


def _select_adaptive_pair(model, gan, sample_batch, dataloader_dict, malicious_client_ids, class_names, args, device, target_source='gan'):
    local_centroids, local_support, local_pred_counts = _collect_local_centroids(
        model,
        dataloader_dict,
        malicious_client_ids,
        device,
        min_support=int(args.min_local_support),
    )
    if target_source == 'gan':
        target_centroids = _collect_gan_centroids(
            model,
            gan,
            sample_batch,
            samples_per_class=int(args.gan_samples_per_class),
            class_count=len(class_names),
            device=device,
        )
    elif target_source == 'local':
        target_centroids = local_centroids
    else:
        raise ValueError(f'Unknown target_source={target_source}')
    local_matrix = _l2_normalize(np.stack([local_centroids[label] for label in sorted(local_centroids)], axis=0))
    local_labels = sorted(local_centroids)
    local_norm = {label: local_matrix[idx] for idx, label in enumerate(local_labels)}
    target_matrix = _l2_normalize(np.stack([target_centroids[label] for label in sorted(target_centroids)], axis=0))
    target_labels = sorted(target_centroids)
    target_norm = {label: target_matrix[idx] for idx, label in enumerate(target_labels)}
    candidates = []
    for source_id, source_vec in local_norm.items():
        for target_id, target_vec in target_norm.items():
            if int(source_id) == int(target_id):
                continue
            cosine_similarity = float(np.dot(source_vec, target_vec))
            cosine_distance = float(1.0 - cosine_similarity)
            source_pred_counts = local_pred_counts.get(str(source_id), {})
            confusion_count = int(source_pred_counts.get(str(target_id), 0))
            confusion_rate = float(confusion_count) * 100.0 / float(max(1, int(local_support[str(source_id)])))
            score = cosine_distance - 0.01 * confusion_rate
            candidates.append({
                'source_id': int(source_id),
                'source_label': class_names[int(source_id)],
                'target_id': int(target_id),
                'target_label': class_names[int(target_id)],
                'target_source': target_source,
                'cosine_similarity': cosine_similarity,
                'cosine_distance': cosine_distance,
                'local_support': int(local_support[str(source_id)]),
                'target_local_support': int(local_support.get(str(target_id), 0)) if target_source == 'local' else '',
                'local_clean_confusion_count': confusion_count,
                'local_clean_confusion_rate': confusion_rate,
                'selection_score': float(score),
            })
    candidates.sort(key=lambda row: row['selection_score'])
    if not candidates:
        raise ValueError('No adaptive source-target candidates were available')
    return candidates[0], candidates[:50], local_support


def _build_dynamic_group_spec(source_id, target_id, class_names, selection_record):
    groups = {
        f'adaptive_pair_{int(source_id):02d}_to_{int(target_id):02d}': {
            'display_name_zh': 'GAN辅助自适应攻击配对',
            'display_name_en': 'GAN-assisted adaptive attack pair',
            'attack_enabled': True,
            'attack_source': class_names[int(source_id)],
            'attack_target': class_names[int(target_id)],
            'labels': [class_names[int(source_id)], class_names[int(target_id)]],
            'selection': selection_record,
        }
    }
    for label_id, label_name in enumerate(class_names):
        if label_id in {int(source_id), int(target_id)}:
            continue
        groups[f'clean_singleton_{label_name}'] = {
            'display_name_zh': '未选中类别',
            'display_name_en': 'Unselected class',
            'attack_enabled': False,
            'labels': [label_name],
        }
    return {
        'version': 'adaptive_gan_selected_pair_v1',
        'dataset': 'ucf101_demo',
        'description': 'One source-target pair selected at the analysis round by cosine distance between malicious-client local real embeddings and GAN synthetic target embeddings under the current global model.',
        'groups': groups,
    }


def _label_group_maps(group_spec, class_names):
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    group_to_label_ids = {}
    label_to_group_id = {}
    for group_id, group in group_spec['groups'].items():
        ids = [int(name_to_id[name]) for name in group.get('labels', [])]
        group_to_label_ids[group_id] = ids
        for label_id in ids:
            label_to_group_id[int(label_id)] = group_id
    return group_to_label_ids, label_to_group_id


def _fallback_cross_group_target_map(attacked_label_ids, class_names):
    return {int(label_id): int((int(label_id) + 1) % len(class_names)) for label_id in attacked_label_ids}


def _build_poisoned_metrics(mode, detailed_result, attacked_label_ids, target_map, label_to_group_id, group_to_label_ids, class_names, clean_reference=None):
    truth = detailed_result['truth']
    pred = detailed_result['pred']
    main_task = compute_main_task_success_excluding_attacked(truth, pred, attacked_label_ids)
    attack_metrics = compute_attack_success_metrics(truth, pred, attacked_label_ids, target_map)
    group_metrics = compute_same_group_attack_success_metrics(truth, pred, attacked_label_ids, label_to_group_id, group_to_label_ids, class_names)
    poisoned_metrics = {
        'mode': mode,
        'mode_display_name_zh': MODE_DISPLAY_NAMES.get(mode, {'zh': mode})['zh'],
        'mode_display_name_en': MODE_DISPLAY_NAMES.get(mode, {'en': mode})['en'],
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


def plot_curves(metrics, save_path):
    plt.figure(figsize=(8, 5))
    for mode, values in metrics.items():
        display = MODE_DISPLAY_NAMES.get(mode, {'en': mode})['en']
        plt.plot(values['rounds'], values['test_acc'], marker='o', label=display)
    plt.xlabel('Round')
    plt.ylabel('Global Test Accuracy')
    plt.title('GAN-assisted Adaptive Label-Shift Evaluation Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _save_candidate_csv(candidates, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['source_id', 'source_label', 'target_id', 'target_label', 'target_source', 'cosine_similarity', 'cosine_distance', 'local_support', 'target_local_support', 'local_clean_confusion_count', 'local_clean_confusion_rate', 'selection_score']
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


def run_mode(mode, args, class_names, output_dir):
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
    sample_batch = next(iter(dataloader_dict[client_ids[0]]))
    sample_batch = [value.to(device) if torch.is_tensor(value) else value for value in sample_batch]
    gan = None
    if mode == 'adaptive_gan_selected_real_label_shift':
        gan = load_demo_gan(Path(args.gan_checkpoint), Path(args.teacher_checkpoint), sample_batch, device)

    selected_pair = None
    top_candidates = []
    local_support = {}
    attacked_label_ids = []
    target_map = {}
    candidate_map = {}
    client_sampling = {
        'mode': mode,
        'mode_display_name_zh': MODE_DISPLAY_NAMES.get(mode, {'zh': mode})['zh'],
        'mode_display_name_en': MODE_DISPLAY_NAMES.get(mode, {'en': mode})['en'],
        'num_total_clients': len(client_ids),
        'clients_per_round': int(args.clients_per_round),
        'num_malicious_clients': len(malicious_client_ids),
        'malicious_client_ids': malicious_client_ids,
        'selection_round': int(args.selection_round) if mode != 'clean' else None,
        'attack_start_round': int(args.selection_round) + 1 if mode != 'clean' else None,
        'rounds': [],
    }

    rounds, test_acc = [], []
    final_result = None
    for epoch in range(int(args.num_epochs)):
        round_idx = epoch + 1
        server.initialize_epoch_updates(epoch)
        rng = np.random.default_rng(args.seed + epoch)
        selected_clients = [client_ids[idx] for idx in rng.choice(len(client_ids), size=int(args.clients_per_round), replace=False)]
        attack_active = mode != 'clean' and selected_pair is not None and round_idx > int(args.selection_round)
        malicious_selected = [client_id for client_id in selected_clients if attack_active and client_id in malicious_client_set]
        client_sampling['rounds'].append({'round': round_idx, 'attack_active': bool(attack_active), 'selected_client_ids': selected_clients, 'malicious_selected_client_ids': malicious_selected})

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
        if mode != 'clean' and round_idx == int(args.selection_round):
            target_source = 'gan' if mode == 'adaptive_gan_selected_real_label_shift' else 'local'
            selected_pair, top_candidates, local_support = _select_adaptive_pair(server.global_model, gan, sample_batch, dataloader_dict, malicious_client_ids, class_names, args, device, target_source=target_source)
            attacked_label_ids = [int(selected_pair['source_id'])]
            target_map = {int(selected_pair['source_id']): int(selected_pair['target_id'])}
            candidate_map = {int(selected_pair['source_id']): [int(selected_pair['target_id'])]}
            client_sampling['selected_pair'] = selected_pair
            logging.info('Adaptive pair selected at round %d: %s -> %s source=%s distance=%.4f confusion=%.2f', round_idx, selected_pair['source_label'], selected_pair['target_label'], target_source, selected_pair['cosine_distance'], selected_pair['local_clean_confusion_rate'])
        if round_idx % args.eval_interval == 0:
            with torch.no_grad():
                server.inference(dataloader_dict['test'])
                rounds.append(round_idx)
                test_acc.append(float(server.result.get('acc', 0.0)))
                final_result = dict(server.result)
                logging.info('Mode=%s Round=%d TestAcc=%.2f', mode, round_idx, test_acc[-1])

    if final_result is None:
        with torch.no_grad():
            server.inference(dataloader_dict['test'])
            final_result = dict(server.result)
    monitor_labels = attacked_label_ids if attacked_label_ids else list(range(len(class_names)))
    detailed_result = server.eval.classification_detailed_summary(monitor_labels=monitor_labels)
    final_result.update({
        'truth': detailed_result['truth'],
        'pred': detailed_result['pred'],
        'confusion_count': detailed_result['confusion_count'],
        'confusion_row_normalized': detailed_result['confusion_row_normalized'],
        'label_support': detailed_result['label_support'],
        'class_names': class_names,
    })
    if mode != 'clean':
        save_json({'selected_pair': selected_pair, 'top_candidates': top_candidates, 'local_support': local_support}, output_dir / 'adaptive_pair_selection.json')
        _save_candidate_csv(top_candidates, output_dir / 'adaptive_pair_candidates_top50.csv')
    return {
        'rounds': rounds,
        'test_acc': test_acc,
        'final_result': final_result,
        'client_sampling': client_sampling,
        'selected_pair': selected_pair,
    }


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
    )
    output_dir = _resolve_output_dir(cfg, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = sorted_label_names()
    selected_modes = list(dict.fromkeys(args.modes))

    mode_results = {}
    selected_pair = None
    for mode in selected_modes:
        mode_results[mode] = run_mode(mode, args, class_names, output_dir)
        if mode_results[mode]['selected_pair'] is not None:
            selected_pair = mode_results[mode]['selected_pair']
    if selected_pair is None:
        raise ValueError('No adaptive pair was selected; include an adaptive attack mode in --modes')

    source_id = int(selected_pair['source_id'])
    target_id = int(selected_pair['target_id'])
    attacked_label_ids = [source_id]
    target_map = {source_id: target_id}
    candidate_map = {source_id: [target_id]}
    group_spec = _build_dynamic_group_spec(source_id, target_id, class_names, selected_pair)
    group_to_label_ids, label_to_group_id = _label_group_maps(group_spec, class_names)
    cross_group_target_map = _fallback_cross_group_target_map(attacked_label_ids, class_names)
    save_json(group_spec, output_dir / 'adaptive_attack_group_spec.json')
    save_json({
        'attacked_label_ids': attacked_label_ids,
        'target_map': target_map,
        'candidate_map': candidate_map,
        'source_label': class_names[source_id],
        'target_label': class_names[target_id],
        'selection_round': int(args.selection_round),
        'attack_start_round': int(args.selection_round) + 1,
    }, output_dir / 'attacked_label_report.json')

    metrics = {}
    detailed_outputs = {}
    comparison_rows = []
    confusion_by_mode = {}
    clean_reference = None
    if args.clean_reference_dir:
        predictions_path = Path(args.clean_reference_dir) / 'modes' / 'clean' / 'test_predictions.json'
        with open(predictions_path, 'r', encoding='utf-8') as handle:
            predictions = json.load(handle)
        clean_reference = {'truth': [int(row['truth_id']) for row in predictions], 'pred': [int(row['pred_id']) for row in predictions], 'main_task_success_excluding_attacked': 0.0}

    for mode in selected_modes:
        mode_result = mode_results[mode]
        detailed_result = mode_result['final_result']
        poisoned_metrics = _build_poisoned_metrics(mode, detailed_result, attacked_label_ids, target_map, label_to_group_id, group_to_label_ids, class_names, clean_reference=clean_reference)
        poisoned_metrics['test_acc'] = float(detailed_result.get('acc', 0.0))
        poisoned_metrics['monitored_label_acc'] = detailed_result.get('monitored_label_acc', {})
        confusion_outputs = _save_mode_outputs(mode, output_dir, detailed_result, poisoned_metrics, class_names, attacked_label_ids, target_map, cross_group_target_map, group_spec, group_to_label_ids, label_to_group_id, mode_result['client_sampling'], save_predictions=args.save_predictions)
        metrics[mode] = {'mode_display_name_zh': MODE_DISPLAY_NAMES.get(mode, {'zh': mode})['zh'], 'mode_display_name_en': MODE_DISPLAY_NAMES.get(mode, {'en': mode})['en'], 'rounds': mode_result['rounds'], 'test_acc': mode_result['test_acc']}
        detailed_outputs[mode] = poisoned_metrics
        comparison_rows.append(poisoned_metrics)
        confusion_by_mode[mode] = confusion_outputs
        if mode == 'clean':
            clean_reference = {'truth': detailed_result['truth'], 'pred': detailed_result['pred'], 'main_task_success_excluding_attacked': poisoned_metrics['main_task_success_excluding_attacked']}

    if 'clean' in selected_modes:
        detailed_outputs = {}
        comparison_rows = []
        confusion_by_mode = {}
        for mode in selected_modes:
            mode_result = mode_results[mode]
            detailed_result = mode_result['final_result']
            poisoned_metrics = _build_poisoned_metrics(mode, detailed_result, attacked_label_ids, target_map, label_to_group_id, group_to_label_ids, class_names, clean_reference=clean_reference if mode != 'clean' else None)
            poisoned_metrics['test_acc'] = float(detailed_result.get('acc', 0.0))
            poisoned_metrics['monitored_label_acc'] = detailed_result.get('monitored_label_acc', {})
            confusion_outputs = _save_mode_outputs(mode, output_dir, detailed_result, poisoned_metrics, class_names, attacked_label_ids, target_map, cross_group_target_map, group_spec, group_to_label_ids, label_to_group_id, mode_result['client_sampling'], save_predictions=args.save_predictions)
            detailed_outputs[mode] = poisoned_metrics
            comparison_rows.append(poisoned_metrics)
            confusion_by_mode[mode] = confusion_outputs

    save_json(metrics, output_dir / 'metrics.json')
    save_json(detailed_outputs, output_dir / 'poisoned_eval_summary.json')
    save_comparison_table(comparison_rows, output_dir / 'comparison_table.csv')
    if not args.skip_plots:
        plot_curves(metrics, output_dir / 'accuracy_curves.png')
    if args.save_heatmaps:
        _save_heatmaps(output_dir, confusion_by_mode, class_names, selected_modes)
    if not args.skip_per_class_analysis and 'clean' in selected_modes and args.save_predictions:
        attack_modes_for_analysis = [mode for mode in selected_modes if mode != 'clean']
        per_class_results = {}
        for attack_mode in attack_modes_for_analysis:
            output_subdir = 'per_class_analysis' if len(attack_modes_for_analysis) == 1 else f'per_class_analysis_{attack_mode}'
            per_class_results[attack_mode] = analyze_experiment_per_class_changes(output_dir, attack_mode=attack_mode, output_subdir=output_subdir)
        if per_class_results:
            save_json(per_class_results if len(per_class_results) > 1 else next(iter(per_class_results.values())), output_dir / 'per_class_analysis.json')
    print(f'Saved adaptive GAN-selected label-shift metrics to {output_dir / "metrics.json"}')


if __name__ == '__main__':
    main()
