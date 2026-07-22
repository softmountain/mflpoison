"""P4: UCF101 合成数据特征碰撞攻击（中途注入）。

攻击语义（用户确认）: 恶意客户端的 target 类真实样本，feature 被替换为 source 类 GAN 合成特征，
label 保持 target。即"内容像 source、标签是 target"的特征碰撞，让全局模型把 source 特征映射到 target。

两阶段: 从 P1 训练好的收敛全局模型 M* (load_state_dict) 出发，注入攻击数据后继续训练 K 轮。
用 test 混淆矩阵 conf 读取 target/source 的 per-class recall，度量攻击效果。
结果写到独立子目录 result/{fed_alg}_poison/.../source{s}_target{t}_m{method}/result.json。
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from fed_multimodal.constants import constants
from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.model.mm_models import MMActionClassifier
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from fed_multimodal.trainers.server_trainer import Server

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _load_base_train_module():
    train_path = Path(__file__).with_name('train.py')
    spec = importlib.util.spec_from_file_location('ucf101_train_base', train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_TRAIN = _load_base_train_module()


def parse_attack_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--m_star_path', type=str, required=True,
                        help='converged global model checkpoint from P1 (dict with model_state_dict+args)')
    parser.add_argument('--synth_data', type=str, default=None,
                        help='GAN synthetic .pt (final50); not needed for the no-attack baseline')
    parser.add_argument('--attack_source_class', type=int, required=True, help='source class id (content provider)')
    parser.add_argument('--attack_target_class', type=int, required=True, help='target class id (label kept, feature replaced)')
    parser.add_argument('--attack_epochs', type=int, default=30, help='K: attack training rounds after injecting')
    parser.add_argument('--malicious_clients', type=str, default='0,1,2,3',
                        help='comma-separated malicious client ids (40%% of 10 = 4)')
    parser.add_argument('--result_subdir', type=str, required=True, help='e.g. source5_target12_m1')
    parser.add_argument('--eval_every', type=int, default=5, help='inference test every N attack rounds (0=only final)')
    parser.add_argument('--attack_n_inject', type=int, default=None,
                        help='fix number of target samples poisoned per malicious client (None=all target samples)')
    parser.add_argument('--seed', type=int, default=8, help='seed for set_seed (baseline x3 uses different seeds)')
    attack_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return attack_args


def load_m_star(path: Path, device):
    ckpt = torch.load(path, map_location='cpu')
    m_args = ckpt['args']
    model = MMActionClassifier(
        num_classes=constants.num_class_dict[m_args.get('dataset', 'ucf101')],
        audio_input_dim=constants.feature_len_dict['mfcc'],
        video_input_dim=constants.feature_len_dict['mobilenet_v2'],
        d_hid=m_args['hid_size'],
        en_att=m_args.get('att', False),
        att_name=m_args.get('att_name', 'base'),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device), m_args, ckpt.get('final_test_acc')


def inject_feature_collision(audio_dict, video_dict, target_class, source_class,
                             synth_audio, synth_video, synth_cond, n_inject=None, rng=None):
    """In-place: replace target-class entries' feature with source-class synthetic features, keep target label.

    If n_inject is set and smaller than the number of target-class entries on this client, only n_inject
    of them are poisoned (sampled without replacement from tgt_idx) — this fixes the inject count across
    targets so cross-target comparison isn't confounded by differing poison load.
    """
    src_mask = (synth_cond == source_class)
    src_a = synth_audio[src_mask]
    src_v = synth_video[src_mask]
    n_synth = src_a.shape[0]
    if n_synth == 0:
        return 0
    tgt_idx = [i for i, it in enumerate(audio_dict) if it[2] == target_class]
    if n_inject is not None and n_inject < len(tgt_idx):
        if rng is None:
            rng = np.random.default_rng()
        tgt_idx = [int(i) for i in rng.choice(tgt_idx, size=n_inject, replace=False)]
    for j, idx in enumerate(tgt_idx):
        k = j % n_synth
        audio_dict[idx] = [audio_dict[idx][0], audio_dict[idx][1], int(target_class), src_a[k].cpu().numpy()]
        video_dict[idx] = [video_dict[idx][0], video_dict[idx][1], int(target_class), src_v[k].cpu().numpy()]
    return len(tgt_idx)


def recall_from_conf(conf, cls):
    """conf is row-normalized confusion matrix (numpy 2D, percent). conf[i][i] = recall of class i."""
    conf = np.asarray(conf)
    return float(conf[cls][cls])


def cell_from_conf(conf, true_cls, pred_cls):
    """conf[true][pred]: percent of true-class samples predicted as pred_cls (row-normalized)."""
    conf = np.asarray(conf)
    return float(conf[true_cls][pred_cls])


def per_class_recall_from_conf(conf, num_classes):
    """Return list of per-class recall (diagonal of row-normalized conf) for all classes.

    Used by the no-attack baseline to measure NATURAL drift of every class over the
    extra training rounds, so attack Δt_recall can be corrected for that drift.
    """
    conf = np.asarray(conf)
    out = []
    for c in range(num_classes):
        if c < conf.shape[0] and c < conf.shape[1]:
            out.append(float(conf[c][c]))
        else:
            out.append(float('nan'))
    return out


def eval_per_class_entropy(model, test_loader, device, num_classes):
    """Run an extra forward pass over test and return per-class mean softmax entropy (nat),
    averaged over CORRECTLY predicted samples of each class (high = model right but unsure).
    Used to track target-class entropy change across attack rounds (early fragility signal).
    """
    model.eval()
    all_lp, all_y = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            x_a, x_b, l_a, l_b, y = batch_data
            x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
            l_a, l_b = l_a.to(device), l_b.to(device)
            outputs, _ = model(x_a.float(), x_b.float(), l_a, l_b)
            all_lp.append(torch.log_softmax(outputs, dim=1).cpu())
            all_y.append(y.cpu())
    log_probs = torch.cat(all_lp)
    labels = torch.cat(all_y)
    preds = log_probs.argmax(dim=1)
    entropies = -(log_probs.exp() * log_probs).sum(dim=1)
    H = [float('nan')] * num_classes
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            correct = mask & (preds == c)
            if correct.sum() > 0:
                H[c] = float(entropies[correct].mean().item())
    return H


def main():
    attack_args = parse_attack_args()
    args = BASE_TRAIN.parse_args()
    # attack training uses exactly K rounds
    args.num_epochs = attack_args.attack_epochs
    args.num_folds = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(
        f'Attack: source={attack_args.attack_source_class} -> target={attack_args.attack_target_class}, '
        f'K={attack_args.attack_epochs} rounds, malicious={attack_args.malicious_clients}'
    )

    # load synthetic data only when there are malicious clients; the no-attack baseline
    # (malicious_clients="") never injects, so it skips the synth load entirely.
    malicious_ids = set(s.strip() for s in attack_args.malicious_clients.split(',') if s.strip())
    synth_audio = synth_video = synth_cond = None
    if malicious_ids:
        if attack_args.synth_data is None:
            raise SystemExit('--synth_data is required when --malicious_clients is non-empty')
        synth = torch.load(attack_args.synth_data, map_location='cpu')
        synth_audio = synth['audio'].float()
        synth_video = synth['video'].float()
        cond_key = 'condition_label' if 'condition_label' in synth else ('labels' if 'labels' in synth else 'train_label')
        synth_cond = synth[cond_key].long() if torch.is_tensor(synth[cond_key]) else torch.as_tensor(synth[cond_key]).long()
        logging.info(f'Loaded synth data from {attack_args.synth_data}')
    else:
        logging.info('No malicious clients — BASELINE (no-attack) run; synth data not loaded')

    # data loading with injection
    dm = DataloadManager(args)
    dm.get_simulation_setting(alpha=args.alpha)
    dm.load_sim_dict(fold_idx=1)
    dm.get_client_ids(fold_idx=1)
    dataloader_dict = dict()
    inject_log = dict()
    inject_rng = np.random.default_rng(8)  # deterministic per-client sampling for fixed-N inject
    logging.info('Loading client data with feature-collision injection on malicious clients %s', malicious_ids)
    for client_id in tqdm(dm.client_ids):
        audio_dict = dm.load_audio_feat(client_id=client_id, fold_idx=1)
        video_dict = dm.load_video_feat(client_id=client_id, fold_idx=1)
        if client_id in malicious_ids:
            n = inject_feature_collision(
                audio_dict, video_dict, attack_args.attack_target_class, attack_args.attack_source_class,
                synth_audio, synth_video, synth_cond,
                n_inject=attack_args.attack_n_inject, rng=inject_rng,
            )
            inject_log[client_id] = n
            logging.info(f'  client {client_id}: replaced {n} target-class samples with source synthetic features')
        dm.get_label_dist(video_dict, client_id)
        shuffle = False if client_id in ['dev', 'test'] else True
        client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
        dataloader_dict[client_id] = dm.set_dataloader(
            audio_dict, video_dict,
            client_sim_dict=client_sim_dict,
            default_feat_shape_a=np.array([500, constants.feature_len_dict['mfcc']]),
            default_feat_shape_b=np.array([10, constants.feature_len_dict['mobilenet_v2']]),
            shuffle=shuffle,
        )

    # load converged global model M*
    global_model, m_args, m_star_acc = load_m_star(Path(attack_args.m_star_path), device)
    # override model-related args with M*'s config to guarantee dimensional consistency
    args.hid_size = m_args['hid_size']
    args.att = m_args.get('att', False)
    args.att_name = m_args.get('att_name', 'base')
    logging.info(f'Loaded M* (hid={m_args["hid_size"]}, att={m_args.get("att")}, att_name={m_args.get("att_name")}, '
                 f'recorded final_test_acc={m_star_acc})')

    client_ids = [c for c in dm.client_ids if c not in ['dev', 'test']]
    num_of_clients = len(client_ids)
    BASE_TRAIN.set_seed(attack_args.seed)
    criterion = nn.NLLLoss().to(device)
    server = Server(args, global_model, device=device, criterion=criterion, client_ids=client_ids)
    server.initialize_log(1)
    server.sample_clients(num_of_clients, sample_rate=args.sample_rate)
    server.get_num_params()

    # baseline (attack-before) evaluation on test
    num_classes = constants.num_class_dict[args.dataset]
    with torch.no_grad():
        server.inference(dataloader_dict['test'])
        base_result = server.result
    acc_base = base_result['acc']
    target_recall_base = recall_from_conf(base_result['conf'], attack_args.attack_target_class)
    source_recall_base = recall_from_conf(base_result['conf'], attack_args.attack_source_class)
    s2t_base = cell_from_conf(base_result['conf'], attack_args.attack_source_class, attack_args.attack_target_class)
    base_per_class = per_class_recall_from_conf(base_result['conf'], num_classes)
    base_per_class_entropy = eval_per_class_entropy(server.global_model, dataloader_dict['test'], device, num_classes)
    # eval_history records per-class recall AND entropy at every eval point so the no-attack baseline
    # (malicious_clients="") can be subtracted from each attack run to attribute true attack damage,
    # and the target-class entropy change can be tracked as an early fragility signal.
    eval_history = {
        'base': {'acc': float(acc_base), 'per_class_recall': base_per_class,
                 'per_class_entropy': base_per_class_entropy},
    }
    logging.info(f'[baseline M*] acc={acc_base:.2f} target_recall={target_recall_base:.2f} source_recall={source_recall_base:.2f}')

    # attack training (K rounds)
    final_test = None
    for epoch in range(int(args.num_epochs)):
        server.initialize_epoch_updates(epoch)
        skip = list()
        for idx in server.clients_list[epoch]:
            client_id = client_ids[idx]
            dataloader = dataloader_dict[client_id]
            if dataloader is None:
                skip.append(client_id)
                continue
            client = ClientFedAvg(
                args, device, criterion, dataloader,
                model=copy.deepcopy(server.global_model),
                label_dict=dm.label_dist_dict[client_id],
                num_class=constants.num_class_dict[args.dataset],
            )
            client.update_weights()
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()), client.result['sample'], client.result
            )
            del client
        logging.info(f'Attack Round {epoch}: skip {skip}')
        if len(server.num_samples_list) == 0:
            continue
        server.average_weights()
        do_eval = (attack_args.eval_every > 0 and (epoch + 1) % attack_args.eval_every == 0) or (epoch == int(args.num_epochs) - 1)
        if do_eval:
            with torch.no_grad():
                server.inference(dataloader_dict['test'])
                server.result_dict[epoch]['test'] = server.result
                r = server.result
                eval_history[f'round{epoch}'] = {
                    'acc': float(r['acc']),
                    'per_class_recall': per_class_recall_from_conf(r['conf'], num_classes),
                    'per_class_entropy': eval_per_class_entropy(server.global_model, dataloader_dict['test'], device, num_classes),
                }
                rt_rcl = recall_from_conf(r['conf'], attack_args.attack_target_class)
                rs_rcl = recall_from_conf(r['conf'], attack_args.attack_source_class)
                logging.info(f'  [round {epoch}] acc={r["acc"]:.2f} target_recall={rt_rcl:.2f} source_recall={rs_rcl:.2f}')
                final_test = r

    if final_test is None:
        logging.error('No attack rounds completed; aborting save.')
        return

    acc_after = final_test['acc']
    target_recall_after = recall_from_conf(final_test['conf'], attack_args.attack_target_class)
    source_recall_after = recall_from_conf(final_test['conf'], attack_args.attack_source_class)
    s2t_after = cell_from_conf(final_test['conf'], attack_args.attack_source_class, attack_args.attack_target_class)

    result = {
        'source_class': attack_args.attack_source_class,
        'target_class': attack_args.attack_target_class,
        'malicious_clients': sorted(list(malicious_ids)),
        'inject_counts': inject_log,
        'attack_epochs': attack_args.attack_epochs,
        'sample_rate': args.sample_rate,
        'baseline': {'acc': acc_base, 'target_recall': target_recall_base, 'source_recall': source_recall_base, 'source_to_target': s2t_base},
        'after': {'acc': acc_after, 'target_recall': target_recall_after, 'source_recall': source_recall_after, 'source_to_target': s2t_after},
        'delta': {
            'acc': acc_after - acc_base,
            'target_recall': target_recall_after - target_recall_base,
            'source_recall': source_recall_after - source_recall_base,
            'source_to_target': s2t_after - s2t_base,
        },
        # eval_history carries per-class recall at every eval point so a no-attack run
        # (is_baseline=True) can be subtracted from attack runs to attribute true damage.
        'is_baseline': len(malicious_ids) == 0,
        'eval_history': eval_history,
    }

    save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
        'result', f'{args.fed_alg}_poison', args.dataset, server.feature, server.att,
        server.model_setting_str, attack_args.result_subdir,
    )
    save_json_path.mkdir(parents=True, exist_ok=True)
    out_file = save_json_path / 'result.json'
    json.dump(result, open(out_file, 'w'), indent=2)
    logging.info(f'Saved result to {out_file}')
    logging.info(
        f'SUMMARY source={attack_args.attack_source_class} target={attack_args.attack_target_class} | '
        f'acc {acc_base:.2f}->{acc_after:.2f} (Δ{acc_after - acc_base:+.2f}) | '
        f'target_recall {target_recall_base:.2f}->{target_recall_after:.2f} (Δ{target_recall_after - target_recall_base:+.2f}) | '
        f'source_recall {source_recall_base:.2f}->{source_recall_after:.2f} (Δ{source_recall_after - source_recall_base:+.2f})'
    )


if __name__ == '__main__':
    main()
