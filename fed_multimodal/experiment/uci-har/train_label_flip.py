import argparse
import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import logging
import torch.multiprocessing
import copy
import time
import pickle
import shutil
import sys
import os
import pdb
import importlib.util

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.constants import constants
from fed_multimodal.trainers.server_trainer import Server
from fed_multimodal.model.mm_models import HARClassifier
from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.trainers.fed_rs_trainer import ClientFedRS
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from fed_multimodal.trainers.scaffold_trainer import ClientScaffold
from fed_multimodal.generator.label_flip_attack import UCILabelFlipAttack

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _load_base_train_module():
    train_path = Path(__file__).with_name('train.py')
    spec = importlib.util.spec_from_file_location('uci_har_train_base', train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_TRAIN = _load_base_train_module()


def _parse_suffix_list(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)):
        tokens = [str(v) for v in raw_value]
    else:
        tokens = str(raw_value).split(',')
    suffixes = [token.strip() for token in tokens if token.strip()]
    return suffixes if suffixes else None


def parse_attack_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--attack_prob',
        type=float,
        default=0.5,
        help='Flip probability applied to targeted training clients.'
    )
    parser.add_argument(
        '--attack_seed',
        type=int,
        default=42,
        help='Random seed used by the label flipping attack.'
    )
    parser.add_argument(
        '--attack_src_label',
        type=int,
        default=1,
        help='Original label id to flip (default: 1, WALKING_UPSTAIRS).'
    )
    parser.add_argument(
        '--attack_dst_label',
        type=int,
        default=2,
        help='Label id after flipping (default: 2, WALKING_DOWNSTAIRS).'
    )
    parser.add_argument(
        '--attack_client_suffix',
        type=str,
        default='-1',
        help='Clients whose id ends with any of these comma-separated suffixes are attacked.'
    )
    parser.add_argument(
        '--attack_dev_ratio',
        type=float,
        default=0.2,
        help='Fraction of dev samples (by count) to flip.'
    )
    raw_argv = sys.argv[1:]
    normalized_argv = list()
    skip_next = False
    for idx, token in enumerate(raw_argv):
        if skip_next:
            skip_next = False
            continue
        if token == '--attack_client_suffix' and idx + 1 < len(raw_argv):
            normalized_argv.append(f'--attack_client_suffix={raw_argv[idx + 1]}')
            skip_next = True
        else:
            normalized_argv.append(token)

    attack_args, remaining = parser.parse_known_args(args=normalized_argv)
    attack_args.attack_client_suffix = _parse_suffix_list(attack_args.attack_client_suffix)
    sys.argv = [sys.argv[0]] + remaining
    return attack_args


def maybe_build_attack(args, attack_cli_args):
    if args.dataset != 'uci-har':
        logging.warning('Label flipping attack is only defined for uci-har; attack disabled.')
        return None
    logging.info(
        'Enabling label flipping attack: %s -> %s (train prob=%.2f, dev ratio=%.2f) on suffixes: %s.',
        attack_cli_args.attack_src_label,
        attack_cli_args.attack_dst_label,
        attack_cli_args.attack_prob,
        attack_cli_args.attack_dev_ratio,
        attack_cli_args.attack_client_suffix or 'ALL',
    )
    return UCILabelFlipAttack(
        flip_prob=attack_cli_args.attack_prob,
        seed=attack_cli_args.attack_seed,
        src_label=attack_cli_args.attack_src_label,
        dst_label=attack_cli_args.attack_dst_label,
        target_client_suffix=attack_cli_args.attack_client_suffix,
        dev_attack_ratio=attack_cli_args.attack_dev_ratio,
    )


def apply_attack_if_needed(attack, client_id, acc_dict, gyro_dict):
    if attack is None:
        return acc_dict, gyro_dict
    return attack.apply(client_id, acc_dict, gyro_dict)


if __name__ == '__main__':

    attack_cli_args = parse_attack_args()
    args = BASE_TRAIN.parse_args()
    attack = maybe_build_attack(args, attack_cli_args)

    log_dir = Path(args.log_dir) if args.log_dir else Path(os.path.realpath(__file__)).parents[2].joinpath('result', 'logs')
    log_file = BASE_TRAIN.setup_file_logger(log_dir, log_prefix=f'{args.dataset}_{args.fed_alg}_label_flip')
    logging.info('Logging to %s', log_file)
    if args.monitor_labels:
        logging.info('Monitoring per-label accuracy for labels: %s', args.monitor_labels)

    dm = DataloadManager(args)
    dm.get_simulation_setting(alpha=args.alpha)

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print('GPU available, use GPU')
    save_result_dict = dict()

    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    elif args.fed_alg in ['fed_rs']:
        Client = ClientFedRS

    dm.load_sim_dict()
    dm.get_client_ids()
    dataloader_dict = dict()
    logging.info('Reading Data with label flipping attack%s', '' if attack is None else ' (enabled)')
    for client_id in tqdm(dm.client_ids):
        acc_dict = dm.load_acc_feat(client_id=client_id)
        gyro_dict = dm.load_gyro_feat(client_id=client_id)
        acc_dict, gyro_dict = apply_attack_if_needed(attack, client_id, acc_dict, gyro_dict)
        dm.get_label_dist(gyro_dict, client_id)
        shuffle = False if client_id in ['dev', 'test'] else True
        client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
        dataloader_dict[client_id] = dm.set_dataloader(
            acc_dict,
            gyro_dict,
            shuffle=shuffle,
            client_sim_dict=client_sim_dict,
            default_feat_shape_a=np.array([128, constants.feature_len_dict[args.acc_feat]]),
            default_feat_shape_b=np.array([128, constants.feature_len_dict[args.gyro_feat]]),
        )

    for fold_idx in range(1, 2):
        client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
        num_of_clients = len(client_ids)
        BASE_TRAIN.set_seed(8*fold_idx)
        criterion = nn.NLLLoss().to(device)
        global_model = HARClassifier(
            num_classes=constants.num_class_dict[args.dataset],
            acc_input_dim=constants.feature_len_dict[args.acc_feat],
            gyro_input_dim=constants.feature_len_dict[args.gyro_feat],
            en_att=args.att,
            d_hid=args.hid_size,
            att_name=args.att_name
        )
        global_model = global_model.to(device)

        server = Server(
            args,
            global_model,
            device=device,
            criterion=criterion,
            client_ids=client_ids
        )
        server.initialize_log(fold_idx)
        server.sample_clients(
            num_of_clients,
            sample_rate=args.sample_rate
        )
        server.get_num_params()

        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result',
            f'{args.fed_alg}_label_flip',
            args.dataset,
            server.feature,
            server.att,
            server.model_setting_str
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)

        server.save_json_file(
            dm.label_dist_dict,
            save_json_path.joinpath('label.json')
        )

        BASE_TRAIN.set_seed(8*fold_idx)
        for epoch in range(int(args.num_epochs)):
            server.initialize_epoch_updates(epoch)
            skip_client_ids = list()
            for idx in server.clients_list[epoch]:
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue

                client = Client(
                    args,
                    device,
                    criterion,
                    dataloader,
                    model=copy.deepcopy(server.global_model),
                    label_dict=dm.label_dist_dict[client_id],
                    num_class=constants.num_class_dict[args.dataset]
                )

                if args.fed_alg == 'scaffold':
                    client.set_control(
                        server_control=copy.deepcopy(server.server_control),
                        client_control=copy.deepcopy(server.client_controls[client_id])
                    )
                    client.update_weights()
                    server.set_client_control(client_id, copy.deepcopy(client.client_control))
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()),
                        client.result['sample'],
                        client.result,
                        delta_control=copy.deepcopy(client.delta_control)
                    )
                else:
                    client.update_weights()
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()),
                        client.result['sample'],
                        client.result
                    )
                del client

            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')

            if len(server.num_samples_list) == 0:
                continue
            server.average_weights()
            logging.info('---------------------------------------------------------')
            server.log_classification_result(
                data_split='train',
                metric='f1'
            )
            if (epoch+1) % args.test_frequency == 0:
                with torch.no_grad():
                    server.inference(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_classification_result(
                        data_split='dev',
                        metric='f1'
                    )

                    server.inference(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    server.log_classification_result(
                        data_split='test',
                        metric='f1'
                    )

                logging.info('---------------------------------------------------------')
                server.log_epoch_result(metric='f1')
            logging.info('---------------------------------------------------------')

        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()

        server.save_json_file(
            save_result_dict,
            save_json_path.joinpath('result.json')
        )

    # save_result_dict['average'] = dict()
    # for metric in ['f1', 'acc', 'top5_acc']:
    #     result_list = list()
    #     for key in save_result_dict:
    #         if metric not in save_result_dict[key]:
    #             continue
    #         result_list.append(save_result_dict[key][metric])
    #     if len(result_list) == 0:
    #         continue
    #     save_result_dict['average'][metric] = np.nanmean(result_list)

    # server.save_json_file(
    #     save_result_dict,
    #     save_json_path.joinpath('result.json')
    # )
