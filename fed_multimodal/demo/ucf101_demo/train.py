import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fed_multimodal.constants import constants
from fed_multimodal.model.mm_models import MMActionClassifier
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from fed_multimodal.trainers.fed_rs_trainer import ClientFedRS
from fed_multimodal.trainers.scaffold_trainer import ClientScaffold
from fed_multimodal.trainers.server_trainer import Server

from .config import resolve_config
from .loader import create_loader

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_seed(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Demo UCF101 federated training')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--audio_feat', type=str, default='mfcc')
    parser.add_argument('--video_feat', type=str, default='mobilenet_v2')
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--global_learning_rate', type=float, default=0.05)
    parser.add_argument('--att', type=bool, default=False)
    parser.add_argument('--en_att', dest='att', action='store_true')
    parser.add_argument('--att_name', type=str, default='base')
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--mu', type=float, default=0.001)
    parser.add_argument('--sample_rate', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--fed_alg', type=str, default='fed_avg')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_clients', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--folds', type=int, nargs='*', default=[1, 2, 3])
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--missing_modality', type=bool, default=False)
    parser.add_argument('--missing_label', type=bool, default=False)
    parser.add_argument('--label_nosiy', type=bool, default=False)
    parser.add_argument('--missing_modailty_rate', type=float, default=0.5)
    parser.add_argument('--missing_label_rate', type=float, default=0.5)
    parser.add_argument('--label_nosiy_level', type=float, default=0.1)
    parser.add_argument('--modality', type=str, default='multimodal')
    parser.add_argument('--dataset', type=str, default='ucf101')
    return parser.parse_args()


def _select_client_class(fed_alg: str):
    if fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        return ClientFedAvg
    if fed_alg == 'scaffold':
        return ClientScaffold
    if fed_alg == 'fed_rs':
        return ClientFedRS
    raise ValueError(f'Unsupported fed_alg: {fed_alg}')


def _checkpoint_tag(args) -> str:
    sr = str(args.sample_rate).replace('.', '')
    lr = str(args.learning_rate).replace('.', '')
    glr = str(args.global_learning_rate).replace('.', '')
    return f'{args.fed_alg}_sr{sr}_ep{args.num_epochs}_lr{lr}_glr{glr}_hid{args.hid_size}'


def _save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, args, fold_idx: int, metric_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'fold_idx': fold_idx,
        'best_test_acc': metric_value,
    }, path)


def main():
    args = parse_args()
    demo_config = resolve_config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        train_batch_size=args.batch_size,
        audio_feature_type=args.audio_feat,
        video_feature_type=args.video_feat,
    )
    args.data_dir = str(demo_config.output_dir)

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print('GPU available, use GPU')

    criterion = nn.NLLLoss().to(device)
    Client = _select_client_class(args.fed_alg)
    save_result_dict = {}
    training_dir = demo_config.demo_root / 'training'
    training_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = _checkpoint_tag(args)

    for fold_idx in args.folds:
        loader = create_loader(
            fold_idx,
            data_dir=demo_config.data_dir,
            output_dir=demo_config.output_dir,
            num_clients=demo_config.num_clients,
            train_batch_size=demo_config.train_batch_size,
            audio_feature_type=demo_config.audio_feature_type,
            video_feature_type=demo_config.video_feature_type,
        )
        client_ids = loader.client_ids()
        dataloader_dict = {client_id: loader.build_dataloader(client_id, shuffle=True) for client_id in client_ids}
        dataloader_dict['test'] = loader.build_dataloader('test', shuffle=False)

        set_seed(args.seed + fold_idx)
        global_model = MMActionClassifier(
            num_classes=constants.num_class_dict[args.dataset],
            audio_input_dim=constants.feature_len_dict[args.audio_feat],
            video_input_dim=constants.feature_len_dict[args.video_feat],
            d_hid=args.hid_size,
            en_att=args.att,
            att_name=args.att_name,
        ).to(device)

        server = Server(
            args,
            global_model,
            device=device,
            criterion=criterion,
            client_ids=client_ids,
        )
        server.initialize_log(fold_idx)
        server.sample_clients(len(client_ids), sample_rate=args.sample_rate)
        server.get_num_params()
        best_test_acc = float('-inf')

        for epoch in range(int(args.num_epochs)):
            server.initialize_epoch_updates(epoch)
            skip_client_ids = []
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
                    label_dict=None,
                    num_class=constants.num_class_dict[args.dataset],
                )
                if args.fed_alg == 'scaffold':
                    client.set_control(
                        server_control=copy.deepcopy(server.server_control),
                        client_control=copy.deepcopy(server.client_controls[client_id]),
                    )
                    client.update_weights()
                    server.set_client_control(client_id, copy.deepcopy(client.client_control))
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()),
                        client.result['sample'],
                        client.result,
                        delta_control=copy.deepcopy(client.delta_control),
                    )
                else:
                    client.update_weights()
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()),
                        client.result['sample'],
                        client.result,
                    )
                del client

            logging.info(f'Fold {fold_idx} Epoch {epoch}, skip client {skip_client_ids}')
            if len(server.num_samples_list) == 0:
                continue
            server.average_weights()
            server.log_classification_result(data_split='train', metric='acc')
            with torch.no_grad():
                server.inference(dataloader_dict['test'])
                server.result_dict[epoch]['test'] = server.result
                server.result_dict[epoch]['dev'] = copy.deepcopy(server.result)
                if epoch == 0:
                    server.best_dev_dict = copy.deepcopy(server.result)
                current_test_acc = float(server.result.get('acc', 0.0))
                if current_test_acc > best_test_acc:
                    best_test_acc = current_test_acc
                    _save_checkpoint(
                        training_dir / f'fold{fold_idx}_{ckpt_tag}_best_model.pt',
                        epoch=epoch + 1,
                        model=server.global_model,
                        args=args,
                        fold_idx=fold_idx,
                        metric_value=current_test_acc,
                    )
                server.log_classification_result(data_split='test', metric='acc')
            server.log_epoch_result(metric='acc')

        _save_checkpoint(
            training_dir / f'fold{fold_idx}_{ckpt_tag}_final_model.pt',
            epoch=args.num_epochs,
            model=server.global_model,
            args=args,
            fold_idx=fold_idx,
            metric_value=best_test_acc if best_test_acc != float('-inf') else 0.0,
        )
        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()

    serializable_result = {}
    for key, value in save_result_dict.items():
        if isinstance(value, dict):
            serializable_result[key] = {
                metric: float(metric_value) if isinstance(metric_value, (np.floating, np.integer)) else metric_value
                for metric, metric_value in value.items()
            }
        else:
            serializable_result[key] = value

    serializable_result['average'] = {}
    for metric in ['uar', 'acc', 'top5_acc']:
        result_list = [serializable_result[key][metric] for key in serializable_result if key.startswith('fold') and metric in serializable_result[key]]
        serializable_result['average'][metric] = float(np.nanmean(result_list)) if result_list else None

    result_path = training_dir / 'result.json'
    with open(result_path, 'w', encoding='utf-8') as handle:
        json.dump(serializable_result, handle, indent=2)
    print(f'Saved training summary to {result_path}')


if __name__ == '__main__':
    main()
