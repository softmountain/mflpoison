"""P7: 计算 fragility score，用于"脆弱 target"选择。

fragility(c) = z(H(c)) + z(gap(c))，两项 z-score 归一化后等权相加。
  H(c)    = 类 c【正确预测】样本的平均 softmax 熵 (nat)。
            高 = 模型表面分对但不自信 = "纸老虎"，决策边界薄，最易被打穿。
  gap(c)  = M* test recall(c) − TSTR recall(c)。
            高 = GAN 对该类复现能力弱 = 注入的 source 合成特征偏离 target 真实表征 = 破坏力强。
高 fragility = 高 recall + 高熵 + 低 TSTR = 最脆弱 target。

输入: --m_star_path (新 M*), --tstr_json (per-class TSTR recall dict 0-100), --alpha, --data_dir
输出: fragility_per_class.json (每类 recall/H/gap/fragility + top10/bottom5 排序)

复用 train.py 的 parse_args (alpha/data_dir/hid/att 等) + train_label_flip 的 load_m_star 模式。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from fed_multimodal.constants import constants
from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.model.mm_models import MMActionClassifier


def _load_train_module():
    train_path = Path(__file__).with_name('train.py')
    spec = importlib.util.spec_from_file_location('ucf101_train_base_fragility', train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_TRAIN = _load_train_module()


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--m_star_path', type=str, required=True)
    parser.add_argument('--tstr_json', type=str, required=True,
                        help='per-class TSTR recall json (dict class_str -> recall 0-100)')
    parser.add_argument('--output', type=str, required=True)
    frag_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return frag_args


def load_m_star(path, device):
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
    acc = ckpt.get('best_test_acc', ckpt.get('final_test_acc'))
    return model.to(device), m_args, acc


def eval_collect_logprobs(model, test_loader, device):
    """Independent eval loop mirroring server_trainer.inference (multimodal branch),
    but retains per-sample log_softmax for entropy computation."""
    model.eval()
    all_lp, all_y = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            x_a, x_b, l_a, l_b, y = batch_data
            x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
            l_a, l_b = l_a.to(device), l_b.to(device)
            outputs, _ = model(x_a.float(), x_b.float(), l_a, l_b)
            lp = torch.log_softmax(outputs, dim=1)
            all_lp.append(lp.cpu())
            all_y.append(y.cpu())
    return torch.cat(all_lp), torch.cat(all_y)


def zscore(x):
    x = np.asarray(x, dtype=float)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / s if s > 0 else np.zeros_like(x)


def main():
    frag_args = parse_args()
    args = BASE_TRAIN.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model, m_args, m_acc = load_m_star(Path(frag_args.m_star_path), device)
    print(f'Loaded M*: acc={m_acc}, hid={m_args["hid_size"]}, att={m_args.get("att_name")}')

    # load test (alpha50 partition of new M*)
    dm = DataloadManager(args)
    dm.get_simulation_setting(alpha=args.alpha)
    dm.load_sim_dict(fold_idx=1)
    dm.get_client_ids(fold_idx=1)
    audio_dict = dm.load_audio_feat(client_id='test', fold_idx=1)
    video_dict = dm.load_video_feat(client_id='test', fold_idx=1)
    test_loader = dm.set_dataloader(
        audio_dict, video_dict, client_sim_dict=None,
        default_feat_shape_a=np.array([500, constants.feature_len_dict['mfcc']]),
        default_feat_shape_b=np.array([10, constants.feature_len_dict['mobilenet_v2']]),
        shuffle=False,
    )

    num_classes = constants.num_class_dict[args.dataset]
    log_probs, labels = eval_collect_logprobs(model, test_loader, device)
    preds = log_probs.argmax(dim=1)
    entropies = -(log_probs.exp() * log_probs).sum(dim=1)  # nat per sample

    recall = np.full(num_classes, np.nan)
    H = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            recall[c] = float((preds[mask] == c).float().mean().item()) * 100
            correct = mask & (preds == c)
            if correct.sum() > 0:
                H[c] = float(entropies[correct].mean().item())

    tstr_raw = json.load(open(frag_args.tstr_json))
    tstr = {int(k): float(v) for k, v in tstr_raw.items()}
    gap = np.array([recall[c] - tstr.get(c, float('nan')) for c in range(num_classes)])

    zH, zgap = zscore(H), zscore(gap)
    fragility = zH + zgap
    # also store raw components for reporting
    zH_corr = np.where(np.isnan(zH), -1e9, zH)
    zgap_corr = np.where(np.isnan(zgap), -1e9, zgap)

    per_class = {}
    for c in range(num_classes):
        per_class[str(c)] = {
            'recall': (None if np.isnan(recall[c]) else float(recall[c])),
            'H_entropy': (None if np.isnan(H[c]) else float(H[c])),
            'tstr': (None if c not in tstr else float(tstr[c])),
            'gap': (None if np.isnan(gap[c]) else float(gap[c])),
            'fragility': float(fragility[c]),
        }

    ranked = sorted(range(num_classes), key=lambda c: (zH_corr[c] + zgap_corr[c]), reverse=True)
    out = {
        'm_star_path': frag_args.m_star_path,
        'm_star_acc': m_acc,
        'num_classes': num_classes,
        'per_class': per_class,
        'ranked_by_fragility': ranked,
        'top10_fragility': ranked[:10],
        'bottom5_fragility': ranked[-5:][::-1],  # most negative first
    }

    Path(frag_args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(frag_args.output, 'w'), indent=2)

    print(f'\n=== fragility ranking (top 12) ===')
    print(f"{'cls':>4} {'frag':>7} {'recall':>7} {'H':>7} {'tstr':>7} {'gap':>7}")
    for c in ranked[:12]:
        pc = per_class[str(c)]
        r = pc['recall']; h = pc['H_entropy']; t = pc['tstr']; g = pc['gap']
        print(f"{c:>4} {pc['fragility']:>+7.2f} {r if r is not None else float('nan'):>7.1f} "
              f"{h if h is not None else float('nan'):>7.3f} {t if t is not None else float('nan'):>7.1f} "
              f"{g if g is not None else float('nan'):>+7.1f}")
    print(f'\nbottom5 (least fragile): {ranked[-5:][::-1]}')
    print(f'Saved to {frag_args.output}')


if __name__ == '__main__':
    main()
