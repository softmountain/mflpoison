"""P2: 用收敛全局模型 M* 提取 penultimate 特征，计算真实类中心与 source 合成类中心。

真实类中心: 合并 10 个 client 的 train pkl，按真实 label 分组求 x_mm 均值（51 个）。
合成类中心: 对给定 source 类，从 GAN 合成数据按 condition_label==s 取样本，过 M* 求 x_mm 均值。

输出 real_centers.pt / synth_centers.pt，供 P3(target 选择) 与 P5(相似度分析) 使用。
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from fed_multimodal.constants import constants
from fed_multimodal.model.mm_models import MMActionClassifier


def load_m_star(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt["args"]
    model = MMActionClassifier(
        num_classes=constants.num_class_dict[args.get("dataset", "ucf101")],
        audio_input_dim=constants.feature_len_dict["mfcc"],
        video_input_dim=constants.feature_len_dict["mobilenet_v2"],
        d_hid=args["hid_size"],
        en_att=args.get("att", False),
        att_name=args.get("att_name", "base"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, args


def iter_real_batches(audio_items, video_items, batch_size, device):
    """yield (x_audio, x_video, len_a, len_v, labels) padded batches from client pkl items."""
    n = len(audio_items)
    for i in range(0, n, batch_size):
        a_batch = audio_items[i:i + batch_size]
        v_batch = video_items[i:i + batch_size]
        B = len(a_batch)
        max_a = max(f.shape[0] for *_, f in a_batch)
        max_v = max(f.shape[0] for *_, f in v_batch)
        a = torch.zeros(B, max_a, constants.feature_len_dict["mfcc"])
        v = torch.zeros(B, max_v, constants.feature_len_dict["mobilenet_v2"])
        la = torch.zeros(B, dtype=torch.long)
        lv = torch.zeros(B, dtype=torch.long)
        y = torch.zeros(B, dtype=torch.long)
        for j, (it_a, it_v) in enumerate(zip(a_batch, v_batch)):
            ta = np.asarray(it_a[3]); tv = np.asarray(it_v[3])
            a[j, :ta.shape[0]] = torch.as_tensor(ta); la[j] = ta.shape[0]
            v[j, :tv.shape[0]] = torch.as_tensor(tv); lv[j] = tv.shape[0]
            y[j] = it_a[2]
        yield a.to(device), v.to(device), la.to(device), lv.to(device), y.to(device)


@torch.no_grad()
def compute_real_centers(model, audio_items, video_items, num_classes, device, batch_size=32):
    model.eval()
    sums = {}
    counts = {}
    feat_dim = None
    for a, v, la, lv, y in iter_real_batches(audio_items, video_items, batch_size, device):
        _, x_mm = model(a, v, la, lv)  # [B, feat_dim]
        if feat_dim is None:
            feat_dim = x_mm.shape[1]
        for c in range(num_classes):
            mask = (y == c)
            if mask.any():
                s = x_mm[mask].sum(dim=0)
                sums[c] = sums[c] + s if c in sums else s
                counts[c] = counts.get(c, 0) + int(mask.sum().item())
    centers = {}
    for c in sums:
        centers[int(c)] = (sums[c] / counts[c]).cpu()
    return centers, feat_dim


@torch.no_grad()
def compute_synth_centers(model, synth, source_classes, device, batch_size=32):
    """synth: dict with audio(N,500,80), video(N,9,1280), condition_label(N,)."""
    model.eval()
    audio = synth["audio"].float()
    video = synth["video"].float()
    cond = synth.get("condition_label", synth.get("labels", synth.get("train_label")))
    if cond is None:
        raise ValueError("synth data missing condition_label/labels/train_label")
    cond = cond.long() if torch.is_tensor(cond) else torch.as_tensor(cond).long()
    centers = {}
    for s in source_classes:
        mask = (cond == s)
        if mask.sum() == 0:
            print(f"  source {s}: no synthetic samples (condition_label), skip")
            continue
        a_s = audio[mask]; v_s = video[mask]
        N = a_s.shape[0]
        acc = None
        for i in range(0, N, batch_size):
            a = a_s[i:i + batch_size].to(device)
            v = v_s[i:i + batch_size].to(device)
            la = torch.full((a.shape[0],), a.shape[1], dtype=torch.long, device=device)
            lv = torch.full((v.shape[0],), v.shape[1], dtype=torch.long, device=device)
            _, x_mm = model(a, v, la, lv)
            acc = x_mm.sum(dim=0) if acc is None else acc + x_mm.sum(dim=0)
        centers[int(s)] = (acc / N).cpu()
        print(f"  source {s}: {N} synthetic samples, center norm={centers[int(s)].norm().item():.3f}")
    return centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_star_path", required=True)
    parser.add_argument("--client_pkl_dir_root", required=True,
                        help="results root containing feature/{audio,video}/{feat}/ucf101/alpha{a}/fold{f}/")
    parser.add_argument("--synth_data", required=True, help="GAN synthetic .pt (final50)")
    parser.add_argument("--source_classes", required=True, help="comma-separated source class ids, e.g. 0,5,12")
    parser.add_argument("--audio_feat", default="mfcc")
    parser.add_argument("--video_feat", default="mobilenet_v2")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading M* from {args.m_star_path}")
    model, m_args = load_m_star(Path(args.m_star_path))
    model = model.to(device)
    num_classes = constants.num_class_dict[m_args.get("dataset", "ucf101")]
    print(f"  num_classes={num_classes}, hid={m_args['hid_size']}, att={m_args.get('att')}, att_name={m_args.get('att_name')}")

    # merge 10 client train pkls
    alpha_str = str(args.alpha).replace(".", "")
    audio_dir = Path(args.client_pkl_dir_root) / "feature" / "audio" / args.audio_feat / "ucf101" / f"alpha{alpha_str}" / f"fold{args.fold}"
    video_dir = Path(args.client_pkl_dir_root) / "feature" / "video" / args.video_feat / "ucf101" / f"alpha{alpha_str}" / f"fold{args.fold}"
    audio_items, video_items = [], []
    for cid in range(args.num_clients):
        audio_items += pickle.load(open(audio_dir / f"{cid}.pkl", "rb"))
        video_items += pickle.load(open(video_dir / f"{cid}.pkl", "rb"))
    print(f"  merged {len(audio_items)} real train samples from {args.num_clients} clients")

    print("Computing real class centers...")
    real_centers, feat_dim = compute_real_centers(model, audio_items, video_items, num_classes, device)
    print(f"  real centers: {len(real_centers)} classes, feat_dim={feat_dim}")

    print("Loading synthetic data...")
    synth = torch.load(args.synth_data, map_location="cpu")
    source_classes = [int(x) for x in args.source_classes.split(",")]
    print(f"Computing synthetic centers for source classes {source_classes}...")
    synth_centers = compute_synth_centers(model, synth, source_classes, device)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"centers": real_centers, "feat_dim": feat_dim}, out / "real_centers.pt")
    torch.save({"centers": synth_centers, "feat_dim": feat_dim, "source_classes": source_classes}, out / "synth_centers.pt")
    print(f"Saved real_centers.pt ({len(real_centers)} classes) and synth_centers.pt ({len(synth_centers)} sources) to {out}")


if __name__ == "__main__":
    main()
