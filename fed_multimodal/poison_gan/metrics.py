import math
from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def classification_metrics(logits, y_target, num_classes, fake_class):
    probs = F.softmax(logits, dim=1)
    pred_original = logits[:, :num_classes].argmax(dim=1)
    pred_all = logits.argmax(dim=1)
    target_among_real = (pred_original == y_target).float().mean()
    discriminator_escape = (pred_all != int(fake_class)).float().mean()
    joint_target_escape = (pred_all == y_target).float().mean()
    return {
        "target_success_rate": float(target_among_real.cpu()),
        "fake_escape_rate": float(discriminator_escape.cpu()),
        "target_among_real_rate": float(target_among_real.cpu()),
        "discriminator_escape_rate": float(discriminator_escape.cpu()),
        "joint_target_escape_rate": float(joint_target_escape.cpu()),
        "fake_class_prob": float(probs[:, int(fake_class)].mean().cpu()),
        "target_prob": float(probs.gather(1, y_target.view(-1, 1)).mean().cpu()),
    }


@torch.no_grad()
def tensor_stats(x, prefix):
    flat = x.detach().float().reshape(-1, x.size(-1))
    return {
        f"{prefix}_mean": float(flat.mean().cpu()),
        f"{prefix}_std": float(flat.std(unbiased=False).cpu()),
        f"{prefix}_min": float(flat.min().cpu()),
        f"{prefix}_max": float(flat.max().cpu()),
    }


@torch.no_grad()
def diversity_ratio(fake, real, labels, max_per_class=16):
    ratios = []
    fake_flat = fake.detach().float().flatten(1)
    real_flat = real.detach().float().flatten(1)
    labels = labels.detach()
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=False).view(-1)[:max_per_class]
        if idx.numel() < 2:
            continue
        fd = torch.pdist(fake_flat[idx], p=2).mean()
        rd = torch.pdist(real_flat[idx], p=2).mean()
        if rd > 1e-8:
            ratios.append(fd / rd)
    if not ratios:
        return float("nan")
    return float(torch.stack(ratios).mean().cpu())


@torch.no_grad()
def embedding_gaps(fake_emb, real_emb):
    fake_mean = fake_emb.mean(dim=0)
    real_mean = real_emb.mean(dim=0)
    fake_var = fake_emb.var(dim=0, unbiased=False)
    real_var = real_emb.var(dim=0, unbiased=False)
    return {
        "embedding_mean_l2_gap": float((fake_mean - real_mean).norm(p=2).cpu()),
        "embedding_var_l1_gap": float((fake_var - real_var).abs().mean().cpu()),
    }


def merge_metric_sums(accum, metrics, n=1):
    for key, value in metrics.items():
        if isinstance(value, float) and math.isnan(value):
            continue
        accum[key] = accum.get(key, 0.0) + float(value) * n
        accum[f"{key}__n"] = accum.get(f"{key}__n", 0.0) + n


def finalize_metric_sums(accum):
    final = {}
    for key, value in accum.items():
        if key.endswith("__n"):
            continue
        count = accum.get(f"{key}__n", 1.0)
        final[key] = value / max(count, 1.0)
    return final
