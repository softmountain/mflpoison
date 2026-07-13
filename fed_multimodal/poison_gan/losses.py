from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def discriminator_loss(logits_real, y_real, logits_fake, fake_class, lambda_d_fake=1.0):
    fake_label = torch.full_like(y_real, int(fake_class))
    loss_real = F.cross_entropy(logits_real, y_real)
    loss_fake = F.cross_entropy(logits_fake, fake_label)
    total = loss_real + lambda_d_fake * loss_fake
    metrics = {
        "d_loss_real": float(loss_real.detach().cpu()),
        "d_loss_fake": float(loss_fake.detach().cpu()),
        "d_loss": float(total.detach().cpu()),
    }
    return total, metrics


def feature_matching_loss(emb_fake, y_target, emb_real=None, y_real=None, bank=None):
    losses = []
    for cls in y_target.unique():
        cls_fake = emb_fake[y_target == cls]
        if cls_fake.numel() == 0:
            continue
        target_mean = None
        if emb_real is not None and y_real is not None and (y_real == cls).any():
            target_mean = emb_real[y_real == cls].detach().mean(dim=0)
        elif bank is not None:
            bank_mean, _, valid = bank.lookup(cls.view(1))
            if valid is not None and bool(valid[0].item()):
                target_mean = bank_mean[0].detach()
        if target_mean is not None:
            losses.append(F.mse_loss(cls_fake.mean(dim=0), target_mean))
    if not losses:
        return emb_fake.new_tensor(0.0)
    return torch.stack(losses).mean()


def variance_matching_loss(emb_fake, y_target, emb_real=None, y_real=None, bank=None):
    losses = []
    for cls in y_target.unique():
        cls_fake = emb_fake[y_target == cls]
        if cls_fake.size(0) < 2:
            continue
        target_var = None
        if emb_real is not None and y_real is not None and (y_real == cls).sum() > 1:
            target_var = emb_real[y_real == cls].detach().var(dim=0, unbiased=False)
        elif bank is not None:
            _, bank_var, valid = bank.lookup(cls.view(1))
            if valid is not None and bool(valid[0].item()):
                target_var = bank_var[0].detach()
        if target_var is not None:
            losses.append(F.l1_loss(cls_fake.var(dim=0, unbiased=False), target_var))
    if not losses:
        return emb_fake.new_tensor(0.0)
    return torch.stack(losses).mean()


def mode_seeking_diversity_loss(generator, z, y_target, len_a=None, len_v=None, eps=1e-6):
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    z2 = torch.randn_like(z)
    fake_a1, fake_v1 = generator(z, y_target, len_a, len_v)
    fake_a2, fake_v2 = generator(z2, y_target, len_a, len_v)
    dist_a = (fake_a1 - fake_a2).flatten(1).norm(p=1, dim=1) / fake_a1[0].numel()
    dist_v = (fake_v1 - fake_v2).flatten(1).norm(p=1, dim=1) / fake_v1[0].numel()
    dist_z = (z - z2).flatten(1).norm(p=1, dim=1) / z.size(1)
    return -((dist_a + dist_v) / (dist_z + eps)).mean()


def stat_matching_loss(fake_audio, fake_video, real_audio, real_video, len_a=None, len_v=None):
    def masked_values(x, lengths):
        if lengths is None:
            return x.reshape(-1, x.size(-1))
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
        return x[mask]

    fa = masked_values(fake_audio, len_a)
    ra = masked_values(real_audio, len_a)
    fv = masked_values(fake_video, len_v)
    rv = masked_values(real_video, len_v)
    loss = fake_audio.new_tensor(0.0)
    if fa.numel() > 0 and ra.numel() > 0:
        loss = loss + F.l1_loss(fa.mean(dim=0), ra.detach().mean(dim=0))
        loss = loss + F.l1_loss(fa.std(dim=0, unbiased=False), ra.detach().std(dim=0, unbiased=False))
    if fv.numel() > 0 and rv.numel() > 0:
        loss = loss + F.l1_loss(fv.mean(dim=0), rv.detach().mean(dim=0))
        loss = loss + F.l1_loss(fv.std(dim=0, unbiased=False), rv.detach().std(dim=0, unbiased=False))
    return loss


def generator_loss(
    logits_fake,
    emb_fake,
    y_target,
    fake_class,
    config,
    emb_real=None,
    y_real=None,
    bank=None,
    generator=None,
    z=None,
    len_a=None,
    len_v=None,
    fake_audio=None,
    fake_video=None,
    real_audio=None,
    real_video=None,
    epoch=0,
):
    loss_target = F.cross_entropy(logits_fake, y_target)
    prob = F.softmax(logits_fake, dim=1)
    loss_avoid = prob[:, int(fake_class)].mean()
    loss_fm = feature_matching_loss(emb_fake, y_target, emb_real, y_real, bank)
    loss_var = variance_matching_loss(emb_fake, y_target, emb_real, y_real, bank)
    loss_div = logits_fake.new_tensor(0.0)
    if generator is not None and z is not None and epoch >= config.diversity_start_epoch:
        loss_div = mode_seeking_diversity_loss(generator, z, y_target, len_a, len_v)
    loss_stat = logits_fake.new_tensor(0.0)
    if fake_audio is not None and fake_video is not None and real_audio is not None and real_video is not None:
        loss_stat = stat_matching_loss(fake_audio, fake_video, real_audio, real_video, len_a, len_v)

    var_weight = config.lambda_var if epoch >= config.diversity_start_epoch else 0.0
    div_weight = config.lambda_div if epoch >= config.diversity_start_epoch else 0.0
    total = (
        config.lambda_adv * loss_target
        + config.lambda_avoid * loss_avoid
        + config.lambda_fm * loss_fm
        + var_weight * loss_var
        + div_weight * loss_div
        + config.lambda_stat * loss_stat
    )
    metrics = {
        "g_loss": float(total.detach().cpu()),
        "g_target": float(loss_target.detach().cpu()),
        "g_avoid": float(loss_avoid.detach().cpu()),
        "g_fm": float(loss_fm.detach().cpu()),
        "g_var": float(loss_var.detach().cpu()),
        "g_div": float(loss_div.detach().cpu()),
        "g_stat": float(loss_stat.detach().cpu()),
        "fake_class_prob": float(prob[:, int(fake_class)].mean().detach().cpu()),
        "target_prob": float(prob.gather(1, y_target.view(-1, 1)).mean().detach().cpu()),
    }
    return total, metrics
