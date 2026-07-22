"""Evaluation + mechanism metrics for the fragility experiment (protocol §12).

One forward pass over the test set yields ALL per-class quantities so we never run the
model twice for the same eval point:
  - acc, balanced_accuracy (== macro recall == UAR), macro_f1
  - per-class recall (confusion diagonal)
  - per-class entropy over ALL samples of the class, and over CORRECTLY predicted samples
    (the protocol distinguishes these; the "correct-only" version is the legacy early signal)
  - per-class top1-top2 margin (softmax prob gap) and per-class NLL

Block update norms (protocol §12.3) split a client delta state_dict into
audio / video / fuse / classifier / total by parameter-name prefix.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score


@torch.no_grad()
def evaluate_full(model, dataloader, device, num_classes: int) -> dict:
    """Single forward pass over `dataloader`; returns the full metric dict."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in dataloader:
        x_a, x_b, l_a, l_b, y = batch
        x_a, x_b = x_a.to(device).float(), x_b.to(device).float()
        l_a, l_b = l_a.to(device), l_b.to(device)
        logits, _ = model(x_a, x_b, l_a, l_b)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits)              # [N, C]
    labels = torch.cat(all_labels).long()       # [N]
    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    preds = logits.argmax(dim=1)

    y_true = labels.numpy()
    y_pred = preds.numpy()

    acc = float(accuracy_score(y_true, y_pred) * 100)
    # balanced accuracy == macro-averaged recall == UAR
    bal_acc = float(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100)

    # per-class entropy (nat), margin, nll
    entropy_all = -(probs * log_probs).sum(dim=1)         # [N]
    top2 = probs.topk(2, dim=1).values                    # [N,2]
    margin = (top2[:, 0] - top2[:, 1])                    # [N]
    nll = -log_probs[torch.arange(len(labels)), labels]   # [N]

    per_class_recall = [float("nan")] * num_classes
    per_class_entropy_all = [float("nan")] * num_classes
    per_class_entropy_correct = [float("nan")] * num_classes
    per_class_margin = [float("nan")] * num_classes
    per_class_nll = [float("nan")] * num_classes
    per_class_support = [0] * num_classes
    per_class_correct = [0] * num_classes

    for c in range(num_classes):
        mask = labels == c
        n = int(mask.sum())
        per_class_support[c] = n
        if n == 0:
            continue
        correct = mask & (preds == c)
        nc = int(correct.sum())
        per_class_correct[c] = nc
        per_class_recall[c] = float(nc / n * 100)
        per_class_entropy_all[c] = float(entropy_all[mask].mean())
        per_class_margin[c] = float(margin[mask].mean())
        per_class_nll[c] = float(nll[mask].mean())
        if nc > 0:
            per_class_entropy_correct[c] = float(entropy_all[correct].mean())

    return {
        "acc": acc,
        "balanced_accuracy": bal_acc,
        "uar": bal_acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "per_class_entropy_all": per_class_entropy_all,
        "per_class_entropy_correct": per_class_entropy_correct,
        "per_class_margin": per_class_margin,
        "per_class_nll": per_class_nll,
        "per_class_support": per_class_support,
        "per_class_correct": per_class_correct,
        "num_samples": int(len(labels)),
    }


def source_to_target_rate(model, dataloader, device, source_class: int, target_class: int) -> float:
    """Fraction of true source-class samples predicted as target (ASR cell), percent.

    Separate helper (needs preds vs a specific true class) kept out of evaluate_full to avoid
    passing source/target into the generic evaluator.
    """
    model.eval()
    n_src, n_hit = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            x_a, x_b, l_a, l_b, y = batch
            x_a, x_b = x_a.to(device).float(), x_b.to(device).float()
            l_a, l_b = l_a.to(device), l_b.to(device)
            logits, _ = model(x_a, x_b, l_a, l_b)
            preds = logits.argmax(dim=1).cpu()
            src_mask = (y == source_class)
            n_src += int(src_mask.sum())
            n_hit += int(((preds == target_class) & src_mask).sum())
    return float(n_hit / n_src * 100) if n_src > 0 else float("nan")


# parameter-name prefixes -> logical block (protocol §12.3)
_BLOCK_PREFIXES = {
    "audio": ("audio_conv", "audio_rnn", "audio_proj", "audio_att"),
    "video": ("video_rnn", "video_proj", "video_att"),
    "fuse": ("fuse_att",),
    "classifier": ("classifier",),
}


def _block_of(param_name: str) -> str:
    for block, prefixes in _BLOCK_PREFIXES.items():
        if param_name.startswith(prefixes):
            return block
    return "other"


def block_update_norms(local_state: Dict[str, torch.Tensor],
                       global_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """L2 norm of the client delta (local - global), total and per logical block.

    Sums squared norms per block over all its parameter tensors, then sqrt. Used to compare
    malicious vs benign client updates and (later) update-level amplification (protocol §12.3).
    """
    sq = {"total": 0.0, "audio": 0.0, "video": 0.0, "fuse": 0.0, "classifier": 0.0, "other": 0.0}
    for name, g in global_state.items():
        if name not in local_state:
            continue
        delta = (local_state[name].float() - g.float())
        s = float((delta * delta).sum())
        sq["total"] += s
        sq[_block_of(name)] += s
    return {k: float(np.sqrt(v)) for k, v in sq.items()}
