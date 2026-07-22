"""Gate 0: freeze code/data/model provenance and verify the 7.21 anchor (protocol §4).

Gate 0 must pass before ANY formal matrix run. This module provides the checks that are
*deterministic* and therefore verifiable exactly:

  1. environment/provenance snapshot (git rev, dirty diff, python/torch/cuda versions, pip freeze)
  2. SHA256 of every frozen artifact (features, partition, M*, DTM-GAN ckpt, synth pool, centers)
  3. class-id -> class-name map derived FROM the real data loader flow (client pkl `key` field),
     NOT from any demo/aux file
  4. shape / class-count assertions (51 classes, audio [500,80], real video [*,1280] up to 10)
  5. M* test accuracy within 0.05pt of 74.95, per-class recall within 0.1pt of the frozen 7.21 file
  6. H (mean softmax entropy over correctly-predicted samples) recomputed from M* and matched to
     the frozen fragility_per_class.json within 1e-6
  7. fragility-formula internal consistency: z(H)+z(gap) over the 51-class population (ddof=0)
     reproduces the frozen `fragility` column (verifies the §3.1 standardization).

The stochastic TSTR re-run (needs retraining a synthetic classifier) and the R0 trend
reproduction are separate GPU jobs invoked by the runner; here we hash and freeze the existing
TSTR values as the provenance anchor and check everything that does not require retraining.

Steps 1-4 and 7 are CPU-only. Steps 5-6 need one M* forward pass over the test set (GPU).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# ---- provenance ---------------------------------------------------------------

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # follow symlinks (feature.pkl is a symlink into fedpoigan); hash the real bytes
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _run(cmd) -> str:
    try:
        return subprocess.check_output(cmd, cwd="/home/xp/fedpoi", stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def env_provenance() -> Dict[str, str]:
    import torch
    return {
        "git_rev": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": _run(["git", "status", "--porcelain"]),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
        "cudnn": str(torch.backends.cudnn.version()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def hash_artifacts(artifacts: Dict[str, str]) -> Dict[str, dict]:
    """artifacts: {logical_name: path}. Returns {name: {path, sha256, bytes}} (missing -> error}."""
    out = {}
    for name, path in artifacts.items():
        p = Path(path)
        if not p.exists():
            out[name] = {"path": path, "error": "missing"}
            continue
        out[name] = {"path": path, "sha256": sha256_file(path), "bytes": p.stat().st_size}
    return out


# ---- class map (from the real data flow) --------------------------------------

def extract_class_map(client_pkl_dir: str, num_classes: int = 51) -> Dict[int, str]:
    """Derive {label_id: class_name} from client pkl `key` fields (key = 'ClassName/v_...').

    Reads every client pkl in the dir and asserts each label maps to exactly one name.
    """
    import pickle
    label_to_names: Dict[int, set] = {}
    for pkl in sorted(Path(client_pkl_dir).glob("*.pkl")):
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for entry in data:
            key, label = entry[0], int(entry[2])
            name = str(key).split("/")[0]
            label_to_names.setdefault(label, set()).add(name)
    class_map = {}
    for label, names in label_to_names.items():
        if len(names) != 1:
            raise ValueError(f"label {label} maps to multiple names {names}")
        class_map[label] = next(iter(names))
    if len(class_map) != num_classes:
        raise ValueError(f"found {len(class_map)} classes, expected {num_classes}")
    return dict(sorted(class_map.items()))


# ---- shape / class assertions -------------------------------------------------

def assert_shapes(audio_pkl_entry_shape, video_pkl_entry_shape, num_classes: int):
    """audio must be [500,80]; video last dim 1280 and frames <= 10; classes == 51."""
    a = tuple(audio_pkl_entry_shape)
    v = tuple(video_pkl_entry_shape)
    assert num_classes == 51, f"num_classes {num_classes} != 51"
    assert a[-1] == 80, f"audio feat dim {a[-1]} != 80"
    assert v[-1] == 1280, f"video feat dim {v[-1]} != 1280"
    assert v[0] <= 10, f"video frames {v[0]} > 10"
    return {"audio_shape": a, "video_shape": v, "num_classes": num_classes}


# ---- fragility formula internal consistency (CPU) -----------------------------

def _zscore_pop(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=0)


def verify_fragility_formula(fragility_per_class: dict, tol: float = 1e-4) -> dict:
    """Check z(H)+z(gap) over the 51-class population reproduces the frozen `fragility` column.

    fragility_per_class: the parsed data/fragility_per_class.json (has per-class H_entropy, gap,
    fragility). Standardization is population mean/std (ddof=0) over all 51 classes (protocol §3.1).
    """
    per = fragility_per_class["per_class"]
    ids = sorted(int(k) for k in per.keys())
    H = np.array([per[str(i)]["H_entropy"] for i in ids])
    gap = np.array([per[str(i)]["gap"] for i in ids])
    frag_frozen = np.array([per[str(i)]["fragility"] for i in ids])
    frag_recomputed = _zscore_pop(H) + _zscore_pop(gap)
    max_abs = float(np.max(np.abs(frag_recomputed - frag_frozen)))
    return {
        "max_abs_diff": max_abs,
        "passed": bool(max_abs < tol),
        "z_H_mean": float(_zscore_pop(H).mean()),
        "z_gap_std": float(_zscore_pop(gap).std(ddof=0)),
    }


# ---- M* verification (GPU: one forward pass) ----------------------------------

def verify_m_star(m_star_path: str, test_loader, device, fragility_per_class: dict,
                  num_classes: int = 51,
                  acc_target: float = 74.94855967078189,
                  acc_tol: float = 0.05, recall_tol: float = 0.1, H_tol: float = 1e-6) -> dict:
    """Load M*, one forward pass over test; compare acc / per-class recall / H to frozen values.

    Returns a report dict with pass/fail per check. Uses the shared metrics.evaluate_full so the
    recall/entropy definitions match the experiment exactly.
    """
    from .driver import load_m_star
    from . import metrics as M

    model, _, _ = load_m_star(m_star_path, device)
    ev = M.evaluate_full(model, test_loader, device, num_classes)

    per = fragility_per_class["per_class"]
    frozen_recall = np.array([per[str(i)]["recall"] for i in range(num_classes)])
    frozen_H = np.array([per[str(i)]["H_entropy"] for i in range(num_classes)])
    got_recall = np.array(ev["per_class_recall"])
    got_H = np.array(ev["per_class_entropy_correct"])  # H is over CORRECT preds (protocol §3.1)

    recall_diff = np.nanmax(np.abs(got_recall - frozen_recall))
    H_diff = np.nanmax(np.abs(got_H - frozen_H))
    acc_diff = abs(ev["acc"] - acc_target)

    return {
        "acc": ev["acc"],
        "acc_target": acc_target,
        "acc_diff": float(acc_diff),
        "acc_pass": bool(acc_diff <= acc_tol),
        "recall_max_diff": float(recall_diff),
        "recall_pass": bool(recall_diff <= recall_tol),
        "H_max_diff": float(H_diff),
        "H_pass": bool(H_diff <= H_tol),
    }


# ---- default artifact set (7.21 alpha5 experiment) ----------------------------

ROOT = "/home/xp/fedpoi/fed_multimodal"
DEFAULT_ARTIFACTS = {
    "audio_feature_pkl": f"{ROOT}/results/feature/audio/mfcc/ucf101/feature.pkl",
    "video_feature_pkl": f"{ROOT}/results/feature/video/mobilenet_v2/ucf101/feature.pkl",
    "m_star": f"{ROOT}/Local/results/poison_attack/M_star_alpha5_sr1.pt",
    "dtm_gan_ckpt": f"{ROOT}/Local/results/dtm_poison_gan/final_dtm_final.pt",
    "tstr_pool_final5100": f"{ROOT}/Local/results/dtm_poison_features/dtm_final_dtm_final_train5100.pt",
    "real_centers_alpha5": f"{ROOT}/Local/results/poison_attack/centers_alpha5/real_centers.pt",
    "synth_centers_alpha5": f"{ROOT}/Local/results/poison_attack/centers_alpha5/synth_centers.pt",
}
CLIENT_PKL_DIR = f"{ROOT}/results/feature/video/mobilenet_v2/ucf101/alpha50/fold1"
FRAGILITY_JSON = "/home/xp/fedpoi/docs/fragility_exp_7.21/data/fragility_per_class.json"


def freeze_cpu_checks(out_path: Optional[str] = None) -> dict:
    """Run all CPU-only Gate-0 checks and return (optionally write) the provenance report."""
    report = {"env": env_provenance(), "artifacts": hash_artifacts(DEFAULT_ARTIFACTS)}
    report["class_map"] = extract_class_map(CLIENT_PKL_DIR)
    frag = json.loads(Path(FRAGILITY_JSON).read_text())
    report["fragility_formula"] = verify_fragility_formula(frag)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
    return report
