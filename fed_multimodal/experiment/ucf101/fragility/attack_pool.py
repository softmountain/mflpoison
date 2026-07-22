"""Attack synthetic pool loader + video frame-length adapter (protocol §6.1, corrected §1.1).

The DTM-GAN emits video as [T=9, 1280]; real UCF101 video is [T=10, 1280]. Per the corrected
protocol we DO NOT regenerate 10-frame data. Instead:
  - freeze the synthetic physical shape at 9 frames;
  - keep the effective sequence length len_v = 9;
  - if a downstream consumer needs a fixed 10-frame tensor, zero-pad ONLY at the adapter layer
    (the model masks positions >= len_v, so a 9-frame sample padded to 10 with len_v=9 is
    forward-equivalent to the unpadded 9-frame sample — verified by tests.padding_equivalence).

The dataloader's collate_mm_fn_padd already pads each batch to its own max frame count and the
model packs by len_v, so within the training pipeline NO explicit padding is required; the
adapter here exists for callers that materialize fixed-shape tensors and for the equivalence test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

REAL_VIDEO_FRAMES = 10
SYNTH_VIDEO_FRAMES = 9  # frozen physical shape of DTM-GAN output


@dataclass
class AttackPool:
    audio: torch.Tensor          # [N, 500, 80]
    video: torch.Tensor          # [N, 9, 1280]  (frozen physical shape)
    condition_label: torch.Tensor  # [N]  (all == source_class for a fixed-source pool)
    train_label: torch.Tensor       # [N]
    meta: dict
    path: str

    @property
    def size(self) -> int:
        return int(self.audio.shape[0])

    def condition_ids(self):
        return self.condition_label.tolist()


def load_attack_pool(path: str, expected_condition: Optional[int] = None) -> AttackPool:
    """Load a synthetic pool .pt; optionally assert every sample's condition == expected_condition.

    A fixed-source attack pool (protocol §6.1) has condition_label all equal to source_class and
    a recorded generator `seed` in meta so it is provably distinct from the TSTR pool.
    """
    d = torch.load(path, map_location="cpu")
    audio = d["audio"].float()
    video = d["video"].float()
    cond_key = "condition_label" if "condition_label" in d else ("labels" if "labels" in d else "train_label")
    cond = d[cond_key]
    cond = cond.long() if torch.is_tensor(cond) else torch.as_tensor(cond).long()
    train = d.get("train_label", cond)
    train = train.long() if torch.is_tensor(train) else torch.as_tensor(train).long()
    meta = d.get("meta", {})

    if video.shape[1] != SYNTH_VIDEO_FRAMES:
        # not fatal, but record: protocol expects the frozen 9-frame synthetic shape
        meta = {**meta, "_video_frames_observed": int(video.shape[1])}

    if expected_condition is not None:
        uniq = torch.unique(cond).tolist()
        if uniq != [expected_condition]:
            raise ValueError(
                f"attack pool {path}: condition labels {uniq} != [expected {expected_condition}]; "
                f"a fixed-source pool must contain only source_class content"
            )
    return AttackPool(audio=audio, video=video, condition_label=cond,
                      train_label=train, meta=meta, path=str(path))


def pad_video_to(frames: np.ndarray, target_frames: int = REAL_VIDEO_FRAMES) -> np.ndarray:
    """Zero-pad a [t, D] video feature to [target_frames, D]; returns as-is if already >= target.

    Adapter-layer only. The effective length (len_v) must still be reported as the ORIGINAL t so
    the model masks the padded tail (see module docstring / padding equivalence test).
    """
    t = frames.shape[0]
    if t >= target_frames:
        return frames
    pad = np.zeros((target_frames - t, frames.shape[1]), dtype=frames.dtype)
    return np.concatenate([frames, pad], axis=0)
