from pathlib import Path

import torch

from mflpoison.core.types import SyntheticBatch


def save_synthetic(batch: SyntheticBatch, path, legacy: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = batch.to_legacy_dict() if legacy else batch.to_dict()
    torch.save(payload, path)
    return path


def synthetic_batch_from_payload(payload) -> SyntheticBatch:
    if not isinstance(payload, dict):
        raise TypeError("synthetic artifact must contain a dictionary")
    if "features" in payload:
        return SyntheticBatch.from_dict(payload)

    audio = payload.get("audio", payload.get("audio_features"))
    video = payload.get("video", payload.get("video_features"))
    train_labels = payload.get("train_label", payload.get("labels"))
    condition_labels = payload.get(
        "condition_label",
        payload.get("condition_labels", train_labels),
    )
    if audio is None or video is None or train_labels is None:
        raise ValueError(
            "synthetic artifact is missing audio, video, or training labels"
        )
    audio = torch.as_tensor(audio)
    video = torch.as_tensor(video)
    train_labels = torch.as_tensor(train_labels)
    condition_labels = torch.as_tensor(condition_labels)
    size = int(train_labels.shape[0])
    audio_lengths = payload.get("len_a", payload.get("audio_lengths"))
    video_lengths = payload.get("len_v", payload.get("video_lengths"))
    if audio_lengths is None:
        audio_lengths = torch.full(
            (size,), int(audio.shape[1]), dtype=torch.long
        )
    if video_lengths is None:
        video_lengths = torch.full(
            (size,), int(video.shape[1]), dtype=torch.long
        )
    return SyntheticBatch(
        features={"audio": audio, "video": video},
        lengths={
            "audio": torch.as_tensor(audio_lengths),
            "video": torch.as_tensor(video_lengths),
        },
        condition_labels=condition_labels,
        train_labels=train_labels,
        source_labels=payload.get("source_label", payload.get("source_labels")),
        metadata=dict(payload.get("meta", payload.get("metadata", {}))),
    ).validate()


def load_synthetic(path, map_location="cpu") -> SyntheticBatch:
    payload = torch.load(Path(path), map_location=map_location)
    return synthetic_batch_from_payload(payload)
