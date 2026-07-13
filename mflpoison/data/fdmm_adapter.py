from mflpoison.core.types import MultimodalBatch


def batch_from_fdmm(batch) -> MultimodalBatch:
    """Convert the current FDMM audio/video tuple into the named schema."""

    if len(batch) != 5:
        raise ValueError("FDMM multimodal batches must contain five tensors")
    audio, video, len_audio, len_video, labels = batch
    return MultimodalBatch(
        features={"audio": audio, "video": video},
        lengths={"audio": len_audio, "video": len_video},
        labels=labels,
    ).validate()
