import random
from typing import Optional

from torch.utils.data import Dataset

from mflpoison.core.types import SyntheticBatch


class SyntheticFeatureDataset(Dataset):
    """Expose a canonical synthetic artifact to legacy FDMM clients."""

    def __init__(self, batch: SyntheticBatch):
        self.batch = batch.validate()
        required = {"audio", "video"}
        if not required.issubset(batch.features):
            raise ValueError("FDMM adapter requires audio and video features")

    def __len__(self):
        return self.batch.num_samples

    def __getitem__(self, index):
        return (
            self.batch.features["audio"][index],
            self.batch.features["video"][index],
            int(self.batch.lengths["audio"][index]),
            int(self.batch.lengths["video"][index]),
            self.batch.train_labels[index].long(),
        )


class MixedPoisonDataset(Dataset):
    """Deterministically replace a fraction of a clean client dataset."""

    def __init__(
        self,
        clean_dataset: Dataset,
        poison_dataset: Dataset,
        poison_ratio: float,
        seed: int = 42,
        length: Optional[int] = None,
    ):
        if not 0.0 <= float(poison_ratio) <= 1.0:
            raise ValueError("poison_ratio must be in [0, 1]")
        if len(clean_dataset) < 1 or len(poison_dataset) < 1:
            raise ValueError("clean and poison datasets must be non-empty")
        self.clean_dataset = clean_dataset
        self.poison_dataset = poison_dataset
        self.length = len(clean_dataset) if length is None else int(length)
        if self.length < 1:
            raise ValueError("mixed dataset length must be positive")
        poison_count = int(round(self.length * float(poison_ratio)))
        indices = list(range(self.length))
        random.Random(int(seed)).shuffle(indices)
        self.poison_indices = set(indices[:poison_count])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index in self.poison_indices:
            return self.poison_dataset[index % len(self.poison_dataset)]
        return self.clean_dataset[index % len(self.clean_dataset)]
