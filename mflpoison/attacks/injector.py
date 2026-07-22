from mflpoison.core.types import SyntheticBatch
from mflpoison.data.synthetic_dataset import MixedPoisonDataset, SyntheticFeatureDataset


def inject_synthetic_dataset(
    clean_dataset,
    synthetic: SyntheticBatch,
    poison_ratio: float = None,
    seed: int = 42,
    length=None,
    mode="replace",
    poison_count=None,
):
    poison_dataset = SyntheticFeatureDataset(synthetic)
    return MixedPoisonDataset(
        clean_dataset,
        poison_dataset,
        poison_ratio=poison_ratio,
        seed=seed,
        length=length,
        mode=mode,
        poison_count=poison_count,
    )
