from pathlib import Path

import numpy as np


class PartitionManager:
    """Build the retained UCF101 fold/client partitions."""

    def __init__(self, args):
        dataset = getattr(args, "dataset", None)
        if dataset != "ucf101":
            raise ValueError(
                f"{type(self).__name__} only supports ucf101, got {dataset!r}"
            )
        self.args = args

    def fetch_filelist(self):
        audio_root = Path(self.args.raw_data_dir).joinpath("ucf101", "audios")
        self.file_list = sorted(str(path) for path in audio_root.glob("*/*.wav"))
        if not self.file_list:
            raise FileNotFoundError(
                f"no UCF101 audio files found under {audio_root}"
            )

    def fetch_label_dict(self):
        if not hasattr(self, "file_list"):
            raise RuntimeError("fetch_filelist must be called before fetch_label_dict")
        labels = sorted({Path(path).parent.name for path in self.file_list})
        self.label_dict = {label: index for index, label in enumerate(labels)}

    def split_train_dev(self, train_val_file_id, seed=8):
        indices = np.arange(len(train_val_file_id))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        dev_length = len(indices) // 5
        train = [train_val_file_id[index] for index in indices[dev_length:]]
        dev = [train_val_file_id[index] for index in indices[:dev_length]]
        return train, dev

    def direchlet_partition(
        self,
        file_label_list,
        seed=8,
        min_sample_size=5,
    ):
        """Retain the historical method name used by the UCF101 script."""
        labels = np.asarray(file_label_list)
        num_clients = int(self.args.num_clients)
        alpha = float(self.args.alpha)
        if num_clients < 1:
            raise ValueError("num_clients must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if labels.size < num_clients * min_sample_size:
            raise ValueError("not enough samples to satisfy min_sample_size")

        rng = np.random.RandomState(seed)
        unique_labels = np.unique(labels)
        for _ in range(10000):
            client_indices = [[] for _ in range(num_clients)]
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                rng.shuffle(label_indices)
                proportions = rng.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.asarray(
                    [
                        proportion
                        * (len(indices) < labels.size / num_clients)
                        for proportion, indices in zip(
                            proportions, client_indices
                        )
                    ]
                )
                if proportions.sum() == 0:
                    proportions = np.ones(num_clients, dtype=np.float64)
                proportions /= proportions.sum()
                split_points = (
                    np.cumsum(proportions) * len(label_indices)
                ).astype(int)[:-1]
                client_indices = [
                    existing + split.tolist()
                    for existing, split in zip(
                        client_indices, np.split(label_indices, split_points)
                    )
                ]
            if min(map(len, client_indices)) >= min_sample_size:
                return client_indices
        raise RuntimeError("unable to produce a valid Dirichlet partition")
