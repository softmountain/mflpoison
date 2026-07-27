import json
import pickle
from pathlib import Path

import numpy as np


class SimulationManager:
    """Create optional missing-data and label-noise metadata for UCF101."""

    def __init__(self, args):
        dataset = getattr(args, "dataset", None)
        if dataset != "ucf101":
            raise ValueError(
                f"{type(self).__name__} only supports ucf101, got {dataset!r}"
            )
        self.args = args

    def fetch_partition(self, fold_idx=1, alpha=0.5, ext="json"):
        if ext not in {"json", "pkl"}:
            raise ValueError("partition extension must be json or pkl")
        alpha_name = str(alpha).replace(".", "")
        partition_path = Path(self.args.output_dir).joinpath(
            "partition",
            "ucf101",
            f"fold{fold_idx}",
            f"partition_alpha{alpha_name}.{ext}",
        )
        if ext == "pkl":
            with partition_path.open("rb") as stream:
                return pickle.load(stream)
        with partition_path.open("r") as stream:
            return json.load(stream)

    def simulate_missing_modality(self, seed):
        return np.random.RandomState(seed).binomial(
            size=1,
            n=1,
            p=self.args.missing_modailty_rate,
        )[0]

    def simulate_missing_label(self, seed, size):
        return np.random.RandomState(seed).binomial(
            size=size,
            n=1,
            p=self.args.missing_label_rate,
        )

    def label_noise_matrix(self, seed, class_num=51):
        """Return a deterministic row-stochastic UCF101 transition matrix."""
        if class_num < 2:
            raise ValueError("class_num must be at least two")
        noisy_level = float(self.args.label_nosiy_level)
        if not 0.0 <= noisy_level <= 1.0:
            raise ValueError("label_nosiy_level must be between zero and one")

        sparse_level = 0.4
        active_count = max(
            1, int(round((class_num - 1) * (1.0 - sparse_level)))
        )
        rng = np.random.RandomState(seed)
        matrix = np.zeros((class_num, class_num), dtype=np.float64)
        for source_class in range(class_num):
            candidates = np.delete(np.arange(class_num), source_class)
            active_targets = rng.choice(
                candidates, size=active_count, replace=False
            )
            matrix[source_class, source_class] = 1.0 - noisy_level
            matrix[source_class, active_targets] = noisy_level / active_count
        return matrix

    def get_simulation_setting(self, alpha=None):
        settings = []
        if self.args.missing_modality:
            settings.append(
                "mm" + str(self.args.missing_modailty_rate).replace(".", "")
            )
        if self.args.label_nosiy:
            settings.append(
                "ln" + str(self.args.label_nosiy_level).replace(".", "")
            )
        if self.args.missing_label:
            settings.append(
                "ml" + str(self.args.missing_label_rate).replace(".", "")
            )
        if settings and alpha is not None:
            settings.append("alpha" + str(self.args.alpha).replace(".", ""))
        self.setting_str = "_".join(settings)

    def simulation(self, data_dict, seed, class_num=51):
        if self.args.missing_modality:
            modality_a_missing = int(self.simulate_missing_modality(seed))
            modality_b_missing = int(self.simulate_missing_modality(seed * 2))
        else:
            modality_a_missing = 0
            modality_b_missing = 0

        if self.args.label_nosiy:
            self.prob_matrix = self.label_noise_matrix(seed, class_num)
            label_rng = np.random.RandomState(seed)
        if self.args.missing_label:
            missing_labels = self.simulate_missing_label(seed, len(data_dict))

        for index in range(len(data_dict)):
            original_label = int(data_dict[index][2])
            if not 0 <= original_label < class_num:
                raise ValueError(
                    f"label {original_label} is outside [0, {class_num})"
                )
            if self.args.label_nosiy:
                new_label = int(
                    label_rng.choice(class_num, p=self.prob_matrix[original_label])
                )
            else:
                new_label = original_label
            missing_label = (
                int(missing_labels[index]) if self.args.missing_label else 0
            )
            data_dict[index].append(
                [
                    modality_a_missing,
                    modality_b_missing,
                    new_label,
                    missing_label,
                ]
            )
        return data_dict
