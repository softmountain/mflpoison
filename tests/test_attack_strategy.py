import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from mflpoison.attacks import (
    AttackSpec,
    GenerativeFeaturePoisoningStrategy,
    InjectionMode,
)
from mflpoison.core.types import SyntheticBatch
from mflpoison.data import SyntheticFeatureDataset, canonical_synthetic_batch
from mflpoison.data.synthetic_dataset import MixedPoisonDataset


class _CleanDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return (
            torch.full((2, 1), float(index)),
            torch.full((3, 1), float(index)),
            2,
            3,
            torch.tensor(9),
        )


class _GeneratorBackend:
    def __init__(self):
        self.calls = []

    def generate(
        self,
        target_labels,
        train_labels=None,
        source_labels=None,
        lengths=None,
        batch_size=64,
        seed=None,
    ):
        self.calls.append(
            {
                "target_labels": target_labels.clone(),
                "train_labels": train_labels.clone(),
                "batch_size": batch_size,
                "seed": seed,
            }
        )
        size = target_labels.shape[0]
        return SyntheticBatch(
            features={
                "audio": torch.zeros(size, 2, 1),
                "video": torch.ones(size, 3, 1),
            },
            lengths={
                "audio": torch.full((size,), 2),
                "video": torch.full((size,), 3),
            },
            condition_labels=target_labels.clone(),
            train_labels=train_labels.clone(),
        )


class GenerativeFeaturePoisoningTest(unittest.TestCase):
    def test_default_generation_seed_comes_from_attack_spec(self):
        clean = _CleanDataset(10)
        spec = AttackSpec(
            condition_class=7,
            assigned_train_label=2,
            victim_eval_class=2,
            goal_prediction_class=7,
            poison_count=2,
            seed=100,
        )
        backend = _GeneratorBackend()

        GenerativeFeaturePoisoningStrategy(spec).apply(
            clean, backend, round_index=4
        )

        self.assertEqual(backend.calls[0]["seed"], 104)

    def test_replace_uses_exact_budget_and_explicit_label_direction(self):
        backend = _GeneratorBackend()
        strategy = GenerativeFeaturePoisoningStrategy(
            AttackSpec(
                condition_class=7,
                assigned_train_label=2,
                victim_eval_class=2,
                goal_prediction_class=7,
                poison_count=3,
            ),
            seed=100,
        )

        view = strategy.apply(_CleanDataset(10), backend, round_index=4)

        self.assertTrue(view.active)
        self.assertEqual(view.poison_sample_count, 3)
        self.assertEqual(view.total_sample_count, 10)
        self.assertEqual(view.aggregation_sample_count, 10)
        self.assertEqual(len(view.dataset.poison_indices), 3)
        self.assertTrue(
            torch.equal(backend.calls[0]["target_labels"], torch.full((3,), 7))
        )
        self.assertTrue(
            torch.equal(backend.calls[0]["train_labels"], torch.full((3,), 2))
        )
        self.assertEqual(backend.calls[0]["seed"], 104)
        self.assertEqual(
            view.synthetic.metadata["attack_semantics"]["victim_eval_class"], 2
        )
        self.assertEqual(
            view.synthetic.metadata["attack_semantics"]["goal_prediction_class"], 7
        )

    def test_append_is_explicit_and_does_not_increase_aggregation_weight(self):
        backend = _GeneratorBackend()
        strategy = GenerativeFeaturePoisoningStrategy(
            AttackSpec(
                condition_class=1,
                assigned_train_label=0,
                victim_eval_class=0,
                goal_prediction_class=1,
                poison_ratio=0.3,
                injection_mode=InjectionMode.APPEND,
            )
        )

        view = strategy.apply(_CleanDataset(10), backend, round_index=0)

        self.assertEqual(view.poison_sample_count, 3)
        self.assertEqual(view.total_sample_count, 13)
        self.assertEqual(view.aggregation_sample_count, 10)
        self.assertEqual(view.dataset.poison_indices, {10, 11, 12})

    def test_inactive_schedule_does_not_generate_or_wrap_data(self):
        clean = _CleanDataset(5)
        backend = _GeneratorBackend()
        strategy = GenerativeFeaturePoisoningStrategy(
            AttackSpec(
                condition_class=1,
                assigned_train_label=0,
                victim_eval_class=0,
                goal_prediction_class=1,
                poison_ratio=0.2,
                start_round=3,
                every=2,
            )
        )

        view = strategy.apply(clean, backend, round_index=2)

        self.assertFalse(view.active)
        self.assertIs(view.dataset, clean)
        self.assertEqual(view.poison_sample_count, 0)
        self.assertEqual(backend.calls, [])

    def test_prepare_dataloader_preserves_clean_aggregation_size(self):
        clean_loader = DataLoader(_CleanDataset(8), batch_size=4, shuffle=False)
        strategy = GenerativeFeaturePoisoningStrategy(
            AttackSpec(
                condition_class=4,
                assigned_train_label=1,
                victim_eval_class=1,
                goal_prediction_class=4,
                poison_count=2,
            )
        )

        poisoned_loader = strategy.prepare_dataloader(
            clean_loader, _GeneratorBackend(), round_index=0
        )

        self.assertEqual(len(poisoned_loader.dataset), 8)
        self.assertEqual(poisoned_loader.dataset.clean_count, 8)
        self.assertEqual(poisoned_loader.dataset.poison_count, 2)
        self.assertEqual(poisoned_loader.batch_size, 4)

    def test_replace_maps_each_selected_position_to_one_unique_poison(self):
        clean = torch.utils.data.TensorDataset(torch.arange(10))
        poison = torch.utils.data.TensorDataset(torch.arange(3) + 100)
        mixed = MixedPoisonDataset(
            clean,
            poison,
            mode="replace",
            poison_count=3,
            seed=0,
        )

        injected = {
            int(mixed[index][0]) for index in sorted(mixed.poison_indices)
        }

        self.assertEqual(injected, {100, 101, 102})

    def test_budget_count_and_ratio_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            AttackSpec(
                condition_class=1,
                assigned_train_label=0,
                victim_eval_class=0,
                goal_prediction_class=1,
                poison_count=1,
                poison_ratio=0.1,
            )


class SyntheticCompatibilityTest(unittest.TestCase):
    def test_legacy_aliases_and_missing_lengths_are_normalized(self):
        batch = canonical_synthetic_batch(
            {
                "audio": torch.zeros(2, 4, 3),
                "video": torch.zeros(2, 5, 2),
                "labels": torch.tensor([3, 4]),
                "metadata": {"legacy": True},
            }
        )

        self.assertTrue(torch.equal(batch.condition_labels, batch.train_labels))
        self.assertTrue(torch.equal(batch.lengths["audio"], torch.tensor([4, 4])))
        self.assertTrue(torch.equal(batch.lengths["video"], torch.tensor([5, 5])))
        self.assertEqual(len(SyntheticFeatureDataset(batch)), 2)

    def test_non_finite_features_and_invalid_lengths_are_rejected(self):
        base = {
            "audio": torch.zeros(1, 2, 1),
            "video": torch.zeros(1, 3, 1),
            "len_a": torch.tensor([2]),
            "len_v": torch.tensor([3]),
            "condition_label": torch.tensor([0]),
            "train_label": torch.tensor([0]),
        }
        invalid_feature = dict(base)
        invalid_feature["audio"] = torch.full((1, 2, 1), float("nan"))
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            canonical_synthetic_batch(invalid_feature)
        invalid_length = dict(base)
        invalid_length["len_v"] = torch.tensor([4])
        with self.assertRaisesRegex(ValueError, "exceed"):
            canonical_synthetic_batch(invalid_length)


if __name__ == "__main__":
    unittest.main()
