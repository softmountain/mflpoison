import unittest

import torch

from mflpoison.core.types import DatasetSpec, SyntheticBatch


class CoreTypesTest(unittest.TestCase):
    def test_synthetic_legacy_round_trip(self):
        batch = SyntheticBatch(
            features={
                "audio": torch.zeros(3, 5, 2),
                "video": torch.ones(3, 2, 4),
            },
            lengths={
                "audio": torch.tensor([5, 4, 3]),
                "video": torch.tensor([2, 2, 1]),
            },
            condition_labels=torch.tensor([0, 1, 2]),
            train_labels=torch.tensor([0, 1, 2]),
            metadata={"generator_variant": "test"},
        )
        restored = SyntheticBatch.from_dict(batch.to_legacy_dict())
        self.assertEqual(restored.num_samples, 3)
        self.assertTrue(torch.equal(restored.train_labels, batch.train_labels))
        self.assertEqual(restored.metadata["generator_variant"], "test")

    def test_dataset_spec_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            DatasetSpec("broken", 2, {"audio": (0, 80)})


if __name__ == "__main__":
    unittest.main()
