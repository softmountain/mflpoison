import unittest

import torch

from mflpoison.core.types import ClientUpdate
from mflpoison.defenses.robust_aggregation import CoordinateMedian, Krum, TrimmedMean, WeightedMean
from mflpoison.defenses.update_filter import NormClipper


def update(client_id, value, samples=1):
    return ClientUpdate(
        client_id=client_id,
        state={"weight": torch.tensor([float(value)]), "counter": torch.tensor(1)},
        num_samples=samples,
    )


class DefensesTest(unittest.TestCase):
    def setUp(self):
        self.global_state = {"weight": torch.tensor([0.0]), "counter": torch.tensor(1)}

    def test_aggregators(self):
        updates = [update("a", 1), update("b", 2), update("c", 100)]
        median = CoordinateMedian().aggregate(updates, self.global_state)
        trimmed = TrimmedMean(trim_ratio=0.34).aggregate(updates, self.global_state)
        weighted = WeightedMean().aggregate(updates[:2], self.global_state)
        self.assertAlmostEqual(float(median["weight"]), 2.0)
        self.assertAlmostEqual(float(trimmed["weight"]), 2.0)
        self.assertAlmostEqual(float(weighted["weight"]), 1.5)
        self.assertEqual(int(median["counter"]), 1)

    def test_krum_ignores_a_distant_update(self):
        updates = [
            update("a", 1.0),
            update("b", 1.1),
            update("c", 0.9),
            update("d", 1.2),
            update("malicious", 100.0),
        ]
        result = Krum(byzantine_clients=1).aggregate(updates, self.global_state)
        self.assertLess(float(result["weight"]), 2.0)


    def test_norm_clipping(self):
        clipped = NormClipper(max_norm=2.0).apply(
            [update("a", 10)], self.global_state
        )[0]
        self.assertAlmostEqual(float(clipped.state["weight"]), 2.0, places=5)
        self.assertAlmostEqual(clipped.metrics["defense_clip_scale"], 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
