import unittest

import torch

from mflpoison.core.types import ClientUpdate
from mflpoison.defenses.robust_aggregation import WeightedMean
from mflpoison.federated import FederatedEngine


class FederatedEngineTest(unittest.TestCase):
    def test_round_uses_single_shared_orchestration_path(self):
        def runner(client_id, global_state, round_index):
            offset = 1.0 if client_id == "a" else 3.0
            return ClientUpdate(
                client_id=client_id,
                state={"weight": global_state["weight"] + offset},
                num_samples=1,
            )

        engine = FederatedEngine(runner, WeightedMean())
        result = engine.run_round(0, {"weight": torch.tensor([0.0])}, ["a", "b"])
        self.assertAlmostEqual(float(result.global_state["weight"]), 2.0)
        self.assertEqual(len(result.updates), 2)


if __name__ == "__main__":
    unittest.main()
