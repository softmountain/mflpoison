import unittest

import torch

from mflpoison.core.types import ClientUpdate
from mflpoison.defenses.robust_aggregation import WeightedMean


def _legacy_fedavg(states, sample_counts):
    total = float(sum(sample_counts))
    return {
        key: sum(
            state[key] * (count / total)
            for state, count in zip(states, sample_counts)
        )
        for key in states[0]
    }


class FedAvgEquivalenceTest(unittest.TestCase):
    def test_single_and_multi_round_delta_aggregation_matches_legacy_states(self):
        generator = torch.Generator().manual_seed(17)
        current = {"weight": torch.randn(5, generator=generator)}
        sample_counts = [3, 7, 11]
        aggregator = WeightedMean()

        for round_index in range(4):
            client_states = [
                {
                    "weight": current["weight"]
                    + torch.randn(5, generator=generator) * 0.05
                }
                for _ in sample_counts
            ]
            updates = [
                ClientUpdate(
                    client_id=f"client-{index}",
                    delta={"weight": state["weight"] - current["weight"]},
                    round_index=round_index,
                    base_snapshot_hash=f"round-{round_index}",
                    clean_num_samples=count,
                    train_num_samples=count,
                    aggregation_weight=count,
                )
                for index, (state, count) in enumerate(
                    zip(client_states, sample_counts)
                )
            ]

            legacy = _legacy_fedavg(client_states, sample_counts)
            current = aggregator.aggregate(updates, current)

            self.assertTrue(
                torch.allclose(current["weight"], legacy["weight"], atol=1e-7)
            )


if __name__ == "__main__":
    unittest.main()
