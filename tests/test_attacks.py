import unittest

import torch

from mflpoison.attacks import (
    AttackSchedule,
    balanced_targets,
    label_flip_labels,
    select_malicious_clients,
)


class AttackUtilitiesTest(unittest.TestCase):
    def test_balanced_targets_and_flip(self):
        targets = balanced_targets(6, 3)
        self.assertTrue(torch.equal(targets, torch.tensor([0, 1, 2, 0, 1, 2])))
        flipped = label_flip_labels(targets, source_class=7)
        self.assertTrue(torch.equal(flipped, torch.full((6,), 7)))

    def test_schedule_and_selection_are_deterministic(self):
        schedule = AttackSchedule(start_round=2, end_round=6, every=2)
        self.assertEqual([schedule.active(index) for index in range(8)], [False, False, True, False, True, False, True, False])
        first = select_malicious_clients(["c", "a", "b"], 2, seed=4)
        second = select_malicious_clients(["b", "c", "a"], 2, seed=4)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
