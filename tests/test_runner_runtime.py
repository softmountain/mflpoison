import math
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mflpoison.runner.runtime import (
    client_round_seed,
    cpu_state,
    scalar_metrics,
    seed_loader,
    seed_runtime,
)


class RunnerRuntimeTest(unittest.TestCase):
    def test_client_round_seed_is_stable_and_context_specific(self):
        first = client_round_seed(42, "client-1", 3, "pretrain")
        self.assertEqual(first, client_round_seed(42, "client-1", 3, "pretrain"))
        self.assertNotEqual(first, client_round_seed(42, "client-1", 3, "branch"))
        self.assertNotEqual(first, client_round_seed(42, "client-2", 3, "pretrain"))

    def test_cpu_state_clones_tensors(self):
        source = {"weight": torch.tensor([1.0], requires_grad=True)}
        normalized = cpu_state(source)
        source["weight"].data.add_(1.0)
        self.assertEqual(normalized["weight"].device.type, "cpu")
        self.assertFalse(normalized["weight"].requires_grad)
        self.assertEqual(normalized["weight"].item(), 1.0)

    def test_scalar_metrics_keeps_only_finite_scalars(self):
        result = scalar_metrics(
            {
                "integer": 2,
                "tensor": torch.tensor(3.0),
                "vector": torch.tensor([1.0, 2.0]),
                "infinite": math.inf,
                "text": "not-a-number",
            }
        )
        self.assertEqual(result, {"integer": 2.0, "tensor": 3.0})

    def test_runtime_and_loader_seeding_are_reproducible(self):
        seed_runtime(17)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        seed_runtime(17)
        second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(first, second)

        sampler = SimpleNamespace(generator=None)
        loader = SimpleNamespace(dataset=object(), generator=None, sampler=sampler)
        seed_loader(loader, 23)
        self.assertIs(loader.generator, sampler.generator)
        expected = torch.rand(1, generator=torch.Generator().manual_seed(23))
        actual = torch.rand(1, generator=loader.generator)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
