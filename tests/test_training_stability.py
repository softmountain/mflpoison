import unittest

import torch

from mflpoison.training.stability import nonfinite_gradient_names, second_order_rnn_context


class StabilityTest(unittest.TestCase):
    def test_second_order_context_disables_cudnn_temporarily(self):
        original = torch.backends.cudnn.enabled
        with second_order_rnn_context(enabled=True):
            self.assertFalse(torch.backends.cudnn.enabled)
        self.assertEqual(torch.backends.cudnn.enabled, original)

    def test_nonfinite_gradient_detection(self):
        layer = torch.nn.Linear(2, 1)
        layer.weight.grad = torch.full_like(layer.weight, float("nan"))
        names = nonfinite_gradient_names(layer.named_parameters())
        self.assertEqual(names, ["weight"])


if __name__ == "__main__":
    unittest.main()
