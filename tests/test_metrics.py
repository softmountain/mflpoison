import unittest

import torch

from fed_multimodal.poison_gan.metrics import classification_metrics
from mflpoison.evaluation import kplus1_classification_metrics


class MetricsTest(unittest.TestCase):
    def test_joint_metric_distinguishes_fake_rejection(self):
        # Row 0 targets class 0 but fake wins. Row 1 targets class 1 and escapes.
        logits = torch.tensor([[3.0, 1.0, 4.0], [0.0, 5.0, 1.0]])
        targets = torch.tensor([0, 1])
        metrics = classification_metrics(logits, targets, num_classes=2, fake_class=2)
        self.assertAlmostEqual(metrics["target_among_real_rate"], 1.0)
        self.assertAlmostEqual(metrics["discriminator_escape_rate"], 0.5)
        self.assertAlmostEqual(metrics["joint_target_escape_rate"], 0.5)
        self.assertEqual(metrics["target_success_rate"], metrics["target_among_real_rate"])

        explicit = kplus1_classification_metrics(logits, targets, 2, 2)
        self.assertAlmostEqual(explicit["joint_target_escape_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
