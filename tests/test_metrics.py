import unittest

import torch

from fed_multimodal.poison_gan.metrics import classification_metrics
from mflpoison.evaluation import targeted_classification_metrics


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

    def test_targeted_classification_metrics_report_direction_and_utility(self):
        metrics = targeted_classification_metrics(
            [0, 0, 1, 1, 2, 2],
            [1, 0, 1, 0, 1, 2],
            victim_class=0,
            goal_class=1,
        )
        self.assertEqual(metrics["attack_source_sample_count"], 2.0)
        self.assertAlmostEqual(metrics["attack_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["attack_success_rate_pct"], 50.0)
        self.assertAlmostEqual(metrics["source_class_accuracy"], 50.0)
        self.assertAlmostEqual(metrics["source_class_recall"], 50.0)
        self.assertAlmostEqual(metrics["goal_class_false_positive_rate"], 50.0)
        self.assertAlmostEqual(metrics["non_source_accuracy"], 50.0)
        self.assertAlmostEqual(metrics["non_source_macro_f1"], 58.3333333333)

    def test_targeted_classification_metrics_reject_mismatched_inputs(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            targeted_classification_metrics(
                [0, 1],
                [1],
                victim_class=0,
                goal_class=1,
            )


if __name__ == "__main__":
    unittest.main()
