import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from _dispatch import config_to_cli_arguments, load_experiment_config


class ExperimentConfigTest(unittest.TestCase):
    def test_generator_config_is_translated_to_legacy_cli(self):
        config = load_experiment_config(ROOT / "configs" / "generators" / "dtm.json")
        arguments = config_to_cli_arguments(config)

        self.assertEqual(config["generator"]["variant"], "dtm")
        self.assertIn("--lambda_distribution", arguments)
        self.assertIn("--lambda_diversity", arguments)
        self.assertIn("--seed", arguments)
        self.assertNotIn("--dataset", arguments)


if __name__ == "__main__":
    unittest.main()
