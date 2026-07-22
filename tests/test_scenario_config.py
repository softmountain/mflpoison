import json
import tempfile
import unittest
from pathlib import Path

from mflpoison.core.config import ScenarioConfig, load_scenario_config


def valid_config():
    return {
        "dataset": {
            "name": "ucf101",
            "root": "/data/ucf101",
            "fold": 1,
            "alpha": 1.0,
            "num_clients": 5,
            "partition_id": "fold1-alpha1-clients5",
            "partition_hash": "partition-sha256",
            "num_classes": 101,
            "modality_shapes": {"audio": [20, 80], "video": [20, 512]},
        },
        "model": {"name": "MMActionClassifier", "kwargs": {"dropout": 0.2}},
        "federation": {
            "rounds": 10,
            "clients_per_round": 3,
            "local_epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.01,
            "seed": 42,
        },
        "generator": {
            "enabled": True,
            "family": "kplus1",
            "variant": "dtm",
            "lifecycle": "offline_once",
        },
        "attack": {
            "enabled": True,
            "malicious_clients": ["0"],
            "poison_ratio": 0.2,
            "injection_mode": "replace",
            "condition_class": 1,
            "assigned_train_label": 1,
            "victim_eval_class": 1,
            "goal_prediction_class": 0,
        },
        "defense": {
            "enabled": True,
            "detectors": [{"name": "norm_mad"}, {"name": "cosine_center"}],
            "aggregator": {"name": "weighted_mean"},
        },
        "evaluation": {"metrics": ["accuracy", "attack_success_rate"]},
        "artifacts": {"root_dir": "artifacts/run"},
    }


class ScenarioConfigTest(unittest.TestCase):
    def test_load_json_and_yaml_without_flattening_sections(self):
        config = valid_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "scenario.json"
            json_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_scenario_config(json_path)
            self.assertIsInstance(loaded, ScenarioConfig)
            self.assertEqual(loaded.generator.variant, "dtm")
            self.assertEqual(loaded.model.kwargs["dropout"], 0.2)
            self.assertEqual(loaded.attack.malicious_clients, ("0",))

            try:
                import yaml
            except ImportError:
                return
            yaml_path = root / "scenario.yaml"
            yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            from_yaml = load_scenario_config(yaml_path)
            self.assertEqual(from_yaml.content_hash, loaded.content_hash)

    def test_rejects_unknown_top_level_and_nested_fields(self):
        config = valid_config()
        config["seed"] = 99
        with self.assertRaisesRegex(ValueError, "unknown scenario section"):
            ScenarioConfig.from_mapping(config)

        config = valid_config()
        config["generator"]["lambda_diversity"] = 0.2
        with self.assertRaisesRegex(ValueError, "unknown generator field"):
            ScenarioConfig.from_mapping(config)

    def test_requires_every_explicit_section(self):
        config = valid_config()
        del config["defense"]
        with self.assertRaisesRegex(ValueError, "missing scenario section"):
            ScenarioConfig.from_mapping(config)

    def test_options_is_the_explicit_extension_boundary(self):
        config = valid_config()
        config["generator"]["options"] = {"lambda_diversity": 0.2}
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.generator.options["lambda_diversity"], 0.2)

    def test_federation_supports_two_phase_round_counts(self):
        config = valid_config()
        config["federation"]["pretrain_rounds"] = 20
        config["federation"]["attack_rounds"] = 5
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.federation.effective_pretrain_rounds, 20)
        self.assertEqual(loaded.federation.attack_rounds, 5)
        self.assertEqual(loaded.federation.convergence_metric, "acc")


if __name__ == "__main__":
    unittest.main()
