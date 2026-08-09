import json
import tempfile
import unittest
from pathlib import Path

import yaml

from mflpoison.core.config import ScenarioConfig, load_scenario_config


ROOT = Path(__file__).resolve().parents[1]


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
        "artifact": {"root_dir": "artifact/test-run"},
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

            yaml_path = root / "scenario.yaml"
            yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            from_yaml = load_scenario_config(yaml_path)
            self.assertEqual(from_yaml.to_dict(), loaded.to_dict())

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

    def test_derived_config_only_overrides_experiment_parameters(self):
        config = load_scenario_config(
            ROOT
            / "configs"
            / "experiments"
            / "ucf101_dtm_poison_strength"
            / "malicious-clients-2_poison-50pct_generator-epochs-20.yaml"
        )
        self.assertEqual(config.attack.malicious_clients, ("0", "1"))
        self.assertEqual(config.attack.poison_ratio, 0.5)
        self.assertEqual(config.generator.epochs, 20)
        self.assertEqual(config.artifact.root_dir, "artifact")
        self.assertEqual(config.selected_branches, ("attack",))

    def test_federation_supports_two_phase_round_counts(self):
        config = valid_config()
        config["federation"]["pretrain_rounds"] = 20
        config["federation"]["attack_rounds"] = 5
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.federation.effective_pretrain_rounds, 20)
        self.assertEqual(loaded.federation.attack_rounds, 5)
        self.assertEqual(loaded.federation.convergence_metric, "acc")

    def test_branch_selection_defaults_and_validation(self):
        config = valid_config()
        config["defense"]["enabled"] = False
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.selected_branches, ("clean", "attack"))

        config["federation"]["branches"] = ["clean"]
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.selected_branches, ("clean",))

        config["federation"]["branches"] = ["clean", "defended"]
        loaded = ScenarioConfig.from_mapping(config)
        with self.assertRaisesRegex(ValueError, "defense.enabled"):
            _ = loaded.selected_branches

        config["federation"]["branches"] = ["clean", "unknown"]
        with self.assertRaisesRegex(ValueError, "unsupported branch"):
            ScenarioConfig.from_mapping(config)

    def test_reused_m_star_accepts_a_checkpoint_path(self):
        config = valid_config()
        config["federation"]["m_star_path"] = "artifact/base/m_star.pt"
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(
            loaded.federation.m_star_path,
            "artifact/base/m_star.pt",
        )

    def test_m_star_only_and_canonical_clean_runtime_fields(self):
        config = valid_config()
        config["federation"]["m_star_only"] = True
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.selected_branches, ())

        config = valid_config()
        config["federation"]["m_star_only"] = True
        config["federation"]["branches"] = ["clean"]
        with self.assertRaisesRegex(ValueError, "cannot select"):
            ScenarioConfig.from_mapping(config)

        config = valid_config()
        config["federation"]["m_star_path"] = "artifact/base/m_star.pt"
        config["evaluation"]["canonical_clean_path"] = (
            "artifact/batches/example/canonical_clean_seed-42.json"
        )
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(
            loaded.evaluation.canonical_clean_path,
            "artifact/batches/example/canonical_clean_seed-42.json",
        )

        config["federation"]["m_star_only"] = True
        with self.assertRaisesRegex(ValueError, "cannot reuse"):
            ScenarioConfig.from_mapping(config)

    def test_production_configs_use_correct_zero_to_one_direction(self):
        expected = (
            (
                "ucf101_fdmm_dtm_poison_0to1_defense.yaml",
                ("clean", "attack", "defended"),
                True,
            ),
            (
                "ucf101_fdmm_dtm_poison_0to1_smoke.yaml",
                ("clean", "attack", "defended"),
                True,
            ),
            (
                "ucf101_fdmm_dtm_poison_0to1.yaml",
                ("clean", "attack"),
                False,
            ),
        )
        for name, branches, defense_enabled in expected:
            with self.subTest(name=name):
                config = load_scenario_config(ROOT / "configs" / "experiments" / name)
                self.assertEqual(config.selected_branches, branches)
                self.assertEqual(config.defense.enabled, defense_enabled)
                self.assertEqual(config.attack.condition_class, 0)
                self.assertEqual(config.attack.assigned_train_label, 1)
                self.assertEqual(config.attack.victim_eval_class, 0)
                self.assertEqual(config.attack.goal_prediction_class, 1)
                self.assertEqual(config.attack.malicious_clients, ("1",))


if __name__ == "__main__":
    unittest.main()
