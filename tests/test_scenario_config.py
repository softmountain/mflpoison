import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import yaml

from mflpoison.core.config import ScenarioConfig, load_scenario_config
from mflpoison.runner.__main__ import main as runner_main


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

    def test_generator_supports_separate_discriminator_learning_rate(self):
        config = valid_config()
        config["generator"].update(
            learning_rate=3e-4,
            discriminator_learning_rate=5e-5,
        )
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.generator.learning_rate, 3e-4)
        self.assertEqual(loaded.generator.discriminator_learning_rate, 5e-5)

        config["generator"]["discriminator_learning_rate"] = 0
        with self.assertRaisesRegex(ValueError, "discriminator_learning_rate"):
            ScenarioConfig.from_mapping(config)

    def test_generator_supports_explicit_steps_per_batch(self):
        loaded = ScenarioConfig.from_mapping(valid_config())
        self.assertEqual(loaded.generator.generator_steps_per_batch, 3)
        self.assertEqual(loaded.generator.discriminator_steps_per_batch, 1)

        config = valid_config()
        config["generator"].update(
            generator_steps_per_batch=10,
            discriminator_steps_per_batch=1,
        )
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.generator.generator_steps_per_batch, 10)
        self.assertEqual(loaded.generator.discriminator_steps_per_batch, 1)

        for field_name in (
            "generator_steps_per_batch",
            "discriminator_steps_per_batch",
        ):
            for invalid in (0, -1, 1.5, True):
                with self.subTest(field_name=field_name, invalid=invalid):
                    config = valid_config()
                    config["generator"][field_name] = invalid
                    with self.assertRaisesRegex(ValueError, field_name):
                        ScenarioConfig.from_mapping(config)

    def test_generator_migrates_legacy_backend_step_aliases(self):
        config = valid_config()
        config["generator"].update(
            loss={"d_steps": 2},
            options={"g_steps": 10},
        )
        loaded = ScenarioConfig.from_mapping(config)
        self.assertEqual(loaded.generator.generator_steps_per_batch, 10)
        self.assertEqual(loaded.generator.discriminator_steps_per_batch, 2)
        self.assertNotIn("g_steps", loaded.generator.options)
        self.assertNotIn("d_steps", loaded.generator.loss)

        config = valid_config()
        config["generator"].update(
            generator_steps_per_batch=10,
            options={"g_steps": 10},
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            ScenarioConfig.from_mapping(config)

    def test_runner_requires_explicit_cli_authorization_for_approved_reuse(self):
        config = valid_config()
        config["evaluation"].update(
            canonical_clean_path="canonical-clean.json",
            canonical_source_policy="approved_reuse",
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "scenario.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                runner_main(["--config", str(config_path)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("must be explicitly authorized", stderr.getvalue())

            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                runner_main(
                    [
                        "--config",
                        str(config_path),
                        "--canonical-clean",
                        "canonical-clean.json",
                        "--canonical-source-policy",
                        "approved_reuse",
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(
                "requires --m-star-path and --canonical-clean",
                stderr.getvalue(),
            )

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

    def test_defended_strength_configs_match_attack_parameter_matrix(self):
        experiment_root = ROOT / "configs" / "experiments"
        attack_root = experiment_root / "ucf101_dtm_poison_strength"
        defended_root = experiment_root / "ucf101_dtm_poison_strength_defense"
        names = sorted(path.name for path in attack_root.glob("*.yaml"))
        self.assertEqual(len(names), 10)
        self.assertEqual(
            names,
            sorted(path.name for path in defended_root.glob("*.yaml")),
        )
        for name in names:
            with self.subTest(name=name):
                attack = load_scenario_config(attack_root / name)
                defended = load_scenario_config(defended_root / name)
                self.assertEqual(attack.selected_branches, ("attack",))
                self.assertEqual(defended.selected_branches, ("defended",))
                self.assertFalse(attack.defense.enabled)
                self.assertTrue(defended.defense.enabled)
                self.assertEqual(
                    defended.attack.malicious_clients,
                    attack.attack.malicious_clients,
                )
                self.assertEqual(
                    defended.attack.malicious_client_count,
                    attack.attack.malicious_client_count,
                )
                self.assertEqual(
                    defended.attack.poison_ratio,
                    attack.attack.poison_ratio,
                )
                self.assertEqual(defended.generator.epochs, attack.generator.epochs)

    def test_separate_gan_learning_rate_configs_cover_eleven_dual_branch_cases(self):
        config_root = (
            ROOT
            / "configs"
            / "experiments"
            / "ucf101_dtm_poison_strength_separate_gan_learning_rates"
        )
        expected = {
            "malicious-clients-1_poison-20pct_generator-epochs-5.yaml": (
                ("1",),
                0.2,
                5,
            ),
            "malicious-clients-1_poison-20pct_generator-epochs-20.yaml": (
                ("1",),
                0.2,
                20,
            ),
            "malicious-clients-1_poison-20pct_generator-epochs-50.yaml": (
                ("1",),
                0.2,
                50,
            ),
            "malicious-clients-1_poison-50pct_generator-epochs-5.yaml": (
                ("1",),
                0.5,
                5,
            ),
            "malicious-clients-1_poison-100pct_generator-epochs-5.yaml": (
                ("1",),
                1.0,
                5,
            ),
            "malicious-clients-2_poison-20pct_generator-epochs-5.yaml": (
                ("0", "1"),
                0.2,
                5,
            ),
            "malicious-clients-2_poison-20pct_generator-epochs-20.yaml": (
                ("0", "1"),
                0.2,
                20,
            ),
            "malicious-clients-2_poison-50pct_generator-epochs-20.yaml": (
                ("0", "1"),
                0.5,
                20,
            ),
            "malicious-clients-3_poison-20pct_generator-epochs-5.yaml": (
                ("0", "1", "4"),
                0.2,
                5,
            ),
            "malicious-clients-3_poison-50pct_generator-epochs-50.yaml": (
                ("0", "1", "4"),
                0.5,
                50,
            ),
            "malicious-clients-3_poison-100pct_generator-epochs-50.yaml": (
                ("0", "1", "4"),
                1.0,
                50,
            ),
        }
        self.assertEqual(
            sorted(expected),
            sorted(path.name for path in config_root.glob("*.yaml")),
        )
        for name, (clients, poison_ratio, epochs) in expected.items():
            with self.subTest(name=name):
                config = load_scenario_config(config_root / name)
                self.assertEqual(config.selected_branches, ("attack", "defended"))
                self.assertTrue(config.defense.enabled)
                self.assertEqual(config.attack.malicious_clients, clients)
                self.assertEqual(config.attack.malicious_client_count, len(clients))
                self.assertEqual(config.attack.poison_ratio, poison_ratio)
                self.assertEqual(config.generator.epochs, epochs)
                self.assertEqual(config.generator.learning_rate, 3e-4)
                self.assertEqual(config.generator.discriminator_learning_rate, 5e-5)
                self.assertEqual(
                    config.evaluation.canonical_source_policy,
                    "exact",
                )

    def test_gan_step_ratio_configs_cover_six_dual_branch_cases(self):
        config_root = (
            ROOT
            / "configs"
            / "experiments"
            / "ucf101_dtm_poison_strength_gan_step_ratios"
        )
        expected = {
            (
                f"malicious-clients-{client_count}_poison-20pct_"
                "generator-epochs-50_generator-to-discriminator-steps-"
                f"{generator_steps}to1.yaml"
            ): (client_count, generator_steps)
            for client_count in (1, 2)
            for generator_steps in (10, 20, 40)
        }
        self.assertEqual(
            sorted(expected),
            sorted(path.name for path in config_root.glob("*.yaml")),
        )
        expected_clients = {1: ("1",), 2: ("0", "1")}
        for name, (client_count, generator_steps) in expected.items():
            with self.subTest(name=name):
                config = load_scenario_config(config_root / name)
                self.assertEqual(config.selected_branches, ("attack", "defended"))
                self.assertTrue(config.defense.enabled)
                self.assertEqual(
                    config.attack.malicious_clients,
                    expected_clients[client_count],
                )
                self.assertEqual(
                    config.attack.malicious_client_count,
                    client_count,
                )
                self.assertEqual(config.attack.poison_ratio, 0.2)
                self.assertEqual(config.attack.condition_class, 0)
                self.assertEqual(config.attack.assigned_train_label, 1)
                self.assertEqual(config.attack.victim_eval_class, 0)
                self.assertEqual(config.attack.goal_prediction_class, 1)
                self.assertEqual(config.generator.epochs, 50)
                self.assertEqual(config.generator.learning_rate, 3e-4)
                self.assertEqual(config.generator.discriminator_learning_rate, 5e-5)
                self.assertEqual(
                    config.generator.generator_steps_per_batch,
                    generator_steps,
                )
                self.assertEqual(
                    config.generator.discriminator_steps_per_batch,
                    1,
                )
                self.assertEqual(
                    config.evaluation.canonical_source_policy,
                    "exact",
                )

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
