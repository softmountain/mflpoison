import copy
import subprocess
import sys
import unittest
from pathlib import Path

from mflpoison.core.config import ScenarioConfig, load_config
from mflpoison.runner import build_default_runner


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "scenarios" / "ucf101_generative_poison_defense.yaml"


def scenario_config(mutator=None):
    payload = copy.deepcopy(load_config(TEMPLATE))
    if mutator is not None:
        mutator(payload)
    return ScenarioConfig.from_mapping(payload)


class RunnerBuilderValidationTest(unittest.TestCase):
    def assert_builder_rejects(self, mutator, message):
        with self.assertRaisesRegex((ValueError, KeyError), message):
            build_default_runner(scenario_config(mutator))

    def test_rejects_unsupported_component_selections(self):
        cases = (
            (lambda value: value["dataset"].update(name="other"), "only UCF101"),
            (lambda value: value["model"].update(name="other"), "MMActionClassifier"),
            (lambda value: value["generator"].update(family="other"), "family=kplus1"),
            (lambda value: value["generator"].update(variant="other"), "variant"),
            (lambda value: value["attack"].update(strategy="other"), "only generative"),
            (lambda value: value["defense"].update(policy="other"), "defense.policy"),
        )
        for mutator, message in cases:
            with self.subTest(message=message):
                self.assert_builder_rejects(mutator, message)

    def test_rejects_ignored_or_lineage_breaking_options(self):
        cases = (
            (
                lambda value: value["model"].update(constructor="other:Model"),
                "model.constructor",
            ),
            (
                lambda value: value["model"]["kwargs"].update(dropout=0.2),
                "model.kwargs",
            ),
            (
                lambda value: value["attack"]["options"].update(unknown=True),
                "attack.options",
            ),
            (
                lambda value: (
                    value["defense"].update(enabled=False),
                    value["defense"]["options"].update(unknown=True),
                ),
                "defense.options",
            ),
            (
                lambda value: value["generator"]["loss"].update(num_classes=999),
                "scenario-owned field",
            ),
            (
                lambda value: value["generator"]["options"].update(lr_g=0.5),
                "scenario-owned field",
            ),
        )
        for mutator, message in cases:
            with self.subTest(message=message):
                self.assert_builder_rejects(mutator, message)

    def test_rejects_out_of_range_attack_classes(self):
        for field_name in (
            "condition_class",
            "assigned_train_label",
            "victim_eval_class",
            "goal_prediction_class",
        ):
            with self.subTest(field_name=field_name):
                self.assert_builder_rejects(
                    lambda value, name=field_name: value["attack"].update(
                        {name: value["dataset"]["num_classes"]}
                    ),
                    "must be in",
                )

    def test_variant_is_normalized_and_default_checkpoint_root_is_not_duplicated(self):
        runner = build_default_runner(
            scenario_config(
                lambda value: value["generator"].update(variant="DTM")
            )
        )
        manager = runner.generator_lifecycle_factory("base")
        trainer = manager.trainer_factory("client-0")
        expected = Path(runner.artifact_root) / "generator_checkpoints" / "base"
        self.assertEqual(trainer.variant, "dtm")
        self.assertEqual(trainer.output_dir, expected)

    def test_compatibility_entry_points_can_import_the_runner(self):
        scripts = (
            "experiments/run_scenario.py",
            "experiments/train_generator.py",
            "fed_multimodal/Local/train_dtm_poison_gan.py",
            "fed_multimodal/Local/train_temporal_adaptive_gan.py",
        )
        for script in scripts:
            with self.subTest(script=script):
                completed = subprocess.run(
                    [sys.executable, str(ROOT / script), "--help"],
                    cwd=str(ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=completed.stdout + completed.stderr,
                )
                self.assertIn("--config", completed.stdout)


if __name__ == "__main__":
    unittest.main()
