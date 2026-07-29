import argparse
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fed_multimodal.legacy_evaluation.checkpoint import (
    validate_legacy_checkpoint,
    validate_module_state,
)
from fed_multimodal.legacy_evaluation.data import (
    UCF101EvaluationData,
    evaluation_loader_from_args,
)
from fed_multimodal.legacy_evaluation.teacher_guided import (
    load_evaluation_checkpoint,
)
from fed_multimodal.legacy_evaluation.tstr import fit_tstr
from fed_multimodal.poison_gan.kplus1 import (
    _argument_value,
    _checkpoint_state_dict,
)
from mflpoison.artifacts.synthetic import synthetic_batch_from_payload
from mflpoison.core.types import SyntheticBatch


ROOT = Path(__file__).resolve().parents[1]


def _record(key, label, feature):
    return [key, f"/raw/{key}", label, np.asarray(feature, dtype=np.float32)]


class LegacyEvaluationDataTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        for partition, label in (("0", 3), ("dev", 4), ("test", 5)):
            self._write_partition(partition, label)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_partition(self, partition, label):
        audio_dir = (
            self.root
            / "feature/audio/mfcc/ucf101/alpha10/fold1"
        )
        video_dir = (
            self.root
            / "feature/video/mobilenet_v2/ucf101/alpha10/fold1"
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        audio = [_record(partition, label, np.ones((2, 80)))]
        video = [_record(partition, label, np.ones((1, 1280)))]
        with (audio_dir / f"{partition}.pkl").open("wb") as stream:
            pickle.dump(audio, stream)
        with (video_dir / f"{partition}.pkl").open("wb") as stream:
            pickle.dump(video, stream)

    def test_test_dev_and_explicit_client_partitions(self):
        data = UCF101EvaluationData(
            data_dir=self.root,
            alpha=1.0,
            fold=1,
            client_id="0",
            batch_size=2,
        )
        self.assertEqual(data.client_ids, ("0",))
        expected_labels = {"test": 5, "dev": 4, "client": 3}
        for partition, expected_label in expected_labels.items():
            with self.subTest(partition=partition):
                batch = next(iter(data.get_loader(partition)))
                self.assertEqual(len(batch), 5)
                self.assertEqual(batch[-1].item(), expected_label)

        loaders = data.get_dataloaders()
        self.assertNotIn("full_train", loaders)
        self.assertIs(loaders["train"], loaders["client"])

    def test_train_alias_requires_one_client(self):
        base = {
            "partition": "test",
            "client_id": None,
            "use_train": True,
            "data_dir": str(self.root),
            "dataset_dir": None,
            "alpha": 1.0,
            "fold": 1,
            "batch_size": 2,
            "num_workers": 0,
        }
        with self.assertRaisesRegex(ValueError, "requires an explicit"):
            evaluation_loader_from_args(SimpleNamespace(**base))

        base["client_id"] = "0"
        _, loader = evaluation_loader_from_args(SimpleNamespace(**base))
        self.assertEqual(next(iter(loader))[-1].item(), 3)

        base["use_train"] = False
        with self.assertRaisesRegex(ValueError, "requires --partition client"):
            evaluation_loader_from_args(SimpleNamespace(**base))

        base["partition"] = "client"
        data, loader = evaluation_loader_from_args(SimpleNamespace(**base))
        self.assertEqual(data.selected_partition, "client")
        self.assertEqual(next(iter(loader))[-1].item(), 3)


class LegacyCheckpointTest(unittest.TestCase):
    def test_checkpoint_schemas_and_optional_teacher_config(self):
        state = {"weight": torch.ones(1)}
        teacher = {"audio_generator": state, "video_generator": state}
        self.assertIs(
            validate_legacy_checkpoint(teacher, "teacher_guided"),
            teacher,
        )
        validate_legacy_checkpoint(
            {**teacher, "config": {}, "joint_discriminator": state},
            "teacher_guided",
        )

        common = {
            "config": {},
            "generator_state_dict": state,
            "discriminator_state_dict": state,
        }
        validate_legacy_checkpoint(common, "kplus1_legacy")
        validate_legacy_checkpoint(common, "temporal_adaptive")
        validate_legacy_checkpoint({**common, "gan_type": "dtm_gan"}, "dtm")

        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_legacy_checkpoint({}, "teacher_guided")
        with self.assertRaisesRegex(ValueError, "gan_type"):
            validate_legacy_checkpoint(common, "dtm")
        with self.assertRaisesRegex(TypeError, "config"):
            validate_legacy_checkpoint({**teacher, "config": []}, "teacher_guided")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            validate_legacy_checkpoint(
                {"audio_generator": {}, "video_generator": state},
                "teacher_guided",
            )

    def test_module_state_rejects_partial_or_mismatched_weights(self):
        module = torch.nn.Linear(2, 1)
        state = module.state_dict()
        self.assertIs(
            validate_module_state(module, state, "discriminator_state_dict"),
            state,
        )
        with self.assertRaisesRegex(ValueError, "missing=bias"):
            validate_module_state(
                module,
                {"weight": state["weight"]},
                "discriminator_state_dict",
            )
        with self.assertRaisesRegex(ValueError, "shape_or_dtype=weight"):
            validate_module_state(
                module,
                {"weight": torch.ones(1), "bias": state["bias"]},
                "discriminator_state_dict",
            )

    def test_teacher_state_wrappers_and_namespace_arguments(self):
        state = {"weight": torch.ones(1)}
        self.assertIs(_checkpoint_state_dict(state), state)
        for key in ("model_state_dict", "state_dict", "model"):
            self.assertIs(_checkpoint_state_dict({key: state}), state)

        self.assertEqual(_argument_value({"hid_size": 17}, "hid_size", 8), 17)
        namespace = argparse.Namespace(hid_size=19)
        self.assertEqual(_argument_value(namespace, "hid_size", 8), 19)
        self.assertEqual(_argument_value(namespace, "att", False), False)

    def test_joint_metric_is_enabled_only_for_saved_discriminator(self):
        class FakeGan:
            def __init__(self):
                self.calls = []

            def load_checkpoint(self, path, **kwargs):
                self.calls.append((path, kwargs))

        gan = FakeGan()
        self.assertFalse(load_evaluation_checkpoint(gan, "old.pt", {}))
        self.assertEqual(
            gan.calls[-1][1],
            {"load_discriminators": False},
        )
        self.assertTrue(
            load_evaluation_checkpoint(
                gan,
                "complete.pt",
                {"joint_discriminator": {}},
            )
        )
        self.assertEqual(
            gan.calls[-1][1],
            {"load_discriminators": True},
        )


class SyntheticCompatibilityTest(unittest.TestCase):
    def test_canonical_legacy_and_early_payloads(self):
        canonical = SyntheticBatch(
            features={
                "audio": torch.ones(2, 3, 4),
                "video": torch.ones(2, 2, 5),
            },
            lengths={
                "audio": torch.tensor([3, 2]),
                "video": torch.tensor([2, 1]),
            },
            condition_labels=torch.tensor([1, 2]),
            train_labels=torch.tensor([2, 3]),
        ).validate()
        restored = synthetic_batch_from_payload(canonical.to_dict())
        self.assertTrue(torch.equal(restored.train_labels, torch.tensor([2, 3])))

        legacy = synthetic_batch_from_payload(
            {
                "audio": torch.ones(2, 3, 4),
                "video": torch.ones(2, 2, 5),
                "len_a": torch.tensor([3, 2]),
                "len_v": torch.tensor([2, 1]),
                "train_label": torch.tensor([4, 5]),
            }
        )
        self.assertTrue(torch.equal(legacy.condition_labels, legacy.train_labels))

        early = synthetic_batch_from_payload(
            {
                "audio_features": torch.ones(2, 3, 4),
                "video_features": torch.ones(2, 2, 5),
                "labels": torch.tensor([6, 7]),
            }
        )
        self.assertTrue(torch.equal(early.lengths["audio"], torch.tensor([3, 3])))
        self.assertTrue(torch.equal(early.lengths["video"], torch.tensor([2, 2])))


class TSTRBoundaryTest(unittest.TestCase):
    def test_dev_selects_and_test_runs_once(self):
        model = torch.nn.Linear(1, 1)
        args = SimpleNamespace(num_epochs=2, log_interval=10)
        calls = []
        dev_scores = iter((20.0, 10.0))
        train_steps = iter((1.0, 2.0))

        def train_epoch_fn(model, *unused):
            with torch.no_grad():
                model.weight.fill_(next(train_steps))
            return 1.0, 50.0, 0.0, 0.0

        def evaluate_fn(model, loader, *unused):
            del model
            calls.append(loader)
            if loader == "dev":
                score = next(dev_scores)
                return 1.0, score, score, score
            self.assertEqual(loader, "test")
            return 2.0, 30.0, 31.0, 32.0

        best, test_metrics, history = fit_tstr(
            model=model,
            synth_loader="synthetic",
            dev_loader="dev",
            test_loader="test",
            criterion=None,
            optimizer=None,
            scheduler=None,
            args=args,
            device="cpu",
            num_classes=2,
            train_epoch_fn=train_epoch_fn,
            evaluate_fn=evaluate_fn,
        )
        self.assertEqual(calls, ["dev", "dev", "test"])
        self.assertEqual(best, 20.0)
        self.assertEqual(test_metrics, (2.0, 30.0, 31.0, 32.0))
        self.assertEqual(history["val_acc"], [20.0, 10.0])
        self.assertTrue(torch.equal(model.weight, torch.ones_like(model.weight)))


class LegacyEvaluationCLITest(unittest.TestCase):
    def test_evaluation_modules_support_help(self):
        modules = (
            "teacher_guided",
            "kplus1",
            "dtm",
            "temporal_adaptive",
            "tstr",
        )
        commands = [
            [
                sys.executable,
                "-m",
                f"fed_multimodal.legacy_evaluation.{module}",
                "--help",
            ]
            for module in modules
        ]
        commands.extend(
            [sys.executable, "-m", module, "--help"]
            for module in (
                "experiments.evaluate_generator",
                "experiments.evaluate_tstr",
            )
        )
        for command in commands:
            with self.subTest(command=command):
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=completed.stdout + completed.stderr,
                )
                self.assertIn("usage:", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
