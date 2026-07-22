import tempfile
import unittest
from pathlib import Path

import torch

from mflpoison.adapters.fedmm.generator import FedMMGeneratorTrainer
from mflpoison.core.types import GlobalSnapshot, ModelSpec
from mflpoison.generators.lifecycle import (
    ClientGeneratorPartition,
    GeneratorTrainer,
    GeneratorTrainingRequest,
)


class _FakeLegacyTrainer:
    def __init__(self):
        self.loaded = None

    def load_checkpoint(self, path, load_optimizers=True):
        self.loaded = (Path(path), bool(load_optimizers))
        return torch.load(path, map_location="cpu")

    def train_epoch(self, epoch, max_batches=None, log_interval=0):
        return {"loss": float(epoch)}

    def save_checkpoint(self, path, epoch, metrics):
        torch.save(
            {
                "config": {"num_classes": 2},
                "generator_state_dict": {"weight": torch.tensor([1.0])},
                "epoch": int(epoch),
                "metrics": dict(metrics),
            },
            path,
        )


class _TestFedMMGeneratorTrainer(FedMMGeneratorTrainer):
    def __init__(self, output_dir):
        super().__init__(
            variant="dtm",
            output_dir=output_dir,
            model_metadata={
                "hid_size": 4,
                "attention": False,
                "attention_name": "base",
            },
            modality_shapes={"audio": (2, 3), "video": (2, 3)},
            num_classes=2,
            epochs=1,
        )
        self.fake_trainer = _FakeLegacyTrainer()

    def _build_trainer(self, teacher_path, dataloader, seed):
        return self.fake_trainer


class FedMMGeneratorAdapterTest(unittest.TestCase):
    def test_implements_lifecycle_protocol_and_records_request_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _TestFedMMGeneratorTrainer(directory)
            self.assertIsInstance(trainer, GeneratorTrainer)
            snapshot = GlobalSnapshot(
                state={"weight": torch.tensor([0.0])},
                round_index=7,
                dev_metrics={"acc": 1.0},
                model_spec=ModelSpec(name="tiny"),
                partition_hash="global-partition",
            )
            request = GeneratorTrainingRequest(
                client_id="client-a",
                partition_hash="client-partition",
                global_snapshot_hash=snapshot.content_hash,
                variant="dtm",
                round_index=7,
                refresh_index=2,
                seed=99,
                global_snapshot=snapshot,
            )

            artifact = trainer.train(
                request,
                ClientGeneratorPartition(
                    client_id="client-a",
                    partition_hash="client-partition",
                    data=object(),
                ),
            )

            self.assertEqual(artifact.parent_snapshot_hash, snapshot.content_hash)
            self.assertEqual(artifact.trained_round, 7)
            self.assertEqual(artifact.refresh_index, 2)
            self.assertTrue(Path(artifact.checkpoint_path).is_file())
            payload = torch.load(artifact.checkpoint_path, map_location="cpu")
            self.assertEqual(payload["lineage"]["client_id"], "client-a")
            self.assertEqual(payload["lineage"]["partition_hash"], "client-partition")

    def test_requires_full_snapshot_for_fedmm_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _TestFedMMGeneratorTrainer(directory)
            request = GeneratorTrainingRequest(
                client_id="client-a",
                partition_hash="client-partition",
                global_snapshot_hash="snapshot-hash",
                variant="dtm",
                round_index=0,
                refresh_index=0,
                seed=1,
            )
            with self.assertRaisesRegex(ValueError, "GlobalSnapshot"):
                trainer.train(
                    request,
                    ClientGeneratorPartition(
                        "client-a", "client-partition", object()
                    ),
                )

    def test_legacy_online_refresh_continues_checkpoint_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _TestFedMMGeneratorTrainer(directory)
            first_snapshot = GlobalSnapshot(
                state={"weight": torch.tensor([0.0])},
                round_index=1,
                dev_metrics={},
                model_spec=ModelSpec(name="tiny"),
                partition_hash="global-partition",
            )
            first = trainer.fit(
                "client-a",
                first_snapshot,
                object(),
                "client-partition",
                seed=1,
            )
            second_snapshot = GlobalSnapshot(
                state={"weight": torch.tensor([1.0])},
                round_index=2,
                dev_metrics={},
                model_spec=ModelSpec(name="tiny"),
                partition_hash="global-partition",
            )

            second = trainer.fit(
                "client-a",
                second_snapshot,
                object(),
                "client-partition",
                seed=2,
                previous_artifact=first,
            )

            payload = torch.load(second.checkpoint_path, map_location="cpu")
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(second.refresh_index, 1)


if __name__ == "__main__":
    unittest.main()
