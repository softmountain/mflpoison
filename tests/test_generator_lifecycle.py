import unittest
import tempfile
from pathlib import Path

import torch

from mflpoison.core.types import GlobalSnapshot, ModelSpec
from mflpoison.core.hashing import file_sha256
from mflpoison.generators import (
    CallbackGeneratorTrainer,
    ClientGeneratorPartition,
    GeneratorLifecycle,
    GeneratorLifecycleManager,
    GeneratorLifecycleMode,
)


class _TrainingRecorder:
    def __init__(self):
        self.requests = []
        self.partitions = []
        self._temporary_directory = tempfile.TemporaryDirectory()

    def __del__(self):
        self._temporary_directory.cleanup()

    def __call__(self, request, partition):
        self.requests.append(request)
        self.partitions.append(partition)
        checkpoint = Path(self._temporary_directory.name) / (
            f"{request.client_id}-{request.refresh_index}.pt"
        )
        torch.save({"refresh_index": request.refresh_index}, checkpoint)
        return request.artifact(
            str(checkpoint),
            file_sha256(checkpoint),
            {"data_identity": id(partition.data)},
        )


def _lifecycle(recorder, mode="offline_once", refresh_interval=1):
    return GeneratorLifecycle(
        client_id="malicious-1",
        partition_hash="partition-abc",
        trainer_factory=lambda: CallbackGeneratorTrainer(recorder),
        variant="dtm",
        mode=mode,
        refresh_interval=refresh_interval,
        seed=17,
    )


class GeneratorLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.data = object()
        self.partition = ClientGeneratorPartition(
            client_id="malicious-1",
            partition_hash="partition-abc",
            data=self.data,
        )

    def test_offline_once_trains_only_from_bound_client_partition(self):
        recorder = _TrainingRecorder()
        lifecycle = _lifecycle(recorder)

        first = lifecycle.ensure_artifact("snapshot-m-star", 6, self.partition)
        second = lifecycle.ensure_artifact("later-snapshot", 20, self.partition)

        self.assertIs(first, second)
        self.assertEqual(len(recorder.requests), 1)
        self.assertEqual(first.parent_snapshot_hash, "snapshot-m-star")
        self.assertIs(recorder.partitions[0].data, self.data)
        self.assertIsNone(recorder.requests[0].warm_start_artifact)

        wrong_client = ClientGeneratorPartition(
            client_id="malicious-2",
            partition_hash="partition-abc",
            data=object(),
        )
        with self.assertRaisesRegex(ValueError, "another client"):
            lifecycle.ensure_artifact("snapshot", 21, wrong_client)
        wrong_partition = ClientGeneratorPartition(
            client_id="malicious-1",
            partition_hash="partition-other",
            data=object(),
        )
        with self.assertRaisesRegex(ValueError, "partition hash"):
            lifecycle.ensure_artifact("snapshot", 21, wrong_partition)

    def test_online_refresh_is_periodic_warm_started_and_deterministic(self):
        recorder = _TrainingRecorder()
        lifecycle = _lifecycle(
            recorder,
            mode=GeneratorLifecycleMode.ONLINE_REFRESH,
            refresh_interval=2,
        )

        artifact_0 = lifecycle.ensure_artifact("snapshot-0", 0, self.partition)
        self.assertIs(
            lifecycle.ensure_artifact("snapshot-1", 1, self.partition), artifact_0
        )
        artifact_2 = lifecycle.ensure_artifact("snapshot-2", 2, self.partition)
        self.assertIs(recorder.requests[1].warm_start_artifact, artifact_0)
        self.assertNotEqual(artifact_0.seed, artifact_2.seed)
        self.assertEqual(artifact_2.seed, artifact_0.seed + 1)

        state = lifecycle.state_dict()
        resumed_recorder = _TrainingRecorder()
        resumed = _lifecycle(
            resumed_recorder,
            mode=GeneratorLifecycleMode.ONLINE_REFRESH,
            refresh_interval=2,
        )
        resumed.load_state_dict(state)
        self.assertEqual(
            resumed.ensure_artifact("snapshot-3", 3, self.partition), artifact_2
        )
        artifact_4 = resumed.ensure_artifact("snapshot-4", 4, self.partition)

        self.assertEqual(len(resumed_recorder.requests), 1)
        self.assertEqual(artifact_4.metadata["refresh_index"], 2)
        self.assertEqual(artifact_4.seed, artifact_2.seed + 1)
        self.assertEqual(artifact_4.parent_snapshot_hash, "snapshot-4")
        self.assertEqual(
            resumed_recorder.requests[0].warm_start_artifact, artifact_2
        )

    def test_resume_rejects_state_from_another_owner(self):
        recorder = _TrainingRecorder()
        lifecycle = _lifecycle(recorder)
        lifecycle.ensure_artifact("snapshot", 0, self.partition)
        state = lifecycle.state_dict()
        state["partition_hash"] = "partition-other"

        with self.assertRaisesRegex(ValueError, "partition_hash"):
            _lifecycle(_TrainingRecorder()).load_state_dict(state)

    def test_resume_rejects_missing_or_tampered_checkpoint(self):
        recorder = _TrainingRecorder()
        lifecycle = _lifecycle(recorder)
        artifact = lifecycle.ensure_artifact("snapshot", 0, self.partition)
        state = lifecycle.state_dict()
        Path(artifact.checkpoint_path).write_bytes(b"tampered")

        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            _lifecycle(_TrainingRecorder()).load_state_dict(state)

    def test_request_carries_read_only_global_snapshot_for_trainer(self):
        recorder = _TrainingRecorder()
        snapshot = GlobalSnapshot(
            state={"weight": torch.tensor([1.0])},
            round_index=5,
            dev_metrics={"accuracy": 0.5},
            model_spec=ModelSpec("test-model"),
            partition_hash="global-partition",
        )

        artifact = _lifecycle(recorder).ensure_artifact(
            snapshot, 6, self.partition
        )

        self.assertIs(recorder.requests[0].global_snapshot, snapshot)
        self.assertEqual(
            recorder.requests[0].global_snapshot_hash, snapshot.snapshot_hash
        )
        self.assertEqual(artifact.parent_snapshot_hash, snapshot.snapshot_hash)

    def test_manager_exposes_per_client_artifacts(self):
        recorders = {}

        def trainer_factory(client_id):
            recorder = recorders.setdefault(client_id, _TrainingRecorder())
            return CallbackGeneratorTrainer(recorder)

        manager = GeneratorLifecycleManager(
            trainer_factory=trainer_factory,
            variant="dtm",
            mode="offline_once",
            seed=11,
        )
        artifact = manager.ensure(
            "malicious-1", "snapshot", self.data, "partition-abc", 0
        )

        self.assertEqual(manager.artifacts, {"malicious-1": artifact})
        self.assertIs(recorders["malicious-1"].partitions[0].data, self.data)

        resumed_recorders = {}

        def resumed_factory(client_id):
            recorder = resumed_recorders.setdefault(client_id, _TrainingRecorder())
            return CallbackGeneratorTrainer(recorder)

        resumed = GeneratorLifecycleManager(
            trainer_factory=resumed_factory,
            variant="dtm",
            mode="offline_once",
            seed=11,
        )
        resumed.load_state_dict(manager.state_dict())
        self.assertEqual(
            resumed.ensure(
                "malicious-1", "later", self.data, "partition-abc", 5
            ),
            artifact,
        )
        self.assertEqual(resumed_recorders, {})


if __name__ == "__main__":
    unittest.main()
