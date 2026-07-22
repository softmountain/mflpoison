import json
import tempfile
import unittest
from pathlib import Path

import torch

from mflpoison.artifacts import (
    create_generator_artifact,
    load_generator_artifact,
    load_round_record,
    load_round_record_bundle,
    load_snapshot,
    save_round_record,
    save_round_record_bundle,
    save_generator_artifact,
    save_snapshot,
)
from mflpoison.core.types import (
    AggregationResult,
    ClientUpdate,
    DefenseDecision,
    GeneratorArtifact,
    GlobalSnapshot,
    ModelSpec,
    RoundRecord,
)


def snapshot(state=None):
    return GlobalSnapshot(
        state=state or {
            "weight": torch.tensor([2.0, 4.0]),
            "counter": torch.tensor(3),
        },
        round_index=4,
        dev_metrics={"accuracy": 0.75},
        model_spec=ModelSpec(
            name="MMActionClassifier",
            constructor="fed_multimodal.model.ucf101:MMActionClassifier",
        ),
        partition_hash="partition-sha256",
    )


class CoreContractsTest(unittest.TestCase):
    @staticmethod
    def _round_record():
        base = snapshot()
        update = ClientUpdate(
            client_id="7",
            delta={
                "weight": torch.tensor([1.0, -1.0]),
                "counter": torch.tensor(0),
            },
            round_index=base.round_index,
            base_snapshot_hash=base.content_hash,
            clean_num_samples=8,
            train_num_samples=10,
            artifact_ids=("generator-7",),
        )
        decision = DefenseDecision(
            client_id="7",
            action="accept",
            final_weight=8.0,
        )
        return RoundRecord(
            round_index=base.round_index,
            base_snapshot_hash=base.content_hash,
            selected_client_ids=("7",),
            raw_updates=(update,),
            defense_decisions=(decision,),
            processed_updates=(update,),
            aggregation_result=AggregationResult(
                state=update.effective_state(base.state),
                decisions=(decision,),
                diagnostics={"aggregator": "weighted_mean"},
            ),
            evaluation={"accuracy": 0.8},
        )

    def test_snapshot_hash_is_stable_and_detects_tampering(self):
        first = snapshot()
        second = snapshot()
        self.assertEqual(first.content_hash, second.content_hash)
        with self.assertRaises(ValueError):
            GlobalSnapshot(
                state={"weight": torch.tensor([9.0, 4.0]), "counter": torch.tensor(3)},
                round_index=4,
                dev_metrics={"accuracy": 0.75},
                model_spec=first.model_spec,
                partition_hash=first.partition_hash,
                content_hash=first.content_hash,
            )

    def test_snapshot_round_trip_revalidates_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.pt"
            original = snapshot()
            save_snapshot(original, path)
            restored = load_snapshot(path)
        self.assertEqual(restored.content_hash, original.content_hash)
        self.assertTrue(torch.equal(restored.state["weight"], original.state["weight"]))
        self.assertEqual(restored.state["weight"].device.type, "cpu")

    def test_typed_client_delta_validates_lineage_schema_and_finite_values(self):
        base = snapshot()
        update = ClientUpdate(
            client_id="7",
            delta={
                "weight": torch.tensor([1.0, -1.0]),
                "counter": torch.tensor(0),
            },
            round_index=4,
            base_snapshot_hash=base.content_hash,
            clean_num_samples=8,
            train_num_samples=10,
            artifact_ids=("generator:client-7", "synthetic:round-5"),
        ).validate_against(base)
        self.assertEqual(update.effective_weight, 8.0)
        self.assertEqual(update.artifact_ids[0], "generator:client-7")
        self.assertTrue(
            torch.equal(update.effective_state(base.state)["weight"], torch.tensor([3.0, 3.0]))
        )
        self.assertEqual(int(update.effective_state(base.state)["counter"]), 3)

        with self.assertRaises(ValueError):
            ClientUpdate(
                client_id="bad",
                delta={"weight": torch.tensor([float("nan"), 0.0])},
                base_snapshot_hash=base.content_hash,
                clean_num_samples=1,
            )
        with self.assertRaises(ValueError):
            ClientUpdate(
                client_id="bad",
                delta={"weight": torch.zeros(3), "counter": torch.tensor(0)},
                base_snapshot_hash=base.content_hash,
                clean_num_samples=1,
            ).validate_against(base)

    def test_legacy_client_state_has_unambiguous_effective_delta(self):
        base = snapshot()
        update = ClientUpdate(
            client_id="legacy",
            state={"weight": torch.tensor([3.0, 1.0]), "counter": torch.tensor(7)},
            num_samples=5,
        )
        self.assertEqual(update.num_samples, 5)
        self.assertTrue(update.is_legacy_state)
        self.assertTrue(
            torch.equal(update.effective_delta(base.state)["weight"], torch.tensor([1.0, -3.0]))
        )
        self.assertEqual(int(update.effective_delta(base.state)["counter"]), 0)

    def test_client_update_can_carry_delta_and_materialized_state(self):
        base = snapshot()
        update = ClientUpdate(
            client_id="typed",
            delta={"weight": torch.tensor([1.0, -3.0]), "counter": torch.tensor(0)},
            state={"weight": torch.tensor([3.0, 1.0]), "counter": torch.tensor(3)},
            round_index=base.round_index,
            base_snapshot_hash=base.content_hash,
            clean_num_samples=5,
        ).validate_against(base)
        self.assertFalse(update.is_legacy_state)
        self.assertTrue(torch.equal(update.effective_delta(base.state)["weight"], torch.tensor([1.0, -3.0])))
        self.assertTrue(torch.equal(update.effective_state(base.state)["weight"], torch.tensor([3.0, 1.0])))

        inconsistent = ClientUpdate(
            client_id="typed",
            delta={"weight": torch.tensor([1.0, -3.0]), "counter": torch.tensor(0)},
            state={"weight": torch.tensor([9.0, 9.0]), "counter": torch.tensor(3)},
            round_index=base.round_index,
            base_snapshot_hash=base.content_hash,
            clean_num_samples=5,
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            inconsistent.validate_against(base)

        with self.assertRaisesRegex(ValueError, "positive"):
            ClientUpdate(
                client_id="zero-weight",
                delta={"weight": torch.zeros(2), "counter": torch.tensor(0)},
                round_index=base.round_index,
                base_snapshot_hash=base.content_hash,
                clean_num_samples=1,
                aggregation_weight=0,
            )

    def test_generator_manifest_enforces_checkpoint_and_lineage(self):
        base = snapshot()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "generator.pt"
            torch.save({"weight": torch.tensor([1.0])}, checkpoint)
            artifact = create_generator_artifact(
                client_id="7",
                partition_hash=base.partition_hash,
                parent_snapshot_hash=base.content_hash,
                variant="dtm",
                seed=42,
                checkpoint_path=checkpoint,
            )
            manifest = root / "generator.json"
            save_generator_artifact(artifact, manifest)
            restored = load_generator_artifact(
                manifest, snapshot=base, client_id="7"
            )
            self.assertEqual(restored.content_hash, artifact.content_hash)

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["client_id"] = "8"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_generator_artifact(manifest, snapshot=base, client_id="7")

    def test_generator_artifact_rejects_wrong_parent(self):
        base = snapshot()
        artifact = GeneratorArtifact(
            client_id="7",
            partition_hash=base.partition_hash,
            parent_snapshot_hash="wrong",
            variant="dtm",
            seed=42,
            checkpoint_path="generator.pt",
            checkpoint_hash="checkpoint-hash",
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "generator.json"
            save_generator_artifact(artifact, manifest)
            with self.assertRaises(ValueError):
                load_generator_artifact(
                    manifest,
                    verify_checkpoint=False,
                    snapshot=base,
                )

    def test_round_record_rejects_processed_update_with_different_lineage(self):
        record = self._round_record()
        raw = record.raw_updates[0]
        wrong = ClientUpdate(
            client_id=raw.client_id,
            delta=raw.delta,
            round_index=raw.round_index,
            base_snapshot_hash="another-snapshot",
            clean_num_samples=raw.clean_num_samples,
            train_num_samples=raw.train_num_samples,
            artifact_ids=raw.artifact_ids,
        )
        with self.assertRaisesRegex(ValueError, "processed update"):
            RoundRecord(
                round_index=record.round_index,
                base_snapshot_hash=record.base_snapshot_hash,
                selected_client_ids=record.selected_client_ids,
                raw_updates=record.raw_updates,
                defense_decisions=record.defense_decisions,
                processed_updates=(wrong,),
                aggregation_result=record.aggregation_result,
                evaluation=record.evaluation,
            )

    def test_round_record_and_bundle_hashes_detect_payload_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_path = root / "record.pt"
            bundle_path = root / "records.pt"
            record = self._round_record()
            save_round_record(record, record_path)
            save_round_record_bundle({"pretrain": [record]}, bundle_path)
            self.assertEqual(load_round_record(record_path).round_index, 4)
            self.assertEqual(
                load_round_record_bundle(bundle_path)["pretrain"][0].round_index,
                4,
            )

            payload = torch.load(record_path, map_location="cpu")
            payload["record"].processed_updates[0].delta["weight"][0] = 99.0
            torch.save(payload, record_path)
            with self.assertRaisesRegex(ValueError, "content hash"):
                load_round_record(record_path)

            payload = torch.load(bundle_path, map_location="cpu")
            payload["phases"]["pretrain"][0].evaluation["accuracy"] = 0.1
            torch.save(payload, bundle_path)
            with self.assertRaisesRegex(ValueError, "hashes"):
                load_round_record_bundle(bundle_path)


if __name__ == "__main__":
    unittest.main()
