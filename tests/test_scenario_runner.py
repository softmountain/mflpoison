import json
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from mflpoison.core.config import ScenarioConfig
from mflpoison.core.hashing import file_sha256
from mflpoison.core.types import ClientUpdate
from mflpoison.defenses import (
    DefensePipeline,
    DetectionResult,
)
from mflpoison.defenses.robust_aggregation import WeightedMean
from mflpoison.generators import (
    CallbackGeneratorTrainer,
    GeneratorLifecycleManager,
)
from mflpoison.runner import ScenarioRunner


@dataclass(frozen=True)
class _Bundle:
    client_id: str
    dataloader: float
    clean_num_samples: int
    partition_hash: str
    malicious: bool = False
    attack_active: bool = False
    poison_sample_count: int = 0
    generator_artifact_id: str = None


class _Adapter:
    def __init__(self):
        self.partition_hash = "global-partition"
        self.client_ids = ("a", "b", "c")
        self.bundles = {
            client_id: _Bundle(client_id, 1.0, 1, "partition-" + client_id)
            for client_id in self.client_ids
        }
        self.evaluations = []

    def prepare(self):
        return self

    def get_client(self, client_id):
        return self.bundles[client_id]

    def evaluate_state(self, state, split):
        weight = float(state["weight"].item())
        self.evaluations.append((split, weight))
        if split == "dev":
            return {"acc": -(weight - 2.0) ** 2, "non_scalar": [weight]}
        return {
            "acc": weight,
            "truth": [0, 0, 1],
            "pred": [1, 1, 1] if weight > 5.0 else [0, 0, 1],
        }


class _RandomModelAdapter(_Adapter):
    def build_model(self, state=None):
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.rand(1)))
        if state is not None:
            model.load_state_dict(dict(state), strict=True)
        return model


class _ClientTrainer:
    def train(
        self,
        client_id,
        snapshot,
        dataloader,
        clean_num_samples,
        artifact_ids=None,
    ):
        return ClientUpdate(
            client_id=client_id,
            delta={"weight": torch.tensor([float(dataloader)])},
            round_index=snapshot.round_index,
            base_snapshot_hash=snapshot.content_hash,
            clean_num_samples=clean_num_samples,
            train_num_samples=clean_num_samples,
            aggregation_weight=clean_num_samples,
            artifact_ids=artifact_ids or (),
        )


class _AttackStrategy:
    def __init__(self):
        self.calls = []

    def prepare_dataloader(
        self, bundle, artifact, snapshot=None, round_index=0, lengths=None
    ):
        del lengths
        self.calls.append(
            (bundle.client_id, artifact.content_hash, snapshot.content_hash, round_index)
        )
        return replace(
            bundle,
            dataloader=10.0,
            malicious=True,
            attack_active=True,
            poison_sample_count=1,
            generator_artifact_id=artifact.content_hash,
        )


class _MagnitudeDetector:
    def __init__(self, name):
        self.name = name

    def detect(self, updates, global_state):
        del global_state
        scores = {
            update.client_id: abs(float(update.delta["weight"].item()))
            for update in updates
        }
        return DetectionResult(
            name=self.name,
            scores=scores,
            threshold=5.0,
            anomalous_clients={
                client_id for client_id, score in scores.items() if score > 5.0
            },
        )


def _config(root):
    return ScenarioConfig.from_mapping(
        {
            "dataset": {
                "name": "ucf101",
                "root": "/unused",
                "num_clients": 3,
                "partition_hash": "global-partition",
                "num_classes": 2,
                "modality_shapes": {"audio": [1, 1], "video": [1, 1]},
            },
            "model": {"name": "toy"},
            "federation": {
                "rounds": 3,
                "pretrain_rounds": 3,
                "attack_rounds": 2,
                "clients_per_round": 3,
                "seed": 7,
                "convergence_metric": "acc",
                "convergence_mode": "max",
            },
            "generator": {
                "enabled": True,
                "variant": "dtm",
                "lifecycle": "online_refresh",
                "refresh_interval": 1,
            },
            "attack": {
                "enabled": True,
                "malicious_clients": ["a"],
                "poison_ratio": 0.5,
                "condition_class": 0,
                "assigned_train_label": 0,
                "victim_eval_class": 0,
                "goal_prediction_class": 1,
            },
            "defense": {
                "enabled": True,
                "aggregator": {"name": "weighted_mean"},
            },
            "evaluation": {"metrics": ["accuracy"], "evaluate_test": True},
            "results": {"root_dir": str(root)},
        }
    )


class ScenarioRunnerTest(unittest.TestCase):
    def test_federation_seed_fixes_random_initial_model(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshots = []
            for run_name in ("first", "second"):
                root = Path(directory) / run_name
                payload = _config(root).to_dict()
                payload["generator"]["enabled"] = False
                payload["attack"]["enabled"] = False
                payload["defense"]["enabled"] = False
                config = ScenarioConfig.from_mapping(payload)
                result = ScenarioRunner(
                    config,
                    adapter=_RandomModelAdapter(),
                    client_trainer=_ClientTrainer(),
                    aggregator=WeightedMean(),
                    results_dir=root,
                ).run()
                snapshots.append(result.initial_snapshot)

            self.assertEqual(
                snapshots[0].content_hash,
                snapshots[1].content_hash,
            )
            self.assertTrue(
                torch.equal(
                    snapshots[0].state["weight"],
                    snapshots[1].state["weight"],
                )
            )

    def test_tiny_end_to_end_uses_one_m_star_schedule_and_server_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = _Adapter()
            attack = _AttackStrategy()
            training_requests = {}

            def manager_factory(phase):
                phase_requests = training_requests.setdefault(phase, [])

                def trainer_factory(client_id):
                    def train(request, partition):
                        phase_requests.append(
                            (
                                client_id,
                                request.global_snapshot_hash,
                                request.round_index,
                                partition.partition_hash,
                            )
                        )
                        checkpoint_path = root / (
                            f"{phase}-{client_id}-{request.refresh_index}.pt"
                        )
                        checkpoint_path.write_bytes(
                            f"{phase}:{client_id}:{request.refresh_index}".encode("ascii")
                        )
                        return request.artifact(
                            str(checkpoint_path),
                            file_sha256(checkpoint_path),
                        )

                    return CallbackGeneratorTrainer(train)

                return GeneratorLifecycleManager(
                    trainer_factory=trainer_factory,
                    variant="dtm",
                    mode="online_refresh",
                    refresh_every=1,
                    seed=9,
                )

            defense = DefensePipeline(
                detectors=(
                    _MagnitudeDetector("magnitude_one"),
                    _MagnitudeDetector("magnitude_two"),
                ),
                aggregator=WeightedMean(),
            )
            runner = ScenarioRunner(
                _config(root),
                adapter=adapter,
                client_trainer=_ClientTrainer(),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([0.0])},
                generator_lifecycle_factory=manager_factory,
                attack_strategy=attack,
                defense_pipeline=defense,
            )

            result = runner.run()

            self.assertEqual(result.m_star.round_index, 2)
            self.assertAlmostEqual(float(result.m_star.state["weight"]), 2.0)
            self.assertEqual(result.malicious_clients, ("a",))
            self.assertEqual(len(result.branch_schedule), 2)
            self.assertEqual(
                result.branches["clean"].training.records[0].selected_client_ids,
                result.branches["attack"].training.records[0].selected_client_ids,
            )
            self.assertEqual(
                result.branches["attack"].training.records[0].selected_client_ids,
                result.branches["defended"].training.records[0].selected_client_ids,
            )
            self.assertAlmostEqual(
                float(result.branches["clean"].final_snapshot.state["weight"]), 4.0
            )
            self.assertAlmostEqual(
                float(result.branches["attack"].final_snapshot.state["weight"]), 10.0
            )
            self.assertAlmostEqual(
                float(result.branches["defended"].final_snapshot.state["weight"]), 4.0
            )
            self.assertEqual(
                result.branches["attack"].test_metrics["attack_success_rate"], 1.0
            )
            self.assertEqual(
                result.branches["defended"].test_metrics["attack_success_rate"], 0.0
            )
            defended_record = result.branches["defended"].training.records[0]
            defended_decisions = {
                item.client_id: item for item in defended_record.defense_decisions
            }
            self.assertEqual(defended_decisions["a"].action, "reject")
            self.assertEqual(
                result.branches["defended"].detection_metrics["precision"], 1.0
            )
            self.assertEqual(
                result.branches["defended"].detection_metrics["recall"], 1.0
            )

            self.assertEqual(len(training_requests["base"]), 1)
            self.assertEqual(len(training_requests["attack"]), 1)
            self.assertEqual(len(training_requests["defended"]), 1)
            attack_parent = training_requests["attack"][0][1]
            defended_parent = training_requests["defended"][0][1]
            self.assertNotEqual(attack_parent, defended_parent)
            self.assertTrue(
                all(item[3] == "partition-a" for item in training_requests["base"])
            )

            test_calls = [item for item in adapter.evaluations if item[0] == "test"]
            self.assertEqual(len(test_calls), 4)
            first_test_index = next(
                index
                for index, item in enumerate(adapter.evaluations)
                if item[0] == "test"
            )
            self.assertEqual(
                len(
                    [
                        item
                        for item in adapter.evaluations[:first_test_index]
                        if item[0] == "dev"
                    ]
                ),
                4,
            )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["m_star"]["round_index"], 2)
            self.assertEqual(
                summary["branches"]["defended"]["detection_metrics"]["fpr"],
                0.0,
            )
            self.assertEqual(
                summary["branches"]["attack"]["clean_utility_drop"], -6.0
            )
            self.assertEqual(
                summary["branches"]["attack"]["delta_asr_percentage_points"],
                100.0,
            )
            self.assertEqual(
                summary["branches"]["attack"]["attack_exposure"],
                {
                    "active_poisoned_updates": 2,
                    "malicious_client_seats": 2,
                    "rounds_with_active_poison": 2,
                    "rounds_with_malicious_clients": 2,
                    "total_client_seats": 6,
                    "total_poison_samples": 2,
                    "total_rounds": 2,
                },
            )
            self.assertNotIn(
                "targeted_attack_screen", summary["branches"]["attack"]
            )
            self.assertEqual(
                summary["branch_schedule"],
                [list(row) for row in result.branch_schedule],
            )
            self.assertTrue((root / "run_info.json").is_file())
            manifest = json.loads((root / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertIn("torch_cuda_version", manifest["runtime"])
            self.assertIn("gpu_devices", manifest["runtime"])
            self.assertIn("argv", manifest["runtime"])
            self.assertTrue((root / "checkpoints" / "m_star.pt").is_file())
            self.assertTrue((root / "rounds.pt").is_file())
            self.assertTrue(
                list((root / "generators" / "attack" / "a").glob("*.json"))
            )

    def test_branch_selection_skips_defended_when_defense_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _config(root).to_dict()
            payload["federation"]["branches"] = ("clean", "attack")
            payload["defense"]["enabled"] = False
            adapter = _Adapter()

            def manager_factory(phase):
                def trainer_factory(client_id):
                    def train(request, partition):
                        del partition
                        checkpoint = root / (
                            f"{phase}-{client_id}-{request.refresh_index}.pt"
                        )
                        checkpoint.write_bytes(
                            f"{phase}:{client_id}:{request.refresh_index}".encode(
                                "ascii"
                            )
                        )
                        return request.artifact(
                            str(checkpoint), file_sha256(checkpoint)
                        )

                    return CallbackGeneratorTrainer(train)

                return GeneratorLifecycleManager(
                    trainer_factory=trainer_factory,
                    variant="dtm",
                    mode="online_refresh",
                    refresh_every=1,
                    seed=9,
                )

            result = ScenarioRunner(
                ScenarioConfig.from_mapping(payload),
                adapter=adapter,
                client_trainer=_ClientTrainer(),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([0.0])},
                generator_lifecycle_factory=manager_factory,
                attack_strategy=_AttackStrategy(),
            ).run()
            self.assertEqual(tuple(result.branches), ("clean", "attack"))
            self.assertNotIn("defended", result.branches)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_branches"], ["clean", "attack"])

    def test_reuses_m_star_checkpoint_without_repeating_pretraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def clean_config(run_dir):
                payload = _config(run_dir).to_dict()
                payload["generator"]["enabled"] = False
                payload["attack"].update(
                    enabled=False,
                    malicious_clients=(),
                    malicious_client_count=0,
                )
                payload["defense"]["enabled"] = False
                return payload

            baseline_payload = clean_config(root / "baseline")
            baseline = ScenarioRunner(
                ScenarioConfig.from_mapping(baseline_payload),
                adapter=_Adapter(),
                client_trainer=_ClientTrainer(),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([0.0])},
            ).run()
            source_path = baseline.run_dir / "checkpoints" / "m_star.pt"

            reused_payload = clean_config(root / "reused")
            reused_payload["federation"]["m_star_path"] = str(source_path)
            reused = ScenarioRunner(
                ScenarioConfig.from_mapping(reused_payload),
                adapter=_Adapter(),
                client_trainer=_ClientTrainer(),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([0.0])},
            ).run()

            self.assertEqual(reused.m_star.content_hash, baseline.m_star.content_hash)
            self.assertEqual(reused.pretraining.records, [])
            summary = json.loads(reused.summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["m_star"]["reused"])
            self.assertEqual(summary["m_star"]["source_path"], str(source_path))

if __name__ == "__main__":
    unittest.main()
