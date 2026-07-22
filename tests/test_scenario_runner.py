import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from mflpoison.core.config import ScenarioConfig
from mflpoison.core.hashing import file_sha256
from mflpoison.artifacts import round_record_hash
from mflpoison.core.types import ClientUpdate, GeneratorArtifact
from mflpoison.defenses import (
    DefensePipeline,
    DetectionResult,
    EWMAReputation,
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


class _InterruptingTrainer(_ClientTrainer):
    def __init__(self, fail_on_call):
        self.calls = 0
        self.fail_on_call = int(fail_on_call)

    def train(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return super().train(*args, **kwargs)


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
        return replace(bundle, dataloader=10.0)


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
            "artifacts": {"root_dir": str(root)},
        }
    )


class ScenarioRunnerTest(unittest.TestCase):
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
                summary["branch_schedule"],
                [list(row) for row in result.branch_schedule],
            )
            self.assertTrue((root / "manifest.json").is_file())
            self.assertTrue((root / "snapshots" / "m_star.pt").is_file())
            self.assertTrue((root / "round_records.pt").is_file())
            self.assertTrue(
                list((root / "generators" / "attack" / "a").glob("*.json"))
            )

    def test_resume_continues_after_last_complete_round_reproducibly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_root = root / "baseline"
            resumed_root = root / "resumed"

            def clean_config(artifact_root, resume_from=None):
                payload = _config(artifact_root).to_dict()
                payload["generator"]["enabled"] = False
                payload["attack"].update(
                    {
                        "enabled": False,
                        "malicious_clients": (),
                        "malicious_client_count": 0,
                    }
                )
                payload["defense"]["enabled"] = False
                payload["federation"]["resume_from"] = resume_from
                return ScenarioConfig.from_mapping(payload)

            baseline = ScenarioRunner(
                clean_config(baseline_root),
                adapter=_Adapter(),
                client_trainer=_ClientTrainer(),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([0.0])},
            ).run()

            interrupted_config = clean_config(resumed_root)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                ScenarioRunner(
                    interrupted_config,
                    adapter=_Adapter(),
                    client_trainer=_InterruptingTrainer(fail_on_call=4),
                    aggregator=WeightedMean(),
                    initial_state={"weight": torch.tensor([0.0])},
                ).run()

            resume_path = resumed_root / "resume_state.pt"
            resumed = ScenarioRunner(
                clean_config(resumed_root, str(resume_path)),
                adapter=_Adapter(),
                client_trainer=_ClientTrainer(),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([999.0])},
            ).run()

            self.assertEqual(resumed.m_star.content_hash, baseline.m_star.content_hash)
            self.assertEqual(
                len(resumed.pretraining.records), len(baseline.pretraining.records)
            )
            for name in ScenarioRunner.BRANCHES:
                self.assertEqual(
                    resumed.branches[name].final_snapshot.content_hash,
                    baseline.branches[name].final_snapshot.content_hash,
                )

    def test_resume_rejects_tampered_runtime_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _config(root).to_dict()
            payload["generator"]["enabled"] = False
            payload["attack"].update(
                enabled=False,
                malicious_clients=(),
                malicious_client_count=0,
            )
            payload["defense"]["enabled"] = False
            interrupted_config = ScenarioConfig.from_mapping(payload)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                ScenarioRunner(
                    interrupted_config,
                    adapter=_Adapter(),
                    client_trainer=_InterruptingTrainer(fail_on_call=4),
                    aggregator=WeightedMean(),
                    initial_state={"weight": torch.tensor([0.0])},
                ).run()

            resume_path = root / "resume_state.pt"
            resume_payload = torch.load(resume_path, map_location="cpu")
            resume_payload["initial_snapshot"].state["weight"][0] = 123.0
            torch.save(resume_payload, resume_path)
            payload["federation"]["resume_from"] = str(resume_path)
            with self.assertRaisesRegex(ValueError, "content hash"):
                ScenarioRunner(
                    ScenarioConfig.from_mapping(payload),
                    adapter=_Adapter(),
                    client_trainer=_ClientTrainer(),
                    aggregator=WeightedMean(),
                    initial_state={"weight": torch.tensor([0.0])},
                ).run()

    def test_online_generator_and_ewma_resume_match_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"

            def manager_factory(requests):
                def build(phase):
                    def trainer_factory(client_id):
                        def train(request, partition):
                            del partition
                            requests.append(
                                (
                                    phase,
                                    client_id,
                                    request.round_index,
                                    request.refresh_index,
                                    request.global_snapshot_hash,
                                )
                            )
                            checkpoint = root / (
                                f"{phase}-{client_id}-{request.refresh_index}.pt"
                            )
                            checkpoint.parent.mkdir(parents=True, exist_ok=True)
                            checkpoint.write_bytes(
                                (
                                    f"{phase}:{client_id}:"
                                    f"{request.refresh_index}:"
                                    f"{request.global_snapshot_hash}"
                                ).encode("ascii")
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

                return build

            def defense():
                return DefensePipeline(
                    detectors=(
                        _MagnitudeDetector("magnitude_one"),
                        _MagnitudeDetector("magnitude_two"),
                    ),
                    reputation=EWMAReputation(
                        decay=0.5,
                        minimum_reputation=0.75,
                    ),
                    aggregator=WeightedMean(),
                )

            baseline_adapter = _Adapter()
            baseline_defense = defense()
            baseline = ScenarioRunner(
                _config(root),
                adapter=baseline_adapter,
                client_trainer=_ClientTrainer(),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([0.0])},
                generator_lifecycle_factory=manager_factory([]),
                attack_strategy=_AttackStrategy(),
                defense_pipeline=baseline_defense,
            ).run()
            baseline_records = {
                name: [
                    round_record_hash(record)
                    for record in branch.training.records
                ]
                for name, branch in baseline.branches.items()
            }
            baseline_artifacts = {
                name: {
                    client_id: (artifact.content_hash, artifact.refresh_index)
                    for client_id, artifact in branch.generator_artifacts.items()
                }
                for name, branch in baseline.branches.items()
            }
            baseline_evaluations = list(baseline_adapter.evaluations)
            baseline_reputation = baseline_defense.reputation.state_dict()

            for fail_on_call in (19, 25):
                with self.subTest(fail_on_call=fail_on_call):
                    shutil.rmtree(root)
                    interrupted_adapter = _Adapter()
                    interrupted_defense = defense()
                    requests = []
                    with self.assertRaisesRegex(
                        RuntimeError, "simulated interruption"
                    ):
                        ScenarioRunner(
                            _config(root),
                            adapter=interrupted_adapter,
                            client_trainer=_InterruptingTrainer(fail_on_call),
                            aggregator=WeightedMean(),
                            initial_state={"weight": torch.tensor([0.0])},
                            generator_lifecycle_factory=manager_factory(requests),
                            attack_strategy=_AttackStrategy(),
                            defense_pipeline=interrupted_defense,
                        ).run()

                    resume_path = root / "resume_state.pt"
                    resumed_payload = _config(root).to_dict()
                    resumed_payload["federation"]["resume_from"] = str(
                        resume_path
                    )
                    resumed_adapter = _Adapter()
                    resumed_defense = defense()
                    resumed = ScenarioRunner(
                        ScenarioConfig.from_mapping(resumed_payload),
                        adapter=resumed_adapter,
                        client_trainer=_ClientTrainer(),
                        aggregator=WeightedMean(),
                        initial_state={"weight": torch.tensor([999.0])},
                        generator_lifecycle_factory=manager_factory(requests),
                        attack_strategy=_AttackStrategy(),
                        defense_pipeline=resumed_defense,
                    ).run()

                    self.assertEqual(
                        resumed.m_star.content_hash, baseline.m_star.content_hash
                    )
                    for name in ScenarioRunner.BRANCHES:
                        self.assertEqual(
                            resumed.branches[name].final_snapshot.content_hash,
                            baseline.branches[name].final_snapshot.content_hash,
                        )
                        self.assertEqual(
                            [
                                round_record_hash(record)
                                for record in resumed.branches[name].training.records
                            ],
                            baseline_records[name],
                        )
                        self.assertEqual(
                            {
                                client_id: (
                                    artifact.content_hash,
                                    artifact.refresh_index,
                                )
                                for client_id, artifact in resumed.branches[
                                    name
                                ].generator_artifacts.items()
                            },
                            baseline_artifacts[name],
                        )
                    self.assertEqual(
                        resumed_defense.reputation.state_dict(),
                        baseline_reputation,
                    )
                    self.assertEqual(
                        interrupted_adapter.evaluations
                        + resumed_adapter.evaluations,
                        baseline_evaluations,
                    )

    def test_completed_resume_does_not_repeat_training_or_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _config(root).to_dict()
            payload["generator"]["enabled"] = False
            payload["attack"].update(
                enabled=False,
                malicious_clients=(),
                malicious_client_count=0,
            )
            payload["defense"]["enabled"] = False
            config = ScenarioConfig.from_mapping(payload)
            baseline = ScenarioRunner(
                config,
                adapter=_Adapter(),
                client_trainer=_ClientTrainer(),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([0.0])},
            ).run()
            baseline_summary = json.loads(
                baseline.summary_path.read_text(encoding="utf-8")
            )

            payload["federation"]["resume_from"] = str(
                root / "resume_state.pt"
            )
            adapter = _Adapter()
            resumed = ScenarioRunner(
                ScenarioConfig.from_mapping(payload),
                adapter=adapter,
                client_trainer=_InterruptingTrainer(fail_on_call=1),
                aggregator=WeightedMean(),
                initial_state={"weight": torch.tensor([999.0])},
            ).run()
            self.assertEqual(adapter.evaluations, [])
            self.assertEqual(
                resumed.m_star.content_hash,
                baseline.m_star.content_hash,
            )
            self.assertEqual(
                json.loads(resumed.summary_path.read_text(encoding="utf-8")),
                baseline_summary,
            )


if __name__ == "__main__":
    unittest.main()
