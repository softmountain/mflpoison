import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from mflpoison.core.types import (
    ClientUpdate,
    DefenseDecision,
    GlobalSnapshot,
    ModelSpec,
)
from mflpoison.defenses.robust_aggregation import WeightedMean
from mflpoison.federated import (
    ConvergencePolicy,
    FedAvgCoordinator,
    build_client_schedule_count,
)


@dataclass
class Bundle:
    dataloader: object
    clean_num_samples: int = 1


class FakeTrainer:
    def train(
        self,
        client_id,
        snapshot,
        dataloader,
        clean_num_samples,
        artifact_ids=None,
    ):
        offset = float(dataloader)
        return ClientUpdate(
            client_id=client_id,
            delta={"weight": torch.tensor([offset])},
            round_index=snapshot.round_index,
            base_snapshot_hash=snapshot.content_hash,
            clean_num_samples=clean_num_samples,
            train_num_samples=clean_num_samples,
            aggregation_weight=clean_num_samples,
            artifact_ids=artifact_ids or (),
        )


class MutatingTrainer(FakeTrainer):
    def train(self, client_id, snapshot, **kwargs):
        snapshot.state["weight"].add_(1.0)
        return super().train(client_id=client_id, snapshot=snapshot, **kwargs)


class StaleTrainer(FakeTrainer):
    def train(self, client_id, snapshot, **kwargs):
        update = super().train(client_id=client_id, snapshot=snapshot, **kwargs)
        update.base_snapshot_hash = "stale"
        return update


class RejectAllPipeline:
    def process(self, updates, snapshot, **kwargs):
        decisions = [
            DefenseDecision(
                client_id=update.client_id,
                action="reject",
                reason="test rejection",
                final_weight=0.0,
            )
            for update in updates
        ]
        return SimpleNamespace(
            aggregated_state=None,
            decisions=decisions,
            sanitized=(),
            aggregation_audit=SimpleNamespace(
                aggregator="weighted_mean",
                submitted_clients=tuple(update.client_id for update in updates),
                accepted_clients=(),
                clipped_clients=(),
                rejected_clients=tuple(update.client_id for update in updates),
                final_weights={update.client_id: 0.0 for update in updates},
                validation_errors={},
                aggregation_performed=False,
            ),
        )


class FederatedCoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec("toy", kwargs={"width": 1})
        self.initial = GlobalSnapshot(
            state={"weight": torch.tensor([0.0])},
            round_index=0,
            dev_metrics={},
            model_spec=self.spec,
            partition_hash="partition",
        )

    def test_schedule_is_deterministic(self):
        first = build_client_schedule_count(["c", "a", "b"], 3, 2, seed=4)
        second = build_client_schedule_count(["b", "c", "a"], 3, 2, seed=4)
        self.assertEqual(first, second)

    def test_rounds_use_delta_and_dev_selection(self):
        coordinator = FedAvgCoordinator(
            client_trainer=FakeTrainer(),
            aggregator=WeightedMean(),
            model_spec=self.spec,
            partition_hash="partition",
        )
        bundles = {"a": Bundle(1.0), "b": Bundle(3.0)}
        result = coordinator.train(
            initial_snapshot=self.initial,
            schedule=[("a", "b"), ("a", "b")],
            data_resolver=lambda client_id, snapshot: bundles[client_id],
            evaluate_dev=lambda snapshot: {
                "acc": float(snapshot.state["weight"].item())
            },
            convergence=ConvergencePolicy(metric="acc", mode="max"),
        )
        self.assertEqual(result.best_snapshot.round_index, 2)
        self.assertAlmostEqual(float(result.final_snapshot.state["weight"]), 4.0)
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[0].base_snapshot_hash, self.initial.content_hash)

    def test_rejecting_every_update_keeps_the_global_state(self):
        coordinator = FedAvgCoordinator(
            client_trainer=FakeTrainer(),
            aggregator=WeightedMean(),
            model_spec=self.spec,
            partition_hash="partition",
            defense_pipeline=RejectAllPipeline(),
        )

        next_snapshot, _, processed, aggregation = coordinator.run_round(
            self.initial,
            ("a",),
            lambda client_id, snapshot: Bundle(5.0),
        )

        self.assertEqual(processed, [])
        self.assertTrue(
            torch.equal(next_snapshot.state["weight"], self.initial.state["weight"])
        )
        self.assertEqual(aggregation.diagnostics["rejected_clients"], ("a",))
        self.assertFalse(aggregation.diagnostics["aggregation_performed"])

    def test_client_cannot_mutate_the_broadcast_snapshot(self):
        coordinator = FedAvgCoordinator(
            client_trainer=MutatingTrainer(),
            aggregator=WeightedMean(),
            model_spec=self.spec,
            partition_hash="partition",
        )

        with self.assertRaisesRegex(RuntimeError, "read-only global snapshot"):
            coordinator.run_round(
                self.initial,
                ("a",),
                lambda client_id, snapshot: Bundle(1.0),
            )
        self.assertEqual(float(self.initial.state["weight"]), 0.0)

    def test_protocol_validation_runs_when_defense_is_disabled(self):
        coordinator = FedAvgCoordinator(
            client_trainer=StaleTrainer(),
            aggregator=WeightedMean(),
            model_spec=self.spec,
            partition_hash="partition",
        )

        next_snapshot, _, processed, aggregation = coordinator.run_round(
            self.initial,
            ("a",),
            lambda client_id, snapshot: Bundle(5.0),
        )

        self.assertEqual(processed, [])
        self.assertEqual(float(next_snapshot.state["weight"]), 0.0)
        self.assertEqual(aggregation.decisions[0].action, "reject")
        self.assertIn(
            "base snapshot hash mismatch",
            aggregation.decisions[0].reason,
        )


if __name__ == "__main__":
    unittest.main()
