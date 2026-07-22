import math
import unittest

import torch

from mflpoison.core.types import ClientUpdate, GlobalSnapshot, ModelSpec
from mflpoison.defenses import (
    CompositeDecisionPolicy,
    CosineMADDetector,
    DefensePipeline,
    NormMADDetector,
    UpdateValidationError,
    UpdateValidator,
)
from mflpoison.defenses.robust_aggregation import (
    CoordinateMedian,
    Krum,
    TrimmedMean,
    WeightedMean,
)
from mflpoison.defenses.update_filter import NormClipper
from mflpoison.evaluation import detection_metrics


BASE_HASH = "snapshot-1"


def delta_update(client_id, vector, *, weight=1.0, base_hash=BASE_HASH):
    return ClientUpdate(
        client_id=client_id,
        delta={
            "weight": torch.tensor(vector, dtype=torch.float32),
            "counter": torch.tensor(0, dtype=torch.int64),
        },
        round_index=1,
        base_snapshot_hash=base_hash,
        clean_num_samples=max(1, int(weight)),
        train_num_samples=max(1, int(weight)),
        aggregation_weight=float(weight),
    )


class DefensePipelineTest(unittest.TestCase):
    def setUp(self):
        self.global_state = {
            "weight": torch.zeros(2, dtype=torch.float32),
            "counter": torch.tensor(3, dtype=torch.int64),
        }

    def test_validator_checks_provenance_schema_dtype_and_finite_values(self):
        validator = UpdateValidator()
        validator.validate(
            delta_update("good", [1.0, 0.0]),
            self.global_state,
            expected_base_snapshot_hash=BASE_HASH,
        )
        with self.assertRaises(UpdateValidationError):
            validator.validate(
                delta_update("bad-hash", [1.0, 0.0], base_hash="other"),
                self.global_state,
                expected_base_snapshot_hash=BASE_HASH,
            )
        wrong_shape = delta_update("shape", [1.0, 0.0])
        wrong_shape.delta["weight"] = torch.ones(3, dtype=torch.float32)
        with self.assertRaises(UpdateValidationError):
            validator.validate(wrong_shape, self.global_state, BASE_HASH)
        wrong_dtype = delta_update("dtype", [1.0, 0.0])
        wrong_dtype.delta["weight"] = torch.ones(2, dtype=torch.float64)
        with self.assertRaises(UpdateValidationError):
            validator.validate(wrong_dtype, self.global_state, BASE_HASH)
        nonfinite = delta_update("nan", [0.0, 0.0])
        nonfinite.delta["weight"][0] = math.nan
        with self.assertRaises(UpdateValidationError):
            validator.validate(nonfinite, self.global_state, BASE_HASH)
        missing = delta_update("missing", [1.0, 0.0])
        missing.delta.pop("counter")
        with self.assertRaises(UpdateValidationError):
            validator.validate(missing, self.global_state, BASE_HASH)

        snapshot = GlobalSnapshot(
            state=self.global_state,
            round_index=1,
            dev_metrics={},
            model_spec=ModelSpec("toy"),
            partition_hash="partition",
        )
        wrong_round = delta_update("wrong-round", [1.0, 0.0])
        wrong_round.round_index = 999
        wrong_round.base_snapshot_hash = snapshot.content_hash
        with self.assertRaisesRegex(UpdateValidationError, "round index mismatch"):
            validator.validate(wrong_round, snapshot)

    def test_two_detector_anomalies_reject_without_oracle_labels(self):
        updates = [
            delta_update("a", [1.00, 0.00]),
            delta_update("b", [1.05, 0.01]),
            delta_update("c", [0.95, -0.01]),
            delta_update("d", [1.02, 0.00]),
            delta_update("malicious", [-100.0, 100.0]),
        ]
        result = DefensePipeline(
            aggregator=WeightedMean(),
            sanitizer=NormClipper(max_norm=None),
        ).process(
            updates,
            self.global_state,
            expected_base_snapshot_hash=BASE_HASH,
            aggregator=WeightedMean(),
        )
        decisions = {item.client_id: item for item in result.decisions}
        self.assertEqual(decisions["malicious"].action, "reject")
        self.assertEqual(decisions["malicious"].final_weight, 0.0)
        self.assertNotIn(
            "malicious", [item.client_id for item in result.sanitized]
        )
        self.assertEqual(result.aggregation_audit.rejected_clients, ("malicious",))
        self.assertEqual(result.aggregation_audit.aggregator, "weighted_mean")
        self.assertIsNotNone(result.aggregated_state)

    def test_one_detector_anomaly_is_clipped_and_audited(self):
        updates = [
            delta_update("a", [1.0, 0.0]),
            delta_update("b", [1.1, 0.0]),
            delta_update("c", [0.9, 0.0]),
            delta_update("large", [10.0, 0.0]),
        ]
        pipeline = DefensePipeline(
            detectors=[NormMADDetector(threshold=3.5)],
            decision_policy=CompositeDecisionPolicy(),
            sanitizer=NormClipper(max_norm=None),
        )
        result = pipeline.process(
            updates,
            self.global_state,
            expected_base_snapshot_hash=BASE_HASH,
        )
        decisions = {item.client_id: item for item in result.decisions}
        self.assertEqual(decisions["large"].action, "clip")
        clipped = next(item for item in result.sanitized if item.client_id == "large")
        self.assertLess(float(torch.linalg.vector_norm(clipped.delta["weight"])), 2.0)
        self.assertLess(clipped.metrics["defense_clip_scale"], 1.0)
        self.assertEqual(result.aggregation_audit.clipped_clients, ("large",))

    def test_pipeline_rejects_invalid_update_and_records_reason(self):
        updates = [
            delta_update("good", [1.0, 0.0]),
            delta_update("bad", [1.0, 0.0], base_hash="stale"),
        ]
        result = DefensePipeline(detectors=[]).process(
            updates,
            self.global_state,
            expected_base_snapshot_hash=BASE_HASH,
        )
        decisions = {item.client_id: item for item in result.decisions}
        self.assertEqual(decisions["bad"].action, "reject")
        self.assertIn("base snapshot hash mismatch", decisions["bad"].reason)
        self.assertIn("bad", result.aggregation_audit.validation_errors)

    def test_explicit_delta_weighted_mean_adds_delta_to_global_state(self):
        base = {
            "weight": torch.tensor([10.0, 10.0]),
            "counter": torch.tensor(7),
        }
        result = WeightedMean().aggregate(
            [
                delta_update("a", [1.0, 1.0], weight=1),
                delta_update("b", [3.0, 3.0], weight=3),
            ],
            base,
        )
        self.assertTrue(torch.allclose(result["weight"], torch.tensor([12.5, 12.5])))
        self.assertEqual(int(result["counter"]), 7)

    def test_robust_aggregators_accept_explicit_deltas(self):
        base = {
            "weight": torch.tensor([10.0, 10.0]),
            "counter": torch.tensor(7),
        }
        updates = [
            delta_update("a", [0.9, 0.9]),
            delta_update("b", [1.0, 1.0]),
            delta_update("c", [1.1, 1.1]),
            delta_update("d", [1.2, 1.2]),
            delta_update("outlier", [100.0, 100.0]),
        ]
        median = CoordinateMedian().aggregate(updates, base)
        trimmed = TrimmedMean(trim_ratio=0.2).aggregate(updates, base)
        krum = Krum(byzantine_clients=1).aggregate(updates, base)
        self.assertTrue(torch.allclose(median["weight"], torch.tensor([11.1, 11.1])))
        self.assertTrue(torch.allclose(trimmed["weight"], torch.tensor([11.1, 11.1])))
        self.assertLess(float(krum["weight"].max()), 12.0)
        self.assertEqual(int(krum["counter"]), 7)

    def test_detection_metrics_include_tie_aware_auroc(self):
        metrics = detection_metrics(
            labels=[0, 0, 1, 1],
            scores=[0.1, 0.8, 0.9, 0.2],
            threshold=0.5,
        )
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["fpr"], 0.5)
        self.assertAlmostEqual(metrics["fnr"], 0.5)
        self.assertAlmostEqual(metrics["auroc"], 0.75)


if __name__ == "__main__":
    unittest.main()
