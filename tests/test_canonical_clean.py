import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mflpoison.artifacts import save_snapshot
from mflpoison.core.types import GlobalSnapshot, ModelSpec
from mflpoison.federated import TrainingResult
from mflpoison.runner.canonical_clean import (
    build_canonical_clean,
    canonical_comparison_protocol,
    load_canonical_clean,
    validate_canonical_clean,
    write_canonical_clean,
)
from mflpoison.runner.persistence import ResultStore


class CanonicalCleanTest(unittest.TestCase):
    def _comparison_config(self):
        return {
            "dataset": {"name": "toy", "partition_id": "partition-one"},
            "model": {"name": "toy", "constructor": None, "kwargs": {}},
            "federation": {
                "attack_rounds": 2,
                "clients_per_round": 2,
                "local_epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.1,
                "options": {"device": "cpu", "mu": 0.0},
            },
            "evaluation": {
                "metrics": ["accuracy", "attack_success_rate"],
                "evaluate_test": True,
                "evaluate_attack": True,
                "canonical_clean_path": None,
                "options": {},
            },
            "attack": {
                "victim_eval_class": 0,
                "goal_prediction_class": 1,
            },
        }

    def _snapshot(self, root):
        snapshot = GlobalSnapshot(
            state={"weight": torch.tensor([1.0])},
            round_index=3,
            dev_metrics={"acc": 0.5},
            model_spec=ModelSpec(name="toy"),
            partition_hash="partition-one",
            metadata={"phase": "m_star"},
        )
        run_dir = Path(root) / "mstar-run"
        path = run_dir / "checkpoints" / "m_star.pt"
        save_snapshot(snapshot, path)
        config = self._comparison_config()
        config["federation"].update(
            {"m_star_only": True, "m_star_path": None, "branches": []}
        )
        manifest = {
            "experiment_id": run_dir.name,
            "status": "completed",
            "seed": 42,
            "extra": {
                "partition_hash": "partition-one",
                "selected_branches": [],
            },
            "results": {
                "m_star_hash": snapshot.content_hash,
                "summary_path": str((run_dir / "summary.json").resolve()),
            },
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_tree_hash": "b" * 64,
            "config": config,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return snapshot, path.resolve()

    def _clean_run(self, root, index, snapshot, m_star_path, successes):
        run_dir = Path(root) / f"clean-{index}"
        run_dir.mkdir()
        summary = {
            "schema_version": 3,
            "m_star": {
                "snapshot_hash": snapshot.content_hash,
                "reused": True,
                "source_path": str(m_star_path),
            },
            "selected_branches": ["clean"],
            "branch_schedule": [["a", "b"], ["b", "a"]],
            "branches": {
                "clean": {
                    "final_snapshot_hash": snapshot.content_hash,
                    "test_metrics": {
                        "attack_success_count": float(successes),
                        "attack_source_sample_count": 4.0,
                        "attack_success_rate": successes / 4.0,
                        "attack_success_rate_pct": successes / 4.0 * 100.0,
                    },
                }
            },
        }
        config = copy.deepcopy(self._comparison_config())
        config["federation"].update(
            {
                "m_star_only": False,
                "m_star_path": str(m_star_path),
                "branches": ["clean"],
            }
        )
        manifest = {
            "experiment_id": run_dir.name,
            "status": "completed",
            "seed": 42,
            "extra": {
                "partition_hash": "partition-one",
                "selected_branches": ["clean"],
                "m_star_source": {"path": str(m_star_path)},
            },
            "results": {
                "m_star_hash": snapshot.content_hash,
                "summary_path": str((run_dir / "summary.json").resolve()),
            },
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_tree_hash": "b" * 64,
            "config": config,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return run_dir / "summary.json"

    def test_builds_five_run_baseline_and_attack_only_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, m_star_path = self._snapshot(root)
            summaries = [
                self._clean_run(root, index, snapshot, m_star_path, successes)
                for index, successes in enumerate((0, 1, 2, 3, 4), start=1)
            ]
            payload = build_canonical_clean(
                summaries,
                seed=42,
                m_star_path=m_star_path,
            )
            self.assertEqual(payload["replica_count"], 5)
            self.assertEqual(
                [item["attack_success_count"] for item in payload["replicas"]],
                [0, 1, 2, 3, 4],
            )
            self.assertAlmostEqual(payload["asr_canonical_clean"], 0.5)
            self.assertEqual(payload["m_star"]["experiment_id"], "mstar-run")
            baseline_path = write_canonical_clean(
                payload, root / "canonical_clean_seed-42.json"
            )
            loaded = load_canonical_clean(baseline_path)
            validate_canonical_clean(
                loaded,
                seed=42,
                partition_hash="partition-one",
                m_star_hash=snapshot.content_hash,
                branch_schedule=(("a", "b"), ("b", "a")),
                victim_eval_class=0,
                goal_prediction_class=1,
                attack_source_sample_count=4,
                comparison_protocol=self._comparison_config(),
                source_identity={
                    "git_commit": "a" * 40,
                    "git_dirty": False,
                    "source_tree_hash": "b" * 64,
                },
            )
            runtime_policy_config = self._comparison_config()
            runtime_policy_config["evaluation"][
                "canonical_source_policy"
            ] = "approved_reuse"
            self.assertEqual(
                canonical_comparison_protocol(runtime_policy_config),
                canonical_comparison_protocol(self._comparison_config()),
            )

            reused_source_identity = {
                "git_commit": "c" * 40,
                "git_dirty": False,
                "source_tree_hash": "d" * 64,
            }
            validation_args = {
                "seed": 42,
                "partition_hash": "partition-one",
                "m_star_hash": snapshot.content_hash,
                "branch_schedule": (("a", "b"), ("b", "a")),
                "victim_eval_class": 0,
                "goal_prediction_class": 1,
                "attack_source_sample_count": 4,
                "comparison_protocol": runtime_policy_config,
                "source_identity": reused_source_identity,
            }
            with self.assertRaisesRegex(ValueError, "source identity"):
                validate_canonical_clean(loaded, **validation_args)
            validate_canonical_clean(
                loaded,
                source_policy="approved_reuse",
                **validation_args,
            )
            dirty_validation_args = dict(validation_args)
            dirty_validation_args["source_identity"] = {
                **reused_source_identity,
                "git_dirty": True,
            }
            with self.assertRaisesRegex(ValueError, "requires clean source identities"):
                validate_canonical_clean(
                    loaded,
                    source_policy="approved_reuse",
                    **dirty_validation_args,
                )
            dirty_baseline = copy.deepcopy(loaded)
            dirty_baseline["source_identity"]["git_dirty"] = True
            dirty_baseline["m_star"]["source_identity"]["git_dirty"] = True
            with self.assertRaisesRegex(ValueError, "requires clean source identities"):
                validate_canonical_clean(
                    dirty_baseline,
                    source_policy="approved_reuse",
                    **validation_args,
                )
            with self.assertRaisesRegex(ValueError, "source policy"):
                validate_canonical_clean(
                    loaded,
                    source_policy="unsafe",
                    **validation_args,
                )
            mismatched_config = self._comparison_config()
            mismatched_config["federation"]["learning_rate"] = 0.2
            with self.assertRaisesRegex(ValueError, "training protocol"):
                validate_canonical_clean(
                    loaded,
                    seed=42,
                    partition_hash="partition-one",
                    m_star_hash=snapshot.content_hash,
                    branch_schedule=(("a", "b"), ("b", "a")),
                    victim_eval_class=0,
                    goal_prediction_class=1,
                    attack_source_sample_count=4,
                    comparison_protocol=mismatched_config,
                    source_identity={
                        "git_commit": "a" * 40,
                        "git_dirty": False,
                        "source_tree_hash": "b" * 64,
                    },
                )

            config = SimpleNamespace(
                federation=SimpleNamespace(m_star_path=str(m_star_path))
            )
            store = ResultStore(config, root / "attack")
            source_provenance = {
                "policy": "approved_reuse",
                "baseline_identity": dict(loaded["source_identity"]),
                "m_star_identity": dict(loaded["m_star"]["source_identity"]),
                "current_identity": reused_source_identity,
                "exact_match": False,
            }
            store.set_canonical_clean(
                loaded,
                source_provenance=source_provenance,
            )
            training = TrainingResult(
                best_snapshot=snapshot,
                final_snapshot=snapshot,
                records=[],
                stopped_early=False,
            )
            attack = SimpleNamespace(
                final_snapshot=snapshot,
                test_metrics={
                    "attack_success_count": 3.0,
                    "attack_source_sample_count": 4.0,
                    "attack_success_rate": 0.75,
                },
                generator_artifacts={},
                detection_metrics={},
                training=SimpleNamespace(records=[]),
            )
            summary_path = store.persist_summary(
                snapshot,
                training,
                snapshot,
                {},
                (),
                (("a", "b"), ("b", "a")),
                (),
                {"attack": attack},
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], 3)
            self.assertEqual(summary["branches"]["attack"]["delta_baseline"], "canonical_clean")
            self.assertAlmostEqual(
                summary["branches"]["attack"]["delta_attack_success_rate"],
                0.25,
            )
            self.assertAlmostEqual(
                summary["branches"]["attack"]["delta_asr_percentage_points"],
                25.0,
            )
            self.assertEqual(
                summary["canonical_clean"]["m_star_experiment_id"],
                "mstar-run",
            )
            self.assertEqual(
                summary["canonical_clean"]["source_provenance"],
                source_provenance,
            )

            attack.test_metrics["attack_success_rate"] = 0.5
            with self.assertRaisesRegex(ValueError, "count and rate"):
                store.persist_summary(
                    snapshot,
                    training,
                    snapshot,
                    {},
                    (),
                    (("a", "b"), ("b", "a")),
                    (),
                    {"attack": attack},
                )

            tampered = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            tampered["asr_canonical_clean"] = 0.1
            baseline_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mean ASR"):
                load_canonical_clean(baseline_path)

    def test_rejects_incomplete_duplicate_and_mismatched_replicas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, m_star_path = self._snapshot(root)
            summaries = [
                self._clean_run(root, index, snapshot, m_star_path, index - 1)
                for index in range(1, 6)
            ]
            with self.assertRaisesRegex(ValueError, "requires 5"):
                build_canonical_clean(
                    summaries[:4], seed=42, m_star_path=m_star_path
                )
            with self.assertRaisesRegex(ValueError, "must be unique"):
                build_canonical_clean(
                    [summaries[0]] * 5, seed=42, m_star_path=m_star_path
                )

            manifest_path = summaries[-1].parent / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["experiment_id"] = summaries[0].parent.name
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "experiment identity"):
                build_canonical_clean(
                    summaries, seed=42, m_star_path=m_star_path
                )
            manifest["experiment_id"] = summaries[-1].parent.name
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            mismatched = json.loads(summaries[-1].read_text(encoding="utf-8"))
            mismatched["branch_schedule"] = [["a"]]
            summaries[-1].write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "branch_schedule"):
                build_canonical_clean(
                    summaries, seed=42, m_star_path=m_star_path
                )

    def test_rejects_mismatched_training_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, m_star_path = self._snapshot(root)
            summaries = [
                self._clean_run(root, index, snapshot, m_star_path, index - 1)
                for index in range(1, 6)
            ]
            baseline = build_canonical_clean(
                summaries, seed=42, m_star_path=m_star_path
            )
            baseline_path = write_canonical_clean(
                baseline, root / "canonical_clean_seed-42.json"
            )
            manifest_path = summaries[-1].parent / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["config"]["federation"]["learning_rate"] = 0.2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "comparison_protocol"):
                build_canonical_clean(
                    summaries, seed=42, m_star_path=m_star_path
                )
            with self.assertRaisesRegex(ValueError, "comparison_protocol"):
                load_canonical_clean(baseline_path)

    def test_rejects_cross_seed_or_different_source_m_star(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, m_star_path = self._snapshot(root)
            summaries = [
                self._clean_run(root, index, snapshot, m_star_path, index - 1)
                for index in range(1, 6)
            ]
            manifest_path = m_star_path.parent.parent / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            manifest["seed"] = 43
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed"):
                build_canonical_clean(
                    summaries, seed=42, m_star_path=m_star_path
                )

            manifest["seed"] = 42
            manifest["source_tree_hash"] = "c" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source identity"):
                build_canonical_clean(
                    summaries, seed=42, m_star_path=m_star_path
                )

    def test_resolves_relative_m_star_from_training_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot, m_star_path = self._snapshot(root)
            summaries = [
                self._clean_run(root, index, snapshot, m_star_path, index - 1)
                for index in range(1, 6)
            ]
            relative_m_star = str(m_star_path.relative_to(root))
            for summary_path in summaries:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["m_star"]["source_path"] = relative_m_star
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                manifest_path = summary_path.parent / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["runtime"] = {"working_directory": str(root)}
                manifest["config"]["federation"]["m_star_path"] = relative_m_star
                manifest["extra"]["m_star_source"]["path"] = relative_m_star
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            payload = build_canonical_clean(
                summaries, seed=42, m_star_path=m_star_path
            )
            self.assertEqual(payload["m_star"]["path"], str(m_star_path))


if __name__ == "__main__":
    unittest.main()
