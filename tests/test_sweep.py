import tempfile
import unittest
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from mflpoison.runner.sweep import (
    _screen_summary,
    _validate_comparison_invariants,
    execute_sweep_runs,
    resolve_sweep_runs,
    validate_execution_gate,
    validate_execution_selection,
)
from mflpoison.runner.scenario import ScenarioRunner


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "sweeps" / "ucf101_poison_strength.yaml"


class SweepPlanTest(unittest.TestCase):
    def test_plan_expands_single_factor_and_combination_matrices(self):
        runs = resolve_sweep_runs(PLAN)
        self.assertEqual(len(runs), 36)
        self.assertEqual(len({str(run.artifact_root) for run in runs}), 36)
        self.assertTrue(
            all(run.config.selected_branches == ("clean", "attack") for run in runs)
        )
        self.assertTrue(all(not run.config.defense.enabled for run in runs))
        self.assertTrue(all(run.screening["min_asr_pct"] == 60.0 for run in runs))
        self.assertTrue(
            all(
                (
                    run.config.attack.condition_class,
                    run.config.attack.assigned_train_label,
                    run.config.attack.victim_eval_class,
                    run.config.attack.goal_prediction_class,
                )
                == (0, 1, 0, 1)
                for run in runs
            )
        )

    def test_seed_42_single_factor_matches_requested_nested_client_sets(self):
        runs = resolve_sweep_runs(
            PLAN,
            stages=("single_factor",),
            seeds=(42,),
        )
        self.assertEqual(len(runs), 7)
        by_name = {run.experiment: run for run in runs}
        self.assertEqual(by_name["B0"].config.attack.malicious_clients, ("1",))
        self.assertIsNone(by_name["B0"].m_star_source_path)
        self.assertEqual(
            by_name["M2"].m_star_source_path,
            Path(
                "artifacts/ucf101_poison_strength/single_factor/B0/seed-42/"
                "snapshots/m_star.pt"
            ),
        )
        self.assertEqual(by_name["M2"].config.attack.malicious_clients, ("0", "1"))
        self.assertEqual(
            by_name["M3"].config.attack.malicious_clients,
            ("0", "1", "4"),
        )
        self.assertEqual(by_name["P50"].config.attack.poison_ratio, 0.5)
        self.assertEqual(by_name["P100"].config.attack.poison_ratio, 1.0)
        self.assertEqual(by_name["E20"].config.generator.epochs, 20)
        self.assertEqual(by_name["E50"].config.generator.epochs, 50)
        self.assertEqual(
            len({run.pretrain_input_hash for run in runs}),
            1,
        )

    def test_execute_refuses_an_existing_artifact_root(self):
        run = resolve_sweep_runs(
            PLAN,
            stages=("single_factor",),
            experiments=("B0",),
            seeds=(42,),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            replaced = type(run)(
                stage=run.stage,
                experiment=run.experiment,
                seed=run.seed,
                config=run.config,
                artifact_root=existing,
                pretrain_input_hash=run.pretrain_input_hash,
                screening=run.screening,
            )
            with patch(
                "mflpoison.runner.sweep.validate_execution_gate",
                return_value="approved",
            ):
                with self.assertRaisesRegex(
                    FileExistsError, "no resolved config"
                ):
                    execute_sweep_runs(
                        (replaced,), approved_commit="approved"
                    )

    def test_execution_gate_requires_approved_clean_head(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "sweep@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Sweep Test"],
                cwd=repository,
                check=True,
            )
            tracked = repository / "tracked.txt"
            tracked.write_text("approved\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "approved"],
                cwd=repository,
                check=True,
            )
            approved = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip()

            self.assertEqual(
                validate_execution_gate(
                    approved, repository_root=repository
                ),
                approved,
            )
            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "must be clean"):
                validate_execution_gate(approved, repository_root=repository)
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "next"],
                cwd=repository,
                check=True,
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_execution_gate(approved, repository_root=repository)

    def test_default_execution_scope_requires_explicit_single_stage_seed(self):
        validate_execution_selection(
            stages=("single_factor",),
            experiments=("B0", "P50"),
            seeds=(42,),
            allow_full_matrix=False,
        )
        with self.assertRaisesRegex(ValueError, "exactly one --stage"):
            validate_execution_selection(
                stages=(),
                experiments=(),
                seeds=(),
                allow_full_matrix=False,
            )
        validate_execution_selection(
            stages=(),
            experiments=(),
            seeds=(),
            allow_full_matrix=True,
        )

    def test_interrupted_root_requires_resume_and_resume_path_is_forwarded(self):
        run = resolve_sweep_runs(
            PLAN,
            stages=("single_factor",),
            experiments=("B0",),
            seeds=(42,),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            replaced = type(run)(
                stage=run.stage,
                experiment=run.experiment,
                seed=run.seed,
                config=run.config,
                artifact_root=root,
                pretrain_input_hash=run.pretrain_input_hash,
                screening=run.screening,
            )
            with patch(
                "mflpoison.runner.sweep.validate_execution_gate",
                return_value="approved",
            ), patch(
                "mflpoison.runner.sweep._load_existing_state",
                return_value={"phase": "branch:attack"},
            ):
                with self.assertRaisesRegex(RuntimeError, "requires --resume"):
                    execute_sweep_runs(
                        (replaced,), approved_commit="approved"
                    )

            result = SimpleNamespace(
                m_star=object(),
                branch_schedule=(),
                branches={},
                summary_path=root / "summary.json",
            )
            runner = SimpleNamespace(run=lambda: result)
            with patch(
                "mflpoison.runner.sweep.validate_execution_gate",
                return_value="approved",
            ), patch(
                "mflpoison.runner.sweep._load_existing_state",
                return_value={"phase": "branch:attack"},
            ), patch(
                "mflpoison.runner.sweep.build_default_runner",
                return_value=runner,
            ) as build, patch(
                "mflpoison.runner.sweep._run_provenance",
                return_value={},
            ), patch(
                "mflpoison.runner.sweep._build_run_payload",
                return_value={"run_id": replaced.run_id},
            ):
                payload = execute_sweep_runs(
                    (replaced,), approved_commit="approved", resume=True
                )
            resumed_config = build.call_args.args[0]
            self.assertEqual(
                resumed_config.federation.resume_from,
                str(root / "resume_state.pt"),
            )
            self.assertEqual(payload[0]["execution_status"], "resumed")

    def test_completed_run_is_skipped_without_rebuilding_runner(self):
        run = resolve_sweep_runs(
            PLAN,
            stages=("single_factor",),
            experiments=("B0",),
            seeds=(42,),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            replaced = type(run)(
                stage=run.stage,
                experiment=run.experiment,
                seed=run.seed,
                config=run.config,
                artifact_root=root,
                pretrain_input_hash=run.pretrain_input_hash,
                screening=run.screening,
            )
            completed = {"run_id": replaced.run_id}
            with patch(
                "mflpoison.runner.sweep.validate_execution_gate",
                return_value="approved",
            ), patch(
                "mflpoison.runner.sweep._load_existing_state",
                return_value={"phase": "complete"},
            ), patch(
                "mflpoison.runner.sweep._completed_run_payload",
                return_value=completed,
            ), patch(
                "mflpoison.runner.sweep.build_default_runner"
            ) as build:
                payload = execute_sweep_runs(
                    (replaced,), approved_commit="approved"
                )

            build.assert_not_called()
            self.assertEqual(payload[0]["execution_status"], "skipped_completed")

    def test_pairing_invariants_reject_clean_or_generator_drift(self):
        run = resolve_sweep_runs(
            PLAN,
            stages=("single_factor",),
            experiments=("P50",),
            seeds=(42,),
        )[0]
        source = {
            "run_id": "single_factor/B0/seed-42",
            "clean_final_snapshot_hash": "clean",
            "generator_epochs": 5,
            "generator_checkpoint_hashes": {"1": "generator"},
        }
        provenance = {
            "clean_final_snapshot_hash": "clean",
            "generator_epochs": 5,
            "generator_checkpoint_hashes": {"1": "generator"},
        }
        checks = _validate_comparison_invariants(run, provenance, source)
        self.assertTrue(checks["clean_final_snapshot_matches_baseline"])
        self.assertEqual(
            checks["generator_checkpoint_matches_by_client_epoch"],
            {"1": True},
        )
        with self.assertRaisesRegex(RuntimeError, "clean final snapshot drift"):
            _validate_comparison_invariants(
                run,
                {**provenance, "clean_final_snapshot_hash": "drift"},
                source,
            )
        with self.assertRaisesRegex(RuntimeError, "generator checkpoint drift"):
            _validate_comparison_invariants(
                run,
                {
                    **provenance,
                    "generator_checkpoint_hashes": {"1": "drift"},
                },
                source,
            )

    def test_screening_thresholds_live_in_sweep_analysis(self):
        run = resolve_sweep_runs(
            PLAN,
            stages=("single_factor",),
            experiments=("P100",),
            seeds=(42,),
        )[0]
        screen = _screen_summary(
            run,
            {
                "branches": {
                    "attack": {
                        "test_metrics": {"attack_success_rate_pct": 70.0},
                        "delta_asr_percentage_points": 50.0,
                        "clean_utility_drops": {
                            "acc": 11.0,
                            "non_source_accuracy": 2.0,
                        },
                    }
                }
            },
        )
        self.assertEqual(
            screen["classification"], "availability_or_model_collapse"
        )
        self.assertEqual(screen["thresholds"]["min_asr_pct"], 60.0)

    def test_seed_42_schedule_matches_documented_malicious_exposure(self):
        schedule = ScenarioRunner._schedule(
            tuple(str(index) for index in range(10)),
            rounds=20,
            clients_per_round=5,
            seed=43,
        )
        expected = {
            ("1",): (9, 9),
            ("0", "1"): (17, 22),
            ("0", "1", "4"): (19, 32),
        }
        for malicious_clients, (round_count, seat_count) in expected.items():
            malicious = set(malicious_clients)
            actual_rounds = sum(
                any(client_id in malicious for client_id in row) for row in schedule
            )
            actual_seats = sum(
                client_id in malicious for row in schedule for client_id in row
            )
            self.assertEqual((actual_rounds, actual_seats), (round_count, seat_count))


if __name__ == "__main__":
    unittest.main()
