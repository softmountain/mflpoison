import csv
import json
import os
import signal
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_experiments.sh"


def _usable_bash():
    bash = shutil.which("bash")
    if os.name == "nt" or bash is None:
        return False
    try:
        return subprocess.run(
            [bash, "-c", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(_usable_bash(), "Bash scheduler test requires a usable Bash")
class BatchSchedulerTest(unittest.TestCase):
    def _executable(self, path, content):
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        path.chmod(0o755)
        return path

    def _prepare_scheduler(
        self,
        root,
        *,
        fail_job="",
        sleep_seconds="0.12",
        manifest_status="completed",
        lock_file=None,
        experiment_branch="attack",
        experiment_branches=None,
        reuse=False,
        reuse_m_star_path=None,
        reuse_canonical_clean=None,
        canonical_source_policy=None,
    ):
        root = Path(root)
        events = root / "events.tsv"
        fake_smi = self._executable(
            root / "nvidia-smi",
            """
            #!/usr/bin/env bash
            if [[ "$*" == *"--query-gpu=index"* ]]; then
              printf '0\n1\n2\n3\n'
            elif [[ "$*" == *"--query-compute-apps=pid"* ]]; then
              exit 0
            elif [[ "$*" == *"--query-gpu=memory.used"* ]]; then
              printf '10\n'
            else
              exit 2
            fi
            """,
        )
        fake_python = self._executable(
            root / "fake-python",
            r"""
            #!/usr/bin/env python3
            import hashlib
            import json
            import os
            import sys
            import time
            from pathlib import Path

            args = sys.argv[1:]
            if args and args[0] == "-c":
                code = args[1]
                if (
                    "mflpoison_reuse_baseline_state_check" in code
                    or "mflpoison_reuse_baseline_file_check" in code
                ):
                    metadata = json.loads(Path(args[-3]).read_text(encoding="utf-8"))
                    canonical_path = Path(args[-2])
                    m_star_path = Path(args[-1])
                    valid = (
                        hashlib.sha256(canonical_path.read_bytes()).hexdigest()
                        == metadata["canonical_clean"]["sha256"]
                        and hashlib.sha256(m_star_path.read_bytes()).hexdigest()
                        == metadata["m_star"]["sha256"]
                    )
                    raise SystemExit(0 if valid else 1)
                if "summary=json.load" in code:
                    manifest = json.loads(Path(args[2]).read_text(encoding="utf-8"))
                    summary = json.loads(Path(args[3]).read_text(encoding="utf-8"))
                    expected = tuple(item for item in args[4].split(",") if item)
                    branches = summary.get("branches", {})
                    valid = (
                        manifest.get("status") == "completed"
                        and isinstance(branches, dict)
                        and set(branches) == set(expected)
                    )
                    if valid and args[5]:
                        metadata = json.loads(
                            Path(args[5]).read_text(encoding="utf-8")
                        )
                        expected_identity = metadata["current_source_identity"]
                        manifest_identity = {
                            key: manifest.get(key)
                            for key in (
                                "git_commit",
                                "git_dirty",
                                "source_tree_hash",
                            )
                        }
                        provenance = manifest.get("extra", {}).get(
                            "canonical_clean_source", {}
                        )
                        valid = (
                            manifest_identity == expected_identity
                            and provenance.get("current_identity")
                            == expected_identity
                            and provenance.get("baseline_identity")
                            == metadata["baseline_source_identity"]
                            and provenance.get("m_star_identity")
                            == metadata["baseline_source_identity"]
                            and provenance.get("policy")
                            == metadata["source_policy"]
                            and provenance.get("exact_match")
                            == metadata["source_identity_exact_match"]
                        )
                    raise SystemExit(0 if valid else 1)
                manifest = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
                raise SystemExit(0 if manifest.get("status") == "completed" else 1)
            if args and args[0] == "-" and args[1] == "mflpoison_reuse_preflight":
                (
                    _stdin_marker,
                    _preflight_marker,
                    canonical_path,
                    m_star_path,
                    seed,
                    source_policy,
                    branch_csv,
                    config_csv,
                    metadata_path,
                    _repo_root,
                ) = args
                branches = [item for item in branch_csv.split(",") if item]
                configs = [item for item in config_csv.split(",") if item]
                baseline_identity = {
                    "git_commit": "b" * 40,
                    "git_dirty": False,
                    "source_tree_hash": "b" * 64,
                }
                current_identity = {
                    "git_commit": "c" * 40,
                    "git_dirty": False,
                    "source_tree_hash": "c" * 64,
                }
                Path(metadata_path).write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "reused_canonical_clean",
                            "seed": int(seed),
                            "source_policy": source_policy,
                            "source_identity_exact_match": False,
                            "baseline_source_identity": baseline_identity,
                            "current_source_identity": current_identity,
                            "m_star": {
                                "path": str(Path(m_star_path).resolve()),
                                "sha256": hashlib.sha256(
                                    Path(m_star_path).read_bytes()
                                ).hexdigest(),
                                "snapshot_hash": "m-star-snapshot",
                            },
                            "canonical_clean": {
                                "path": str(Path(canonical_path).resolve()),
                                "sha256": hashlib.sha256(
                                    Path(canonical_path).read_bytes()
                                ).hexdigest(),
                            },
                            "experiment_branches": branches,
                            "config_count": len(configs),
                            "configs": configs,
                        }
                    ),
                    encoding="utf-8",
                )
                if os.environ.get("SCHEDULER_TEST_MUTATE_REUSE_AFTER_PREFLIGHT"):
                    Path(canonical_path).write_text(
                        json.dumps({"kind": "changed"}),
                        encoding="utf-8",
                    )
                raise SystemExit(0)
            module = args[args.index("-m") + 1]
            if module.endswith("canonical_clean"):
                output = Path(args[args.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"kind": "canonical_clean"}), encoding="utf-8")
                raise SystemExit(0)

            run_dir = Path(args[args.index("--run-dir") + 1])
            branches = [
                args[index + 1]
                for index, argument in enumerate(args)
                if argument == "--branch"
            ]
            if "--m-star-only" in args:
                stage = "mstar"
            elif branches:
                stage = "_".join(branches)
            else:
                stage = "scenario"
            gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            events = Path(os.environ["SCHEDULER_TEST_EVENTS"])
            with events.open("a", encoding="utf-8") as handle:
                handle.write(f"start\t{stage}\t{gpu}\t{run_dir.name}\n")
            time.sleep(float(os.environ.get("SCHEDULER_TEST_SLEEP", "0.12")))
            if os.environ.get("SCHEDULER_TEST_DIRTY_REPOSITORY_DURING_JOB"):
                (Path.cwd() / "scheduler-dirty-marker").write_text(
                    "changed during job",
                    encoding="utf-8",
                )
            if os.environ.get("SCHEDULER_TEST_FAIL_JOB") in run_dir.name and os.environ.get("SCHEDULER_TEST_FAIL_JOB"):
                with events.open("a", encoding="utf-8") as handle:
                    handle.write(f"end\t{stage}\t{gpu}\t{run_dir.name}\n")
                raise SystemExit(7)
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "status": os.environ.get(
                    "SCHEDULER_TEST_MANIFEST_STATUS", "completed"
                )
            }
            if "--canonical-source-policy" in args:
                metadata_path = (
                    Path(os.environ["ARTIFACT_ROOT"])
                    / "batches"
                    / os.environ["BATCH_ID"]
                    / "reused_baseline.json"
                )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                current_identity = metadata["current_source_identity"]
                manifest.update(current_identity)
                manifest["extra"] = {
                    "canonical_clean_source": {
                        "policy": metadata["source_policy"],
                        "baseline_identity": metadata[
                            "baseline_source_identity"
                        ],
                        "m_star_identity": metadata["baseline_source_identity"],
                        "current_identity": current_identity,
                        "exact_match": metadata["source_identity_exact_match"],
                    }
                }
                if os.environ.get("SCHEDULER_TEST_INVALID_REUSE_PROVENANCE"):
                    manifest["source_tree_hash"] = "invalid-source-tree"
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (run_dir / "summary.json").write_text(
                json.dumps({"branches": {branch: {} for branch in branches}}),
                encoding="utf-8",
            )
            if stage == "mstar":
                checkpoint = run_dir / "checkpoints" / "m_star.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"m-star")
            with events.open("a", encoding="utf-8") as handle:
                handle.write(f"end\t{stage}\t{gpu}\t{run_dir.name}\n")
            """,
        )
        canonical = ROOT / "configs" / "experiments" / "ucf101_fdmm_dtm_poison_0to1.yaml"
        strength_root = ROOT / "configs" / "experiments" / "ucf101_dtm_poison_strength"
        attack_one = strength_root / "malicious-clients-1_poison-100pct_generator-epochs-5.yaml"
        attack_two = strength_root / "malicious-clients-1_poison-20pct_generator-epochs-20.yaml"
        artifact_root = root / "artifact"
        env = os.environ.copy()
        env.update(
            {
                "ARTIFACT_ROOT": str(artifact_root),
                "BATCH_ID": "scheduler-test",
                "NVIDIA_SMI_BIN": str(fake_smi),
                "PYTHON_BIN": str(fake_python),
                "SCHEDULER_TEST_EVENTS": str(events),
                "SCHEDULER_TEST_FAIL_JOB": fail_job,
                "SCHEDULER_TEST_SLEEP": str(sleep_seconds),
                "SCHEDULER_TEST_MANIFEST_STATUS": manifest_status,
                "SCHEDULER_LOCK_FILE": str(lock_file or (root / "scheduler.lock")),
            }
        )
        command = [
            "bash",
            str(SCRIPT),
        ]
        if experiment_branches is None:
            command.extend(["--experiment-branch", experiment_branch])
        else:
            command.extend(["--experiment-branches", experiment_branches])
        if reuse:
            baseline_root = root / "reused-baseline"
            baseline_root.mkdir(parents=True, exist_ok=True)
            if reuse_m_star_path is None:
                reuse_m_star_path = baseline_root / "m_star.pt"
                reuse_m_star_path.write_bytes(b"reused-m-star")
            if reuse_canonical_clean is None:
                reuse_canonical_clean = baseline_root / "canonical_clean_seed-42.json"
                reuse_canonical_clean.write_text(
                    json.dumps({"kind": "canonical_clean", "seed": 42}),
                    encoding="utf-8",
                )
        if reuse_m_star_path is not None:
            command.extend(["--reuse-m-star-path", str(reuse_m_star_path)])
        if reuse_canonical_clean is not None:
            command.extend(["--reuse-canonical-clean", str(reuse_canonical_clean)])
        if canonical_source_policy is not None:
            command.extend(["--canonical-source-policy", canonical_source_policy])
        command.extend(
            [
                "--gpus",
                "0,1,2,3",
                "--monitor-interval",
                "0.02",
                "--canonical-clean-config",
                str(canonical),
                f"{attack_one}:42",
                f"{attack_two}:42",
            ]
        )
        return command, env, artifact_root, events

    def _read_status(self, artifact_root):
        status_path = artifact_root / "batches" / "scheduler-test" / "status.tsv"
        with status_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def _prepare_clean_repository(self, root, command):
        clean_repo = Path(root) / "clean-repo"
        copied_arguments = {}
        for argument in command:
            value, separator, seed = argument.rpartition(":")
            source = Path(value) if separator and seed.isdigit() else None
            if source is None or not source.is_file():
                continue
            relative = source.resolve().relative_to(ROOT)
            destination = clean_repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_arguments[argument] = f"{destination}:{seed}"
        command = [copied_arguments.get(item, item) for item in command]
        subprocess.run(["git", "init", "-q"], cwd=clean_repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "scheduler-test@example.invalid"],
            cwd=clean_repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Scheduler Test"],
            cwd=clean_repo,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=clean_repo, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "scheduler fixture"],
            cwd=clean_repo,
            check=True,
        )
        return clean_repo, command

    def _run_scheduler(
        self,
        root,
        fail_job="",
        manifest_status="completed",
        experiment_branch="attack",
        experiment_branches=None,
    ):
        command, env, artifact_root, events = self._prepare_scheduler(
            root,
            fail_job=fail_job,
            manifest_status=manifest_status,
            experiment_branch=experiment_branch,
            experiment_branches=experiment_branches,
        )
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        rows = self._read_status(artifact_root)
        event_rows = [line.split("\t") for line in events.read_text(encoding="utf-8").splitlines()]
        return completed, rows, event_rows

    def test_four_gpus_run_four_clean_replicas_before_fifth(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, rows, events = self._run_scheduler(directory)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row["status"] == "completed" for row in rows))
            self.assertEqual(sum(row["stage"] == "clean" for row in rows), 5)
            self.assertEqual(sum(row["stage"] == "attack" for row in rows), 2)

            first_clean_end = next(
                index
                for index, row in enumerate(events)
                if row[0] == "end" and row[1] == "clean"
            )
            clean_starts_before_first_end = [
                row
                for row in events[:first_clean_end]
                if row[0] == "start" and row[1] == "clean"
            ]
            self.assertEqual(len(clean_starts_before_first_end), 4)

            active_gpus = set()
            for action, _stage, gpu, _job in events:
                if action == "start":
                    self.assertNotIn(gpu, active_gpus)
                    active_gpus.add(gpu)
                else:
                    active_gpus.remove(gpu)
            self.assertEqual(active_gpus, set())

    def test_fresh_baseline_schedules_paired_attack_defended_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, rows, events = self._run_scheduler(
                directory,
                experiment_branches="attack,defended",
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertEqual(len(rows), 9)
            stage_counts = {
                stage: sum(row["stage"] == stage for row in rows)
                for stage in {
                    "mstar",
                    "clean",
                    "canonical_aggregate",
                    "attack_defended",
                }
            }
            self.assertEqual(
                stage_counts,
                {
                    "mstar": 1,
                    "clean": 5,
                    "canonical_aggregate": 1,
                    "attack_defended": 2,
                },
            )
            paired_rows = [
                row for row in rows if row["stage"] == "attack_defended"
            ]
            self.assertTrue(
                all(
                    row["depends_on"] == "canonical-aggregate-seed-42"
                    and row["status"] == "completed"
                    for row in paired_rows
                )
            )
            self.assertEqual(
                sum(
                    row[0] == "start" and row[1] == "attack_defended"
                    for row in events
                ),
                2,
            )
            for row in paired_rows:
                summary = json.loads(
                    (Path(row["run_dir"]) / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(set(summary["branches"]), {"attack", "defended"})

    def test_defended_branch_is_scheduled_and_forwarded_to_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, rows, events = self._run_scheduler(
                directory, experiment_branch="defended"
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertEqual(sum(row["stage"] == "defended" for row in rows), 2)
            self.assertFalse(any(row["stage"] == "attack" for row in rows))
            defended_rows = [row for row in rows if row["stage"] == "defended"]
            self.assertTrue(
                all(row["job_id"].startswith("defended-") for row in defended_rows)
            )
            self.assertEqual(
                sum(row[0] == "start" and row[1] == "defended" for row in events),
                2,
            )

    def test_invalid_experiment_branch_is_rejected_before_scheduling(self):
        with tempfile.TemporaryDirectory() as directory:
            command, env, artifact_root, _events = self._prepare_scheduler(
                directory, experiment_branch="unknown"
            )
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn(
                "--experiment-branch must be attack or defended",
                completed.stdout,
            )
            self.assertFalse((artifact_root / "batches").exists())

    def test_reuse_paths_must_be_supplied_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m_star_path = root / "m_star.pt"
            m_star_path.write_bytes(b"reused-m-star")
            command, env, artifact_root, _events = self._prepare_scheduler(
                root,
                reuse_m_star_path=m_star_path,
            )
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn(
                "--reuse-m-star-path and --reuse-canonical-clean must be supplied together",
                completed.stdout,
            )
            self.assertFalse((artifact_root / "batches").exists())

    def test_canonical_source_policy_requires_reuse_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            command, env, artifact_root, _events = self._prepare_scheduler(
                directory,
                canonical_source_policy="approved_reuse",
            )
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn(
                "--canonical-source-policy requires reused M* and canonical clean paths",
                completed.stdout,
            )
            self.assertFalse((artifact_root / "batches").exists())

    def test_reused_baseline_schedules_only_paired_attack_defended_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, env, artifact_root, events = self._prepare_scheduler(
                root,
                experiment_branches="attack,defended",
                reuse=True,
                canonical_source_policy="approved_reuse",
            )
            clean_repo, command = self._prepare_clean_repository(root, command)
            completed = subprocess.run(
                command,
                cwd=str(clean_repo),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)

            rows = self._read_status(artifact_root)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["stage"] == "attack_defended" for row in rows))
            self.assertTrue(all(row["status"] == "completed" for row in rows))
            self.assertTrue(all(row["depends_on"] == "" for row in rows))
            self.assertFalse(
                any(
                    row["stage"] in {"mstar", "clean", "canonical_aggregate"}
                    for row in rows
                )
            )

            event_rows = [
                line.split("\t")
                for line in events.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sum(
                    row[0] == "start" and row[1] == "attack_defended"
                    for row in event_rows
                ),
                2,
            )
            for row in rows:
                summary = json.loads(
                    (Path(row["run_dir"]) / "summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(set(summary["branches"]), {"attack", "defended"})
                manifest = json.loads(
                    (Path(row["run_dir"]) / "run_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest["status"], "completed")

            metadata_path = (
                artifact_root
                / "batches"
                / "scheduler-test"
                / "reused_baseline.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["kind"], "reused_canonical_clean")
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["source_policy"], "approved_reuse")
            self.assertEqual(metadata["experiment_branches"], ["attack", "defended"])
            self.assertEqual(metadata["config_count"], 2)

    def test_reused_baseline_change_after_preflight_aborts_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, env, artifact_root, events = self._prepare_scheduler(
                root,
                experiment_branches="attack,defended",
                reuse=True,
                canonical_source_policy="approved_reuse",
            )
            env["SCHEDULER_TEST_MUTATE_REUSE_AFTER_PREFLIGHT"] = "1"
            clean_repo, command = self._prepare_clean_repository(root, command)
            completed = subprocess.run(
                command,
                cwd=str(clean_repo),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
            self.assertEqual(completed.returncode, 125, completed.stdout)
            self.assertIn("source_or_reused_baseline_changed", completed.stdout)
            rows = self._read_status(artifact_root)
            self.assertTrue(all(row["status"] == "failed" for row in rows))
            self.assertTrue(
                all(
                    row["failure_reason"] == "source_or_reused_baseline_changed"
                    for row in rows
                )
            )
            self.assertFalse(events.exists())

    def test_repository_change_during_reused_job_aborts_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, env, artifact_root, _events = self._prepare_scheduler(
                root,
                experiment_branches="attack,defended",
                reuse=True,
                canonical_source_policy="approved_reuse",
            )
            env["SCHEDULER_TEST_DIRTY_REPOSITORY_DURING_JOB"] = "1"
            clean_repo, command = self._prepare_clean_repository(root, command)
            completed = subprocess.run(
                command,
                cwd=str(clean_repo),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
            self.assertEqual(completed.returncode, 125, completed.stdout)
            self.assertIn("source_or_reused_baseline_changed", completed.stdout)
            rows = self._read_status(artifact_root)
            self.assertTrue(all(row["status"] == "failed" for row in rows))
            self.assertTrue(
                all(
                    row["failure_reason"] == "source_or_reused_baseline_changed"
                    for row in rows
                )
            )

    def test_reused_job_manifest_must_match_preflight_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, env, artifact_root, _events = self._prepare_scheduler(
                root,
                experiment_branches="attack,defended",
                reuse=True,
                canonical_source_policy="approved_reuse",
            )
            env["SCHEDULER_TEST_INVALID_REUSE_PROVENANCE"] = "1"
            clean_repo, command = self._prepare_clean_repository(root, command)
            completed = subprocess.run(
                command,
                cwd=str(clean_repo),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            rows = self._read_status(artifact_root)
            self.assertTrue(all(row["status"] == "failed" for row in rows))
            self.assertTrue(
                all(
                    row["failure_reason"]
                    == "invalid_or_incomplete_run_manifest"
                    for row in rows
                )
            )

    def test_clean_failure_blocks_aggregate_and_attacks(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, rows, events = self._run_scheduler(
                directory, fail_job="clean-seed-42-repeat-3"
            )
            self.assertNotEqual(completed.returncode, 0)
            by_stage = {}
            for row in rows:
                by_stage.setdefault(row["stage"], []).append(row)
            self.assertEqual(
                [row["status"] for row in by_stage["clean"]].count("failed"),
                1,
            )
            self.assertEqual(by_stage["canonical_aggregate"][0]["status"], "failed")
            self.assertTrue(
                by_stage["canonical_aggregate"][0]["failure_reason"].startswith(
                    "dependency_failed:"
                )
            )
            self.assertTrue(all(row["status"] == "failed" for row in by_stage["attack"]))
            self.assertFalse(any(row[1] == "attack" for row in events))

    def test_completed_exit_requires_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, rows, events = self._run_scheduler(
                directory, manifest_status="running"
            )
            self.assertNotEqual(completed.returncode, 0)
            mstar = next(row for row in rows if row["stage"] == "mstar")
            self.assertEqual(mstar["status"], "failed")
            self.assertEqual(
                mstar["failure_reason"], "invalid_or_incomplete_run_manifest"
            )
            self.assertFalse(any(row[1] in {"clean", "attack"} for row in events))

    def test_term_stops_children_and_finalizes_status(self):
        with tempfile.TemporaryDirectory() as directory:
            command, env, artifact_root, _events = self._prepare_scheduler(
                directory, sleep_seconds="10"
            )
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            try:
                for _ in range(200):
                    try:
                        rows = self._read_status(artifact_root)
                    except FileNotFoundError:
                        rows = []
                    if any(row["status"] == "running" for row in rows):
                        break
                    time.sleep(0.02)
                else:
                    self.fail("scheduler did not start a child")
                process.send_signal(signal.SIGTERM)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 143, output)
                rows = self._read_status(artifact_root)
                self.assertTrue(
                    all(row["status"] in {"completed", "failed"} for row in rows)
                )
                interrupted = [
                    row
                    for row in rows
                    if row["failure_reason"] == "scheduler_interrupted:TERM"
                ]
                self.assertTrue(interrupted)
                for row in interrupted:
                    if row["pid"]:
                        with self.assertRaises(ProcessLookupError):
                            os.kill(int(row["pid"]), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_host_lock_blocks_scheduler_with_other_artifact_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_lock = root / "host.lock"
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            command_one, env_one, artifact_one, _ = self._prepare_scheduler(
                first, sleep_seconds="10", lock_file=shared_lock
            )
            command_two, env_two, _artifact_two, _ = self._prepare_scheduler(
                second, sleep_seconds="0.01", lock_file=shared_lock
            )
            process = subprocess.Popen(
                command_one,
                cwd=str(ROOT),
                env=env_one,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            try:
                for _ in range(200):
                    try:
                        rows = self._read_status(artifact_one)
                    except FileNotFoundError:
                        rows = []
                    if any(row["status"] == "running" for row in rows):
                        break
                    time.sleep(0.02)
                else:
                    self.fail("first scheduler did not acquire the lock")
                second_result = subprocess.run(
                    command_two,
                    cwd=str(ROOT),
                    env=env_two,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=5,
                )
                self.assertEqual(second_result.returncode, 2, second_result.stdout)
                self.assertIn("another experiment scheduler", second_result.stdout)
            finally:
                process.send_signal(signal.SIGTERM)
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
