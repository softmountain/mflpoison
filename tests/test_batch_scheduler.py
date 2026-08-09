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
            import json
            import os
            import sys
            import time
            from pathlib import Path

            args = sys.argv[1:]
            if args and args[0] == "-c":
                manifest = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
                raise SystemExit(0 if manifest.get("status") == "completed" else 1)
            module = args[args.index("-m") + 1]
            if module.endswith("canonical_clean"):
                output = Path(args[args.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"kind": "canonical_clean"}), encoding="utf-8")
                raise SystemExit(0)

            run_dir = Path(args[args.index("--run-dir") + 1])
            if "--m-star-only" in args:
                stage = "mstar"
            elif "--branch" in args:
                stage = args[args.index("--branch") + 1]
            else:
                stage = "scenario"
            gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            events = Path(os.environ["SCHEDULER_TEST_EVENTS"])
            with events.open("a", encoding="utf-8") as handle:
                handle.write(f"start\t{stage}\t{gpu}\t{run_dir.name}\n")
            time.sleep(float(os.environ.get("SCHEDULER_TEST_SLEEP", "0.12")))
            if os.environ.get("SCHEDULER_TEST_FAIL_JOB") in run_dir.name and os.environ.get("SCHEDULER_TEST_FAIL_JOB"):
                with events.open("a", encoding="utf-8") as handle:
                    handle.write(f"end\t{stage}\t{gpu}\t{run_dir.name}\n")
                raise SystemExit(7)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"status": os.environ.get("SCHEDULER_TEST_MANIFEST_STATUS", "completed")}), encoding="utf-8"
            )
            (run_dir / "summary.json").write_text("{}", encoding="utf-8")
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
            "--experiment-branch",
            experiment_branch,
            "--gpus",
            "0,1,2,3",
            "--monitor-interval",
            "0.02",
            "--canonical-clean-config",
            str(canonical),
            f"{attack_one}:42",
            f"{attack_two}:42",
        ]
        return command, env, artifact_root, events

    def _read_status(self, artifact_root):
        status_path = artifact_root / "batches" / "scheduler-test" / "status.tsv"
        with status_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def _run_scheduler(
        self,
        root,
        fail_job="",
        manifest_status="completed",
        experiment_branch="attack",
    ):
        command, env, artifact_root, events = self._prepare_scheduler(
            root,
            fail_job=fail_job,
            manifest_status=manifest_status,
            experiment_branch=experiment_branch,
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
