"""Build and validate a five-run canonical clean ASR baseline."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from mflpoison.artifacts import load_snapshot


SCHEMA_VERSION = 1
CANONICAL_REPLICA_COUNT = 5


def canonical_comparison_protocol(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the clean/attack fields that must match for a valid Delta ASR."""

    if not isinstance(config, Mapping):
        raise TypeError("comparison config must be a mapping")

    def section(name: str) -> Mapping[str, Any]:
        value = config.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"comparison config is missing {name}")
        return value

    dataset = section("dataset")
    model = section("model")
    federation = section("federation")
    evaluation = section("evaluation")
    protocol = {
        "dataset": dict(dataset),
        "model": {
            key: value
            for key, value in model.items()
            if key != "checkpoint_path"
        },
        "federation": {
            key: federation.get(key)
            for key in (
                "attack_rounds",
                "clients_per_round",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "options",
            )
        },
        "evaluation": {
            key: value
            for key, value in evaluation.items()
            if key != "canonical_clean_path"
        },
    }
    return json.loads(json.dumps(protocol, sort_keys=True))


def manifest_source_identity(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    identity = {
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "source_tree_hash": manifest.get("source_tree_hash"),
    }
    if (
        not identity["git_commit"]
        or not isinstance(identity["git_dirty"], bool)
        or not identity["source_tree_hash"]
    ):
        raise ValueError("run manifest source identity is incomplete")
    return identity


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be a mapping: {path}")
    return payload


def _integer_metric(metrics: Mapping[str, Any], name: str) -> int:
    value = float(metrics[name])
    rounded = int(round(value))
    if not math.isfinite(value) or not math.isclose(value, rounded, abs_tol=1e-9):
        raise ValueError(f"{name} must be a finite integer-valued metric")
    return rounded


def _resolved_record_path(value: Any, manifest: Mapping[str, Any]) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    runtime = manifest.get("runtime")
    if isinstance(runtime, Mapping) and runtime.get("working_directory"):
        return (Path(str(runtime["working_directory"])) / path).resolve()
    return path.resolve()


def _m_star_record(m_star_path: Path, seed: int) -> Dict[str, Any]:
    """Validate the run that produced the common M* checkpoint."""

    m_star_path = Path(m_star_path).resolve()
    run_dir = m_star_path.parent.parent
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"common M* run is not completed: {run_dir}")
    if int(manifest.get("seed")) != int(seed):
        raise ValueError("common M* run seed does not match the canonical clean seed")
    experiment_id = str(manifest.get("experiment_id", ""))
    if not experiment_id or experiment_id != run_dir.name:
        raise ValueError("common M* experiment identity does not match its run directory")

    config = manifest.get("config")
    extra = manifest.get("extra")
    results = manifest.get("results")
    if (
        not isinstance(config, Mapping)
        or not isinstance(extra, Mapping)
        or not isinstance(results, Mapping)
    ):
        raise ValueError("common M* run provenance is incomplete")
    federation = config.get("federation")
    if not isinstance(federation, Mapping) or federation.get("m_star_only") is not True:
        raise ValueError("common M* must come from an m-star-only run")
    if federation.get("m_star_path") is not None or federation.get("branches"):
        raise ValueError("common M* run must not reuse M* or select experiment branches")
    if extra.get("selected_branches") not in ([], ()):
        raise ValueError("common M* run contains unexpected experiment branches")

    snapshot = load_snapshot(m_star_path)
    partition_hash = str(extra.get("partition_hash", ""))
    result_hash = str(results.get("m_star_hash", ""))
    if not partition_hash or not result_hash:
        raise ValueError("common M* run provenance is incomplete")
    if snapshot.content_hash != result_hash:
        raise ValueError("common M* checkpoint hash does not match its run manifest")
    if snapshot.partition_hash != partition_hash:
        raise ValueError("common M* checkpoint uses a different partition than its run")
    return {
        "path": str(m_star_path),
        "run_dir": str(run_dir),
        "experiment_id": experiment_id,
        "snapshot_hash": snapshot.content_hash,
        "partition_hash": partition_hash,
        "source_identity": manifest_source_identity(manifest),
    }


def _clean_record(summary_path: Path, seed: int) -> Dict[str, Any]:
    summary_path = summary_path.resolve()
    summary = _read_json(summary_path)
    manifest_path = summary_path.parent / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"clean run is not completed: {summary_path.parent}")
    if int(manifest.get("seed")) != int(seed):
        raise ValueError(f"clean run seed mismatch: {summary_path.parent}")
    experiment_id = str(manifest.get("experiment_id", ""))
    if not experiment_id or experiment_id != summary_path.parent.name:
        raise ValueError(
            f"clean experiment identity does not match its run directory: {summary_path.parent}"
        )
    if summary.get("selected_branches") != ["clean"]:
        raise ValueError(f"canonical input must be clean-only: {summary_path}")
    branches = summary.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != {"clean"}:
        raise ValueError(f"canonical input must contain only the clean branch: {summary_path}")
    m_star = summary.get("m_star")
    if not isinstance(m_star, Mapping) or not bool(m_star.get("reused")):
        raise ValueError(f"clean replica must reuse the common M*: {summary_path}")
    metrics = branches["clean"].get("test_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"clean test metrics are missing: {summary_path}")
    source_count = _integer_metric(metrics, "attack_source_sample_count")
    if source_count < 1:
        raise ValueError("attack_source_sample_count must be positive")
    asr = float(metrics["attack_success_rate"])
    if not math.isfinite(asr) or not 0.0 <= asr <= 1.0:
        raise ValueError("attack_success_rate must be in [0, 1]")
    if "attack_success_count" not in metrics:
        raise ValueError("clean run is missing attack_success_count")
    success_count = _integer_metric(metrics, "attack_success_count")
    if not 0 <= success_count <= source_count:
        raise ValueError("attack_success_count is outside the source sample range")
    if not math.isclose(asr, success_count / source_count, abs_tol=1e-9):
        raise ValueError("attack success count and rate are inconsistent")

    extra = manifest.get("extra")
    results = manifest.get("results")
    config = manifest.get("config")
    if (
        not isinstance(extra, Mapping)
        or not isinstance(results, Mapping)
        or not isinstance(config, Mapping)
    ):
        raise ValueError(f"run manifest provenance is incomplete: {manifest_path}")
    if str(results.get("m_star_hash", "")) != str(m_star.get("snapshot_hash", "")):
        raise ValueError(f"clean summary and manifest disagree on M*: {summary_path}")
    if _resolved_record_path(results.get("summary_path", ""), manifest) != summary_path:
        raise ValueError(f"clean summary path does not match its manifest: {summary_path}")
    attack = config.get("attack")
    federation = config.get("federation")
    if not isinstance(attack, Mapping) or not isinstance(federation, Mapping):
        raise ValueError(f"run manifest attack/federation config is missing: {manifest_path}")
    configured_m_star_path = federation.get("m_star_path")
    if configured_m_star_path is None:
        raise ValueError(f"clean run did not configure the common M*: {manifest_path}")
    configured_m_star_path = _resolved_record_path(configured_m_star_path, manifest)
    summary_m_star_path = _resolved_record_path(m_star.get("source_path", ""), manifest)
    if configured_m_star_path != summary_m_star_path:
        raise ValueError(f"clean summary and manifest disagree on the common M*: {summary_path}")
    m_star_source = extra.get("m_star_source")
    if (
        not isinstance(m_star_source, Mapping)
        or _resolved_record_path(m_star_source.get("path", ""), manifest)
        != summary_m_star_path
    ):
        raise ValueError(f"clean M* provenance is incomplete: {manifest_path}")
    branch_schedule = summary.get("branch_schedule")
    if (
        not isinstance(branch_schedule, list)
        or not branch_schedule
        or any(not isinstance(row, list) or not row for row in branch_schedule)
    ):
        raise ValueError(f"clean branch schedule is incomplete: {summary_path}")
    return {
        "summary_path": str(summary_path),
        "run_dir": str(summary_path.parent),
        "experiment_id": experiment_id,
        "partition_hash": str(extra.get("partition_hash", "")),
        "m_star_hash": str(m_star.get("snapshot_hash", "")),
        "m_star_source_path": str(summary_m_star_path),
        "branch_schedule": branch_schedule,
        "victim_eval_class": int(attack["victim_eval_class"]),
        "goal_prediction_class": int(attack["goal_prediction_class"]),
        "attack_success_count": success_count,
        "attack_source_sample_count": source_count,
        "attack_success_rate": asr,
        "comparison_protocol": canonical_comparison_protocol(config),
        "source_identity": manifest_source_identity(manifest),
    }


def build_canonical_clean(
    summary_paths: Sequence[Path],
    *,
    seed: int,
    m_star_path: Path,
) -> Dict[str, Any]:
    """Validate clean replicas and return their canonical ASR baseline."""

    normalized_paths = [Path(path).resolve() for path in summary_paths]
    if len(normalized_paths) != CANONICAL_REPLICA_COUNT:
        raise ValueError(
            f"canonical clean requires {CANONICAL_REPLICA_COUNT} replicas, "
            f"got {len(normalized_paths)}"
        )
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("canonical clean summary paths must be unique")

    records = [_clean_record(path, int(seed)) for path in normalized_paths]
    experiment_ids = [record["experiment_id"] for record in records]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("canonical clean replicas must have unique experiment identities")
    comparison_fields = (
        "partition_hash",
        "m_star_hash",
        "m_star_source_path",
        "branch_schedule",
        "victim_eval_class",
        "goal_prediction_class",
        "attack_source_sample_count",
        "comparison_protocol",
        "source_identity",
    )
    first = records[0]
    for field in comparison_fields:
        if any(record[field] != first[field] for record in records[1:]):
            raise ValueError(f"clean replicas disagree on {field}")
    if not first["partition_hash"] or not first["m_star_hash"]:
        raise ValueError("canonical clean provenance is incomplete")

    m_star_path = Path(m_star_path).resolve()
    if any(Path(record["m_star_source_path"]).resolve() != m_star_path for record in records):
        raise ValueError("clean replicas do not reference the requested common M*")
    m_star_record = _m_star_record(m_star_path, int(seed))
    if m_star_record["snapshot_hash"] != first["m_star_hash"]:
        raise ValueError("common M* checkpoint hash does not match clean summaries")
    if m_star_record["partition_hash"] != first["partition_hash"]:
        raise ValueError("common M* checkpoint uses a different partition")
    if m_star_record["source_identity"] != first["source_identity"]:
        raise ValueError("common M* source identity differs from clean replicas")

    rates = [float(record["attack_success_rate"]) for record in records]
    replicas = []
    for index, record in enumerate(records, start=1):
        replicas.append(
            {
                "replica": index,
                "summary_path": record["summary_path"],
                "run_dir": record["run_dir"],
                "experiment_id": record["experiment_id"],
                "attack_success_count": record["attack_success_count"],
                "attack_source_sample_count": record["attack_source_sample_count"],
                "attack_success_rate": record["attack_success_rate"],
                "attack_success_rate_pct": record["attack_success_rate"] * 100.0,
            }
        )
    mean_asr = statistics.fmean(rates)
    stddev_asr = statistics.pstdev(rates)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "canonical_clean",
        "seed": int(seed),
        "replica_count": len(replicas),
        "partition_hash": first["partition_hash"],
        "m_star": {
            "path": str(m_star_path),
            "snapshot_hash": first["m_star_hash"],
            "run_dir": m_star_record["run_dir"],
            "experiment_id": m_star_record["experiment_id"],
            "source_identity": m_star_record["source_identity"],
        },
        "victim_eval_class": first["victim_eval_class"],
        "goal_prediction_class": first["goal_prediction_class"],
        "attack_source_sample_count": first["attack_source_sample_count"],
        "asr_canonical_clean": mean_asr,
        "asr_canonical_clean_pct": mean_asr * 100.0,
        "asr_population_stddev": stddev_asr,
        "asr_population_stddev_pct": stddev_asr * 100.0,
        "branch_schedule": first["branch_schedule"],
        "comparison_protocol": first["comparison_protocol"],
        "source_identity": first["source_identity"],
        "replicas": replicas,
    }


def write_canonical_clean(payload: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
    return path


def load_canonical_clean(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    payload = _read_json(path)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported canonical clean schema version")
    if payload.get("kind") != "canonical_clean":
        raise ValueError("canonical clean artifact has the wrong kind")
    replicas = payload.get("replicas")
    if (
        not isinstance(replicas, list)
        or len(replicas) != CANONICAL_REPLICA_COUNT
        or int(payload.get("replica_count", -1)) != CANONICAL_REPLICA_COUNT
    ):
        raise ValueError("canonical clean replica metadata is inconsistent")
    source_count = _integer_metric(payload, "attack_source_sample_count")
    if source_count < 1:
        raise ValueError("canonical clean source sample count must be positive")
    summary_paths = set()
    run_dirs = set()
    experiment_ids = set()
    rates = []
    for expected_replica, replica in enumerate(replicas, start=1):
        if not isinstance(replica, Mapping):
            raise ValueError("canonical clean replica must be a mapping")
        if _integer_metric(replica, "replica") != expected_replica:
            raise ValueError("canonical clean replica indices are inconsistent")
        summary_path = str(replica.get("summary_path", ""))
        run_dir = str(replica.get("run_dir", ""))
        experiment_id = str(replica.get("experiment_id", ""))
        if not summary_path or not run_dir or not experiment_id:
            raise ValueError("canonical clean replica provenance is incomplete")
        if str(Path(summary_path).parent) != run_dir:
            raise ValueError("canonical clean summary and run directory disagree")
        if (
            summary_path in summary_paths
            or run_dir in run_dirs
            or experiment_id in experiment_ids
        ):
            raise ValueError("canonical clean replica provenance must be unique")
        if Path(run_dir).name != experiment_id:
            raise ValueError("canonical clean experiment identity is inconsistent")
        summary_paths.add(summary_path)
        run_dirs.add(run_dir)
        experiment_ids.add(experiment_id)
        replica_source_count = _integer_metric(
            replica, "attack_source_sample_count"
        )
        success_count = _integer_metric(replica, "attack_success_count")
        if replica_source_count != source_count:
            raise ValueError("canonical clean source sample counts disagree")
        if not 0 <= success_count <= source_count:
            raise ValueError("canonical clean success count is outside the source range")
        rate = float(replica["attack_success_rate"])
        if (
            not math.isfinite(rate)
            or not math.isclose(rate, success_count / source_count, abs_tol=1e-12)
        ):
            raise ValueError("canonical clean replica count and ASR disagree")
        rate_pct = float(replica["attack_success_rate_pct"])
        if not math.isclose(rate_pct, rate * 100.0, abs_tol=1e-10):
            raise ValueError("canonical clean replica ASR percentage is inconsistent")
        rates.append(rate)
    expected_mean = statistics.fmean(rates)
    expected_stddev = statistics.pstdev(rates)
    if not math.isclose(
        float(payload["asr_canonical_clean"]), expected_mean, abs_tol=1e-12
    ):
        raise ValueError("canonical clean mean ASR is inconsistent")
    if not math.isclose(
        float(payload["asr_canonical_clean_pct"]),
        expected_mean * 100.0,
        abs_tol=1e-10,
    ):
        raise ValueError("canonical clean mean ASR percentage is inconsistent")
    if not math.isclose(
        float(payload["asr_population_stddev"]),
        expected_stddev,
        abs_tol=1e-12,
    ):
        raise ValueError("canonical clean ASR standard deviation is inconsistent")
    if not math.isclose(
        float(payload["asr_population_stddev_pct"]),
        expected_stddev * 100.0,
        abs_tol=1e-10,
    ):
        raise ValueError("canonical clean ASR standard deviation percentage is inconsistent")
    if not isinstance(payload.get("comparison_protocol"), Mapping):
        raise ValueError("canonical clean comparison protocol is missing")
    if not isinstance(payload.get("source_identity"), Mapping):
        raise ValueError("canonical clean source identity is missing")
    m_star = payload.get("m_star")
    if (
        not isinstance(m_star, Mapping)
        or not str(m_star.get("path", ""))
        or not str(m_star.get("run_dir", ""))
        or not str(m_star.get("experiment_id", ""))
        or not isinstance(m_star.get("source_identity"), Mapping)
    ):
        raise ValueError("canonical clean M* provenance is incomplete")
    if Path(str(m_star["run_dir"])).name != str(m_star["experiment_id"]):
        raise ValueError("canonical clean M* experiment identity is inconsistent")
    rebuilt = build_canonical_clean(
        [Path(str(replica["summary_path"])) for replica in replicas],
        seed=int(payload["seed"]),
        m_star_path=Path(str(m_star["path"])),
    )
    if payload != rebuilt:
        raise ValueError(
            "canonical clean artifact differs from its source run artifacts"
        )
    payload["artifact_path"] = str(path)
    return payload


def validate_canonical_clean(
    payload: Mapping[str, Any],
    *,
    seed: int,
    partition_hash: str,
    m_star_hash: str,
    branch_schedule: Sequence[Sequence[str]],
    victim_eval_class: int,
    goal_prediction_class: int,
    attack_source_sample_count: int,
    comparison_protocol: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> None:
    if int(payload["seed"]) != int(seed):
        raise ValueError("canonical clean seed does not match the attack run")
    if str(payload["partition_hash"]) != str(partition_hash):
        raise ValueError("canonical clean partition does not match the attack run")
    m_star = payload.get("m_star")
    if not isinstance(m_star, Mapping) or str(m_star.get("snapshot_hash")) != str(
        m_star_hash
    ):
        raise ValueError("canonical clean M* does not match the attack run")
    if payload.get("branch_schedule") != [list(row) for row in branch_schedule]:
        raise ValueError("canonical clean schedule does not match the attack run")
    if int(payload["victim_eval_class"]) != int(victim_eval_class):
        raise ValueError("canonical clean victim class does not match the attack run")
    if int(payload["goal_prediction_class"]) != int(goal_prediction_class):
        raise ValueError("canonical clean goal class does not match the attack run")
    if int(payload["attack_source_sample_count"]) != int(
        attack_source_sample_count
    ):
        raise ValueError(
            "canonical clean source sample count does not match the attack run"
        )
    if payload["comparison_protocol"] != canonical_comparison_protocol(
        comparison_protocol
    ):
        raise ValueError(
            "canonical clean training protocol does not match the attack run"
        )
    if dict(payload["source_identity"]) != dict(source_identity):
        raise ValueError("canonical clean source identity does not match the attack run")
    if dict(payload["m_star"]["source_identity"]) != dict(source_identity):
        raise ValueError("canonical clean M* source identity does not match the attack run")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build one canonical clean ASR baseline from clean-only summaries"
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--m-star-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("summaries", nargs="+")
    args = parser.parse_args(argv)
    payload = build_canonical_clean(
        [Path(path) for path in args.summaries],
        seed=args.seed,
        m_star_path=Path(args.m_star_path),
    )
    output = write_canonical_clean(payload, Path(args.output))
    print(
        json.dumps(
            {
                "asr_canonical_clean": payload["asr_canonical_clean"],
                "output": str(output),
                "replica_count": payload["replica_count"],
                "seed": payload["seed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
