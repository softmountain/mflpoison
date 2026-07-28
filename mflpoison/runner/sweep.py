"""Resolve and execute reproducible scenario sweep plans without overwriting runs."""

import argparse
import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mflpoison.artifacts import load_snapshot
from mflpoison.core.config import ScenarioConfig, load_config
from mflpoison.core.hashing import mapping_hash

from .builder import build_default_runner
from .persistence import write_json
from .resume import ScenarioResumeStore
from .scenario import BranchResult


_ALLOWED_OVERRIDE_PATHS = {
    "attack.malicious_clients",
    "attack.malicious_client_count",
    "attack.poison_ratio",
    "generator.epochs",
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SCREENING_FIELDS = {
    "min_asr_pct",
    "min_delta_asr_percentage_points",
    "max_global_accuracy_drop_percentage_points",
    "max_non_source_accuracy_drop_percentage_points",
    "collapse_global_accuracy_drop_percentage_points",
}


@dataclass(frozen=True)
class SweepRun:
    stage: str
    experiment: str
    seed: int
    config: ScenarioConfig
    artifact_root: Path
    pretrain_input_hash: str
    screening: Mapping[str, float]
    m_star_source_path: Optional[Path] = None

    @property
    def run_id(self) -> str:
        return f"{self.stage}/{self.experiment}/seed-{self.seed}"

    def plan_payload(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "experiment": self.experiment,
            "seed": self.seed,
            "artifact_root": str(self.artifact_root),
            "pretrain_input_hash": self.pretrain_input_hash,
            "m_star_source_path": (
                None
                if self.m_star_source_path is None
                else str(self.m_star_source_path)
            ),
            "branches": list(self.config.selected_branches),
            "malicious_clients": list(self.config.attack.malicious_clients),
            "poison_ratio": float(self.config.attack.poison_ratio),
            "generator_epochs": int(self.config.generator.epochs),
            "screening": dict(self.screening),
        }


def _strict_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _override_paths(value: Mapping[str, Any], prefix: str = "") -> Tuple[str, ...]:
    paths = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            paths.extend(_override_paths(item, path))
        else:
            paths.append(path)
    return tuple(paths)


def _deep_update(target: Dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            current = target.get(key)
            if not isinstance(current, Mapping):
                raise ValueError(f"cannot merge sweep override into {key}")
            nested = dict(current)
            _deep_update(nested, value)
            target[key] = nested
        else:
            target[key] = copy.deepcopy(value)


def _screening_thresholds(value: Any) -> Dict[str, float]:
    screening = _strict_mapping(value, "screening")
    if set(screening) != _SCREENING_FIELDS:
        missing = sorted(_SCREENING_FIELDS - set(screening))
        unknown = sorted(set(screening) - _SCREENING_FIELDS)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError("invalid sweep screening fields: " + "; ".join(details))
    normalized = {name: float(screening[name]) for name in _SCREENING_FIELDS}
    if any(value < 0.0 for value in normalized.values()):
        raise ValueError("sweep screening thresholds cannot be negative")
    return normalized


def _pretrain_input_hash(config: ScenarioConfig) -> str:
    payload = {
        "dataset": config.to_dict()["dataset"],
        "model": config.to_dict()["model"],
        "federation": config.to_dict()["federation"],
    }
    payload["federation"]["resume_from"] = None
    payload["federation"]["m_star_path"] = None
    payload["federation"]["m_star_snapshot_hash"] = None
    return mapping_hash(payload, length=64)


def _validate_research_invariants(config: ScenarioConfig) -> None:
    expected_semantics = {
        "condition_class": 0,
        "assigned_train_label": 1,
        "victim_eval_class": 0,
        "goal_prediction_class": 1,
    }
    actual_semantics = {
        name: getattr(config.attack, name) for name in expected_semantics
    }
    if actual_semantics != expected_semantics:
        raise ValueError(
            "sweep attack semantics must be condition/train/victim/goal = 0/1/0/1"
        )
    if config.selected_branches != ("clean", "attack"):
        raise ValueError("sweep branches must be exactly clean and attack")
    if config.defense.enabled:
        raise ValueError("sweep requires defense.enabled=false")
    if int(config.federation.attack_rounds) != 20:
        raise ValueError("sweep requires federation.attack_rounds=20")
    if int(config.federation.clients_per_round or 0) != 5:
        raise ValueError("sweep requires federation.clients_per_round=5")
    if config.attack.injection_mode != "replace":
        raise ValueError("sweep requires attack.injection_mode=replace")
    if not config.attack.malicious_clients:
        raise ValueError("sweep requires explicit attack.malicious_clients")
    if int(config.attack.malicious_client_count) != len(
        config.attack.malicious_clients
    ):
        raise ValueError(
            "sweep malicious_client_count must match explicit malicious_clients"
        )
    if int(config.generator.seed) != int(config.federation.seed):
        raise ValueError("generator.seed must match federation.seed in sweep runs")


def resolve_sweep_runs(
    plan_path: Path,
    *,
    stages: Sequence[str] = (),
    experiments: Sequence[str] = (),
    seeds: Sequence[int] = (),
) -> Tuple[SweepRun, ...]:
    plan_path = Path(plan_path)
    plan = _strict_mapping(load_config(plan_path), "sweep plan")
    allowed_top_level = {
        "base_config",
        "artifact_root",
        "m_star_source",
        "screening",
        "stages",
    }
    unknown = sorted(set(plan) - allowed_top_level)
    if unknown:
        raise ValueError("unknown sweep plan field(s): " + ", ".join(unknown))
    missing = sorted(allowed_top_level - set(plan))
    if missing:
        raise ValueError("missing sweep plan field(s): " + ", ".join(missing))

    base_config_path = Path(str(plan["base_config"]))
    if not base_config_path.is_absolute():
        base_config_path = plan_path.parent / base_config_path
    base_payload = load_config(base_config_path)
    artifact_base = Path(str(plan["artifact_root"]))
    screening = _screening_thresholds(plan["screening"])
    m_star_source = _strict_mapping(plan["m_star_source"], "m_star_source")
    if set(m_star_source) != {"stage", "experiment"}:
        raise ValueError("m_star_source must contain exactly stage and experiment")
    source_stage = str(m_star_source["stage"])
    source_experiment = str(m_star_source["experiment"])
    stage_filter = set(str(item) for item in stages)
    experiment_filter = set(str(item) for item in experiments)
    seed_filter = set(int(item) for item in seeds)

    resolved = []
    seen_roots = set()
    stage_payloads = _strict_mapping(plan["stages"], "sweep stages")
    for stage_name, raw_stage in stage_payloads.items():
        stage_name = str(stage_name)
        if stage_filter and stage_name not in stage_filter:
            continue
        stage = _strict_mapping(raw_stage, f"stage {stage_name}")
        if set(stage) != {"seeds", "experiments"}:
            raise ValueError(
                f"stage {stage_name} must contain exactly seeds and experiments"
            )
        stage_seeds = tuple(int(item) for item in stage["seeds"])
        if not stage_seeds or len(set(stage_seeds)) != len(stage_seeds):
            raise ValueError(f"stage {stage_name} seeds must be unique and non-empty")
        experiment_payloads = _strict_mapping(
            stage["experiments"], f"stage {stage_name} experiments"
        )
        for experiment_name, raw_experiment in experiment_payloads.items():
            experiment_name = str(experiment_name)
            if experiment_filter and experiment_name not in experiment_filter:
                continue
            experiment = _strict_mapping(
                raw_experiment, f"experiment {stage_name}/{experiment_name}"
            )
            if set(experiment) != {"overrides"}:
                raise ValueError(
                    f"experiment {stage_name}/{experiment_name} must contain overrides"
                )
            overrides = _strict_mapping(
                experiment["overrides"],
                f"experiment {stage_name}/{experiment_name} overrides",
            )
            disallowed = sorted(set(_override_paths(overrides)) - _ALLOWED_OVERRIDE_PATHS)
            if disallowed:
                raise ValueError(
                    "sweep experiment changes disallowed field(s): "
                    + ", ".join(disallowed)
                )
            for seed in stage_seeds:
                if seed_filter and seed not in seed_filter:
                    continue
                payload = copy.deepcopy(base_payload)
                _deep_update(payload, overrides)
                payload["federation"]["seed"] = seed
                payload["federation"]["resume_from"] = None
                payload["generator"]["seed"] = seed
                artifact_root = (
                    artifact_base
                    / stage_name
                    / experiment_name
                    / f"seed-{seed}"
                )
                payload["artifacts"]["root_dir"] = str(artifact_root)
                config = ScenarioConfig.from_mapping(payload)
                _validate_research_invariants(config)
                source_path = None
                if (stage_name, experiment_name) != (
                    source_stage,
                    source_experiment,
                ):
                    source_path = (
                        artifact_base
                        / source_stage
                        / source_experiment
                        / f"seed-{seed}"
                        / "snapshots"
                        / "m_star.pt"
                    )
                root_key = str(artifact_root.absolute())
                if root_key in seen_roots:
                    raise ValueError("duplicate sweep artifact root: " + root_key)
                seen_roots.add(root_key)
                resolved.append(
                    SweepRun(
                        stage=stage_name,
                        experiment=experiment_name,
                        seed=seed,
                        config=config,
                        artifact_root=artifact_root,
                        pretrain_input_hash=_pretrain_input_hash(config),
                        screening=screening,
                        m_star_source_path=source_path,
                    )
                )
    if not resolved:
        raise ValueError("sweep filters selected no runs")
    return tuple(resolved)


def _git_output(args: Sequence[str], repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repository_root),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError("unable to validate sweep Git state: " + detail.strip())
    return completed.stdout.strip()


def validate_execution_gate(
    approved_commit: str,
    *,
    repository_root: Path = _REPOSITORY_ROOT,
) -> str:
    if not str(approved_commit).strip():
        raise ValueError("--approved-commit is required for sweep execution")
    repository_root = Path(repository_root)
    approved = _git_output(
        ["rev-parse", "--verify", str(approved_commit) + "^{commit}"],
        repository_root,
    )
    head = _git_output(["rev-parse", "HEAD"], repository_root)
    if head != approved:
        raise RuntimeError(
            "current HEAD does not match the approved commit: "
            f"{head} != {approved}"
        )
    tracked_status = _git_output(
        ["status", "--porcelain", "--untracked-files=no"],
        repository_root,
    )
    if tracked_status:
        raise RuntimeError(
            "tracked worktree and staging area must be clean before sweep execution"
        )
    return head


def validate_execution_selection(
    *,
    stages: Sequence[str],
    experiments: Sequence[str],
    seeds: Sequence[int],
    allow_full_matrix: bool,
) -> None:
    if allow_full_matrix:
        return
    if len(stages) != 1 or len(seeds) != 1 or not experiments:
        raise ValueError(
            "--execute requires exactly one --stage, exactly one --seed, and "
            "at least one explicit --experiment; use --allow-full-matrix to "
            "confirm a broader run"
        )


def _load_json(path: Path, name: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a mapping: {path}")
    return dict(payload)


def _resolve_execution_config(
    run: SweepRun,
) -> Tuple[ScenarioConfig, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if run.m_star_source_path is None:
        return run.config, None, None
    if not run.m_star_source_path.is_file():
        raise FileNotFoundError(
            "shared M* artifact does not exist: " + str(run.m_star_source_path)
        )
    source_snapshot = load_snapshot(run.m_star_source_path)
    source_root = run.m_star_source_path.parents[1]
    source_metadata_path = source_root / "sweep_run.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(
            "shared M* sweep metadata does not exist: "
            + str(source_metadata_path)
        )
    source_metadata = _load_json(source_metadata_path, "shared M* metadata")
    if source_metadata.get("pretrain_input_hash") != run.pretrain_input_hash:
        raise ValueError("shared M* uses different pretrain inputs")
    source_provenance = {
        "m_star_hash": source_snapshot.content_hash,
        "partition_hash": source_snapshot.partition_hash,
        "branch_schedule": source_metadata.get("branch_schedule"),
    }
    payload = run.config.to_dict()
    payload["federation"]["m_star_path"] = str(run.m_star_source_path)
    payload["federation"]["m_star_snapshot_hash"] = source_snapshot.content_hash
    return (
        ScenarioConfig.from_mapping(payload),
        source_metadata,
        source_provenance,
    )


def _resume_config(config: ScenarioConfig, resume_path: Path) -> ScenarioConfig:
    payload = config.to_dict()
    payload["federation"]["resume_from"] = str(resume_path)
    return ScenarioConfig.from_mapping(payload)


def _load_existing_state(
    run: SweepRun,
    execution_config: ScenarioConfig,
) -> Dict[str, Any]:
    resolved_config_path = run.artifact_root / "resolved_config.json"
    if not resolved_config_path.is_file():
        raise FileExistsError(
            "existing artifact root has no resolved config: "
            + str(run.artifact_root)
        )
    if _load_json(resolved_config_path, "resolved sweep config") != (
        execution_config.to_dict()
    ):
        raise ValueError(
            "existing artifact root belongs to a different resolved config: "
            + str(run.artifact_root)
        )
    resume_path = run.artifact_root / "resume_state.pt"
    if not resume_path.is_file():
        raise FileExistsError(
            "existing artifact root has no resumable state: "
            + str(run.artifact_root)
        )
    store = ScenarioResumeStore(
        _resume_config(execution_config, resume_path),
        run.artifact_root,
        branch_result_type=BranchResult,
    )
    state = store.load()
    if state is None:
        raise RuntimeError("existing sweep resume state was not loaded")
    return state


def _screen_summary(
    run: SweepRun,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    attack = _strict_mapping(
        _strict_mapping(summary.get("branches"), "summary branches").get("attack"),
        "summary attack branch",
    )
    metrics = _strict_mapping(attack.get("test_metrics"), "attack test metrics")
    utility_drops = _strict_mapping(
        attack.get("clean_utility_drops", {}), "clean utility drops"
    )
    global_drop = next(
        (utility_drops[name] for name in ("acc", "accuracy") if name in utility_drops),
        None,
    )
    non_source_drop = utility_drops.get("non_source_accuracy")
    asr_pct = metrics.get("attack_success_rate_pct")
    delta_asr_pp = attack.get("delta_asr_percentage_points")
    thresholds = dict(run.screening)
    checks = {
        "asr": asr_pct is not None
        and float(asr_pct) >= thresholds["min_asr_pct"],
        "delta_asr": delta_asr_pp is not None
        and float(delta_asr_pp)
        >= thresholds["min_delta_asr_percentage_points"],
        "global_accuracy_drop": global_drop is not None
        and float(global_drop)
        <= thresholds["max_global_accuracy_drop_percentage_points"],
        "non_source_accuracy_drop": non_source_drop is not None
        and float(non_source_drop)
        <= thresholds["max_non_source_accuracy_drop_percentage_points"],
    }
    classification = "below_strong_targeted_threshold"
    if all(checks.values()):
        classification = "strong_targeted_candidate"
    if (
        float(run.config.attack.poison_ratio) == 1.0
        and run.config.attack.injection_mode == "replace"
        and global_drop is not None
        and float(global_drop)
        > thresholds["collapse_global_accuracy_drop_percentage_points"]
    ):
        classification = "availability_or_model_collapse"
    return {
        "thresholds": thresholds,
        "checks": checks,
        "passes": all(checks.values()),
        "classification": classification,
    }


def _run_provenance(
    run: SweepRun,
    *,
    m_star,
    branch_schedule: Sequence[Sequence[str]],
    branches: Mapping[str, BranchResult],
) -> Dict[str, Any]:
    clean = branches.get("clean")
    attack = branches.get("attack")
    if clean is None or attack is None:
        raise ValueError("sweep results must contain clean and attack branches")
    return {
        "m_star_hash": m_star.content_hash,
        "partition_hash": m_star.partition_hash,
        "branch_schedule": [list(row) for row in branch_schedule],
        "clean_final_snapshot_hash": clean.final_snapshot.content_hash,
        "generator_epochs": int(run.config.generator.epochs),
        "generator_artifact_ids": {
            client_id: artifact.content_hash
            for client_id, artifact in attack.generator_artifacts.items()
        },
        "generator_checkpoint_hashes": {
            client_id: artifact.checkpoint_hash
            for client_id, artifact in attack.generator_artifacts.items()
        },
    }


def _validate_comparison_invariants(
    run: SweepRun,
    provenance: Mapping[str, Any],
    source_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if source_metadata is None:
        return {
            "baseline_run_id": run.run_id,
            "clean_final_snapshot_matches_baseline": True,
            "generator_checkpoint_matches_by_client_epoch": {},
        }
    required = {
        "run_id",
        "clean_final_snapshot_hash",
        "generator_epochs",
        "generator_checkpoint_hashes",
    }
    missing = sorted(required - set(source_metadata))
    if missing:
        raise ValueError(
            "baseline sweep provenance is missing field(s): " + ", ".join(missing)
        )
    clean_matches = (
        provenance["clean_final_snapshot_hash"]
        == source_metadata["clean_final_snapshot_hash"]
    )
    if not clean_matches:
        raise RuntimeError(
            f"clean final snapshot drift detected for {run.run_id} against B0"
        )
    checkpoint_checks = {}
    if int(provenance["generator_epochs"]) == int(
        source_metadata["generator_epochs"]
    ):
        baseline_hashes = _strict_mapping(
            source_metadata["generator_checkpoint_hashes"],
            "baseline generator checkpoint hashes",
        )
        current_hashes = _strict_mapping(
            provenance["generator_checkpoint_hashes"],
            "generator checkpoint hashes",
        )
        for client_id in sorted(set(baseline_hashes) & set(current_hashes)):
            matches = baseline_hashes[client_id] == current_hashes[client_id]
            checkpoint_checks[client_id] = matches
            if not matches:
                raise RuntimeError(
                    "generator checkpoint drift detected for client "
                    f"{client_id} at {provenance['generator_epochs']} epochs"
                )
    return {
        "baseline_run_id": str(source_metadata["run_id"]),
        "clean_final_snapshot_matches_baseline": clean_matches,
        "generator_checkpoint_matches_by_client_epoch": checkpoint_checks,
    }


def _build_run_payload(
    run: SweepRun,
    *,
    provenance: Mapping[str, Any],
    summary_path: Path,
    resolved_config_path: Path,
    approved_commit: str,
    source_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = _load_json(summary_path, "scenario summary")
    invariants = _validate_comparison_invariants(
        run, provenance, source_metadata
    )
    return {
        **run.plan_payload(),
        **dict(provenance),
        "approved_commit": approved_commit,
        "summary_path": str(summary_path),
        "resolved_config_path": str(resolved_config_path),
        "targeted_attack_screen": _screen_summary(run, summary),
        "paired_invariants": invariants,
    }


def _validate_core_provenance(
    run: SweepRun,
    provenance: Mapping[str, Any],
    source_provenance: Optional[Mapping[str, Any]],
) -> None:
    if source_provenance is None:
        return
    actual = {
        name: provenance[name]
        for name in ("m_star_hash", "partition_hash", "branch_schedule")
    }
    if actual != dict(source_provenance):
        raise RuntimeError(
            f"comparison provenance drift detected for seed {run.seed}"
        )


def _completed_run_payload(
    run: SweepRun,
    state: Mapping[str, Any],
    *,
    approved_commit: str,
    source_metadata: Optional[Mapping[str, Any]],
    source_provenance: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    required_paths = (
        run.artifact_root / "summary.json",
        run.artifact_root / run.config.artifacts.manifest_name,
        run.artifact_root / "snapshots" / "m_star.pt",
        run.artifact_root / "snapshots" / "clean" / "final.pt",
        run.artifact_root / "snapshots" / "attack" / "final.pt",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(
            "completed resume state is missing artifact(s): " + ", ".join(missing)
        )
    manifest = _load_json(required_paths[1], "scenario manifest")
    if manifest.get("git_commit") != approved_commit:
        raise RuntimeError(
            "completed artifact was not produced from the approved commit"
        )
    m_star = load_snapshot(required_paths[2])
    provenance = _run_provenance(
        run,
        m_star=m_star,
        branch_schedule=state["branch_schedule"],
        branches=_strict_mapping(state.get("branches"), "resume branches"),
    )
    _validate_core_provenance(run, provenance, source_provenance)
    return _build_run_payload(
        run,
        provenance=provenance,
        summary_path=required_paths[0],
        resolved_config_path=run.artifact_root / "resolved_config.json",
        approved_commit=approved_commit,
        source_metadata=source_metadata,
    )


def execute_sweep_runs(
    runs: Sequence[SweepRun],
    *,
    approved_commit: str,
    resume: bool = False,
    repository_root: Path = _REPOSITORY_ROOT,
) -> Tuple[Mapping[str, Any], ...]:
    approved = validate_execution_gate(
        approved_commit, repository_root=repository_root
    )
    results = []
    for run in runs:
        execution_config, source_metadata, source_provenance = (
            _resolve_execution_config(run)
        )
        resolved_config_path = run.artifact_root / "resolved_config.json"
        if run.artifact_root.exists():
            state = _load_existing_state(run, execution_config)
            if state.get("phase") == "complete":
                payload = _completed_run_payload(
                    run,
                    state,
                    approved_commit=approved,
                    source_metadata=source_metadata,
                    source_provenance=source_provenance,
                )
                metadata_path = run.artifact_root / "sweep_run.json"
                write_json(payload, metadata_path)
                results.append({**payload, "execution_status": "skipped_completed"})
                continue
            if not resume:
                raise RuntimeError(
                    "interrupted sweep run requires --resume: "
                    + str(run.artifact_root)
                )
            runner_config = _resume_config(
                execution_config, run.artifact_root / "resume_state.pt"
            )
            execution_status = "resumed"
        else:
            run.artifact_root.mkdir(parents=True, exist_ok=False)
            write_json(execution_config.to_dict(), resolved_config_path)
            runner_config = execution_config
            execution_status = "completed"
        result = build_default_runner(
            runner_config,
            artifact_root=run.artifact_root,
        ).run()
        provenance = _run_provenance(
            run,
            m_star=result.m_star,
            branch_schedule=result.branch_schedule,
            branches=result.branches,
        )
        _validate_core_provenance(run, provenance, source_provenance)
        payload = _build_run_payload(
            run,
            provenance=provenance,
            summary_path=result.summary_path,
            resolved_config_path=resolved_config_path,
            approved_commit=approved,
            source_metadata=source_metadata,
        )
        write_json(payload, run.artifact_root / "sweep_run.json")
        results.append({**payload, "execution_status": execution_status})
    return tuple(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute reproducible UCF101 poisoning sweeps"
    )
    parser.add_argument("--plan", required=True, help="Sweep YAML or JSON path")
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run selected experiments; without this flag only print the plan",
    )
    parser.add_argument(
        "--approved-commit",
        help="Locally approved Git commit required for execution",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matching interrupted runs from validated resume_state.pt",
    )
    parser.add_argument(
        "--allow-full-matrix",
        action="store_true",
        help="Explicitly confirm execution broader than one stage and one seed",
    )
    args = parser.parse_args(argv)
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.allow_full_matrix and not args.execute:
        parser.error("--allow-full-matrix requires --execute")
    if args.execute:
        try:
            validate_execution_selection(
                stages=args.stage,
                experiments=args.experiment,
                seeds=args.seed,
                allow_full_matrix=args.allow_full_matrix,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if not args.approved_commit:
            parser.error("--approved-commit is required with --execute")
    runs = resolve_sweep_runs(
        Path(args.plan),
        stages=args.stage,
        experiments=args.experiment,
        seeds=args.seed,
    )
    if args.execute:
        payload = execute_sweep_runs(
            runs,
            approved_commit=args.approved_commit,
            resume=args.resume,
        )
    else:
        payload = tuple(run.plan_payload() for run in runs)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
