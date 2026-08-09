"""Persist scenario round records, generator lineage, and summaries."""

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mflpoison.artifacts import (
    load_round_record_bundle,
    save_generator_artifact,
    save_round_record_bundle,
)
from mflpoison.core.types import GeneratorArtifact, GlobalSnapshot
from mflpoison.federated import TrainingResult


def write_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomically write strict JSON without non-finite numeric values."""

    def json_safe(value):
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)
    return path


class ResultStore:
    """Write the stable files inside one human-readable run directory."""

    def __init__(self, config, run_dir: Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self._seen_generator_artifacts = set()
        self._canonical_clean = None

    def set_canonical_clean(self, payload: Mapping[str, Any]) -> None:
        if payload.get("kind") != "canonical_clean":
            raise ValueError("canonical clean artifact has the wrong kind")
        self._canonical_clean = dict(payload)

    def persist_records(self, phase: str, records: Sequence[Any]) -> None:
        bundle_path = self.run_dir / "round_records.pt"
        phases = {}
        if bundle_path.exists():
            phases = load_round_record_bundle(bundle_path)
        phases[phase] = list(records)
        save_round_record_bundle(phases, bundle_path)

    def persist_generator_artifact(
        self, phase: str, artifact: GeneratorArtifact
    ) -> None:
        identity = (phase, artifact.content_hash)
        if identity in self._seen_generator_artifacts:
            return
        checkpoint_path = Path(artifact.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))
        self._seen_generator_artifacts.add(identity)
        path = (
            self.run_dir
            / "generators"
            / phase
            / artifact.client_id
            / f"{artifact.content_hash}.json"
        )
        save_generator_artifact(artifact, path)

    def persist_summary(
        self,
        initial_snapshot: GlobalSnapshot,
        pretraining: TrainingResult,
        m_star: GlobalSnapshot,
        m_star_test: Mapping[str, float],
        pretrain_schedule: Sequence[Sequence[str]],
        branch_schedule: Sequence[Sequence[str]],
        malicious_clients: Sequence[str],
        branches: Mapping[str, Any],
    ) -> Path:
        payload = {
            "schema_version": 3,
            "initial_snapshot_hash": initial_snapshot.content_hash,
            "m_star": {
                "snapshot_hash": m_star.content_hash,
                "round_index": m_star.round_index,
                "dev_metrics": dict(m_star.dev_metrics),
                "test_metrics": dict(m_star_test),
                "stopped_early": pretraining.stopped_early,
                "reused": self.config.federation.m_star_path is not None,
                "source_path": self.config.federation.m_star_path,
            },
            "malicious_clients": list(malicious_clients),
            "selected_branches": list(branches),
            "pretrain_schedule": [list(row) for row in pretrain_schedule],
            "branch_schedule": [list(row) for row in branch_schedule],
            "branches": {
                name: {
                    "final_snapshot_hash": result.final_snapshot.content_hash,
                    "final_round_index": result.final_snapshot.round_index,
                    "dev_metrics": dict(result.final_snapshot.dev_metrics),
                    "test_metrics": dict(result.test_metrics),
                    "generator_artifacts": {
                        client_id: artifact.content_hash
                        for client_id, artifact in result.generator_artifacts.items()
                    },
                    "generator_checkpoint_hashes": {
                        client_id: artifact.checkpoint_hash
                        for client_id, artifact in result.generator_artifacts.items()
                    },
                    "detection_metrics": dict(result.detection_metrics),
                }
                for name, result in branches.items()
            },
        }
        canonical_clean = self._canonical_clean
        if canonical_clean is not None:
            payload["canonical_clean"] = {
                "artifact_path": canonical_clean["artifact_path"],
                "schema_version": canonical_clean["schema_version"],
                "seed": canonical_clean["seed"],
                "replica_count": canonical_clean["replica_count"],
                "partition_hash": canonical_clean["partition_hash"],
                "m_star_snapshot_hash": canonical_clean["m_star"][
                    "snapshot_hash"
                ],
                "m_star_run_dir": canonical_clean["m_star"]["run_dir"],
                "m_star_experiment_id": canonical_clean["m_star"][
                    "experiment_id"
                ],
                "m_star_source_identity": canonical_clean["m_star"][
                    "source_identity"
                ],
                "attack_source_sample_count": canonical_clean[
                    "attack_source_sample_count"
                ],
                "asr_canonical_clean": canonical_clean[
                    "asr_canonical_clean"
                ],
                "asr_canonical_clean_pct": canonical_clean[
                    "asr_canonical_clean_pct"
                ],
                "asr_population_stddev": canonical_clean[
                    "asr_population_stddev"
                ],
                "asr_population_stddev_pct": canonical_clean[
                    "asr_population_stddev_pct"
                ],
                "comparison_protocol": canonical_clean[
                    "comparison_protocol"
                ],
                "source_identity": canonical_clean["source_identity"],
            }
        for name in ("attack", "defended"):
            if name in branches:
                payload["branches"][name]["attack_exposure"] = (
                    self._attack_exposure(
                        branches[name].training.records,
                    )
                )

        clean_result = branches.get("clean")
        clean_metrics = None if clean_result is None else clean_result.test_metrics
        for name, result in branches.items():
            if name == "clean":
                continue
            branch_metrics = result.test_metrics
            if clean_metrics is not None:
                utility_drops = {}
                for metric_name in (
                    "acc",
                    "accuracy",
                    "uar",
                    "f1",
                    "source_class_accuracy",
                    "source_class_recall",
                    "non_source_accuracy",
                    "non_source_macro_f1",
                ):
                    if metric_name in clean_metrics and metric_name in branch_metrics:
                        utility_drops[metric_name] = float(
                            clean_metrics[metric_name] - branch_metrics[metric_name]
                        )
                payload["branches"][name]["clean_utility_drops"] = utility_drops
                for metric_name in ("acc", "accuracy"):
                    if metric_name in utility_drops:
                        payload["branches"][name]["clean_utility_drop"] = (
                            utility_drops[metric_name]
                        )
                        break
                if (
                    "attack_success_rate" in clean_metrics
                    and "attack_success_rate" in branch_metrics
                ):
                    delta_asr = float(
                        branch_metrics["attack_success_rate"]
                        - clean_metrics["attack_success_rate"]
                    )
                    if canonical_clean is None:
                        payload["branches"][name]["delta_baseline"] = "run_clean"
                        payload["branches"][name][
                            "delta_attack_success_rate"
                        ] = delta_asr
                        payload["branches"][name][
                            "delta_asr_percentage_points"
                        ] = delta_asr * 100.0
                    else:
                        payload["branches"][name][
                            "within_run_delta_attack_success_rate"
                        ] = delta_asr
                        payload["branches"][name][
                            "within_run_delta_asr_percentage_points"
                        ] = delta_asr * 100.0
            if canonical_clean is not None:
                if (
                    "attack_success_count" not in branch_metrics
                    or "attack_source_sample_count" not in branch_metrics
                    or "attack_success_rate" not in branch_metrics
                ):
                    raise ValueError(
                        "canonical clean comparison requires attack count, source count, and rate"
                    )
                source_count_value = float(
                    branch_metrics["attack_source_sample_count"]
                )
                source_count = int(round(source_count_value))
                if (
                    not math.isfinite(source_count_value)
                    or not math.isclose(
                        source_count_value, source_count, abs_tol=1e-9
                    )
                    or source_count < 1
                ):
                    raise ValueError(
                        "attack_source_sample_count must be a positive integer"
                    )
                success_count_value = float(branch_metrics["attack_success_count"])
                success_count = int(round(success_count_value))
                if (
                    not math.isfinite(success_count_value)
                    or not math.isclose(
                        success_count_value, success_count, abs_tol=1e-9
                    )
                    or not 0 <= success_count <= source_count
                ):
                    raise ValueError(
                        "attack_success_count must be an integer within the source sample range"
                    )
                attack_asr = float(branch_metrics["attack_success_rate"])
                if (
                    not math.isfinite(attack_asr)
                    or not 0.0 <= attack_asr <= 1.0
                    or not math.isclose(
                        attack_asr, success_count / source_count, abs_tol=1e-9
                    )
                ):
                    raise ValueError("attack success count and rate are inconsistent")
                if source_count != int(
                    canonical_clean["attack_source_sample_count"]
                ):
                    raise ValueError(
                        "attack source sample count differs from canonical clean"
                    )
                baseline_asr = float(canonical_clean["asr_canonical_clean"])
                delta_asr = attack_asr - baseline_asr
                payload["branches"][name]["delta_baseline"] = "canonical_clean"
                payload["branches"][name][
                    "canonical_clean_attack_success_rate"
                ] = baseline_asr
                payload["branches"][name][
                    "canonical_clean_attack_success_rate_pct"
                ] = baseline_asr * 100.0
                payload["branches"][name][
                    "delta_attack_success_rate"
                ] = delta_asr
                payload["branches"][name][
                    "delta_asr_percentage_points"
                ] = delta_asr * 100.0
        return write_json(payload, self.run_dir / "summary.json")

    def _attack_exposure(
        self,
        records: Sequence[Any],
    ) -> Mapping[str, Any]:
        total_client_seats = 0
        malicious_client_seats = 0
        rounds_with_malicious_clients = 0
        rounds_with_active_poison = 0
        active_poisoned_updates = 0
        total_poison_samples = 0
        for record in records:
            selected_malicious = [
                update
                for update in record.raw_updates
                if bool(update.malicious)
            ]
            total_client_seats += len(record.selected_client_ids)
            malicious_client_seats += len(selected_malicious)
            if selected_malicious:
                rounds_with_malicious_clients += 1
            poisoned_updates = [
                update
                for update in selected_malicious
                if bool(update.attack_active)
            ]
            active_poisoned_updates += len(poisoned_updates)
            total_poison_samples += sum(
                int(update.poison_sample_count) for update in poisoned_updates
            )
            if poisoned_updates:
                rounds_with_active_poison += 1
        return {
            "total_rounds": len(records),
            "total_client_seats": total_client_seats,
            "rounds_with_malicious_clients": rounds_with_malicious_clients,
            "malicious_client_seats": malicious_client_seats,
            "rounds_with_active_poison": rounds_with_active_poison,
            "active_poisoned_updates": active_poisoned_updates,
            "total_poison_samples": total_poison_samples,
        }
