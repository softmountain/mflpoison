"""Persist scenario round records, generator lineage, and summaries."""

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mflpoison.artifacts import (
    load_round_record_bundle,
    save_generator_artifact,
    save_round_record,
    save_round_record_bundle,
)
from mflpoison.core.hashing import file_sha256
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


class ScenarioArtifactStore:
    """Own all scenario outputs below a configured artifact root."""

    def __init__(self, config, artifact_root: Path, *, config_hash: str):
        self.config = config
        self.artifact_root = Path(artifact_root)
        self.config_hash = str(config_hash)
        self._seen_generator_artifacts = set()

    def persist_records(self, phase: str, records: Sequence[Any]) -> None:
        records_root = self.artifact_root / "round_records" / phase
        if self.config.artifacts.save_every_round:
            for index, record in enumerate(records):
                save_round_record(record, records_root / f"round-{index:04d}.pt")
        bundle_path = self.artifact_root / self.config.artifacts.round_records_name
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
        if file_sha256(checkpoint_path) != artifact.checkpoint_hash:
            raise ValueError("generator checkpoint hash does not match its artifact")
        self._seen_generator_artifacts.add(identity)
        path = (
            self.artifact_root
            / self.config.artifacts.generator_dir
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
            "schema_version": 1,
            "config_hash": self.config_hash,
            "initial_snapshot_hash": initial_snapshot.content_hash,
            "m_star": {
                "snapshot_hash": m_star.content_hash,
                "round_index": m_star.round_index,
                "dev_metrics": dict(m_star.dev_metrics),
                "test_metrics": dict(m_star_test),
                "stopped_early": pretraining.stopped_early,
            },
            "malicious_clients": list(malicious_clients),
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
                    "detection_metrics": dict(result.detection_metrics),
                }
                for name, result in branches.items()
            },
        }
        clean_metrics = branches["clean"].test_metrics
        for name in ("attack", "defended"):
            branch_metrics = branches[name].test_metrics
            for metric_name in ("acc", "accuracy"):
                if metric_name in clean_metrics and metric_name in branch_metrics:
                    payload["branches"][name]["clean_utility_drop"] = float(
                        clean_metrics[metric_name] - branch_metrics[metric_name]
                    )
                    break
        return write_json(payload, self.artifact_root / "summary.json")
