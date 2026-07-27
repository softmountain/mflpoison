"""Persist and revalidate scenario resume state."""

import copy
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type

import torch

from mflpoison.artifacts import revalidate_round_record
from mflpoison.core.hashing import mapping_hash, semantic_hash
from mflpoison.core.types import GeneratorArtifact, GlobalSnapshot
from mflpoison.federated import TrainingResult

from .runtime import scalar_metrics


class ScenarioResumeStore:
    """Own the versioned resume payload and its integrity checks."""

    def __init__(
        self,
        config,
        artifact_root: Path,
        *,
        branch_result_type: Type[Any],
    ):
        self.config = config
        self.artifact_root = Path(artifact_root)
        self.branch_result_type = branch_result_type

    @property
    def path(self) -> Path:
        configured = self.config.federation.resume_from
        return (
            Path(configured)
            if configured is not None
            else self.artifact_root / "resume_state.pt"
        )

    @property
    def config_hash(self) -> str:
        payload = copy.deepcopy(self.config.to_dict())
        payload["federation"]["resume_from"] = None
        return mapping_hash(payload)

    def load(self) -> Optional[Dict[str, Any]]:
        if self.config.federation.resume_from is None:
            return None
        path = self.path
        if not path.is_file():
            raise FileNotFoundError(str(path))
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError("scenario resume state must contain a mapping")
        if int(payload.get("schema_version", -1)) != 2:
            raise ValueError("unsupported scenario resume schema version")
        expected_content_hash = str(payload.get("content_hash", ""))
        hashed_payload = {
            key: value for key, value in payload.items() if key != "content_hash"
        }
        if (
            not expected_content_hash
            or semantic_hash(hashed_payload) != expected_content_hash
        ):
            raise ValueError("scenario resume content hash does not match its payload")
        if payload.get("config_hash") != self.config_hash:
            raise ValueError("resume state belongs to a different scenario config")
        return self.revalidate_state(dict(payload))

    def save(self, **values) -> Path:
        payload = {
            "schema_version": 2,
            "config_hash": self.config_hash,
            **values,
        }
        payload["content_hash"] = semantic_hash(payload)
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    @staticmethod
    def revalidate_snapshot(snapshot: GlobalSnapshot) -> GlobalSnapshot:
        if not isinstance(snapshot, GlobalSnapshot):
            raise TypeError("resume state contains an invalid global snapshot")
        return GlobalSnapshot(
            state=snapshot.state,
            round_index=snapshot.round_index,
            dev_metrics=snapshot.dev_metrics,
            model_spec=snapshot.model_spec,
            partition_hash=snapshot.partition_hash,
            metadata=snapshot.metadata,
            content_hash=snapshot.content_hash,
        )

    @classmethod
    def revalidate_training_result(cls, result: TrainingResult) -> TrainingResult:
        if not isinstance(result, TrainingResult):
            raise TypeError("resume state contains an invalid training result")
        return TrainingResult(
            best_snapshot=cls.revalidate_snapshot(result.best_snapshot),
            final_snapshot=cls.revalidate_snapshot(result.final_snapshot),
            records=[revalidate_round_record(record) for record in result.records],
            stopped_early=bool(result.stopped_early),
        )

    @staticmethod
    def revalidate_generator_artifact(artifact) -> GeneratorArtifact:
        if not isinstance(artifact, GeneratorArtifact):
            raise TypeError("resume state contains an invalid generator artifact")
        return GeneratorArtifact.from_dict(artifact.to_dict())

    def revalidate_branch_result(self, result):
        if not isinstance(result, self.branch_result_type):
            raise TypeError("resume state contains an invalid branch result")
        return self.branch_result_type(
            name=str(result.name),
            training=self.revalidate_training_result(result.training),
            test_metrics=scalar_metrics(result.test_metrics),
            generator_artifacts={
                str(client_id): self.revalidate_generator_artifact(artifact)
                for client_id, artifact in result.generator_artifacts.items()
            },
            detection_metrics=dict(result.detection_metrics),
        )

    def revalidate_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        phase = str(payload.get("phase", ""))
        allowed = {
            "pretrain",
            "pretrain_complete",
            "base_generators",
            "base_complete",
            "branches_complete",
            "complete",
        }
        if phase not in allowed and not phase.startswith("branch:"):
            raise ValueError("scenario resume state has an invalid phase")
        payload["initial_snapshot"] = self.revalidate_snapshot(
            payload.get("initial_snapshot")
        )
        if "pretraining" in payload:
            payload["pretraining"] = self.revalidate_training_result(
                payload["pretraining"]
            )
        if "base_generator_artifacts" in payload:
            artifacts = payload["base_generator_artifacts"]
            if not isinstance(artifacts, Mapping):
                raise TypeError("base generator artifacts must be a mapping")
            payload["base_generator_artifacts"] = {
                str(client_id): self.revalidate_generator_artifact(artifact)
                for client_id, artifact in artifacts.items()
            }
        if "branches" in payload:
            branches = payload["branches"]
            if not isinstance(branches, Mapping):
                raise TypeError("resume branches must be a mapping")
            payload["branches"] = {
                str(name): self.revalidate_branch_result(result)
                for name, result in branches.items()
            }
        if "active" in payload:
            active = payload["active"]
            if not isinstance(active, Mapping):
                raise TypeError("resume active progress must be a mapping")
            active = dict(active)
            active["current_snapshot"] = self.revalidate_snapshot(
                active.get("current_snapshot")
            )
            active["best_snapshot"] = self.revalidate_snapshot(
                active.get("best_snapshot")
            )
            best_value = float(active.get("best_value"))
            if not math.isfinite(best_value):
                raise ValueError("resume best_value must be finite")
            active["best_value"] = best_value
            stale_rounds = int(active.get("stale_rounds", 0))
            if stale_rounds < 0:
                raise ValueError("resume stale_rounds cannot be negative")
            active["stale_rounds"] = stale_rounds
            active["records"] = [
                revalidate_round_record(record)
                for record in active.get("records", ())
            ]
            payload["active"] = active
        return payload

    @staticmethod
    def progress_payload(progress) -> Dict[str, Any]:
        return {
            "current_snapshot": progress.current_snapshot,
            "best_snapshot": progress.best_snapshot,
            "best_value": progress.best_value,
            "stale_rounds": progress.stale_rounds,
            "records": list(progress.records),
        }
