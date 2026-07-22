import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from mflpoison.artifacts import (
    build_manifest,
    load_round_record_bundle,
    revalidate_round_record,
    save_generator_artifact,
    save_round_record,
    save_round_record_bundle,
    save_snapshot,
    write_manifest,
)
from mflpoison.attacks import AttackSpec, select_malicious_clients
from mflpoison.core.config import ScenarioConfig
from mflpoison.core.hashing import file_sha256
from mflpoison.core.hashing import mapping_hash
from mflpoison.core.hashing import semantic_hash
from mflpoison.core.types import GeneratorArtifact, GlobalSnapshot, ModelSpec
from mflpoison.evaluation import detection_metrics
from mflpoison.federated import (
    ConvergencePolicy,
    FedAvgCoordinator,
    TrainingResult,
    build_client_schedule_count,
)


@dataclass(frozen=True)
class BranchResult:
    name: str
    training: TrainingResult
    test_metrics: Mapping[str, float]
    generator_artifacts: Mapping[str, GeneratorArtifact]
    detection_metrics: Mapping[str, float]

    @property
    def final_snapshot(self) -> GlobalSnapshot:
        return self.training.final_snapshot


@dataclass(frozen=True)
class ScenarioResult:
    initial_snapshot: GlobalSnapshot
    pretraining: TrainingResult
    m_star: GlobalSnapshot
    pretrain_schedule: Tuple[Tuple[str, ...], ...]
    branch_schedule: Tuple[Tuple[str, ...], ...]
    malicious_clients: Tuple[str, ...]
    branches: Mapping[str, BranchResult]
    m_star_test_metrics: Mapping[str, float]
    artifact_root: Path
    summary_path: Path


def _cpu_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Keep only finite scalar metrics suitable for hashes and selection."""

    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[str(key)] = number
    return result


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
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


class ScenarioRunner:
    """Orchestrate the complete scenario without crossing client data boundaries.

    The injected adapter owns partition and evaluation access. The runner only
    asks for one selected client's bundle at a time, while defense receives
    typed updates at the server boundary.
    """

    BRANCHES = ("clean", "attack", "defended")

    def __init__(
        self,
        config: ScenarioConfig,
        *,
        adapter,
        client_trainer,
        aggregator,
        initial_state: Optional[Mapping[str, torch.Tensor]] = None,
        model_spec: Optional[ModelSpec] = None,
        generator_lifecycle_factory: Optional[Callable[[str], Any]] = None,
        attack_strategy=None,
        defense_pipeline=None,
        artifact_root: Optional[Path] = None,
    ):
        if not isinstance(config, ScenarioConfig):
            raise TypeError("config must be a ScenarioConfig")
        self.config = config
        self.adapter = adapter
        self.client_trainer = client_trainer
        self.aggregator = aggregator
        self.initial_state = None if initial_state is None else _cpu_state(initial_state)
        self.model_spec = model_spec or ModelSpec(
            name=config.model.name,
            constructor=config.model.constructor,
            kwargs=dict(config.model.kwargs),
        )
        self.generator_lifecycle_factory = generator_lifecycle_factory
        self.attack_strategy = attack_strategy
        self.defense_pipeline = defense_pipeline
        self.artifact_root = Path(
            config.artifacts.root_dir if artifact_root is None else artifact_root
        )
        self._seen_generator_artifacts = set()

    @property
    def _resume_path(self) -> Path:
        configured = self.config.federation.resume_from
        return (
            Path(configured)
            if configured is not None
            else self.artifact_root / "resume_state.pt"
        )

    @property
    def _resume_config_hash(self) -> str:
        payload = copy.deepcopy(self.config.to_dict())
        payload["federation"]["resume_from"] = None
        return mapping_hash(payload)

    def _load_resume_state(self) -> Optional[Dict[str, Any]]:
        if self.config.federation.resume_from is None:
            return None
        path = self._resume_path
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
        if payload.get("config_hash") != self._resume_config_hash:
            raise ValueError("resume state belongs to a different scenario config")
        return self._revalidate_resume_state(dict(payload))

    def _save_resume_state(self, **values) -> Path:
        payload = {
            "schema_version": 2,
            "config_hash": self._resume_config_hash,
            **values,
        }
        payload["content_hash"] = semantic_hash(payload)
        path = self._resume_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    @staticmethod
    def _revalidate_snapshot(snapshot: GlobalSnapshot) -> GlobalSnapshot:
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
    def _revalidate_training_result(cls, result: TrainingResult) -> TrainingResult:
        if not isinstance(result, TrainingResult):
            raise TypeError("resume state contains an invalid training result")
        return TrainingResult(
            best_snapshot=cls._revalidate_snapshot(result.best_snapshot),
            final_snapshot=cls._revalidate_snapshot(result.final_snapshot),
            records=[revalidate_round_record(record) for record in result.records],
            stopped_early=bool(result.stopped_early),
        )

    @staticmethod
    def _revalidate_generator_artifact(artifact) -> GeneratorArtifact:
        if not isinstance(artifact, GeneratorArtifact):
            raise TypeError("resume state contains an invalid generator artifact")
        return GeneratorArtifact.from_dict(artifact.to_dict())

    @classmethod
    def _revalidate_branch_result(cls, result: BranchResult) -> BranchResult:
        if not isinstance(result, BranchResult):
            raise TypeError("resume state contains an invalid branch result")
        return BranchResult(
            name=str(result.name),
            training=cls._revalidate_training_result(result.training),
            test_metrics=_scalar_metrics(result.test_metrics),
            generator_artifacts={
                str(client_id): cls._revalidate_generator_artifact(artifact)
                for client_id, artifact in result.generator_artifacts.items()
            },
            detection_metrics=dict(result.detection_metrics),
        )

    @classmethod
    def _revalidate_resume_state(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
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
        payload["initial_snapshot"] = cls._revalidate_snapshot(
            payload.get("initial_snapshot")
        )
        if "pretraining" in payload:
            payload["pretraining"] = cls._revalidate_training_result(
                payload["pretraining"]
            )
        if "base_generator_artifacts" in payload:
            artifacts = payload["base_generator_artifacts"]
            if not isinstance(artifacts, Mapping):
                raise TypeError("base generator artifacts must be a mapping")
            payload["base_generator_artifacts"] = {
                str(client_id): cls._revalidate_generator_artifact(artifact)
                for client_id, artifact in artifacts.items()
            }
        if "branches" in payload:
            branches = payload["branches"]
            if not isinstance(branches, Mapping):
                raise TypeError("resume branches must be a mapping")
            payload["branches"] = {
                str(name): cls._revalidate_branch_result(result)
                for name, result in branches.items()
            }
        if "active" in payload:
            active = payload["active"]
            if not isinstance(active, Mapping):
                raise TypeError("resume active progress must be a mapping")
            active = dict(active)
            active["current_snapshot"] = cls._revalidate_snapshot(
                active.get("current_snapshot")
            )
            active["best_snapshot"] = cls._revalidate_snapshot(
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
    def _progress_payload(progress) -> Dict[str, Any]:
        return {
            "current_snapshot": progress.current_snapshot,
            "best_snapshot": progress.best_snapshot,
            "best_value": progress.best_value,
            "stale_rounds": progress.stale_rounds,
            "records": list(progress.records),
        }

    def run(self) -> ScenarioResult:
        prepared = self.adapter.prepare()
        if prepared is not None:
            self.adapter = prepared
        client_ids = tuple(sorted(str(item) for item in self.adapter.client_ids))
        if not client_ids:
            raise ValueError("scenario adapter exposes no training clients")
        if (
            self.config.dataset.num_clients is not None
            and int(self.config.dataset.num_clients) != len(client_ids)
        ):
            raise ValueError(
                "configured dataset.num_clients does not match the adapter: "
                f"{self.config.dataset.num_clients} != {len(client_ids)}"
            )
        partition_hash = str(self.adapter.partition_hash)
        if not partition_hash:
            raise ValueError("scenario adapter exposes an empty partition hash")
        self._validate_adapter_contract(partition_hash)

        initial_state = self._resolve_initial_state()
        self._validate_model_state(initial_state)
        initial_snapshot = GlobalSnapshot(
            state=initial_state,
            round_index=0,
            dev_metrics={},
            model_spec=self.model_spec,
            partition_hash=partition_hash,
            metadata={"phase": "initial"},
        )
        clients_per_round = self.config.federation.clients_per_round or len(client_ids)
        pretrain_schedule = self._schedule(
            client_ids,
            self.config.federation.effective_pretrain_rounds,
            clients_per_round,
            self.config.federation.seed,
        )
        branch_schedule = self._schedule(
            client_ids,
            self.config.federation.attack_rounds,
            clients_per_round,
            self.config.federation.seed + 1,
        )
        malicious_clients = self._resolve_malicious_clients(client_ids)
        resume_state = self._load_resume_state()
        if resume_state is not None:
            saved_initial = resume_state.get("initial_snapshot")
            if not isinstance(saved_initial, GlobalSnapshot):
                raise TypeError("resume state is missing its initial snapshot")
            initial_snapshot = saved_initial
            expected_resume_values = {
                "partition_hash": partition_hash,
                "pretrain_schedule": pretrain_schedule,
                "branch_schedule": branch_schedule,
                "malicious_clients": malicious_clients,
            }
            for name, expected in expected_resume_values.items():
                if resume_state.get(name) != expected:
                    raise ValueError(f"resume state has mismatched {name}")

        def save_resume(phase: str, **values):
            return self._save_resume_state(
                phase=phase,
                partition_hash=partition_hash,
                initial_snapshot=initial_snapshot,
                pretrain_schedule=pretrain_schedule,
                branch_schedule=branch_schedule,
                malicious_clients=malicious_clients,
                **values,
            )

        manifest_config = copy.deepcopy(self.config.to_dict())
        manifest_config["federation"]["resume_from"] = None
        manifest = build_manifest(
            experiment_id=self._resume_config_hash[:12],
            config=manifest_config,
            seed=self.config.federation.seed,
            extra={
                "partition_hash": partition_hash,
                "client_ids": list(client_ids),
                "malicious_clients": list(malicious_clients),
                "pretrain_schedule": [list(row) for row in pretrain_schedule],
                "branch_schedule": [list(row) for row in branch_schedule],
            },
        )
        write_manifest(manifest, self.artifact_root / self.config.artifacts.manifest_name)
        save_snapshot(initial_snapshot, self.artifact_root / "snapshots" / "initial.pt")

        pretrain_runtime_seeds = {}

        def pretrain_data(client_id: str, snapshot: GlobalSnapshot):
            phase_round = int(snapshot.round_index)
            seed = self._client_round_seed(client_id, phase_round, "pretrain")
            pretrain_runtime_seeds[client_id] = seed
            bundle = self.adapter.get_client(client_id)
            self._seed_loader(bundle, seed)
            return bundle

        def pretrain_artifacts(client_id: str) -> Iterable[str]:
            self._seed_runtime(pretrain_runtime_seeds[client_id])
            return ()

        resume_phase = None if resume_state is None else str(resume_state.get("phase"))
        if resume_state is not None and resume_phase != "pretrain":
            pretraining = resume_state["pretraining"]
        else:
            pretrain_progress = (
                dict(resume_state.get("active", {}))
                if resume_phase == "pretrain"
                else {}
            )

            def save_pretrain_progress(progress):
                save_resume(
                    "pretrain",
                    active=self._progress_payload(progress),
                )

            existing_pretrain_records = list(
                pretrain_progress.get("records", ())
            )
            clean_coordinator = self._coordinator(partition_hash)
            pretraining = clean_coordinator.train(
                initial_snapshot=pretrain_progress.get(
                    "current_snapshot", initial_snapshot
                ),
                schedule=pretrain_schedule[len(existing_pretrain_records) :],
                data_resolver=pretrain_data,
                evaluate_dev=self._evaluate_dev,
                convergence=self._convergence_policy(),
                artifact_resolver=pretrain_artifacts,
                resume_best_snapshot=pretrain_progress.get("best_snapshot"),
                resume_best_value=pretrain_progress.get("best_value"),
                resume_stale_rounds=int(
                    pretrain_progress.get("stale_rounds", 0)
                ),
                existing_records=existing_pretrain_records,
                evaluate_initial=not bool(pretrain_progress),
                on_round_complete=save_pretrain_progress,
            )
        m_star = pretraining.best_snapshot
        save_snapshot(m_star, self.artifact_root / "snapshots" / "m_star.pt")
        save_snapshot(
            m_star,
            self.artifact_root / self.config.artifacts.snapshot_name,
        )
        self._persist_records("pretrain", pretraining.records)
        m_star_test = (
            dict(resume_state.get("m_star_test", {}))
            if resume_state is not None
            and resume_phase not in {"pretrain", "pretrain_complete"}
            else self._evaluate_test(m_star)
        )
        if resume_state is None or resume_phase in {"pretrain", "pretrain_complete"}:
            save_resume(
                "pretrain_complete",
                pretraining=pretraining,
                m_star_test=m_star_test,
            )

        lifecycle_state = None
        base_generator_artifacts = {}
        resume_after_base = resume_phase in {
            "base_complete",
            "branches_complete",
            "complete",
        } or (resume_phase is not None and resume_phase.startswith("branch:"))
        if resume_state is not None and resume_after_base:
            lifecycle_state = resume_state.get("lifecycle_state")
            base_generator_artifacts = dict(
                resume_state.get("base_generator_artifacts", {})
            )
        elif malicious_clients:
            if self.generator_lifecycle_factory is None:
                raise ValueError("an enabled generative attack requires a lifecycle factory")
            base_manager = self.generator_lifecycle_factory("base")
            next_client = 0
            if resume_phase == "base_generators":
                base_manager.load_state_dict(resume_state["base_manager_state"])
                base_generator_artifacts = dict(
                    resume_state.get("base_generator_artifacts", {})
                )
                next_client = int(resume_state.get("next_client", 0))
            for client_index, client_id in enumerate(
                malicious_clients[next_client:], start=next_client
            ):
                bundle = self.adapter.get_client(client_id)
                artifact = base_manager.ensure(
                    client_id,
                    m_star,
                    bundle.dataloader,
                    bundle.partition_hash,
                    m_star.round_index,
                )
                base_generator_artifacts[client_id] = artifact
                self._persist_generator_artifact("base", artifact)
                save_resume(
                    "base_generators",
                    pretraining=pretraining,
                    m_star_test=m_star_test,
                    base_manager_state=copy.deepcopy(base_manager.state_dict()),
                    base_generator_artifacts=dict(base_generator_artifacts),
                    next_client=client_index + 1,
                )
            if not hasattr(base_manager, "state_dict"):
                raise TypeError("generator lifecycle manager must support state_dict")
            lifecycle_state = copy.deepcopy(base_manager.state_dict())
        if resume_state is None or not resume_after_base:
            save_resume(
                "base_complete",
                pretraining=pretraining,
                m_star_test=m_star_test,
                lifecycle_state=lifecycle_state,
                base_generator_artifacts=dict(base_generator_artifacts),
                branches={},
            )

        branches = (
            dict(resume_state.get("branches", {}))
            if resume_state is not None and resume_after_base
            else {}
        )
        for branch_name in self.BRANCHES:
            if branch_name in branches:
                continue
            use_attack = branch_name != "clean" and bool(malicious_clients)
            use_defense = branch_name == "defended" and self.config.defense.enabled
            manager = None
            if use_attack:
                manager = self.generator_lifecycle_factory(branch_name)
                if not hasattr(manager, "load_state_dict"):
                    raise TypeError(
                        "generator lifecycle manager must support load_state_dict"
                )
                manager.load_state_dict(copy.deepcopy(lifecycle_state))
            active_progress = None
            if resume_phase == "branch:" + branch_name:
                active_progress = dict(resume_state.get("active", {}))
                if manager is not None:
                    manager.load_state_dict(resume_state["branch_manager_state"])
                reputation = getattr(self.defense_pipeline, "reputation", None)
                reputation_state = resume_state.get("reputation_state")
                if reputation is not None and reputation_state is not None:
                    reputation.load_state_dict(reputation_state)

            def save_branch_progress(progress, branch_name=branch_name, manager=manager):
                reputation = getattr(self.defense_pipeline, "reputation", None)
                save_resume(
                    "branch:" + branch_name,
                    pretraining=pretraining,
                    m_star_test=m_star_test,
                    lifecycle_state=lifecycle_state,
                    base_generator_artifacts=dict(base_generator_artifacts),
                    branches=dict(branches),
                    active=self._progress_payload(progress),
                    branch_manager_state=(
                        None
                        if manager is None
                        else copy.deepcopy(manager.state_dict())
                    ),
                    reputation_state=(
                        None
                        if reputation is None
                        else copy.deepcopy(reputation.state_dict())
                    ),
                )

            branches[branch_name] = self._run_branch(
                branch_name,
                m_star,
                branch_schedule,
                malicious_clients,
                manager,
                use_attack=use_attack,
                use_defense=use_defense,
                base_generator_artifacts=base_generator_artifacts,
                resume_progress=active_progress,
                on_round_complete=save_branch_progress,
            )
            save_resume(
                "branches_complete",
                pretraining=pretraining,
                m_star_test=m_star_test,
                lifecycle_state=lifecycle_state,
                base_generator_artifacts=dict(base_generator_artifacts),
                branches=dict(branches),
            )

        summary_path = self._persist_summary(
            initial_snapshot,
            pretraining,
            m_star,
            m_star_test,
            pretrain_schedule,
            branch_schedule,
            malicious_clients,
            branches,
        )
        manifest["results"] = {
            "m_star_hash": m_star.content_hash,
            "summary_path": str(summary_path),
            "branch_final_hashes": {
                name: result.final_snapshot.content_hash
                for name, result in branches.items()
            },
        }
        write_manifest(manifest, self.artifact_root / self.config.artifacts.manifest_name)
        save_resume(
            "complete",
            pretraining=pretraining,
            m_star_test=m_star_test,
            lifecycle_state=lifecycle_state,
            base_generator_artifacts=dict(base_generator_artifacts),
            branches=dict(branches),
        )
        return ScenarioResult(
            initial_snapshot=initial_snapshot,
            pretraining=pretraining,
            m_star=m_star,
            pretrain_schedule=pretrain_schedule,
            branch_schedule=branch_schedule,
            malicious_clients=malicious_clients,
            branches=branches,
            m_star_test_metrics=m_star_test,
            artifact_root=self.artifact_root,
            summary_path=summary_path,
        )

    def _run_branch(
        self,
        name: str,
        m_star: GlobalSnapshot,
        schedule: Sequence[Sequence[str]],
        malicious_clients: Sequence[str],
        lifecycle_manager,
        *,
        use_attack: bool,
        use_defense: bool,
        base_generator_artifacts: Mapping[str, GeneratorArtifact],
        resume_progress: Optional[Mapping[str, Any]] = None,
        on_round_complete: Optional[Callable[[Any], None]] = None,
    ) -> BranchResult:
        current_artifacts = dict(base_generator_artifacts if use_attack else {})
        if lifecycle_manager is not None and hasattr(lifecycle_manager, "artifacts"):
            current_artifacts.update(lifecycle_manager.artifacts)
        malicious_set = set(malicious_clients)
        runtime_seeds = {}

        def resolve_data(client_id: str, snapshot: GlobalSnapshot):
            bundle = self.adapter.get_client(client_id)
            phase_round = int(snapshot.round_index) - int(m_star.round_index)
            seed = self._client_round_seed(client_id, phase_round, "branch")
            runtime_seeds[client_id] = seed
            self._seed_loader(bundle, seed)
            if not use_attack or client_id not in malicious_set:
                current_artifacts.pop(client_id, None)
                return bundle
            artifact = lifecycle_manager.ensure(
                client_id,
                snapshot,
                bundle.dataloader,
                bundle.partition_hash,
                snapshot.round_index,
            )
            current_artifacts[client_id] = artifact
            self._persist_generator_artifact(name, artifact)
            return self.attack_strategy.prepare_dataloader(
                bundle,
                artifact,
                snapshot=snapshot,
                round_index=phase_round,
            )

        def artifact_ids(client_id: str) -> Iterable[str]:
            # Generator refresh can touch global RNG state. Reset immediately
            # before local victim training so all three branches are comparable.
            self._seed_runtime(runtime_seeds[client_id])
            artifact = current_artifacts.get(client_id)
            return () if artifact is None else (str(artifact.content_hash),)

        defense = self.defense_pipeline if use_defense else None
        branch_aggregator = (
            getattr(defense, "aggregator", None) or self.aggregator
            if defense is not None
            else self.aggregator
        )
        coordinator = self._coordinator(
            str(m_star.partition_hash),
            defense_pipeline=defense,
            aggregator=branch_aggregator,
        )
        progress = dict(resume_progress or {})
        existing_records = list(progress.get("records", ()))
        remaining_schedule = schedule[len(existing_records) :]
        training = coordinator.train(
            initial_snapshot=progress.get("current_snapshot", m_star),
            schedule=remaining_schedule,
            data_resolver=resolve_data,
            evaluate_dev=self._evaluate_dev,
            convergence=ConvergencePolicy(
                metric=self.config.federation.convergence_metric,
                mode=self.config.federation.convergence_mode,
                patience=0,
                min_delta=self.config.federation.min_delta,
            ),
            artifact_resolver=artifact_ids,
            resume_best_snapshot=progress.get("best_snapshot"),
            resume_best_value=progress.get("best_value"),
            resume_stale_rounds=int(progress.get("stale_rounds", 0)),
            existing_records=existing_records,
            evaluate_initial=not bool(progress),
            on_round_complete=on_round_complete,
        )
        self._persist_records(name, training.records)
        save_snapshot(
            training.final_snapshot,
            self.artifact_root / "snapshots" / name / "final.pt",
        )
        artifacts = (
            dict(lifecycle_manager.artifacts)
            if lifecycle_manager is not None and hasattr(lifecycle_manager, "artifacts")
            else dict(current_artifacts)
        )
        return BranchResult(
            name=name,
            training=training,
            test_metrics=self._evaluate_test(training.final_snapshot),
            generator_artifacts=artifacts,
            detection_metrics=(
                self._detection_metrics(training.records, malicious_set)
                if use_defense and malicious_set
                else {}
            ),
        )

    def _coordinator(
        self,
        partition_hash: str,
        defense_pipeline=None,
        aggregator=None,
    ) -> FedAvgCoordinator:
        return FedAvgCoordinator(
            client_trainer=self.client_trainer,
            aggregator=self.aggregator if aggregator is None else aggregator,
            model_spec=self.model_spec,
            partition_hash=partition_hash,
            defense_pipeline=defense_pipeline,
        )

    def _resolve_initial_state(self) -> Mapping[str, torch.Tensor]:
        if self.initial_state is not None:
            return _cpu_state(self.initial_state)
        model = self.adapter.build_model()
        if not hasattr(model, "state_dict"):
            raise TypeError("adapter.build_model must return a torch model")
        return _cpu_state(model.state_dict())

    def _validate_adapter_contract(self, partition_hash: str) -> None:
        configured_hash = self.config.dataset.partition_hash
        if configured_hash and str(configured_hash) != partition_hash:
            raise ValueError("configured partition_hash does not match the adapter")
        if hasattr(self.adapter, "num_classes"):
            actual_classes = int(self.adapter.num_classes)
            if int(self.config.dataset.num_classes) != actual_classes:
                raise ValueError(
                    "configured dataset.num_classes does not match the adapter: "
                    f"{self.config.dataset.num_classes} != {actual_classes}"
                )
        configured_shapes = {
            str(name): tuple(int(item) for item in shape)
            for name, shape in self.config.dataset.modality_shapes.items()
        }
        if configured_shapes and hasattr(self.adapter, "modality_shapes"):
            actual_shapes = {
                str(name): tuple(int(item) for item in shape)
                for name, shape in self.adapter.modality_shapes.items()
            }
            if configured_shapes != actual_shapes:
                raise ValueError(
                    "configured modality_shapes do not match the adapter"
                )

    def _validate_model_state(self, state: Mapping[str, torch.Tensor]) -> None:
        if not hasattr(self.adapter, "build_model"):
            return
        try:
            self.adapter.build_model(state)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError("initial model state is incompatible with the adapter") from exc

    def _evaluate_dev(self, snapshot: GlobalSnapshot) -> Mapping[str, float]:
        metrics = _scalar_metrics(self.adapter.evaluate_state(snapshot.state, "dev"))
        required = self.config.federation.convergence_metric
        if required not in metrics:
            raise KeyError("dev evaluation is missing convergence metric: " + required)
        return metrics

    def _evaluate_test(self, snapshot: GlobalSnapshot) -> Mapping[str, float]:
        if not self.config.evaluation.evaluate_test:
            return {}
        raw = self.adapter.evaluate_state(snapshot.state, "test")
        metrics = _scalar_metrics(raw)
        victim_class = self.config.attack.victim_eval_class
        goal_class = self.config.attack.goal_prediction_class
        truth = raw.get("truth") if isinstance(raw, Mapping) else None
        predictions = raw.get("pred") if isinstance(raw, Mapping) else None
        if (
            self.config.evaluation.evaluate_attack
            and
            victim_class is not None
            and goal_class is not None
            and truth is not None
            and predictions is not None
        ):
            selected = [
                int(prediction)
                for label, prediction in zip(truth, predictions)
                if int(label) == int(victim_class)
            ]
            if selected:
                metrics["attack_success_rate"] = sum(
                    prediction == int(goal_class) for prediction in selected
                ) / float(len(selected))
        return metrics

    @staticmethod
    def _detection_metrics(records, malicious_clients) -> Mapping[str, float]:
        labels = []
        scores = []
        predictions = []
        for record in records:
            for decision in record.defense_decisions:
                client_id = str(decision.client_id)
                labels.append(int(client_id in malicious_clients))
                normalized_scores = [
                    float(score) / max(float(decision.thresholds.get(name, 1.0)), 1e-12)
                    for name, score in decision.scores.items()
                ]
                score = max(normalized_scores) if normalized_scores else 0.0
                scores.append(score)
                predictions.append(int(decision.action != "accept"))
        if not labels:
            return {}
        return detection_metrics(
            labels,
            scores,
            predictions=predictions,
        )

    def _client_round_seed(self, client_id: str, round_index: int, phase: str) -> int:
        identity = (
            f"{self.config.federation.seed}\0{phase}\0{round_index}\0{client_id}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") % (2**31)

    @staticmethod
    def _seed_runtime(seed: int) -> None:
        random.seed(int(seed))
        try:
            import numpy as np

            np.random.seed(int(seed))
        except ImportError:
            pass
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    @staticmethod
    def _seed_loader(bundle, seed: int) -> None:
        loader = getattr(bundle, "dataloader", bundle)
        if not hasattr(loader, "dataset"):
            return
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if hasattr(loader, "generator"):
            loader.generator = generator
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "generator"):
            sampler.generator = generator

    def _convergence_policy(self) -> ConvergencePolicy:
        return ConvergencePolicy(
            metric=self.config.federation.convergence_metric,
            mode=self.config.federation.convergence_mode,
            patience=int(self.config.federation.patience or 0),
            min_delta=self.config.federation.min_delta,
        )

    @staticmethod
    def _schedule(
        client_ids: Sequence[str],
        rounds: int,
        clients_per_round: int,
        seed: int,
    ) -> Tuple[Tuple[str, ...], ...]:
        return tuple(
            tuple(row)
            for row in build_client_schedule_count(
                client_ids,
                rounds=int(rounds),
                clients_per_round=int(clients_per_round),
                seed=int(seed),
            )
        )

    def _resolve_malicious_clients(
        self, client_ids: Sequence[str]
    ) -> Tuple[str, ...]:
        if not self.config.attack.enabled:
            return ()
        if self.attack_strategy is None:
            raise ValueError("an enabled attack requires an attack strategy")
        if not self.config.generator.enabled:
            raise ValueError("generative poisoning requires generator.enabled=true")
        population = set(client_ids)
        explicit = tuple(sorted(self.config.attack.malicious_clients))
        if explicit:
            unknown = sorted(set(explicit) - population)
            if unknown:
                raise ValueError("unknown malicious client(s): " + ", ".join(unknown))
            configured_count = int(self.config.attack.malicious_client_count)
            if configured_count not in (0, len(explicit)):
                raise ValueError(
                    "malicious_client_count conflicts with malicious_clients"
                )
            return explicit
        count = int(self.config.attack.malicious_client_count)
        if count < 1:
            raise ValueError("an enabled attack requires at least one malicious client")
        return tuple(
            select_malicious_clients(
                client_ids, count=count, seed=self.config.federation.seed
            )
        )

    def _persist_records(self, phase: str, records: Sequence[Any]) -> None:
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

    def _persist_generator_artifact(
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

    def _persist_summary(
        self,
        initial_snapshot: GlobalSnapshot,
        pretraining: TrainingResult,
        m_star: GlobalSnapshot,
        m_star_test: Mapping[str, float],
        pretrain_schedule: Sequence[Sequence[str]],
        branch_schedule: Sequence[Sequence[str]],
        malicious_clients: Sequence[str],
        branches: Mapping[str, BranchResult],
    ) -> Path:
        payload = {
            "schema_version": 1,
            "config_hash": self._resume_config_hash,
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
        return _write_json(payload, self.artifact_root / "summary.json")


def _checkpoint_state_from_payload(payload) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("model checkpoint must contain a mapping")
    for key in ("model_state_dict", "state_dict", "state", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate:
            return candidate
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise ValueError("model checkpoint does not contain a model state")


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    return _checkpoint_state_from_payload(torch.load(path, map_location="cpu"))


def build_default_runner(
    config: ScenarioConfig,
    *,
    artifact_root: Optional[Path] = None,
) -> ScenarioRunner:
    """Build the production UCF101/FedMM scenario from strict configuration."""

    if config.dataset.name.lower() != "ucf101":
        raise ValueError("the first scenario runner release supports only UCF101")
    if config.model.name != "MMActionClassifier":
        raise ValueError(
            "the first scenario runner release supports only MMActionClassifier"
        )
    supported_constructor = "fed_multimodal.model.mm_models:MMActionClassifier"
    if config.model.constructor not in {None, supported_constructor}:
        raise ValueError(
            "unsupported model.constructor: " + str(config.model.constructor)
        )
    if not config.dataset.root:
        raise ValueError("dataset.root is required for the UCF101 adapter")
    if config.dataset.partition_path is not None:
        raise ValueError(
            "dataset.partition_path is not supported; UCF101 uses FedMM paths under dataset.root"
        )
    generator_family = config.generator.family.lower()
    generator_variant = config.generator.variant.lower()
    if generator_family != "kplus1":
        raise ValueError("generative poisoning currently supports generator.family=kplus1")
    if generator_variant not in {"dtm", "temporal_adaptive"}:
        raise ValueError("generator.variant must be dtm or temporal_adaptive")
    if config.attack.strategy != "generative_feature_poisoning":
        raise ValueError("only generative_feature_poisoning is implemented")
    if config.defense.policy != "reject_if_both_clip_if_one":
        raise ValueError("unsupported defense.policy: " + config.defense.policy)
    allowed_metrics = {
        "accuracy",
        "acc",
        "uar",
        "f1",
        "loss",
        "top5_acc",
        "attack_success_rate",
    }
    unknown_metrics = sorted(set(config.evaluation.metrics) - allowed_metrics)
    if unknown_metrics:
        raise ValueError(
            "unsupported evaluation metric(s): " + ", ".join(unknown_metrics)
        )
    if config.evaluation.options:
        raise ValueError("evaluation.options are not implemented in the first release")
    if config.artifacts.options:
        raise ValueError("artifacts.options are not implemented in the first release")

    allowed_dataset_options = {
        "missing_modality",
        "missing_modality_rate",
        "missing_label",
        "missing_label_rate",
        "label_noisy",
        "label_noise_level",
        "audio_seq_len",
        "video_seq_len",
    }
    unknown_dataset_options = sorted(
        set(config.dataset.options) - allowed_dataset_options
    )
    if unknown_dataset_options:
        raise ValueError(
            "unsupported dataset.options: " + ", ".join(unknown_dataset_options)
        )

    allowed_federation_options = {"device", "mu"}
    unknown_federation_options = sorted(
        set(config.federation.options) - allowed_federation_options
    )
    if unknown_federation_options:
        raise ValueError(
            "unsupported federation.options: "
            + ", ".join(unknown_federation_options)
        )
    allowed_model_options = {"checkpoint_sha256", "checkpoint_hash"}
    unknown_model_options = sorted(set(config.model.options) - allowed_model_options)
    if unknown_model_options:
        raise ValueError(
            "unsupported model.options: " + ", ".join(unknown_model_options)
        )
    allowed_model_kwargs = {
        "hid_size",
        "attention",
        "attention_name",
        "att",
        "att_name",
    }
    unknown_model_kwargs = sorted(set(config.model.kwargs) - allowed_model_kwargs)
    if unknown_model_kwargs:
        raise ValueError(
            "unsupported model.kwargs: " + ", ".join(unknown_model_kwargs)
        )
    if (
        "attention" in config.model.kwargs
        and "att" in config.model.kwargs
        and bool(config.model.kwargs["attention"])
        != bool(config.model.kwargs["att"])
    ):
        raise ValueError("model attention and att aliases conflict")
    if (
        "attention_name" in config.model.kwargs
        and "att_name" in config.model.kwargs
        and str(config.model.kwargs["attention_name"])
        != str(config.model.kwargs["att_name"])
    ):
        raise ValueError("model attention_name and att_name aliases conflict")

    allowed_attack_options = {"generation_batch_size"}
    unknown_attack_options = sorted(
        set(config.attack.options) - allowed_attack_options
    )
    if unknown_attack_options:
        raise ValueError(
            "unsupported attack.options: " + ", ".join(unknown_attack_options)
        )
    allowed_defense_options = {"minimum_reputation", "initial_reputation"}
    unknown_defense_options = sorted(
        set(config.defense.options) - allowed_defense_options
    )
    if unknown_defense_options:
        raise ValueError(
            "unsupported defense.options: " + ", ".join(unknown_defense_options)
        )
    for field_name in (
        "condition_class",
        "assigned_train_label",
        "victim_eval_class",
        "goal_prediction_class",
    ):
        label = getattr(config.attack, field_name)
        if label is not None and int(label) >= int(config.dataset.num_classes):
            raise ValueError(
                f"attack.{field_name} must be in [0, "
                f"{int(config.dataset.num_classes) - 1}]"
            )
    if config.attack.enabled and not config.generator.enabled:
        raise ValueError("generative poisoning requires generator.enabled=true")

    generator_options = dict(config.generator.options)
    max_batches = generator_options.pop("max_batches", None)
    log_interval = int(generator_options.pop("log_interval", 0))
    generator_overrides = dict(config.generator.loss)
    conflicts = sorted(set(generator_overrides) & set(generator_options))
    if conflicts:
        raise ValueError(
            "generator.loss and generator.options overlap: " + ", ".join(conflicts)
        )
    generator_overrides.update(generator_options)
    protected_generator_fields = {
        "num_classes",
        "fake_class",
        "audio_seq_len",
        "audio_feat_dim",
        "video_seq_len",
        "video_feat_dim",
        "seed",
        "lr_g",
        "lr_d",
    }
    protected_overrides = sorted(
        set(generator_overrides) & protected_generator_fields
    )
    if protected_overrides:
        raise ValueError(
            "generator options cannot override scenario-owned field(s): "
            + ", ".join(protected_overrides)
        )
    generator_overrides["lr_g"] = config.generator.learning_rate
    generator_overrides["lr_d"] = config.generator.learning_rate
    if generator_variant == "dtm":
        from fed_multimodal.dtm_poison_gan import DTMGANConfig

        allowed_generator_fields = set(DTMGANConfig.__dataclass_fields__)
    else:
        from fed_multimodal.temporal_adaptive_gan import TemporalAdaptiveGANConfig

        allowed_generator_fields = set(TemporalAdaptiveGANConfig.__dataclass_fields__)
    unknown_generator_options = sorted(
        set(generator_overrides) - allowed_generator_fields
    )
    if unknown_generator_options:
        raise ValueError(
            "unsupported generator option(s): "
            + ", ".join(unknown_generator_options)
        )

    from mflpoison.adapters.fedmm import (
        FedAvgClientTrainer,
        FedMMGeneratorTrainer,
        UCF101FedMMAdapter,
    )
    from mflpoison.attacks import GenerativeFeaturePoisoningStrategy
    from mflpoison.defenses import (
        CosineMADDetector,
        DefensePipeline,
        EWMAReputation,
        NormMADDetector,
    )
    from mflpoison.defenses.registry import AGGREGATOR_REGISTRY
    from mflpoison.defenses.update_filter import NormClipper
    from mflpoison.generators import GeneratorLifecycleManager, load_generator_backend

    model_kwargs = dict(config.model.kwargs)
    adapter = UCF101FedMMAdapter(
        data_dir=config.dataset.root,
        alpha=1.0 if config.dataset.alpha is None else config.dataset.alpha,
        fold=config.dataset.fold,
        batch_size=config.federation.batch_size,
        hid_size=int(model_kwargs.get("hid_size", 64)),
        attention=bool(model_kwargs.get("attention", model_kwargs.get("att", False))),
        attention_name=str(
            model_kwargs.get("attention_name", model_kwargs.get("att_name", "base"))
        ),
        **dict(config.dataset.options),
    )
    if int(config.dataset.num_classes) != int(adapter.num_classes):
        raise ValueError(
            "configured dataset.num_classes does not match UCF101: "
            f"{config.dataset.num_classes} != {adapter.num_classes}"
        )
    device = str(config.federation.options.get("device", "cpu"))
    client_trainer = FedAvgClientTrainer(
        adapter.build_model,
        device=device,
        learning_rate=config.federation.learning_rate,
        local_epochs=config.federation.local_epochs,
        mu=float(config.federation.options.get("mu", 0.0)),
        seed=config.federation.seed,
    )
    clean_aggregator = AGGREGATOR_REGISTRY.create("weighted_mean")
    resolved_root = Path(config.artifacts.root_dir if artifact_root is None else artifact_root)

    lifecycle_factory = None
    attack_strategy = None
    if config.attack.enabled:
        if config.generator.checkpoint_dir is None:
            checkpoint_root = resolved_root / "generator_checkpoints"
        else:
            checkpoint_root = Path(config.generator.checkpoint_dir)
            if not checkpoint_root.is_absolute():
                checkpoint_root = resolved_root / checkpoint_root

        def lifecycle_factory(phase: str):
            output_dir = checkpoint_root / phase

            def trainer_factory(client_id: str):
                del client_id
                trainer = FedMMGeneratorTrainer(
                    variant=generator_variant,
                    output_dir=output_dir,
                    model_metadata=adapter.model_metadata(),
                    modality_shapes=adapter.modality_shapes,
                    num_classes=adapter.num_classes,
                    epochs=config.generator.epochs,
                    max_batches=max_batches,
                    log_interval=log_interval,
                    device=device,
                    config_overrides=generator_overrides,
                    batch_size=config.generator.batch_size,
                )
                return trainer

            return GeneratorLifecycleManager(
                trainer_factory=trainer_factory,
                variant=generator_variant,
                mode=config.generator.lifecycle,
                refresh_every=config.generator.refresh_interval,
                seed=config.generator.seed,
            )

        attack_spec = AttackSpec(
            condition_class=config.attack.condition_class,
            assigned_train_label=config.attack.assigned_train_label,
            victim_eval_class=config.attack.victim_eval_class,
            goal_prediction_class=config.attack.goal_prediction_class,
            poison_ratio=config.attack.poison_ratio,
            poison_count=config.attack.poison_count,
            injection_mode=config.attack.injection_mode,
            start_round=config.attack.start_round,
            end_round=config.attack.end_round,
            every=config.attack.every,
            seed=config.generator.seed,
            metadata=dict(config.attack.options),
        )
        attack_options = dict(config.attack.options)
        generation_batch_size = int(attack_options.pop("generation_batch_size", 64))
        attack_strategy = GenerativeFeaturePoisoningStrategy(
            attack_spec,
            seed=config.generator.seed,
            generation_batch_size=generation_batch_size,
            backend_factory=lambda artifact: load_generator_backend(
                artifact.variant,
                artifact.checkpoint_path,
                device=device,
            ),
        )

    defense_pipeline = None
    if config.defense.enabled:
        detector_specs = config.defense.detectors or (
            {"name": "norm_mad"},
            {"name": "cosine_mad"},
        )
        detectors = []
        detector_types = {
            "norm_mad": NormMADDetector,
            "cosine_mad": CosineMADDetector,
            "cosine_center": CosineMADDetector,
        }
        for detector_spec in detector_specs:
            values = dict(detector_spec)
            name = str(values.pop("name")).lower()
            try:
                detector_type = detector_types[name]
            except KeyError as exc:
                raise KeyError("unknown defense detector: " + name) from exc
            detectors.append(detector_type(**values))
        sanitizer_values = dict(config.defense.sanitizer)
        sanitizer_name = str(sanitizer_values.pop("name", "norm_clipping"))
        if sanitizer_name != "norm_clipping":
            raise KeyError("unknown defense sanitizer: " + sanitizer_name)
        sanitizer = NormClipper(**{"max_norm": None, **sanitizer_values})
        aggregator_values = dict(config.defense.aggregator)
        aggregator_name = str(aggregator_values.pop("name", "weighted_mean"))
        defended_aggregator = AGGREGATOR_REGISTRY.create(
            aggregator_name, **aggregator_values
        )
        reputation = None
        if config.defense.ewma_decay is not None:
            reputation = EWMAReputation(
                decay=config.defense.ewma_decay,
                minimum_reputation=float(
                    config.defense.options.get("minimum_reputation", 0.5)
                ),
                initial_reputation=float(
                    config.defense.options.get("initial_reputation", 1.0)
                ),
            )
        defense_pipeline = DefensePipeline(
            detectors=detectors,
            sanitizer=sanitizer,
            reputation=reputation,
            aggregator=defended_aggregator,
        )

    initial_state = None
    if config.model.checkpoint_path:
        checkpoint_path = Path(config.model.checkpoint_path)
        expected_hash = config.model.options.get(
            "checkpoint_sha256", config.model.options.get("checkpoint_hash")
        )
        if expected_hash is not None and file_sha256(checkpoint_path) != str(expected_hash):
            raise ValueError("model checkpoint hash does not match configuration")
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        initial_state = _checkpoint_state_from_payload(checkpoint_payload)
        legacy_args = (
            dict(checkpoint_payload.get("args", {}))
            if isinstance(checkpoint_payload, Mapping)
            and isinstance(checkpoint_payload.get("args", {}), Mapping)
            else {}
        )
        expected_legacy_args = {
            "hid_size": adapter.hid_size,
            "att": adapter.attention,
            "att_name": adapter.attention_name,
        }
        for name, expected in expected_legacy_args.items():
            if name in legacy_args and legacy_args[name] != expected:
                raise ValueError(
                    f"legacy checkpoint {name} does not match model configuration"
                )
    return ScenarioRunner(
        config,
        adapter=adapter,
        client_trainer=client_trainer,
        aggregator=clean_aggregator,
        initial_state=initial_state,
        model_spec=ModelSpec(
            name=config.model.name,
            constructor=config.model.constructor,
            kwargs=dict(config.model.kwargs),
            metadata=adapter.model_metadata(),
        ),
        generator_lifecycle_factory=lifecycle_factory,
        attack_strategy=attack_strategy,
        defense_pipeline=defense_pipeline,
        artifact_root=resolved_root,
    )
