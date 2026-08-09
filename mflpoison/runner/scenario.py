"""Orchestrate clean, attacked, and defended branches from one M* snapshot."""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from mflpoison.artifacts import (
    build_manifest,
    load_snapshot,
    save_snapshot,
    write_manifest,
)
from mflpoison.attacks import select_malicious_clients
from mflpoison.core.config import ScenarioConfig
from mflpoison.core.types import GeneratorArtifact, GlobalSnapshot, ModelSpec
from mflpoison.evaluation import detection_metrics, targeted_classification_metrics
from mflpoison.federated import (
    ConvergencePolicy,
    FedAvgCoordinator,
    TrainingResult,
    build_client_schedule_count,
)

from .canonical_clean import (
    load_canonical_clean,
    manifest_source_identity,
    validate_canonical_clean,
)
from .persistence import ResultStore
from .runtime import (
    client_round_seed,
    cpu_state,
    scalar_metrics,
    seed_loader,
    seed_runtime,
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
    run_dir: Path
    summary_path: Path


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
        artifact_dir: Optional[Path] = None,
    ):
        if not isinstance(config, ScenarioConfig):
            raise TypeError("config must be a ScenarioConfig")
        self.config = config
        self.adapter = adapter
        self.client_trainer = client_trainer
        self.aggregator = aggregator
        self.initial_state = None if initial_state is None else cpu_state(initial_state)
        self.model_spec = model_spec or ModelSpec(
            name=config.model.name,
            constructor=config.model.constructor,
            kwargs=dict(config.model.kwargs),
        )
        self.generator_lifecycle_factory = generator_lifecycle_factory
        self.attack_strategy = attack_strategy
        self.defense_pipeline = defense_pipeline
        self.run_dir = Path(
            config.artifact.root_dir if artifact_dir is None else artifact_dir
        )
        self._result_store = ResultStore(config, self.run_dir)

    def run(self) -> ScenarioResult:
        selected_branches = self.config.selected_branches
        canonical_clean_path = self.config.evaluation.canonical_clean_path
        non_clean_selected = any(
            name in {"attack", "defended"} for name in selected_branches
        )
        if non_clean_selected and "clean" not in selected_branches:
            if self.config.federation.m_star_path is None:
                raise ValueError(
                    "attack-only or defended-only requires federation.m_star_path"
                )
            if canonical_clean_path is None:
                raise ValueError(
                    "attack-only or defended-only requires a canonical clean baseline"
                )
        if canonical_clean_path is not None:
            if not non_clean_selected:
                raise ValueError(
                    "canonical clean comparison requires an attack or defended branch"
                )
            if "clean" in selected_branches:
                raise ValueError(
                    "canonical clean comparison must use attack-only or defended-only branches"
                )
        seed_runtime(self.config.federation.seed)
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
        malicious_clients = (
            self._resolve_malicious_clients(client_ids)
            if any(name != "clean" for name in selected_branches)
            else ()
        )

        manifest_config = copy.deepcopy(self.config.to_dict())
        manifest = build_manifest(
            experiment_id=self.run_dir.name,
            config=manifest_config,
            seed=self.config.federation.seed,
            extra={
                "partition_hash": partition_hash,
                "client_ids": list(client_ids),
                "malicious_clients": list(malicious_clients),
                "pretrain_schedule": [list(row) for row in pretrain_schedule],
                "branch_schedule": [list(row) for row in branch_schedule],
                "selected_branches": list(selected_branches),
                "m_star_source": (
                    None
                    if self.config.federation.m_star_path is None
                    else {"path": self.config.federation.m_star_path}
                ),
            },
        )
        manifest["status"] = "running"
        write_manifest(manifest, self.run_dir / "run_manifest.json")
        save_snapshot(initial_snapshot, self.run_dir / "checkpoints" / "initial.pt")

        pretrain_runtime_seeds = {}

        def pretrain_data(client_id: str, snapshot: GlobalSnapshot):
            phase_round = int(snapshot.round_index)
            seed = client_round_seed(
                self.config.federation.seed,
                client_id,
                phase_round,
                "pretrain",
            )
            pretrain_runtime_seeds[client_id] = seed
            bundle = self.adapter.get_client(client_id)
            seed_loader(bundle, seed)
            return bundle

        def pretrain_artifacts(client_id: str) -> Iterable[str]:
            seed_runtime(pretrain_runtime_seeds[client_id])
            return ()

        configured_m_star = self._configured_m_star(partition_hash)
        if configured_m_star is not None:
            pretraining = TrainingResult(
                best_snapshot=configured_m_star,
                final_snapshot=configured_m_star,
                records=[],
                stopped_early=False,
            )
        else:
            clean_coordinator = self._coordinator(partition_hash)
            pretraining = clean_coordinator.train(
                initial_snapshot=initial_snapshot,
                schedule=pretrain_schedule,
                data_resolver=pretrain_data,
                evaluate_dev=self._evaluate_dev,
                convergence=self._convergence_policy(),
                artifact_resolver=pretrain_artifacts,
            )
        m_star = pretraining.best_snapshot
        save_snapshot(m_star, self.run_dir / "checkpoints" / "m_star.pt")
        self._result_store.persist_records("pretrain", pretraining.records)
        m_star_test = self._evaluate_test(m_star)

        if canonical_clean_path is not None:
            canonical_clean = load_canonical_clean(Path(canonical_clean_path))
            if "attack_source_sample_count" not in m_star_test:
                raise ValueError(
                    "canonical clean comparison requires attack source metrics"
                )
            source_count_value = float(
                m_star_test["attack_source_sample_count"]
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
            validate_canonical_clean(
                canonical_clean,
                seed=self.config.federation.seed,
                partition_hash=partition_hash,
                m_star_hash=m_star.content_hash,
                branch_schedule=branch_schedule,
                victim_eval_class=self.config.attack.victim_eval_class,
                goal_prediction_class=self.config.attack.goal_prediction_class,
                attack_source_sample_count=source_count,
                comparison_protocol=manifest_config,
                source_identity=manifest_source_identity(manifest),
            )
            self._result_store.set_canonical_clean(canonical_clean)

        lifecycle_state = None
        base_generator_artifacts = {}
        if malicious_clients:
            if self.generator_lifecycle_factory is None:
                raise ValueError("an enabled generative attack requires a lifecycle factory")
            base_manager = self.generator_lifecycle_factory("base")
            for client_id in malicious_clients:
                bundle = self.adapter.get_client(client_id)
                artifact = base_manager.ensure(
                    client_id,
                    m_star,
                    bundle.dataloader,
                    bundle.partition_hash,
                    m_star.round_index,
                )
                base_generator_artifacts[client_id] = artifact
                self._result_store.persist_generator_artifact("base", artifact)
            if not hasattr(base_manager, "state_dict"):
                raise TypeError("generator lifecycle manager must support state_dict")
            lifecycle_state = copy.deepcopy(base_manager.state_dict())

        branches = {}
        for branch_name in selected_branches:
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

            branches[branch_name] = self._run_branch(
                branch_name,
                m_star,
                branch_schedule,
                malicious_clients,
                manager,
                use_attack=use_attack,
                use_defense=use_defense,
                base_generator_artifacts=base_generator_artifacts,
            )

        summary_path = self._result_store.persist_summary(
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
        manifest["status"] = "completed"
        write_manifest(manifest, self.run_dir / "run_manifest.json")
        return ScenarioResult(
            initial_snapshot=initial_snapshot,
            pretraining=pretraining,
            m_star=m_star,
            pretrain_schedule=pretrain_schedule,
            branch_schedule=branch_schedule,
            malicious_clients=malicious_clients,
            branches=branches,
            m_star_test_metrics=m_star_test,
            run_dir=self.run_dir,
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
    ) -> BranchResult:
        current_artifacts = dict(base_generator_artifacts if use_attack else {})
        if lifecycle_manager is not None and hasattr(lifecycle_manager, "artifacts"):
            current_artifacts.update(lifecycle_manager.artifacts)
        malicious_set = set(malicious_clients)
        runtime_seeds = {}

        def resolve_data(client_id: str, snapshot: GlobalSnapshot):
            bundle = self.adapter.get_client(client_id)
            phase_round = int(snapshot.round_index) - int(m_star.round_index)
            seed = client_round_seed(
                self.config.federation.seed,
                client_id,
                phase_round,
                "branch",
            )
            runtime_seeds[client_id] = seed
            seed_loader(bundle, seed)
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
            self._result_store.persist_generator_artifact(name, artifact)
            return self.attack_strategy.prepare_dataloader(
                bundle,
                artifact,
                snapshot=snapshot,
                round_index=phase_round,
            )

        def artifact_ids(client_id: str) -> Iterable[str]:
            # Generator refresh can touch global RNG state. Reset immediately
            # before local victim training so all three branches are comparable.
            seed_runtime(runtime_seeds[client_id])
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
        training = coordinator.train(
            initial_snapshot=m_star,
            schedule=schedule,
            data_resolver=resolve_data,
            evaluate_dev=self._evaluate_dev,
            convergence=ConvergencePolicy(
                metric=self.config.federation.convergence_metric,
                mode=self.config.federation.convergence_mode,
                patience=0,
                min_delta=self.config.federation.min_delta,
            ),
            artifact_resolver=artifact_ids,
        )
        self._result_store.persist_records(name, training.records)
        save_snapshot(
            training.final_snapshot,
            self.run_dir / "checkpoints" / f"{name}_last.pt",
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
            return cpu_state(self.initial_state)
        model = self.adapter.build_model()
        if not hasattr(model, "state_dict"):
            raise TypeError("adapter.build_model must return a torch model")
        return cpu_state(model.state_dict())

    def _configured_m_star(self, partition_hash: str) -> Optional[GlobalSnapshot]:
        configured_path = self.config.federation.m_star_path
        if configured_path is None:
            return None
        snapshot = load_snapshot(configured_path)
        if snapshot.partition_hash != str(partition_hash):
            raise ValueError("configured M* belongs to a different data partition")
        if snapshot.model_spec.to_dict() != self.model_spec.to_dict():
            raise ValueError("configured M* model specification does not match the runner")
        self._validate_model_state(snapshot.state)
        return snapshot

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
        metrics = scalar_metrics(self.adapter.evaluate_state(snapshot.state, "dev"))
        required = self.config.federation.convergence_metric
        if required not in metrics:
            raise KeyError("dev evaluation is missing convergence metric: " + required)
        return metrics

    def _evaluate_test(self, snapshot: GlobalSnapshot) -> Mapping[str, float]:
        if not self.config.evaluation.evaluate_test:
            return {}
        raw = self.adapter.evaluate_state(snapshot.state, "test")
        metrics = scalar_metrics(raw)
        victim_class = self.config.attack.victim_eval_class
        goal_class = self.config.attack.goal_prediction_class
        truth = raw.get("truth") if isinstance(raw, Mapping) else None
        predictions = raw.get("pred") if isinstance(raw, Mapping) else None
        if (
            self.config.evaluation.evaluate_attack
            and victim_class is not None
            and goal_class is not None
            and truth is not None
            and predictions is not None
        ):
            metrics.update(
                targeted_classification_metrics(
                    truth,
                    predictions,
                    victim_class=int(victim_class),
                    goal_class=int(goal_class),
                )
            )
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
