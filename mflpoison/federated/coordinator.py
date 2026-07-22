from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from mflpoison.core.types import AggregationResult, GlobalSnapshot, RoundRecord


@dataclass
class ConvergencePolicy:
    metric: str = "acc"
    mode: str = "max"
    patience: int = 0
    min_delta: float = 0.0

    def __post_init__(self):
        if self.mode not in ("max", "min"):
            raise ValueError("convergence mode must be max or min")
        if self.patience < 0:
            raise ValueError("patience cannot be negative")

    def improved(self, value: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        if self.mode == "max":
            return value > best + float(self.min_delta)
        return value < best - float(self.min_delta)


@dataclass
class TrainingResult:
    best_snapshot: GlobalSnapshot
    final_snapshot: GlobalSnapshot
    records: List[RoundRecord] = field(default_factory=list)
    stopped_early: bool = False


@dataclass
class TrainingProgress:
    current_snapshot: GlobalSnapshot
    best_snapshot: GlobalSnapshot
    best_value: float
    stale_rounds: int
    records: List[RoundRecord]


def _cpu_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


class FedAvgCoordinator:
    """Stateful multi-round coordinator around typed client deltas."""

    def __init__(
        self,
        client_trainer,
        aggregator,
        model_spec,
        partition_hash: str,
        defense_pipeline=None,
    ):
        self.client_trainer = client_trainer
        self.aggregator = aggregator
        self.model_spec = model_spec
        self.partition_hash = str(partition_hash)
        if defense_pipeline is None:
            from mflpoison.defenses import DefensePipeline

            defense_pipeline = DefensePipeline(
                detectors=[],
                aggregator=aggregator,
            )
        self.defense_pipeline = defense_pipeline

    def make_snapshot(
        self,
        state: Mapping[str, torch.Tensor],
        round_index: int,
        dev_metrics: Optional[Mapping[str, float]] = None,
    ) -> GlobalSnapshot:
        return GlobalSnapshot(
            state=_cpu_state(state),
            round_index=int(round_index),
            dev_metrics=dict(dev_metrics or {}),
            model_spec=self.model_spec,
            partition_hash=self.partition_hash,
        )

    def run_round(
        self,
        snapshot: GlobalSnapshot,
        selected_clients: Sequence[str],
        data_resolver: Callable[[str, GlobalSnapshot], object],
        artifact_resolver: Optional[Callable[[str], Iterable[str]]] = None,
    ):
        updates = []
        for client_id in selected_clients:
            client_snapshot = GlobalSnapshot(
                state=snapshot.state,
                round_index=snapshot.round_index,
                dev_metrics=snapshot.dev_metrics,
                model_spec=snapshot.model_spec,
                partition_hash=snapshot.partition_hash,
                metadata=snapshot.metadata,
                content_hash=snapshot.content_hash,
            )
            bundle = data_resolver(str(client_id), client_snapshot)
            artifacts = [] if artifact_resolver is None else list(
                artifact_resolver(str(client_id))
            )
            update = self.client_trainer.train(
                client_id=str(client_id),
                snapshot=client_snapshot,
                dataloader=bundle.dataloader,
                clean_num_samples=bundle.clean_num_samples,
                artifact_ids=artifacts,
            )
            if client_snapshot.calculate_hash() != client_snapshot.content_hash:
                raise RuntimeError(
                    "client code mutated its read-only global snapshot: "
                    + str(client_id)
                )
            updates.append(update)
        if not updates:
            raise RuntimeError("federated round produced no client updates")

        defended = self.defense_pipeline.process(
            updates,
            snapshot,
            expected_base_snapshot_hash=snapshot.content_hash,
            aggregator=self.aggregator,
        )
        next_state = (
            snapshot.state
            if defended.aggregated_state is None
            else defended.aggregated_state
        )
        decisions = list(defended.decisions)
        processed_updates = list(defended.sanitized)
        audit = defended.aggregation_audit
        diagnostics = asdict(audit) if is_dataclass(audit) else dict(vars(audit))
        aggregation_result = AggregationResult(
            state=next_state,
            decisions=decisions,
            diagnostics=diagnostics,
        )
        next_snapshot = self.make_snapshot(
            next_state,
            round_index=int(snapshot.round_index) + 1,
        )
        return next_snapshot, updates, processed_updates, aggregation_result

    def train(
        self,
        initial_snapshot: GlobalSnapshot,
        schedule: Sequence[Sequence[str]],
        data_resolver: Callable[[str, GlobalSnapshot], object],
        evaluate_dev: Callable[[GlobalSnapshot], Mapping[str, float]],
        convergence: Optional[ConvergencePolicy] = None,
        artifact_resolver: Optional[Callable[[str], Iterable[str]]] = None,
        resume_best_snapshot: Optional[GlobalSnapshot] = None,
        resume_best_value: Optional[float] = None,
        resume_stale_rounds: int = 0,
        existing_records: Sequence[RoundRecord] = (),
        evaluate_initial: bool = True,
        on_round_complete: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> TrainingResult:
        policy = convergence or ConvergencePolicy(patience=0)
        current = initial_snapshot
        if evaluate_initial:
            initial_metrics = dict(evaluate_dev(initial_snapshot))
            if policy.metric not in initial_metrics:
                raise KeyError("missing convergence metric: " + policy.metric)
            best = self.make_snapshot(
                initial_snapshot.state,
                initial_snapshot.round_index,
                initial_metrics,
            )
            best_value = float(initial_metrics[policy.metric])
            stale_rounds = 0
        else:
            if resume_best_snapshot is None or resume_best_value is None:
                raise ValueError(
                    "resumed training requires best snapshot and metric value"
                )
            best = resume_best_snapshot
            best_value = float(resume_best_value)
            stale_rounds = int(resume_stale_rounds)
            if stale_rounds < 0:
                raise ValueError("resume_stale_rounds cannot be negative")
        records = list(existing_records)
        stopped_early = bool(
            policy.patience and stale_rounds >= policy.patience
        )

        for selected_clients in (() if stopped_early else schedule):
            next_snapshot, updates, processed_updates, aggregation_result = self.run_round(
                current,
                selected_clients,
                data_resolver,
                artifact_resolver=artifact_resolver,
            )
            dev_metrics = dict(evaluate_dev(next_snapshot))
            next_snapshot = self.make_snapshot(
                next_snapshot.state,
                next_snapshot.round_index,
                dev_metrics,
            )
            if policy.metric not in dev_metrics:
                raise KeyError("missing convergence metric: " + policy.metric)
            value = float(dev_metrics[policy.metric])
            if policy.improved(value, best_value):
                best = next_snapshot
                best_value = value
                stale_rounds = 0
            else:
                stale_rounds += 1

            records.append(
                RoundRecord(
                    round_index=int(current.round_index),
                    base_snapshot_hash=current.content_hash,
                    selected_client_ids=list(selected_clients),
                    raw_updates=list(updates),
                    defense_decisions=list(aggregation_result.decisions),
                    processed_updates=list(processed_updates),
                    aggregation_result=aggregation_result,
                    evaluation=dev_metrics,
                )
            )
            current = next_snapshot
            if on_round_complete is not None:
                on_round_complete(
                    TrainingProgress(
                        current_snapshot=current,
                        best_snapshot=best,
                        best_value=float(best_value),
                        stale_rounds=stale_rounds,
                        records=list(records),
                    )
                )
            if policy.patience and stale_rounds >= policy.patience:
                stopped_early = True
                break

        return TrainingResult(
            best_snapshot=best,
            final_snapshot=current,
            records=records,
            stopped_early=stopped_early,
        )
