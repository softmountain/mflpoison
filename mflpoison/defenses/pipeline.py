from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

try:
    from mflpoison.core.types import AggregationResult, DefenseDecision
except ImportError:

    @dataclass(frozen=True)
    class DefenseDecision:  # type: ignore[no-redef]
        client_id: str
        action: str
        scores: Mapping[str, float]
        thresholds: Mapping[str, float]
        reason: str
        final_weight: float

    @dataclass(frozen=True)
    class AggregationResult:  # type: ignore[no-redef]
        state: Mapping[str, torch.Tensor]
        decisions: Sequence[DefenseDecision]
        diagnostics: Mapping[str, Any]

from .common import update_weight
from .detection import (
    CosineMADDetector,
    DetectionResult,
    EWMAReputation,
    NormMADDetector,
)
from .update_filter.norm_clipping import NormClipper
from .validation import UpdateValidationError, UpdateValidator


@dataclass(frozen=True)
class AggregationAudit:
    aggregator: Optional[str]
    submitted_clients: Sequence[str]
    valid_clients: Sequence[str]
    accepted_clients: Sequence[str]
    clipped_clients: Sequence[str]
    rejected_clients: Sequence[str]
    original_weights: Mapping[str, float]
    final_weights: Mapping[str, float]
    aggregation_performed: bool
    validation_errors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DefensePipelineResult:
    decisions: Sequence[DefenseDecision]
    accepted: Sequence[Any]
    sanitized: Sequence[Any]
    aggregation_audit: AggregationAudit
    aggregated_state: Optional[Mapping[str, torch.Tensor]] = None


class CompositeDecisionPolicy:
    """Reject on two anomaly signals, clip on one, accept on none."""

    def decide(
        self,
        updates: Sequence[Any],
        detection_results: Sequence[DetectionResult],
    ) -> List[DefenseDecision]:
        decisions = []
        for update in updates:
            client_id = str(getattr(update, "client_id"))
            anomalous = [
                result.name
                for result in detection_results
                if client_id in result.anomalous_clients
            ]
            action = (
                "reject"
                if len(anomalous) >= 2
                else "clip"
                if anomalous
                else "accept"
            )
            original_weight = update_weight(update)
            scores = {
                result.name: float(result.scores[client_id])
                for result in detection_results
                if client_id in result.scores
            }
            thresholds = {
                result.name: float(result.threshold) for result in detection_results
            }
            decisions.append(
                DefenseDecision(
                    client_id=client_id,
                    action=action,
                    scores=scores,
                    thresholds=thresholds,
                    reason=", ".join(f"{name} anomaly" for name in anomalous),
                    final_weight=0.0 if action == "reject" else original_weight,
                )
            )
        return decisions


class DefensePipeline:
    """Server-only validate, detect, decide, sanitize, and aggregate pipeline."""

    def __init__(
        self,
        *,
        validator: Optional[UpdateValidator] = None,
        detectors: Optional[Sequence[Any]] = None,
        decision_policy: Optional[CompositeDecisionPolicy] = None,
        sanitizer: Optional[NormClipper] = None,
        reputation: Optional[EWMAReputation] = None,
        aggregator: Optional[Any] = None,
    ):
        self.validator = validator or UpdateValidator()
        self.detectors = list(
            detectors
            if detectors is not None
            else (NormMADDetector(), CosineMADDetector())
        )
        self.decision_policy = decision_policy or CompositeDecisionPolicy()
        self.sanitizer = sanitizer or NormClipper(max_norm=None)
        self.reputation = reputation
        self.aggregator = aggregator
        self.last_result: Optional[DefensePipelineResult] = None

    @staticmethod
    def _validation_decision(update: Any, reason: str) -> DefenseDecision:
        client_id = str(getattr(update, "client_id", "<unknown>"))
        return DefenseDecision(
            client_id=client_id,
            action="reject",
            scores={},
            thresholds={},
            reason=f"validation: {reason}",
            final_weight=0.0,
        )

    def process(
        self,
        updates: Sequence[Any],
        global_state: Any,
        *,
        expected_base_snapshot_hash: Optional[str] = None,
        aggregator: Optional[Any] = None,
        run_aggregation: bool = True,
    ) -> DefensePipelineResult:
        if not updates:
            raise ValueError("at least one client update is required")

        valid = []
        validation_decisions: List[DefenseDecision] = []
        validation_errors: Dict[str, str] = {}
        original_weights: Dict[str, float] = {}
        for update in updates:
            client_id = str(getattr(update, "client_id", "<unknown>"))
            try:
                original_weights[client_id] = update_weight(update)
                self.validator.validate(
                    update,
                    global_state,
                    expected_base_snapshot_hash=expected_base_snapshot_hash,
                )
                valid.append(update)
            except (UpdateValidationError, ValueError) as exc:
                validation_errors[client_id] = str(exc)
                validation_decisions.append(self._validation_decision(update, str(exc)))

        detection_results = [
            detector.detect(valid, global_state) for detector in self.detectors
        ] if valid else []
        if valid and self.reputation is not None:
            current_anomaly = {
                str(getattr(update, "client_id")): any(
                    str(getattr(update, "client_id")) in result.anomalous_clients
                    for result in detection_results
                )
                for update in valid
            }
            detection_results.append(self.reputation.update(current_anomaly))

        detected_decisions = self.decision_policy.decide(valid, detection_results)
        all_decisions = {
            item.client_id: item for item in validation_decisions + detected_decisions
        }
        decisions = [
            all_decisions[str(getattr(update, "client_id", "<unknown>"))]
            for update in updates
        ]
        decision_by_client = {item.client_id: item for item in detected_decisions}
        accepted = [
            update
            for update in valid
            if decision_by_client[str(getattr(update, "client_id"))].action != "reject"
        ]
        clip_limit = self.sanitizer.resolve_max_norm(valid, global_state) if valid else None
        sanitized = []
        for update in accepted:
            decision = decision_by_client[str(getattr(update, "client_id"))]
            if decision.action == "clip":
                sanitized.append(
                    self.sanitizer.sanitize(
                        update,
                        global_state,
                        max_norm=clip_limit,
                        aggregation_weight=decision.final_weight,
                    )
                )
            else:
                sanitized.append(update)

        chosen_aggregator = (
            aggregator if aggregator is not None else self.aggregator
        ) if run_aggregation else None
        aggregated_state = None
        if chosen_aggregator is not None and sanitized:
            aggregated_state = chosen_aggregator.aggregate(sanitized, global_state)

        rejected_clients = [
            item.client_id for item in decisions if item.action == "reject"
        ]
        clipped_clients = [item.client_id for item in decisions if item.action == "clip"]
        final_weights = {item.client_id: float(item.final_weight) for item in decisions}
        audit = AggregationAudit(
            aggregator=(
                str(getattr(chosen_aggregator, "name", type(chosen_aggregator).__name__))
                if chosen_aggregator is not None
                else None
            ),
            submitted_clients=tuple(
                str(getattr(item, "client_id", "<unknown>")) for item in updates
            ),
            valid_clients=tuple(str(getattr(item, "client_id")) for item in valid),
            accepted_clients=tuple(str(getattr(item, "client_id")) for item in accepted),
            clipped_clients=tuple(clipped_clients),
            rejected_clients=tuple(rejected_clients),
            original_weights=original_weights,
            final_weights=final_weights,
            aggregation_performed=aggregated_state is not None,
            validation_errors=validation_errors,
        )
        self.last_result = DefensePipelineResult(
            decisions=tuple(decisions),
            accepted=tuple(accepted),
            sanitized=tuple(sanitized),
            aggregation_audit=audit,
            aggregated_state=aggregated_state,
        )
        return self.last_result

    def apply(self, updates: Sequence[Any], global_state: Any) -> Sequence[Any]:
        """Compatibility adapter for the existing FederatedEngine filter API."""

        return self.process(
            updates, global_state, aggregator=None, run_aggregation=False
        ).sanitized

    def aggregate(self, updates: Sequence[Any], global_state: Any) -> AggregationResult:
        """Run the full pipeline and return the canonical aggregation contract."""

        if self.aggregator is None:
            raise ValueError("DefensePipeline.aggregate requires a configured aggregator")
        result = self.process(updates, global_state, aggregator=self.aggregator)
        if result.aggregated_state is None:
            raise RuntimeError("defense pipeline did not produce an aggregated state")
        return AggregationResult(
            state=dict(result.aggregated_state),
            decisions=tuple(result.decisions),
            diagnostics=asdict(result.aggregation_audit),
        )
