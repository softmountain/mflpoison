import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from .hashing import mapping_hash, tensor_map_hash


TensorMap = Mapping[str, torch.Tensor]


def _normalized_tensor_map(values: TensorMap, name: str) -> Dict[str, torch.Tensor]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} cannot be empty")
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}[{key!r}] must be a torch.Tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{name}[{key!r}] must use strided layout")
        normalized = value.detach().cpu().clone()
        if (normalized.is_floating_point() or normalized.is_complex()) and not bool(
            torch.isfinite(normalized).all()
        ):
            raise ValueError(f"{name}[{key!r}] contains NaN or Inf")
        result[key] = normalized
    return result


def _validate_tensor_schema(
    values: TensorMap,
    reference: TensorMap,
    name: str,
) -> None:
    if set(values) != set(reference):
        missing = sorted(set(reference) - set(values))
        extra = sorted(set(values) - set(reference))
        raise ValueError(f"{name} schema mismatch; missing={missing}, extra={extra}")
    for key, value in values.items():
        expected = reference[key]
        if value.shape != expected.shape:
            raise ValueError(
                f"{name} shape mismatch for {key}: {tuple(value.shape)} != "
                f"{tuple(expected.shape)}"
            )
        if value.dtype != expected.dtype:
            raise ValueError(
                f"{name} dtype mismatch for {key}: {value.dtype} != {expected.dtype}"
            )


def _finite_metrics(values: Mapping[str, Any], name: str) -> Dict[str, float]:
    result = {}
    for key, value in values.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{key!r}] must be finite")
        result[str(key)] = number
    return result


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset identity, feature schema, labels, and immutable partition lineage."""

    name: str
    num_classes: int
    modality_shapes: Mapping[str, Tuple[int, int]]
    split: str = "1"
    partition_id: Optional[str] = None
    partition_hash: Optional[str] = None
    label_mapping: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("dataset name cannot be empty")
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if not self.modality_shapes:
            raise ValueError("at least one modality is required")
        normalized_shapes = {}
        for name, shape in self.modality_shapes.items():
            if not name or len(shape) != 2 or any(int(value) < 1 for value in shape):
                raise ValueError(
                    "modality_shapes must map names to positive (sequence, feature) pairs"
                )
            normalized_shapes[str(name)] = (int(shape[0]), int(shape[1]))
        normalized_labels = {str(key): int(value) for key, value in self.label_mapping.items()}
        if any(value < 0 or value >= self.num_classes for value in normalized_labels.values()):
            raise ValueError("label_mapping values must be valid class indices")
        if len(set(normalized_labels.values())) != len(normalized_labels):
            raise ValueError("label_mapping class indices must be unique")
        if self.partition_hash is not None and not str(self.partition_hash):
            raise ValueError("partition_hash cannot be empty")
        object.__setattr__(self, "modality_shapes", normalized_shapes)
        object.__setattr__(self, "label_mapping", normalized_labels)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def feature_schema(self) -> Mapping[str, Tuple[int, int]]:
        return self.modality_shapes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_classes": int(self.num_classes),
            "modality_shapes": {
                name: list(shape) for name, shape in self.modality_shapes.items()
            },
            "split": self.split,
            "partition_id": self.partition_id,
            "partition_hash": self.partition_hash,
            "label_mapping": dict(self.label_mapping),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelSpec:
    """Portable model identity stored with every global snapshot."""

    name: str
    constructor: Optional[str] = None
    version: Optional[str] = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("model name cannot be empty")
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "constructor": self.constructor,
            "version": self.version,
            "kwargs": dict(self.kwargs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        return cls(
            name=str(data["name"]),
            constructor=data.get("constructor"),
            version=data.get("version"),
            kwargs=dict(data.get("kwargs", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MultimodalBatch:
    """Dataset-independent batch with named modalities and sequence lengths."""

    features: Dict[str, torch.Tensor]
    lengths: Dict[str, torch.Tensor]
    labels: torch.Tensor

    def validate(self) -> "MultimodalBatch":
        if not self.features:
            raise ValueError("features cannot be empty")
        batch_size = int(self.labels.shape[0])
        for name, tensor in self.features.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(f"feature batch mismatch for {name}")
            if name not in self.lengths:
                raise ValueError(f"missing lengths for modality {name}")
            if self.lengths[name].shape[0] != batch_size:
                raise ValueError(f"length batch mismatch for {name}")
        return self


@dataclass
class SyntheticBatch:
    """Canonical artifact exchanged between generators and poisoning code."""

    features: Dict[str, torch.Tensor]
    lengths: Dict[str, torch.Tensor]
    condition_labels: torch.Tensor
    train_labels: torch.Tensor
    source_labels: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return int(self.condition_labels.shape[0])

    def validate(self) -> "SyntheticBatch":
        size = self.num_samples
        if self.train_labels.shape[0] != size:
            raise ValueError("condition_labels and train_labels must have equal length")
        if self.source_labels is not None and self.source_labels.shape[0] != size:
            raise ValueError("source_labels must match the synthetic batch size")
        if not self.features:
            raise ValueError("synthetic features cannot be empty")
        for name, tensor in self.features.items():
            if tensor.shape[0] != size:
                raise ValueError(f"feature batch mismatch for {name}")
            if name not in self.lengths:
                raise ValueError(f"missing lengths for modality {name}")
            if self.lengths[name].shape[0] != size:
                raise ValueError(f"length batch mismatch for {name}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "features": dict(self.features),
            "lengths": dict(self.lengths),
            "condition_labels": self.condition_labels,
            "train_labels": self.train_labels,
            "source_labels": self.source_labels,
            "metadata": dict(self.metadata),
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        self.validate()
        if "audio" not in self.features or "video" not in self.features:
            raise ValueError("legacy format requires audio and video modalities")
        result = {
            "audio": self.features["audio"],
            "video": self.features["video"],
            "len_a": self.lengths["audio"],
            "len_v": self.lengths["video"],
            "condition_label": self.condition_labels,
            "train_label": self.train_labels,
            "meta": dict(self.metadata),
            "schema_version": 1,
        }
        if self.source_labels is not None:
            result["source_label"] = self.source_labels
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SyntheticBatch":
        if "features" in data:
            return cls(
                features=dict(data["features"]),
                lengths=dict(data["lengths"]),
                condition_labels=data["condition_labels"],
                train_labels=data["train_labels"],
                source_labels=data.get("source_labels"),
                metadata=dict(data.get("metadata", {})),
            ).validate()
        return cls(
            features={"audio": data["audio"], "video": data["video"]},
            lengths={"audio": data["len_a"], "video": data["len_v"]},
            condition_labels=data["condition_label"],
            train_labels=data["train_label"],
            source_labels=data.get("source_label"),
            metadata=dict(data.get("meta", {})),
        ).validate()


@dataclass
class GlobalSnapshot:
    """Immutable-by-convention global model checkpoint and its lineage."""

    state: Dict[str, torch.Tensor]
    round_index: int
    dev_metrics: Mapping[str, float]
    model_spec: ModelSpec
    partition_hash: str
    content_hash: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.state = _normalized_tensor_map(self.state, "snapshot state")
        self.round_index = int(self.round_index)
        if self.round_index < 0:
            raise ValueError("snapshot round_index cannot be negative")
        self.dev_metrics = _finite_metrics(self.dev_metrics, "dev_metrics")
        if isinstance(self.model_spec, Mapping):
            self.model_spec = ModelSpec.from_dict(self.model_spec)
        if not isinstance(self.model_spec, ModelSpec):
            raise TypeError("model_spec must be a ModelSpec")
        self.partition_hash = str(self.partition_hash)
        if not self.partition_hash:
            raise ValueError("snapshot partition_hash cannot be empty")
        self.metadata = dict(self.metadata)
        calculated = self.calculate_hash()
        if self.content_hash is not None and str(self.content_hash) != calculated:
            raise ValueError("snapshot content hash does not match its payload")
        self.content_hash = calculated

    @property
    def model_state(self) -> Mapping[str, torch.Tensor]:
        return self.state

    @property
    def snapshot_hash(self) -> str:
        return str(self.content_hash)

    def calculate_hash(self) -> str:
        return mapping_hash(
            {
                "state_hash": tensor_map_hash(self.state),
                "round_index": self.round_index,
                "dev_metrics": dict(self.dev_metrics),
                "model_spec": self.model_spec.to_dict(),
                "partition_hash": self.partition_hash,
                "metadata": dict(self.metadata),
            }
        )

    def validate_state(self, state: TensorMap) -> None:
        _validate_tensor_schema(state, self.state, "model state")


@dataclass(init=False)
class ClientUpdate:
    """Typed client delta with provenance and legacy full-state compatibility.

    ``malicious`` is oracle-only ground truth for evaluation. Detection and
    defense code must never consult it when deciding how to process an update.
    """

    client_id: str
    delta: Dict[str, torch.Tensor]
    round_index: int
    base_snapshot_hash: str
    clean_num_samples: int
    train_num_samples: int
    aggregation_weight: float
    metrics: Dict[str, float]
    artifact_ids: Tuple[str, ...]
    malicious: bool
    _legacy_state: Optional[Dict[str, torch.Tensor]]
    _legacy_only: bool

    def __init__(
        self,
        client_id: str,
        delta: Optional[TensorMap] = None,
        round_index: Optional[int] = None,
        base_snapshot_hash: Optional[str] = None,
        clean_num_samples: Optional[int] = None,
        train_num_samples: Optional[int] = None,
        aggregation_weight: Optional[float] = None,
        metrics: Optional[Mapping[str, float]] = None,
        artifact_ids: Sequence[str] = (),
        malicious: bool = False,
        state: Optional[TensorMap] = None,
        num_samples: Optional[int] = None,
    ):
        if not str(client_id):
            raise ValueError("client_id cannot be empty")
        if delta is None and state is None:
            raise ValueError("client update requires delta or legacy state")
        legacy_only = delta is None
        self.client_id = str(client_id)
        if delta is None:
            self.delta = _normalized_tensor_map(state, "legacy client state")
        else:
            self.delta = _normalized_tensor_map(delta, "client delta")
        self._legacy_state = (
            _normalized_tensor_map(state, "client state") if state is not None else None
        )
        if self._legacy_state is not None:
            _validate_tensor_schema(self.delta, self._legacy_state, "client delta/state")
        self._legacy_only = legacy_only
        self.round_index = 0 if round_index is None else int(round_index)
        if self.round_index < 0:
            raise ValueError("round_index cannot be negative")
        if base_snapshot_hash is None:
            if legacy_only:
                base_snapshot_hash = "legacy"
            else:
                raise ValueError("base_snapshot_hash is required for delta updates")
        self.base_snapshot_hash = str(base_snapshot_hash)
        if not self.base_snapshot_hash:
            raise ValueError("base_snapshot_hash cannot be empty")
        if clean_num_samples is None:
            clean_num_samples = num_samples
        if clean_num_samples is None:
            raise ValueError("clean_num_samples is required")
        self.clean_num_samples = int(clean_num_samples)
        if self.clean_num_samples < 1:
            raise ValueError("clean_num_samples must be positive")
        self.train_num_samples = int(
            self.clean_num_samples if train_num_samples is None else train_num_samples
        )
        if self.train_num_samples < 1:
            raise ValueError("train_num_samples must be positive")
        self.aggregation_weight = float(
            self.clean_num_samples
            if aggregation_weight is None
            else aggregation_weight
        )
        if not math.isfinite(self.aggregation_weight) or self.aggregation_weight <= 0:
            raise ValueError("aggregation_weight must be finite and positive")
        self.metrics = _finite_metrics(metrics or {}, "metrics")
        if isinstance(artifact_ids, (str, bytes)):
            raise TypeError("artifact_ids must be a sequence of strings")
        self.artifact_ids = tuple(str(item) for item in artifact_ids)
        if any(not item for item in self.artifact_ids):
            raise ValueError("artifact_ids cannot contain empty strings")
        self.malicious = bool(malicious)

    @property
    def state(self) -> Mapping[str, torch.Tensor]:
        """Legacy payload view; new code should use ``delta`` helpers."""

        return self._legacy_state if self._legacy_state is not None else self.delta

    @property
    def num_samples(self) -> int:
        """Legacy alias for the clean sample count used by FedAvg."""

        return self.clean_num_samples

    @property
    def effective_weight(self) -> float:
        return self.aggregation_weight

    @property
    def is_legacy_state(self) -> bool:
        return self._legacy_only

    def effective_delta(self, global_state: TensorMap) -> Dict[str, torch.Tensor]:
        _validate_tensor_schema(self.delta, global_state, "client update")
        if not self._legacy_only:
            return {key: value.clone() for key, value in self.delta.items()}
        result = {}
        for key, value in self._legacy_state.items():
            base = global_state[key].detach().cpu()
            if value.is_floating_point() or value.is_complex():
                result[key] = value - base
            else:
                result[key] = torch.zeros_like(value)
        return result

    def effective_state(self, global_state: TensorMap) -> Dict[str, torch.Tensor]:
        _validate_tensor_schema(self.delta, global_state, "client update")
        if self._legacy_state is not None:
            return {key: value.clone() for key, value in self._legacy_state.items()}
        result = {}
        for key, value in self.delta.items():
            base = global_state[key].detach().cpu()
            if value.is_floating_point() or value.is_complex():
                result[key] = base + value
            else:
                result[key] = base.clone()
        return result

    def validate_against(self, snapshot: Any) -> "ClientUpdate":
        if isinstance(snapshot, GlobalSnapshot):
            if self.base_snapshot_hash not in {"legacy", snapshot.content_hash}:
                raise ValueError("client update references the wrong base snapshot")
            if self.round_index != snapshot.round_index:
                raise ValueError("client update round does not match its base snapshot")
            reference = snapshot.state
        elif isinstance(snapshot, Mapping):
            reference = snapshot
        else:
            raise TypeError("snapshot must be GlobalSnapshot or a tensor mapping")
        _validate_tensor_schema(self.delta, reference, "client update")
        if self._legacy_state is not None and not self._legacy_only:
            materialized = {}
            for key, delta in self.delta.items():
                base = reference[key].detach().cpu()
                materialized[key] = (
                    base + delta
                    if base.is_floating_point() or base.is_complex()
                    else base
                )
            for key, value in self._legacy_state.items():
                if not torch.equal(value, materialized[key]):
                    raise ValueError(
                        "client state is inconsistent with base snapshot and delta: "
                        + key
                    )
        return self


@dataclass(frozen=True)
class GeneratorArtifact:
    client_id: str
    partition_hash: str
    parent_snapshot_hash: str
    variant: str
    seed: int
    checkpoint_path: str
    checkpoint_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None

    def __post_init__(self):
        for name in (
            "client_id",
            "partition_hash",
            "parent_snapshot_hash",
            "variant",
            "checkpoint_path",
            "checkpoint_hash",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"generator artifact {name} cannot be empty")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "metadata", dict(self.metadata))
        calculated = self.calculate_hash()
        if self.content_hash is not None and str(self.content_hash) != calculated:
            raise ValueError("generator artifact content hash does not match")
        object.__setattr__(self, "content_hash", calculated)

    def calculate_hash(self) -> str:
        return mapping_hash(
            {
                "client_id": self.client_id,
                "partition_hash": self.partition_hash,
                "parent_snapshot_hash": self.parent_snapshot_hash,
                "variant": self.variant,
                "seed": self.seed,
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_hash": self.checkpoint_hash,
                "metadata": dict(self.metadata),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "client_id": self.client_id,
            "partition_hash": self.partition_hash,
            "parent_snapshot_hash": self.parent_snapshot_hash,
            "variant": self.variant,
            "seed": self.seed,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
        }

    @property
    def trained_round(self) -> int:
        """Lifecycle compatibility view stored canonically in metadata."""

        return int(self.metadata.get("trained_round", 0))

    @property
    def refresh_index(self) -> int:
        """Lifecycle compatibility view stored canonically in metadata."""

        return int(self.metadata.get("refresh_index", 0))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratorArtifact":
        return cls(
            client_id=str(data["client_id"]),
            partition_hash=str(data["partition_hash"]),
            parent_snapshot_hash=str(data["parent_snapshot_hash"]),
            variant=str(data["variant"]),
            seed=int(data["seed"]),
            checkpoint_path=str(data["checkpoint_path"]),
            checkpoint_hash=str(data["checkpoint_hash"]),
            metadata=dict(data.get("metadata", {})),
            content_hash=data.get("content_hash"),
        )


@dataclass(frozen=True)
class AttackSpec:
    condition_class: Optional[int]
    assigned_train_label: Optional[int]
    victim_eval_class: Optional[int]
    goal_prediction_class: Optional[int]
    poison_ratio: float = 0.0
    poison_count: Optional[int] = None
    budget: Optional[Any] = None
    injection_mode: str = "replace"
    start_round: int = 0
    end_round: Optional[int] = None
    every: int = 1
    schedule: Optional[Any] = None
    seed: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        labels = (
            self.condition_class,
            self.assigned_train_label,
            self.victim_eval_class,
            self.goal_prediction_class,
        )
        if any(label is not None and int(label) < 0 for label in labels):
            raise ValueError("attack class labels cannot be negative")
        poison_ratio = float(self.poison_ratio)
        poison_count = self.poison_count
        if self.budget is not None:
            budget_count = getattr(self.budget, "count", None)
            budget_fraction = getattr(self.budget, "fraction", None)
            if (budget_count is None) == (budget_fraction is None):
                raise ValueError("attack budget must define exactly one count or fraction")
            if poison_count is not None or poison_ratio != 0.0:
                raise ValueError("use either budget or poison_count/poison_ratio")
            if budget_count is not None:
                poison_count = int(budget_count)
            else:
                poison_ratio = float(budget_fraction)
        if not 0.0 <= poison_ratio <= 1.0:
            raise ValueError("poison_ratio must be in [0, 1]")
        if poison_count is not None and int(poison_count) < 0:
            raise ValueError("poison_count cannot be negative")
        if poison_count is not None and poison_ratio != 0.0:
            raise ValueError("set poison_count or poison_ratio, not both")
        injection_mode = str(getattr(self.injection_mode, "value", self.injection_mode))
        if injection_mode not in {"replace", "append"}:
            raise ValueError("injection_mode must be replace or append")
        if int(self.start_round) < 0 or int(self.every) < 1:
            raise ValueError("attack schedule values are invalid")
        if self.end_round is not None and int(self.end_round) < int(self.start_round):
            raise ValueError("end_round cannot precede start_round")
        if self.schedule is not None and not callable(getattr(self.schedule, "active", None)):
            raise TypeError("schedule must provide active(round_index)")
        object.__setattr__(self, "poison_ratio", poison_ratio)
        object.__setattr__(self, "poison_count", poison_count)
        object.__setattr__(self, "injection_mode", injection_mode)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def active(self, round_index: int) -> bool:
        round_index = int(round_index)
        if self.schedule is not None:
            return bool(self.schedule.active(round_index))
        if round_index < self.start_round:
            return False
        if self.end_round is not None and round_index > self.end_round:
            return False
        return (round_index - self.start_round) % self.every == 0


@dataclass(frozen=True)
class DefenseDecision:
    client_id: str
    action: str
    scores: Mapping[str, float] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""
    final_weight: float = 0.0

    def __post_init__(self):
        if not self.client_id:
            raise ValueError("defense decision client_id cannot be empty")
        if self.action not in {"accept", "clip", "reject", "quarantine"}:
            raise ValueError("unsupported defense action")
        object.__setattr__(self, "scores", _finite_metrics(self.scores, "scores"))
        object.__setattr__(self, "thresholds", _finite_metrics(self.thresholds, "thresholds"))
        weight = float(self.final_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("final_weight must be finite and non-negative")
        if self.action in {"reject", "quarantine"} and weight != 0.0:
            raise ValueError("rejected or quarantined updates must have zero weight")
        object.__setattr__(self, "final_weight", weight)

    @property
    def accepted(self) -> bool:
        return self.action in {"accept", "clip"}


@dataclass
class AggregationResult:
    state: Dict[str, torch.Tensor]
    decisions: Sequence[DefenseDecision] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.state = _normalized_tensor_map(self.state, "aggregation state")
        self.decisions = tuple(self.decisions)
        if any(not isinstance(item, DefenseDecision) for item in self.decisions):
            raise TypeError("decisions must contain DefenseDecision values")
        self.diagnostics = dict(self.diagnostics)

    @property
    def accepted_client_ids(self) -> Tuple[str, ...]:
        return tuple(item.client_id for item in self.decisions if item.accepted)


@dataclass
class RoundRecord:
    round_index: int
    base_snapshot_hash: str
    selected_client_ids: Sequence[str]
    raw_updates: Sequence[ClientUpdate]
    defense_decisions: Sequence[DefenseDecision]
    processed_updates: Sequence[ClientUpdate]
    aggregation_result: AggregationResult
    evaluation: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.round_index = int(self.round_index)
        if self.round_index < 0 or not self.base_snapshot_hash:
            raise ValueError("round record provenance is invalid")
        self.selected_client_ids = tuple(str(item) for item in self.selected_client_ids)
        if len(set(self.selected_client_ids)) != len(self.selected_client_ids):
            raise ValueError("selected_client_ids cannot contain duplicates")
        self.raw_updates = tuple(self.raw_updates)
        self.defense_decisions = tuple(self.defense_decisions)
        self.processed_updates = tuple(self.processed_updates)
        if not isinstance(self.aggregation_result, AggregationResult):
            raise TypeError("aggregation_result must be AggregationResult")
        self.evaluation = _finite_metrics(self.evaluation, "evaluation")
        raw_ids = tuple(update.client_id for update in self.raw_updates)
        if raw_ids != self.selected_client_ids:
            raise ValueError("raw update order must match selected_client_ids")
        if any(update.round_index != self.round_index for update in self.raw_updates):
            raise ValueError("raw update round does not match round record")
        if any(
            update.base_snapshot_hash not in {"legacy", self.base_snapshot_hash}
            for update in self.raw_updates
        ):
            raise ValueError("raw update references the wrong base snapshot")
        decision_ids = tuple(item.client_id for item in self.defense_decisions)
        if decision_ids and decision_ids != self.selected_client_ids:
            raise ValueError("defense decision order must match selected_client_ids")
        processed_ids = tuple(update.client_id for update in self.processed_updates)
        if len(set(processed_ids)) != len(processed_ids):
            raise ValueError("processed updates cannot contain duplicate clients")
        if not set(processed_ids).issubset(set(self.selected_client_ids)):
            raise ValueError("processed updates must come from selected clients")
        if any(update.round_index != self.round_index for update in self.processed_updates):
            raise ValueError("processed update round does not match round record")
        if any(
            update.base_snapshot_hash not in {"legacy", self.base_snapshot_hash}
            for update in self.processed_updates
        ):
            raise ValueError("processed update references the wrong base snapshot")
        raw_by_client = {update.client_id: update for update in self.raw_updates}
        for update in self.processed_updates:
            raw = raw_by_client[update.client_id]
            if (
                update.round_index != raw.round_index
                or update.base_snapshot_hash != raw.base_snapshot_hash
                or update.clean_num_samples != raw.clean_num_samples
                or update.train_num_samples != raw.train_num_samples
                or update.artifact_ids != raw.artifact_ids
            ):
                raise ValueError(
                    "processed update does not preserve its raw update lineage"
                )
        accepted_ids = tuple(
            item.client_id for item in self.defense_decisions if item.accepted
        )
        if decision_ids and processed_ids != accepted_ids:
            raise ValueError("processed updates do not match accepted decisions")
        if tuple(self.aggregation_result.decisions) != tuple(self.defense_decisions):
            raise ValueError("aggregation decisions do not match the round record")
