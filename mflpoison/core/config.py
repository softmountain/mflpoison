import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar

from .hashing import canonical_json, mapping_hash


def config_hash(data: Mapping[str, Any], length: int = 12) -> str:
    return mapping_hash(data, length=length)


def load_config(path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            result = json.load(handle)
            if not isinstance(result, dict):
                raise ValueError("configuration root must be a mapping")
            return result
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML config support requires PyYAML; JSON configs work without it"
                ) from exc
            result = yaml.safe_load(handle)
            if result is None:
                return {}
            if not isinstance(result, dict):
                raise ValueError("configuration root must be a mapping")
            return result
    raise ValueError(f"unsupported config format: {path.suffix}")


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _string_tuple(value: Any, name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ValueError(f"{name} cannot contain empty values")
    return result


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "ucf101"
    root: Optional[str] = None
    split: str = "1"
    fold: int = 1
    alpha: Optional[float] = None
    num_clients: Optional[int] = None
    partition_path: Optional[str] = None
    partition_id: Optional[str] = None
    partition_hash: Optional[str] = None
    num_classes: int = 101
    modality_shapes: Mapping[str, Tuple[int, int]] = field(default_factory=dict)
    label_mapping: Mapping[str, int] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("dataset.name cannot be empty")
        if int(self.fold) < 1 or int(self.num_classes) < 1:
            raise ValueError("dataset.fold and dataset.num_classes must be positive")
        if self.num_clients is not None and int(self.num_clients) < 1:
            raise ValueError("dataset.num_clients must be positive")
        if self.alpha is not None and float(self.alpha) <= 0:
            raise ValueError("dataset.alpha must be positive")
        for name, shape in self.modality_shapes.items():
            if not name or len(shape) != 2 or any(int(item) < 1 for item in shape):
                raise ValueError("dataset.modality_shapes contains an invalid feature shape")
        labels = {str(key): int(value) for key, value in self.label_mapping.items()}
        if any(value < 0 or value >= int(self.num_classes) for value in labels.values()):
            raise ValueError("dataset.label_mapping contains an invalid class index")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "MMActionClassifier"
    constructor: Optional[str] = None
    checkpoint_path: Optional[str] = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("model.name cannot be empty")


@dataclass(frozen=True)
class FederationConfig:
    rounds: int = 1
    pretrain_rounds: Optional[int] = None
    attack_rounds: int = 1
    clients_per_round: Optional[int] = None
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    convergence_metric: str = "acc"
    convergence_mode: str = "max"
    patience: Optional[int] = None
    min_delta: float = 0.0
    resume_from: Optional[str] = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if min(
            int(self.rounds),
            int(self.attack_rounds),
            int(self.local_epochs),
            int(self.batch_size),
        ) < 1:
            raise ValueError("federation round, epoch, and batch counts must be positive")
        if self.pretrain_rounds is not None and int(self.pretrain_rounds) < 1:
            raise ValueError("federation.pretrain_rounds must be positive")
        if self.clients_per_round is not None and int(self.clients_per_round) < 1:
            raise ValueError("federation.clients_per_round must be positive")
        if float(self.learning_rate) <= 0 or float(self.min_delta) < 0:
            raise ValueError("federation learning_rate must be positive and min_delta non-negative")
        if self.convergence_mode not in {"min", "max"}:
            raise ValueError("federation.convergence_mode must be 'min' or 'max'")
        if self.patience is not None and int(self.patience) < 1:
            raise ValueError("federation.patience must be positive")

    @property
    def effective_pretrain_rounds(self) -> int:
        return int(self.rounds if self.pretrain_rounds is None else self.pretrain_rounds)


@dataclass(frozen=True)
class GeneratorConfig:
    enabled: bool = True
    family: str = "kplus1"
    variant: str = "dtm"
    lifecycle: str = "offline_once"
    refresh_interval: int = 1
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.0002
    seed: int = 42
    checkpoint_dir: Optional[str] = None
    loss: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.family or not self.variant:
            raise ValueError("generator family and variant cannot be empty")
        if self.lifecycle not in {"offline_once", "online_refresh"}:
            raise ValueError("generator.lifecycle must be offline_once or online_refresh")
        if min(int(self.refresh_interval), int(self.epochs), int(self.batch_size)) < 1:
            raise ValueError("generator interval, epoch, and batch counts must be positive")
        if float(self.learning_rate) <= 0:
            raise ValueError("generator.learning_rate must be positive")


@dataclass(frozen=True)
class AttackConfig:
    enabled: bool = False
    strategy: str = "generative_feature_poisoning"
    malicious_clients: Tuple[str, ...] = ()
    malicious_client_count: int = 0
    poison_ratio: float = 0.0
    poison_count: Optional[int] = None
    injection_mode: str = "replace"
    condition_class: Optional[int] = None
    assigned_train_label: Optional[int] = None
    victim_eval_class: Optional[int] = None
    goal_prediction_class: Optional[int] = None
    start_round: int = 0
    end_round: Optional[int] = None
    every: int = 1
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "malicious_clients",
            _string_tuple(self.malicious_clients, "attack.malicious_clients"),
        )
        if self.injection_mode not in {"replace", "append"}:
            raise ValueError("attack.injection_mode must be replace or append")
        if not 0.0 <= float(self.poison_ratio) <= 1.0:
            raise ValueError("attack.poison_ratio must be in [0, 1]")
        if self.poison_count is not None and int(self.poison_count) < 0:
            raise ValueError("attack.poison_count cannot be negative")
        if self.poison_count is not None and float(self.poison_ratio) != 0.0:
            raise ValueError("set attack.poison_count or attack.poison_ratio, not both")
        if int(self.malicious_client_count) < 0 or int(self.start_round) < 0 or int(self.every) < 1:
            raise ValueError("attack counts and schedule values are invalid")
        if self.end_round is not None and int(self.end_round) < int(self.start_round):
            raise ValueError("attack.end_round cannot precede start_round")


@dataclass(frozen=True)
class DefenseConfig:
    enabled: bool = False
    detectors: Tuple[Mapping[str, Any], ...] = ()
    sanitizer: Mapping[str, Any] = field(default_factory=dict)
    aggregator: Mapping[str, Any] = field(
        default_factory=lambda: {"name": "weighted_mean"}
    )
    policy: str = "reject_if_both_clip_if_one"
    ewma_decay: Optional[float] = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.detectors, (str, bytes)) or not isinstance(self.detectors, Sequence):
            raise TypeError("defense.detectors must be a sequence")
        object.__setattr__(self, "detectors", tuple(_mapping(item, "detector") for item in self.detectors))
        if self.ewma_decay is not None and not 0.0 <= float(self.ewma_decay) < 1.0:
            raise ValueError("defense.ewma_decay must be in [0, 1)")


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: Tuple[str, ...] = ("accuracy",)
    evaluate_test: bool = True
    evaluate_attack: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metrics", _string_tuple(self.metrics, "evaluation.metrics"))


@dataclass(frozen=True)
class ArtifactsConfig:
    root_dir: str = "artifacts"
    save_every_round: bool = True
    manifest_name: str = "manifest.json"
    snapshot_name: str = "global_snapshot.pt"
    generator_dir: str = "generators"
    round_records_name: str = "round_records.pt"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.root_dir:
            raise ValueError("artifacts.root_dir cannot be empty")


SectionType = TypeVar("SectionType")


def _strict_section(section_type: Type[SectionType], value: Any, name: str) -> SectionType:
    values = _mapping(value, name)
    allowed = {item.name for item in fields(section_type)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")
    try:
        return section_type(**values)
    except TypeError as exc:
        raise TypeError(f"invalid {name} configuration: {exc}") from exc


@dataclass(frozen=True)
class ScenarioConfig:
    dataset: DatasetConfig
    model: ModelConfig
    federation: FederationConfig
    generator: GeneratorConfig
    attack: AttackConfig
    defense: DefenseConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactsConfig

    SECTION_TYPES = {
        "dataset": DatasetConfig,
        "model": ModelConfig,
        "federation": FederationConfig,
        "generator": GeneratorConfig,
        "attack": AttackConfig,
        "defense": DefenseConfig,
        "evaluation": EvaluationConfig,
        "artifacts": ArtifactsConfig,
    }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ScenarioConfig":
        values = _mapping(data, "scenario")
        required = set(cls.SECTION_TYPES)
        unknown = sorted(set(values) - required)
        missing = sorted(required - set(values))
        if unknown:
            raise ValueError(f"unknown scenario section(s): {', '.join(unknown)}")
        if missing:
            raise ValueError(f"missing scenario section(s): {', '.join(missing)}")
        sections = {
            name: _strict_section(section_type, values[name], name)
            for name, section_type in cls.SECTION_TYPES.items()
        }
        return cls(**sections)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        return config_hash(self.to_dict(), length=64)


def load_scenario_config(path) -> ScenarioConfig:
    return ScenarioConfig.from_mapping(load_config(path))
