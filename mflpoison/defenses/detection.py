from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Set

import torch
import torch.nn.functional as F

from .common import flatten_delta


@dataclass(frozen=True)
class DetectionResult:
    name: str
    scores: Mapping[str, float]
    threshold: float
    anomalous_clients: Set[str]


@dataclass(frozen=True)
class DeltaFeatures:
    client_id: str
    l2_norm: float
    cosine_distance_to_center: float


def _client_id(update: Any) -> str:
    return str(getattr(update, "client_id"))


def _robust_z_scores(values: torch.Tensor, *, two_sided: bool) -> torch.Tensor:
    median = values.median()
    residual = values - median
    mad = residual.abs().median()
    scale = max(float(1.4826 * mad), torch.finfo(torch.float64).eps)
    numerator = residual.abs() if two_sided else residual.clamp_min(0)
    return numerator / scale


def extract_delta_features(
    updates: Sequence[Any],
    global_state: Any,
    *,
    epsilon: float = 1e-12,
) -> Mapping[str, DeltaFeatures]:
    """Extract detector-ready features without using labels or trusted data."""

    if not updates:
        raise ValueError("at least one update is required")
    vectors = torch.stack([flatten_delta(item, global_state) for item in updates])
    center = vectors.median(dim=0).values
    center_norm = float(torch.linalg.vector_norm(center))
    result = {}
    for update, vector in zip(updates, vectors):
        vector_norm = float(torch.linalg.vector_norm(vector))
        if center_norm <= epsilon and vector_norm <= epsilon:
            similarity = 1.0
        elif center_norm <= epsilon or vector_norm <= epsilon:
            similarity = 0.0
        else:
            similarity = float(F.cosine_similarity(vector, center, dim=0, eps=epsilon))
        client_id = _client_id(update)
        result[client_id] = DeltaFeatures(
            client_id=client_id,
            l2_norm=vector_norm,
            cosine_distance_to_center=1.0 - similarity,
        )
    return result


class NormMADDetector:
    """Flag unusually large or small update norms using median/MAD only."""

    name = "norm_mad"

    def __init__(self, threshold: float = 3.5):
        if float(threshold) <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)

    def detect(self, updates: Sequence[Any], global_state: Any) -> DetectionResult:
        if not updates:
            raise ValueError("at least one update is required")
        features = extract_delta_features(updates, global_state)
        norms = torch.tensor(
            [features[_client_id(item)].l2_norm for item in updates], dtype=torch.float64
        )
        robust_z = _robust_z_scores(norms, two_sided=True)
        scores = {
            _client_id(item): float(score) for item, score in zip(updates, robust_z)
        }
        anomalous = {
            client_id for client_id, score in scores.items() if score > self.threshold
        }
        return DetectionResult(self.name, scores, self.threshold, anomalous)


class CosineMADDetector:
    """Flag deltas whose direction diverges from the coordinate median center."""

    name = "cosine_mad"

    def __init__(self, threshold: float = 3.5, epsilon: float = 1e-12):
        if float(threshold) <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)

    def detect(self, updates: Sequence[Any], global_state: Any) -> DetectionResult:
        if not updates:
            raise ValueError("at least one update is required")
        features = extract_delta_features(
            updates, global_state, epsilon=self.epsilon
        )
        distances = [
            features[_client_id(item)].cosine_distance_to_center for item in updates
        ]
        distance_tensor = torch.tensor(distances, dtype=torch.float64)
        robust_z = _robust_z_scores(distance_tensor, two_sided=False)
        scores = {
            _client_id(item): float(score) for item, score in zip(updates, robust_z)
        }
        anomalous = {
            client_id for client_id, score in scores.items() if score > self.threshold
        }
        return DetectionResult(self.name, scores, self.threshold, anomalous)


class EWMAReputation:
    """Track optional cross-round client reputation from detector evidence."""

    name = "ewma_reputation"

    def __init__(
        self,
        decay: float = 0.9,
        minimum_reputation: float = 0.5,
        initial_reputation: float = 1.0,
    ):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if not 0.0 <= float(minimum_reputation) <= 1.0:
            raise ValueError("minimum_reputation must be in [0, 1]")
        if not 0.0 <= float(initial_reputation) <= 1.0:
            raise ValueError("initial_reputation must be in [0, 1]")
        self.decay = float(decay)
        self.minimum_reputation = float(minimum_reputation)
        self.initial_reputation = float(initial_reputation)
        self.reputations: Dict[str, float] = {}

    def update(self, anomaly_evidence: Mapping[str, bool]) -> DetectionResult:
        for client_id, anomalous in anomaly_evidence.items():
            previous = self.reputations.get(str(client_id), self.initial_reputation)
            observation = 0.0 if anomalous else 1.0
            self.reputations[str(client_id)] = (
                self.decay * previous + (1.0 - self.decay) * observation
            )
        anomalous_clients = {
            client_id
            for client_id, reputation in self.reputations.items()
            if client_id in anomaly_evidence and reputation < self.minimum_reputation
        }
        return DetectionResult(
            self.name,
            {
                client_id: 1.0 - self.reputations[client_id]
                for client_id in anomaly_evidence
            },
            1.0 - self.minimum_reputation,
            anomalous_clients,
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "minimum_reputation": self.minimum_reputation,
            "initial_reputation": self.initial_reputation,
            "reputations": dict(self.reputations),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.minimum_reputation = float(
            state.get("minimum_reputation", self.minimum_reputation)
        )
        self.initial_reputation = float(
            state.get("initial_reputation", self.initial_reputation)
        )
        self.reputations = {
            str(client_id): float(value)
            for client_id, value in dict(state.get("reputations", {})).items()
        }
