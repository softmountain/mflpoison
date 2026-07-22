from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from mflpoison.core.types import GeneratorArtifact, GlobalSnapshot
from mflpoison.core.hashing import file_sha256


class GeneratorLifecycleMode(str, Enum):
    OFFLINE_ONCE = "offline_once"
    ONLINE_REFRESH = "online_refresh"


@dataclass(frozen=True)
class ClientGeneratorPartition:
    """Opaque client-local data coupled to its stable partition identity."""

    client_id: str
    partition_hash: str
    data: Any

    def __post_init__(self):
        if not str(self.client_id):
            raise ValueError("client_id cannot be empty")
        if not str(self.partition_hash):
            raise ValueError("partition_hash cannot be empty")


@dataclass(frozen=True)
class GeneratorTrainingRequest:
    client_id: str
    partition_hash: str
    global_snapshot_hash: str
    variant: str
    round_index: int
    refresh_index: int
    seed: int
    global_snapshot: Optional[GlobalSnapshot] = None
    warm_start_artifact: Optional[GeneratorArtifact] = None

    def artifact(
        self,
        checkpoint_path: str,
        checkpoint_hash: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> GeneratorArtifact:
        """Build a canonical artifact with provenance from this request."""

        artifact_metadata = dict(metadata or {})
        artifact_metadata.update(
            {
                "trained_round": int(self.round_index),
                "refresh_index": int(self.refresh_index),
            }
        )
        return GeneratorArtifact(
            client_id=self.client_id,
            partition_hash=self.partition_hash,
            parent_snapshot_hash=self.global_snapshot_hash,
            variant=self.variant,
            seed=self.seed,
            checkpoint_path=str(checkpoint_path),
            checkpoint_hash=str(checkpoint_hash),
            metadata=artifact_metadata,
        )


class GeneratorTrainer(ABC):
    """Adapter point for DTM, temporal-adaptive, or future trainers."""

    @abstractmethod
    def train(
        self,
        request: GeneratorTrainingRequest,
        partition: ClientGeneratorPartition,
    ) -> GeneratorArtifact:
        raise NotImplementedError


class CallbackGeneratorTrainer(GeneratorTrainer):
    """Wrap a training callback without coupling lifecycle code to FedMM."""

    def __init__(self, callback):
        self.callback = callback

    def train(
        self,
        request: GeneratorTrainingRequest,
        partition: ClientGeneratorPartition,
    ) -> GeneratorArtifact:
        result = self.callback(request, partition)
        if isinstance(result, GeneratorArtifact):
            return result
        if isinstance(result, Mapping):
            return GeneratorArtifact.from_dict(result)
        raise TypeError("generator training callback must return GeneratorArtifact")


class GeneratorLifecycle:
    """Deterministic generator state owned by exactly one malicious client."""

    def __init__(
        self,
        client_id: str,
        partition_hash: str,
        trainer_factory: Callable[[], GeneratorTrainer],
        variant: str,
        mode: GeneratorLifecycleMode = GeneratorLifecycleMode.OFFLINE_ONCE,
        refresh_interval: int = 1,
        seed: int = 42,
    ):
        if not str(client_id) or not str(partition_hash) or not str(variant):
            raise ValueError("client_id, partition_hash, and variant are required")
        if not isinstance(mode, GeneratorLifecycleMode):
            mode = GeneratorLifecycleMode(str(mode).lower())
        if int(refresh_interval) < 1:
            raise ValueError("refresh_interval must be positive")
        self.client_id = str(client_id)
        self.partition_hash = str(partition_hash)
        self.trainer_factory = trainer_factory
        self.variant = str(variant)
        self.mode = mode
        self.refresh_interval = int(refresh_interval)
        self.seed = int(seed)
        self._trainer = None
        self._artifact = None
        self._refresh_count = 0

    @property
    def artifact(self) -> Optional[GeneratorArtifact]:
        return self._artifact

    def should_refresh(self, round_index: int) -> bool:
        round_index = int(round_index)
        if round_index < 0:
            raise ValueError("round_index cannot be negative")
        if self._artifact is None:
            return True
        trained_round = int(self._artifact.metadata["trained_round"])
        if round_index < trained_round:
            raise ValueError("round_index precedes the current generator artifact")
        if self.mode == GeneratorLifecycleMode.OFFLINE_ONCE:
            return False
        return round_index - trained_round >= self.refresh_interval

    def ensure_artifact(
        self,
        global_snapshot: Union[GlobalSnapshot, str],
        round_index: int,
        partition: ClientGeneratorPartition,
    ) -> GeneratorArtifact:
        self._validate_partition(partition)
        snapshot, snapshot_hash = self._snapshot(global_snapshot)
        if not self.should_refresh(round_index):
            return self._artifact

        request = GeneratorTrainingRequest(
            client_id=self.client_id,
            partition_hash=self.partition_hash,
            global_snapshot_hash=snapshot_hash,
            global_snapshot=snapshot,
            variant=self.variant,
            round_index=int(round_index),
            refresh_index=self._refresh_count,
            seed=self._training_seed(self._refresh_count),
            warm_start_artifact=(
                self._artifact
                if self.mode == GeneratorLifecycleMode.ONLINE_REFRESH
                else None
            ),
        )
        if self._trainer is None:
            self._trainer = self.trainer_factory()
            if not isinstance(self._trainer, GeneratorTrainer):
                raise TypeError("trainer_factory must return a GeneratorTrainer")
        artifact = self._trainer.train(request, partition)
        self._validate_artifact(artifact, request)
        self._artifact = artifact
        self._refresh_count += 1
        return artifact

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "client_id": self.client_id,
            "partition_hash": self.partition_hash,
            "variant": self.variant,
            "mode": self.mode.value,
            "refresh_interval": self.refresh_interval,
            "seed": self.seed,
            "refresh_count": self._refresh_count,
            "artifact": None if self._artifact is None else self._artifact.to_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "client_id": self.client_id,
            "partition_hash": self.partition_hash,
            "variant": self.variant,
            "mode": self.mode.value,
            "refresh_interval": self.refresh_interval,
            "seed": self.seed,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"generator lifecycle state has mismatched {name}")
        refresh_count = int(state.get("refresh_count", 0))
        artifact_data = state.get("artifact")
        artifact = (
            None
            if artifact_data is None
            else GeneratorArtifact.from_dict(artifact_data)
        )
        if refresh_count < 0 or (artifact is None and refresh_count != 0):
            raise ValueError("invalid generator lifecycle refresh state")
        if artifact is not None:
            if artifact.client_id != self.client_id:
                raise ValueError("artifact belongs to another client")
            if artifact.partition_hash != self.partition_hash:
                raise ValueError("artifact belongs to another data partition")
            if artifact.variant != self.variant:
                raise ValueError("artifact uses another generator variant")
            if int(artifact.metadata.get("refresh_index", -1)) + 1 != refresh_count:
                raise ValueError("artifact refresh index is inconsistent with state")
            self._verify_checkpoint(artifact)
        self._artifact = artifact
        self._refresh_count = refresh_count

    def _validate_partition(self, partition: ClientGeneratorPartition) -> None:
        if partition.client_id != self.client_id:
            raise ValueError("generator lifecycle cannot access another client")
        if partition.partition_hash != self.partition_hash:
            raise ValueError("generator lifecycle partition hash mismatch")

    @staticmethod
    def _validate_artifact(
        artifact: GeneratorArtifact, request: GeneratorTrainingRequest
    ) -> None:
        if not isinstance(artifact, GeneratorArtifact):
            raise TypeError("generator trainer must return GeneratorArtifact")
        expected = {
            "client_id": request.client_id,
            "partition_hash": request.partition_hash,
            "parent_snapshot_hash": request.global_snapshot_hash,
            "variant": request.variant,
            "seed": request.seed,
        }
        for name, value in expected.items():
            if getattr(artifact, name) != value:
                raise ValueError(f"generator artifact has mismatched {name}")
        if int(artifact.metadata.get("trained_round", -1)) != request.round_index:
            raise ValueError("generator artifact has mismatched trained_round")
        if int(artifact.metadata.get("refresh_index", -1)) != request.refresh_index:
            raise ValueError("generator artifact has mismatched refresh_index")
        GeneratorLifecycle._verify_checkpoint(artifact)

    @staticmethod
    def _verify_checkpoint(artifact: GeneratorArtifact) -> None:
        checkpoint = Path(artifact.checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        if file_sha256(checkpoint) != artifact.checkpoint_hash:
            raise ValueError("generator checkpoint hash does not match its artifact")

    def _training_seed(self, refresh_index: int) -> int:
        identity = f"{self.client_id}\0{self.partition_hash}".encode("utf-8")
        client_offset = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")
        return (self.seed + client_offset + int(refresh_index)) % (2**31)

    @staticmethod
    def _snapshot(
        value: Union[GlobalSnapshot, str]
    ) -> Tuple[Optional[GlobalSnapshot], str]:
        if isinstance(value, GlobalSnapshot):
            return value, value.snapshot_hash
        snapshot_hash = str(value)
        if not snapshot_hash:
            raise ValueError("global snapshot hash cannot be empty")
        return None, snapshot_hash


class GeneratorLifecycleManager:
    """Runner-facing registry of isolated per-client generator lifecycles."""

    def __init__(
        self,
        trainer_factory: Callable[[str], GeneratorTrainer],
        variant: str,
        mode: GeneratorLifecycleMode = GeneratorLifecycleMode.OFFLINE_ONCE,
        refresh_every: int = 1,
        seed: int = 42,
    ):
        if int(refresh_every) < 1:
            raise ValueError("refresh_every must be positive")
        self.trainer_factory = trainer_factory
        self.variant = str(variant)
        self.mode = GeneratorLifecycleMode(mode)
        self.refresh_every = int(refresh_every)
        self.seed = int(seed)
        self._lifecycles: Dict[str, GeneratorLifecycle] = {}

    @property
    def artifacts(self) -> Dict[str, GeneratorArtifact]:
        return {
            client_id: lifecycle.artifact
            for client_id, lifecycle in self._lifecycles.items()
            if lifecycle.artifact is not None
        }

    def ensure(
        self,
        client_id: str,
        snapshot: Union[GlobalSnapshot, str],
        dataloader,
        partition_hash: str,
        round_index: int,
    ) -> GeneratorArtifact:
        client_id = str(client_id)
        partition_hash = str(partition_hash)
        lifecycle = self._lifecycles.get(client_id)
        if lifecycle is None:
            lifecycle = GeneratorLifecycle(
                client_id=client_id,
                partition_hash=partition_hash,
                trainer_factory=lambda: self.trainer_factory(client_id),
                variant=self.variant,
                mode=self.mode,
                refresh_interval=self.refresh_every,
                seed=self.seed,
            )
            self._lifecycles[client_id] = lifecycle
        elif lifecycle.partition_hash != partition_hash:
            raise ValueError("client partition hash changed during generator lifecycle")
        return lifecycle.ensure_artifact(
            snapshot,
            round_index,
            ClientGeneratorPartition(client_id, partition_hash, dataloader),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "variant": self.variant,
            "mode": self.mode.value,
            "refresh_every": self.refresh_every,
            "seed": self.seed,
            "clients": {
                client_id: lifecycle.state_dict()
                for client_id, lifecycle in self._lifecycles.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "variant": self.variant,
            "mode": self.mode.value,
            "refresh_every": self.refresh_every,
            "seed": self.seed,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"generator manager state has mismatched {name}")
        clients = state.get("clients", {})
        if not isinstance(clients, Mapping):
            raise TypeError("generator manager clients state must be a mapping")
        restored = {}
        for client_id, lifecycle_state in clients.items():
            partition_hash = str(lifecycle_state["partition_hash"])
            lifecycle = GeneratorLifecycle(
                client_id=str(client_id),
                partition_hash=partition_hash,
                trainer_factory=lambda client_id=str(client_id): self.trainer_factory(
                    client_id
                ),
                variant=self.variant,
                mode=self.mode,
                refresh_interval=self.refresh_every,
                seed=self.seed,
            )
            lifecycle.load_state_dict(lifecycle_state)
            restored[str(client_id)] = lifecycle
        self._lifecycles = restored
