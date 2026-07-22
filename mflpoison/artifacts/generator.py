import json
from pathlib import Path
from typing import Optional

from mflpoison.core.hashing import file_sha256
from mflpoison.core.types import GeneratorArtifact, GlobalSnapshot


def create_generator_artifact(
    client_id: str,
    partition_hash: str,
    parent_snapshot_hash: str,
    variant: str,
    seed: int,
    checkpoint_path,
    metadata=None,
) -> GeneratorArtifact:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    return GeneratorArtifact(
        client_id=client_id,
        partition_hash=partition_hash,
        parent_snapshot_hash=parent_snapshot_hash,
        variant=variant,
        seed=seed,
        checkpoint_path=str(checkpoint_path),
        checkpoint_hash=file_sha256(checkpoint_path),
        metadata=dict(metadata or {}),
    )


def verify_generator_lineage(
    artifact: GeneratorArtifact,
    snapshot: GlobalSnapshot,
    client_id: Optional[str] = None,
    partition_hash: Optional[str] = None,
) -> GeneratorArtifact:
    if artifact.parent_snapshot_hash != snapshot.content_hash:
        raise ValueError("generator artifact belongs to a different global snapshot")
    expected_partition = snapshot.partition_hash if partition_hash is None else partition_hash
    if artifact.partition_hash != expected_partition:
        raise ValueError("generator artifact belongs to a different data partition")
    if client_id is not None and artifact.client_id != str(client_id):
        raise ValueError("generator artifact belongs to a different client")
    return artifact


def save_generator_artifact(artifact: GeneratorArtifact, path) -> Path:
    if not isinstance(artifact, GeneratorArtifact):
        raise TypeError("artifact must be a GeneratorArtifact")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)
    return path


def load_generator_artifact(
    path,
    verify_checkpoint: bool = True,
    snapshot: Optional[GlobalSnapshot] = None,
    client_id: Optional[str] = None,
    partition_hash: Optional[str] = None,
) -> GeneratorArtifact:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("generator artifact manifest must contain a mapping")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported generator artifact schema version")
    artifact = GeneratorArtifact.from_dict(payload)
    if verify_checkpoint:
        checkpoint_path = Path(artifact.checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = manifest_path.parent / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))
        if file_sha256(checkpoint_path) != artifact.checkpoint_hash:
            raise ValueError("generator checkpoint hash does not match its manifest")
    if snapshot is not None:
        verify_generator_lineage(
            artifact,
            snapshot,
            client_id=client_id,
            partition_hash=partition_hash,
        )
    elif client_id is not None or partition_hash is not None:
        raise ValueError("snapshot is required for lineage verification")
    return artifact
