"""Save round records used for analysis and reproducibility."""

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch

from mflpoison.core.hashing import mapping_hash, tensor_map_hash
from mflpoison.core.types import RoundRecord


def _update_payload(update):
    return {
        "client_id": update.client_id,
        "round_index": update.round_index,
        "base_snapshot_hash": update.base_snapshot_hash,
        "clean_num_samples": update.clean_num_samples,
        "train_num_samples": update.train_num_samples,
        "aggregation_weight": update.aggregation_weight,
        "metrics": dict(update.metrics),
        "artifact_ids": list(update.artifact_ids),
        "malicious": bool(update.malicious),
        "attack_active": bool(update.attack_active),
        "poison_sample_count": int(update.poison_sample_count),
        "delta_hash": tensor_map_hash(update.delta),
    }


def round_record_hash(record: RoundRecord) -> str:
    """Return a stable record identifier for result comparisons."""

    return mapping_hash(
        {
            "round_index": record.round_index,
            "base_snapshot_hash": record.base_snapshot_hash,
            "selected_client_ids": list(record.selected_client_ids),
            "raw_updates": [_update_payload(item) for item in record.raw_updates],
            "defense_decisions": [
                asdict(item) for item in record.defense_decisions
            ],
            "processed_updates": [
                _update_payload(item) for item in record.processed_updates
            ],
            "aggregation_state_hash": tensor_map_hash(
                record.aggregation_result.state
            ),
            "aggregation_diagnostics": dict(
                record.aggregation_result.diagnostics
            ),
            "evaluation": dict(record.evaluation),
        }
    )


def _atomic_torch_save(payload, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def save_round_record_bundle(phases: Mapping[str, object], path) -> Path:
    normalized = {
        str(phase): list(records)
        for phase, records in phases.items()
    }
    return _atomic_torch_save(
        {"schema_version": 1, "phases": normalized},
        path,
    )


def load_round_record_bundle(path, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("unsupported round record bundle")
    phases = payload.get("phases")
    if not isinstance(phases, Mapping):
        raise TypeError("round record bundle has no phase mapping")
    return {str(phase): list(records) for phase, records in phases.items()}
