from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch

from mflpoison.core.types import (
    AggregationResult,
    ClientUpdate,
    DefenseDecision,
    RoundRecord,
)
from mflpoison.core.hashing import mapping_hash, tensor_map_hash


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
        "delta_hash": tensor_map_hash(update.delta),
    }


def round_record_hash(record: RoundRecord) -> str:
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


def _revalidate_update(update: ClientUpdate) -> ClientUpdate:
    if not isinstance(update, ClientUpdate):
        raise TypeError("round record update has an invalid type")
    legacy_state = getattr(update, "_legacy_state", None)
    return ClientUpdate(
        client_id=update.client_id,
        delta=None if update.is_legacy_state else update.delta,
        state=(
            update.state
            if update.is_legacy_state
            else legacy_state
        ),
        round_index=update.round_index,
        base_snapshot_hash=update.base_snapshot_hash,
        clean_num_samples=update.clean_num_samples,
        train_num_samples=update.train_num_samples,
        aggregation_weight=update.aggregation_weight,
        metrics=update.metrics,
        artifact_ids=update.artifact_ids,
        malicious=update.malicious,
    )


def _revalidate_decision(decision: DefenseDecision) -> DefenseDecision:
    if not isinstance(decision, DefenseDecision):
        raise TypeError("round record decision has an invalid type")
    return DefenseDecision(
        client_id=decision.client_id,
        action=decision.action,
        scores=decision.scores,
        thresholds=decision.thresholds,
        reason=decision.reason,
        final_weight=decision.final_weight,
    )


def revalidate_round_record(record: RoundRecord) -> RoundRecord:
    """Rebuild a pickled record so every constructor invariant runs on load."""

    if not isinstance(record, RoundRecord):
        raise TypeError("round record payload has an invalid type")
    decisions = tuple(
        _revalidate_decision(item) for item in record.defense_decisions
    )
    aggregation = record.aggregation_result
    if not isinstance(aggregation, AggregationResult):
        raise TypeError("round record aggregation result has an invalid type")
    aggregation = AggregationResult(
        state=aggregation.state,
        decisions=tuple(
            _revalidate_decision(item) for item in aggregation.decisions
        ),
        diagnostics=aggregation.diagnostics,
    )
    return RoundRecord(
        round_index=record.round_index,
        base_snapshot_hash=record.base_snapshot_hash,
        selected_client_ids=record.selected_client_ids,
        raw_updates=tuple(_revalidate_update(item) for item in record.raw_updates),
        defense_decisions=decisions,
        processed_updates=tuple(
            _revalidate_update(item) for item in record.processed_updates
        ),
        aggregation_result=aggregation,
        evaluation=record.evaluation,
    )


def save_round_record(record: RoundRecord, path) -> Path:
    if not isinstance(record, RoundRecord):
        raise TypeError("record must be a RoundRecord")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "record": record,
            "content_hash": round_record_hash(record),
        },
        temporary,
    )
    temporary.replace(path)
    return path


def load_round_record(path, map_location="cpu") -> RoundRecord:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("round record artifact must contain a mapping")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported round record schema version")
    record = revalidate_round_record(payload.get("record"))
    expected_hash = str(payload.get("content_hash", ""))
    if not expected_hash or round_record_hash(record) != expected_hash:
        raise ValueError("round record content hash does not match its payload")
    return record


def save_round_record_bundle(phases: Mapping[str, object], path) -> Path:
    normalized = {
        str(phase): [revalidate_round_record(record) for record in records]
        for phase, records in phases.items()
    }
    record_hashes = {
        phase: [round_record_hash(record) for record in records]
        for phase, records in normalized.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 2,
            "phases": normalized,
            "record_hashes": record_hashes,
            "content_hash": mapping_hash({"record_hashes": record_hashes}),
        },
        temporary,
    )
    temporary.replace(path)
    return path


def load_round_record_bundle(path, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("round record bundle must contain a mapping")
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("unsupported round record bundle schema version")
    phases = payload.get("phases")
    hashes = payload.get("record_hashes")
    if not isinstance(phases, Mapping) or not isinstance(hashes, Mapping):
        raise TypeError("round record bundle has an invalid phase mapping")
    normalized = {
        str(phase): [revalidate_round_record(record) for record in records]
        for phase, records in phases.items()
    }
    actual_hashes = {
        phase: [round_record_hash(record) for record in records]
        for phase, records in normalized.items()
    }
    if actual_hashes != dict(hashes):
        raise ValueError("round record bundle hashes do not match its records")
    expected = str(payload.get("content_hash", ""))
    if not expected or mapping_hash({"record_hashes": actual_hashes}) != expected:
        raise ValueError("round record bundle content hash does not match")
    return normalized
