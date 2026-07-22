"""Deterministic fixed-exposure poison sampler with nested data-level amplification.

Protocol §7. This REPLACES the legacy `attack_n_inject` cap semantics (which only replaced
a client's existing target-class entries, so the injected count was decided by the natural
class distribution and could not be fixed across targets).

Fixed-exposure rules:
  - malicious clients fixed = {0,1,2,3}
  - base exposure per malicious client per local epoch: N0 = 8
  - data-level amplification A_data in {1,2,4}: exposure = N0 * A_data = 8/16/32
  - poison samples REPLACE an equal number of clean positions -> training length and FedAvg
    sample weight unchanged (poison replaces clean exposure, does not add aggregation weight)
  - nested dose: for a fixed (client, master_seed), the A=1 replaced-position set is a subset
    of A=2, which is a subset of A=4 (and likewise for the synthetic IDs used)
  - the 4 malicious clients use mutually DISJOINT synthetic sample IDs within a round
    (A_data=4 -> 32/client -> 128 unique synthetic IDs total)

Label semantics (protocol §6.3), asserted before training:
    condition_class == source_class != train_label(=target_class)
Each poisoned item records (local_index, synthetic_id, condition_class, source_class,
train_label) so the attack semantics are recoverable from the manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np

N0_DEFAULT = 8
MALICIOUS_DEFAULT = (0, 1, 2, 3)
DOSE_MAX = 4  # A_data=4 => N0*4 exposure is the maximal nested set


@dataclass
class ClientPoisonPlan:
    """Per-client poison plan at the MAXIMAL dose (A_data=4). Lower doses are prefixes."""
    client_id: str
    # local dataset positions (into this client's clean example list) that get replaced,
    # ordered so [:N0*A] is the A_data=A dose. len == N0 * DOSE_MAX.
    replaced_positions: List[int]
    # synthetic sample IDs (indices into the attack pool) aligned 1:1 with replaced_positions.
    synthetic_ids: List[int]

    def positions_for_dose(self, a_data: int) -> List[int]:
        return self.replaced_positions[: N0_DEFAULT * a_data]

    def synthetic_ids_for_dose(self, a_data: int) -> List[int]:
        return self.synthetic_ids[: N0_DEFAULT * a_data]


@dataclass
class PoisonPlan:
    source_class: int
    target_class: int
    condition_class: int          # == source_class (content generation condition)
    n0: int
    malicious_clients: List[str]
    attack_pool_size: int
    per_client: Dict[str, ClientPoisonPlan] = field(default_factory=dict)

    def assert_label_semantics(self):
        # protocol §6.3 hard check: condition==source!=train_label(target)
        assert self.condition_class == self.source_class, (
            f"condition_class({self.condition_class}) != source_class({self.source_class})"
        )
        assert self.source_class != self.target_class, (
            f"source_class({self.source_class}) == target_class({self.target_class}) (s==t forbidden)"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def build_poison_plan(
    source_class: int,
    target_class: int,
    client_train_lengths: Dict[str, int],
    poison_index_seed: int,
    synthetic_sample_seed: int,
    attack_pool_size: int,
    n0: int = N0_DEFAULT,
    malicious_clients=MALICIOUS_DEFAULT,
) -> PoisonPlan:
    """Construct a deterministic, nested, disjoint-across-clients poison plan.

    client_train_lengths: {client_id: number of clean training examples on that client}.
        Every malicious client must have >= n0*DOSE_MAX trainable examples, else the caller
        must fail the Gate (protocol §7.2: "if a client's effective length < 32, Gate fails
        and N0 is re-chosen; do not truncate and continue").
    attack_pool_size: number of synthetic source-class features available in the attack pool.
        Must be >= n0*DOSE_MAX*len(malicious_clients) so the 4 clients get disjoint IDs.
    """
    malicious = [str(c) for c in malicious_clients]
    max_exposure = n0 * DOSE_MAX

    # 1. validate capacities (hard errors -> Gate failure upstream)
    for c in malicious:
        if c not in client_train_lengths:
            raise ValueError(f"malicious client {c} not found in client_train_lengths")
        if client_train_lengths[c] < max_exposure:
            raise ValueError(
                f"client {c} has {client_train_lengths[c]} train examples < required "
                f"{max_exposure} (=N0*{DOSE_MAX}); re-choose N0 rather than truncate (protocol §7.2)"
            )
    need_ids = max_exposure * len(malicious)
    if attack_pool_size < need_ids:
        raise ValueError(
            f"attack_pool_size {attack_pool_size} < required {need_ids} "
            f"(={max_exposure}*{len(malicious)} disjoint synthetic IDs); regenerate a larger pool"
        )

    # 2. assign DISJOINT synthetic ID blocks to the 4 clients (deterministic order).
    #    Block i = [i*max_exposure : (i+1)*max_exposure). Within a block the order is a
    #    deterministic permutation so nested prefixes are still "random" but reproducible.
    synth_rng = np.random.default_rng(synthetic_sample_seed)
    pos_rng = np.random.default_rng(poison_index_seed)

    per_client: Dict[str, ClientPoisonPlan] = {}
    for i, c in enumerate(sorted(malicious, key=lambda x: int(x))):
        block_start = i * max_exposure
        block_ids = np.arange(block_start, block_start + max_exposure)
        # permute the block so [:N0*A] doses aren't a monotonic slice of raw IDs
        synth_perm = synth_rng.permutation(max_exposure)
        client_synth_ids = [int(block_ids[j]) for j in synth_perm]

        # choose max_exposure distinct clean positions to replace, from this client's range
        n_clean = client_train_lengths[c]
        chosen = pos_rng.choice(n_clean, size=max_exposure, replace=False)
        replaced_positions = [int(p) for p in chosen]

        per_client[c] = ClientPoisonPlan(
            client_id=c,
            replaced_positions=replaced_positions,
            synthetic_ids=client_synth_ids,
        )

    plan = PoisonPlan(
        source_class=int(source_class),
        target_class=int(target_class),
        condition_class=int(source_class),
        n0=n0,
        malicious_clients=malicious,
        attack_pool_size=int(attack_pool_size),
        per_client=per_client,
    )
    plan.assert_label_semantics()
    return plan


def apply_poison(
    clean_audio: list,
    clean_video: list,
    plan: PoisonPlan,
    client_id: str,
    a_data: int,
    pool_audio,
    pool_video,
):
    """Return NEW (audio_list, video_list, exposure_records) with poison applied at dose A_data.

    Replaces clean entries in-place on COPIES (caller owns clean lists). Each replaced entry
    keeps the client-list structure [key, path, label, feature] but:
      - label  <- target_class  (train_label)
      - feature <- source-class synthetic feature (content of condition==source)

    exposure_records: list of dicts recording the attack semantics per poisoned item, so the
    manifest can recover condition/source/train for every injected sample (protocol §6.3, §15).

    Uses a SHALLOW list copy (new list object, shared element references) then replaces only the
    poisoned slots with fresh entries. The dataset only READS features, so unpoisoned entries can
    be shared by reference; this avoids deep-copying every feature array on every round.
    """
    if client_id not in plan.per_client:
        return list(clean_audio), list(clean_video), []

    audio = list(clean_audio)
    video = list(clean_video)

    cp = plan.per_client[client_id]
    positions = cp.positions_for_dose(a_data)
    synth_ids = cp.synthetic_ids_for_dose(a_data)
    records = []
    for pos, sid in zip(positions, synth_ids):
        orig_key = audio[pos][0]
        orig_label = int(audio[pos][2])
        a_feat = pool_audio[sid].cpu().numpy() if hasattr(pool_audio[sid], "cpu") else np.asarray(pool_audio[sid])
        v_feat = pool_video[sid].cpu().numpy() if hasattr(pool_video[sid], "cpu") else np.asarray(pool_video[sid])
        audio[pos] = [orig_key, audio[pos][1], int(plan.target_class), a_feat]
        video[pos] = [video[pos][0], video[pos][1], int(plan.target_class), v_feat]
        records.append({
            "local_index": int(pos),
            "synthetic_id": int(sid),
            "condition_class": int(plan.condition_class),
            "source_class": int(plan.source_class),
            "train_label": int(plan.target_class),
            "replaced_clean_label": orig_label,
        })
    return audio, video, records
