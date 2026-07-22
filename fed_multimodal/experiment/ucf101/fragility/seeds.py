"""6-way random-stream split for paired benign/attack trajectories.

Protocol §10 requires splitting the single global seed used by the legacy code into
six named, recorded streams:

    partition_seed         - which client partition is used (fixed on disk; recorded as provenance)
    client_sampling_seed   - per-round client sampling
    local_data_order_seed  - per-client DataLoader shuffle order
    model_training_seed    - model init / dropout during local training
    poison_index_seed      - which clean positions are replaced by poison
    synthetic_sample_seed  - which synthetic sample IDs are used

The benign and attack conditions of the same master seed SHARE the first four streams;
only `poison_index_seed` and `synthetic_sample_seed` exist because of the attack. This
guarantees the paired benign/attack runs see the same client order and the same clean-sample
shuffle, so per-target destruction can be attributed by subtraction (protocol §3.2).

Per-(round, client) sub-seeds are derived deterministically from a base stream seed via a
stable hash, NOT from a running RNG state. This makes checkpoint-resume bit-identical: a run
resumed from round 30 derives exactly the same round-31 seeds as an uninterrupted run, because
each seed depends only on (base, round, client), never on how many draws happened before.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict


# numpy legacy RNG seeds must be in [0, 2**32). torch accepts 64-bit but we clamp to
# 2**31 for a single safe range usable by both numpy and torch.
_SEED_MOD = 2 ** 31


def derive_seed(*parts) -> int:
    """Deterministically derive a sub-seed in [0, 2**31) from arbitrary labeled parts.

    Uses SHA256 over a stable string join so results are stable across processes and
    Python hash-randomization. Order of parts matters.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MOD


@dataclass
class SeedBundle:
    """The six named seed streams for one (master_seed, condition) trajectory.

    `master_seed` is the human-facing seed (8..12 in the protocol). The first four
    streams are derived so benign and attack of the same master_seed share them; the
    last two are attack-only (equal to 0-labeled derivations for benign, but never used
    there since benign injects nothing).
    """
    master_seed: int
    partition_seed: int
    client_sampling_seed: int
    local_data_order_seed: int
    model_training_seed: int
    poison_index_seed: int
    synthetic_sample_seed: int

    @classmethod
    def from_master(cls, master_seed: int, partition_tag: str = "alpha50_fold1") -> "SeedBundle":
        # partition is materialized on disk; partition_seed is a recorded provenance tag,
        # derived from a fixed string so it does NOT change with master_seed (all seeds
        # reuse the same physical partition — a protocol requirement, §10 "share first four").
        partition_seed = derive_seed("partition", partition_tag)
        return cls(
            master_seed=int(master_seed),
            partition_seed=partition_seed,
            client_sampling_seed=derive_seed("client_sampling", master_seed),
            local_data_order_seed=derive_seed("local_data_order", master_seed),
            model_training_seed=derive_seed("model_training", master_seed),
            poison_index_seed=derive_seed("poison_index", master_seed),
            synthetic_sample_seed=derive_seed("synthetic_sample", master_seed),
        )

    def client_sampling(self, round_idx: int) -> int:
        return derive_seed(self.client_sampling_seed, "round", round_idx)

    def local_data_order(self, round_idx: int, client_idx) -> int:
        return derive_seed(self.local_data_order_seed, "round", round_idx, "client", client_idx)

    def model_training(self, round_idx: int, client_idx) -> int:
        return derive_seed(self.model_training_seed, "round", round_idx, "client", client_idx)

    def to_dict(self) -> dict:
        return asdict(self)
