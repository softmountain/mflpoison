"""RunSpec + RunManifest: unique run identity, resolved seeds, poison plan, provenance, status.

Every trajectory (benign or attack) is fully described by a RunSpec. The RunManifest is what gets
written to disk and must let a reader recover (protocol §15):
  - the parsed experiment condition and status (planned/running/completed/failed),
  - code/env/data/model/generator/partition provenance summary,
  - per-client poison indices, synthetic IDs, actual exposure and aggregation weights,
  - per-round global/per-class/per-client update metrics (in the paired result.json),
  - round 0/15/30/60 model snapshots (paths),
  - final summary, stdout/stderr log path and failure reason.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# schedule types
CONTINUOUS = "continuous"   # attack rounds 1..horizon
PULSE = "pulse"             # attack rounds 1..pulse_end, clean pulse_end+1..horizon

# condition types (protocol §11 C1 uses several; milestone-1 needs benign + g_poison)
BENIGN = "benign"
G_POISON = "g_poison"       # source synthetic feature + target label (main attack)

EVAL_POINTS_DEFAULT = (0, 5, 10, 15, 20, 30, 45, 60)
CKPT_POINTS_DEFAULT = (0, 15, 30, 60)


@dataclass
class RunSpec:
    run_id: str
    phase: str                     # R0 / S1 / M1 / M2 / C1 / smoke ...
    condition: str                 # benign / g_poison
    master_seed: int               # human-facing seed (8..12); benign & attack share the first 4 streams
    m_star_path: str = ""          # frozen M* checkpoint
    target_class: int = -1         # -1 for benign (target-agnostic baseline)
    source_class: int = -1
    a_data: int = 1                # data-level amplification {1,2,4}
    schedule: str = CONTINUOUS
    pulse_end: int = 15            # last attack round in pulse mode
    horizon: int = 60              # total rounds
    fold: int = 1
    n0: int = 8
    malicious_clients: Tuple[int, ...] = (0, 1, 2, 3)
    eval_points: Tuple[int, ...] = EVAL_POINTS_DEFAULT
    checkpoint_points: Tuple[int, ...] = CKPT_POINTS_DEFAULT
    early_dense_eval_until: int = 0   # if >0, also eval every round r<=this (protocol §9.3 early signal)

    def is_benign(self) -> bool:
        return self.condition == BENIGN or not self.malicious_clients or self.target_class < 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunManifest:
    spec: RunSpec
    status: str = "planned"                       # planned/running/completed/failed
    stop_reason: Optional[str] = None
    seeds: Dict[str, int] = field(default_factory=dict)      # resolved 6-way seed streams
    provenance: Dict[str, str] = field(default_factory=dict)  # hashes / versions (Gate 0)
    poison_plan: Optional[dict] = None            # PoisonPlan.to_dict()
    per_client_exposure: Dict[str, int] = field(default_factory=dict)  # actual poison count/client (max dose)
    per_client_train_length: Dict[str, int] = field(default_factory=dict)
    ckpt_paths: Dict[str, str] = field(default_factory=dict)           # {round: path}
    log_path: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @classmethod
    def new(cls, spec: RunSpec, seeds, plan, train_lengths: Dict[str, int]) -> "RunManifest":
        """Build a manifest from a spec + resolved SeedBundle + (optional) PoisonPlan.

        per_client_exposure records the ACTUAL number of poisoned positions at this run's dose
        (a_data), so a reader can confirm 8/16/32 without re-deriving the plan.
        """
        exposure: Dict[str, int] = {}
        if plan is not None:
            for cid, cp in plan.per_client.items():
                exposure[cid] = len(cp.positions_for_dose(spec.a_data))
        return cls(
            spec=spec,
            status="planned",
            seeds=seeds.to_dict() if hasattr(seeds, "to_dict") else dict(seeds),
            poison_plan=plan.to_dict() if plan is not None else None,
            per_client_exposure=exposure,
            per_client_train_length={k: int(v) for k, v in train_lengths.items()},
            started_at=time.time(),
        )

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spec"] = self.spec.to_dict()
        return d
