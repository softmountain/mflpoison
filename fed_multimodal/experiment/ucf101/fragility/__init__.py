"""Fragility z(H)/z(gap) independent-contribution experiment (pre-registered protocol).

This package implements the milestone-1 engineering contract for the
`docs/fragility_H_gap_实验实施方案.md` protocol on UCF101 (51-class multimodal FL,
targeted availability poisoning):

- 6-way random-stream split with reproducible per-(round, client) seed derivation (seeds.py)
- explicit condition/source/train label semantics with runtime assertions (poison_plan.py)
- deterministic fixed-exposure injection with nested doses A_data in {1,2,4} (poison_plan.py)
- paired benign/attack trajectories from a shared M* / partition / seed streams (driver.py)
- continuous vs pulse attack scheduling (driver.py)
- per-round / per-class / per-client metrics incl. actual poison counts and update norms (metrics.py, driver.py)
- checkpointing, resume, run status + failure logging, run manifests (run_config.py, driver.py)
- Gate 0 provenance freeze + 7.21 anchor verification (gate0.py)

Deferred to later milestones (explicitly out of scope here): gamma_update / U1,
gradient alignment, R1 source robustness, and heavy mechanism metrics.
"""
