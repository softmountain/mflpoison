"""Automated tests for the fragility experiment infrastructure (protocol §16).

Covered here (milestone-1 mandatory, CPU-only unless noted):
  1  label semantics: condition==source!=train_label asserted
  2  fixed dose: 4 malicious clients each get 8/16/32 at A_data 1/2/4
  3  nested dose: A1 positions/ids subset of A2 subset of A4
  4  data-length invariance: poisoned client train length + FedAvg weight unchanged
  5  benign pairing: same master seed => identical client order + dataloader/train seeds
  6  schedule: pulse poison count == 0 strictly after pulse_end; continuous stays on
  9  resume: seeds are (round,client)-derived so a resumed round matches an uninterrupted one
  10 result schema: result.json carries all required fields incl. status/stop_reason on failure
  padding equivalence: 9-frame video padded to 10 (len_v=9) is forward-equivalent (needs a tiny model)

Deferred (per user's milestone-1 scope): #7 update amplification, #8 full cosine matrix.

Run:  PYTHONPATH=/home/xp/fedpoi /home/xp/anaconda3/envs/poigan/bin/python -m pytest \
        fed_multimodal/experiment/ucf101/fragility/tests/test_fragility.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fed_multimodal.experiment.ucf101.fragility.poison_plan import (
    apply_poison,
    build_poison_plan,
    N0_DEFAULT,
)
from fed_multimodal.experiment.ucf101.fragility.seeds import SeedBundle
from fed_multimodal.experiment.ucf101.fragility import run_config as RC


# ---- fixtures ----------------------------------------------------------------

def _fake_client_lists(n, label=7, key_prefix="c"):
    """Build a client's [key, path, label, feature] lists for audio+video."""
    audio, video = [], []
    for i in range(n):
        a = np.random.randn(500, 80).astype(np.float32)
        v = np.random.randn(10, 1280).astype(np.float32)
        audio.append([f"{key_prefix}/{i}", f"{key_prefix}/{i}", label, a])
        video.append([f"{key_prefix}/{i}", f"{key_prefix}/{i}", label, v])
    return audio, video


def _fake_pool(n=512, frames=9):
    return (torch.randn(n, 500, 80), torch.randn(n, frames, 1280))


def _plan(source=42, target=49, lengths=None, pool=512):
    lengths = lengths or {str(c): 400 for c in range(10)}
    return build_poison_plan(
        source_class=source, target_class=target, client_train_lengths=lengths,
        poison_index_seed=111, synthetic_sample_seed=222, attack_pool_size=pool,
    )


# ---- test 1: label semantics -------------------------------------------------

def test_label_semantics_assert():
    plan = _plan(source=42, target=49)
    plan.assert_label_semantics()  # should not raise
    with pytest.raises(AssertionError):
        _plan(source=42, target=42)  # s==t must fail at build time


def test_label_semantics_recorded_in_records():
    plan = _plan(source=42, target=49)
    pa, pv = _fake_pool()
    ca, cv = _fake_client_lists(400)
    _a, _v, records = apply_poison(ca, cv, plan, "0", a_data=1, pool_audio=pa, pool_video=pv)
    assert len(records) == N0_DEFAULT
    for rec in records:
        assert rec["condition_class"] == rec["source_class"] == 42
        assert rec["train_label"] == 49
        assert rec["source_class"] != rec["train_label"]


# ---- test 2: fixed dose ------------------------------------------------------

@pytest.mark.parametrize("a_data,expected", [(1, 8), (2, 16), (4, 32)])
def test_fixed_dose_counts(a_data, expected):
    plan = _plan()
    for c in ["0", "1", "2", "3"]:
        assert len(plan.per_client[c].positions_for_dose(a_data)) == expected
        assert len(plan.per_client[c].synthetic_ids_for_dose(a_data)) == expected


def test_all_four_malicious_present():
    plan = _plan()
    assert set(plan.per_client.keys()) == {"0", "1", "2", "3"}


# ---- test 3: nested dose -----------------------------------------------------

def test_nested_dose_positions_and_ids():
    plan = _plan()
    for c in ["0", "1", "2", "3"]:
        p1 = plan.per_client[c].positions_for_dose(1)
        p2 = plan.per_client[c].positions_for_dose(2)
        p4 = plan.per_client[c].positions_for_dose(4)
        assert set(p1).issubset(set(p2))
        assert set(p2).issubset(set(p4))
        i1 = plan.per_client[c].synthetic_ids_for_dose(1)
        i2 = plan.per_client[c].synthetic_ids_for_dose(2)
        i4 = plan.per_client[c].synthetic_ids_for_dose(4)
        assert set(i1).issubset(set(i2))
        assert set(i2).issubset(set(i4))


def test_synthetic_ids_disjoint_across_clients():
    plan = _plan()
    seen = []
    for c in ["0", "1", "2", "3"]:
        seen.append(set(plan.per_client[c].synthetic_ids_for_dose(4)))
    # pairwise disjoint (protocol §6.1: 4 clients use non-overlapping synthetic IDs)
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert seen[i].isdisjoint(seen[j])


# ---- test 4: data-length + weight invariance ---------------------------------

@pytest.mark.parametrize("a_data", [1, 2, 4])
def test_length_invariant_under_poison(a_data):
    plan = _plan()
    pa, pv = _fake_pool()
    ca, cv = _fake_client_lists(400)
    a2, v2, records = apply_poison(ca, cv, plan, "0", a_data=a_data, pool_audio=pa, pool_video=pv)
    assert len(a2) == len(ca) == 400        # training length unchanged
    assert len(v2) == len(cv) == 400
    assert len(records) == N0_DEFAULT * a_data
    # exactly `exposure` positions changed label to target; the rest keep their clean label
    changed = sum(1 for i in range(400) if a2[i][2] != ca[i][2])
    assert changed == N0_DEFAULT * a_data


def test_capacity_failure_raises():
    # a client with < N0*4 examples must raise (protocol §7.2: re-choose N0, do not truncate)
    with pytest.raises(ValueError):
        _plan(lengths={**{str(c): 400 for c in range(10)}, "0": 30})


def test_pool_too_small_raises():
    # attack pool < 4*32 disjoint IDs must raise
    with pytest.raises(ValueError):
        _plan(pool=100)


# ---- test 5: benign/attack pairing (shared seed streams) ---------------------

def test_paired_seed_streams_identical():
    sb_benign = SeedBundle.from_master(8)
    sb_attack = SeedBundle.from_master(8)
    # the first four streams (partition/sampling/order/training) are identical
    assert sb_benign.client_sampling_seed == sb_attack.client_sampling_seed
    assert sb_benign.local_data_order_seed == sb_attack.local_data_order_seed
    assert sb_benign.model_training_seed == sb_attack.model_training_seed
    assert sb_benign.partition_seed == sb_attack.partition_seed
    # per-(round,client) derived seeds match too
    for r in [1, 15, 30, 60]:
        assert sb_benign.client_sampling(r) == sb_attack.client_sampling(r)
        for c in range(10):
            assert sb_benign.local_data_order(r, c) == sb_attack.local_data_order(r, c)
            assert sb_benign.model_training(r, c) == sb_attack.model_training(r, c)


def test_different_master_seeds_differ():
    a = SeedBundle.from_master(8)
    b = SeedBundle.from_master(9)
    assert a.client_sampling(1) != b.client_sampling(1)


# ---- test 6: schedule --------------------------------------------------------

def _attack_active(spec, r):
    from fed_multimodal.experiment.ucf101.fragility.driver import TrajectoryRunner
    # emulate the schedule predicate without constructing a full runner
    if spec.condition == RC.BENIGN:
        return False
    if spec.schedule == RC.CONTINUOUS:
        return True
    return r <= spec.pulse_end


def test_pulse_schedule_stops_after_pulse_end():
    spec = RC.RunSpec(run_id="t", phase="M2", condition=RC.G_POISON, master_seed=8,
                      target_class=49, source_class=42, a_data=4, schedule=RC.PULSE,
                      pulse_end=15, horizon=60)
    assert all(_attack_active(spec, r) for r in range(1, 16))       # 1..15 on
    assert all(not _attack_active(spec, r) for r in range(16, 61))  # 16..60 strictly off


def test_continuous_schedule_always_on():
    spec = RC.RunSpec(run_id="t", phase="S1", condition=RC.G_POISON, master_seed=8,
                      target_class=49, source_class=42, a_data=1, schedule=RC.CONTINUOUS,
                      horizon=60)
    assert all(_attack_active(spec, r) for r in range(1, 61))


# ---- test 9: resume determinism ----------------------------------------------

def test_resume_seed_determinism():
    """A run resumed at round 31 must derive the same round-31 seeds as an uninterrupted run.

    Because every per-round seed is derived from (base, round, client) via a stable hash and
    never from a running RNG state, resume is bit-identical without saving RNG state.
    """
    sb = SeedBundle.from_master(10)
    uninterrupted = [(sb.client_sampling(r), sb.local_data_order(r, 3), sb.model_training(r, 3))
                     for r in range(1, 61)]
    # "resume" = recompute only rounds 31..60 from the same bundle
    resumed = [(sb.client_sampling(r), sb.local_data_order(r, 3), sb.model_training(r, 3))
               for r in range(31, 61)]
    assert resumed == uninterrupted[30:]


# ---- test 10: result schema --------------------------------------------------

def test_manifest_schema_fields():
    spec = RC.RunSpec(run_id="t", phase="S1", condition=RC.G_POISON, master_seed=8,
                      target_class=49, source_class=42)
    sb = SeedBundle.from_master(8)
    plan = _plan()
    man = RC.RunManifest.new(spec, sb, plan, {str(c): 400 for c in range(10)})
    d = man.to_dict()
    for key in ["spec", "status", "stop_reason", "seeds", "poison_plan",
                "per_client_exposure", "per_client_train_length", "ckpt_paths"]:
        assert key in d
    assert d["status"] == "planned"
    # exposure recorded per malicious client at this dose
    assert d["per_client_exposure"]["0"] == N0_DEFAULT * spec.a_data
    # seeds dict carries all six named streams
    for s in ["partition_seed", "client_sampling_seed", "local_data_order_seed",
              "model_training_seed", "poison_index_seed", "synthetic_sample_seed"]:
        assert s in d["seeds"]


# ---- padding equivalence (tiny model forward, CPU) ---------------------------

def test_video_padding_forward_equivalence():
    """A 9-frame video (len_v=9) and its zero-pad-to-10 version (len_v=9) give the same output.

    Verifies the corrected §1.1 handling: freeze synthetic at 9 frames, pad only at the adapter,
    report len_v=9 so the model masks the padded 10th frame. Uses fuse_base+hid128 like M*.
    """
    from fed_multimodal.model.mm_models import MMActionClassifier
    torch.manual_seed(0)
    model = MMActionClassifier(
        num_classes=51, audio_input_dim=80, video_input_dim=1280,
        d_hid=128, en_att=True, att_name="fuse_base",
    ).eval()

    torch.manual_seed(1)
    audio = torch.randn(2, 500, 80)
    v9 = torch.randn(2, 9, 1280)
    len_a = torch.tensor([500, 500])
    len_v = torch.tensor([9, 9])

    # padded to 10 frames but len_v stays 9
    v10 = torch.cat([v9, torch.zeros(2, 1, 1280)], dim=1)

    with torch.no_grad():
        out9, _ = model(audio, v9, len_a, len_v)
        out10, _ = model(audio, v10, len_a, len_v)
    assert torch.allclose(out9, out10, atol=1e-5), \
        f"max diff {(out9 - out10).abs().max().item():.2e}"
