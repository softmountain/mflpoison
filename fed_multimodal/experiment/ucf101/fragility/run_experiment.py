"""CLI entry point for a single fragility trajectory (protocol §11, §17).

Builds a RunSpec from CLI args, resolves the frozen M* + attack pool, and runs one paired-able
trajectory via the driver. Deterministic run_id encodes the full condition so the manifest set is
self-describing and duplicate/missing combinations can be detected before launch (protocol §15.1).

Examples
--------
Benign 60-round baseline for seed 8:
    python -m fed_multimodal.experiment.ucf101.fragility.run_experiment \
        --phase baseline --condition benign --seed 8 --horizon 60

Attack (S1): target 49, source 42, A_data=1, continuous, seed 8:
    python -m fed_multimodal.experiment.ucf101.fragility.run_experiment \
        --phase S1 --condition g_poison --seed 8 --target 49 --source 42 \
        --a_data 1 --schedule continuous --horizon 60 \
        --attack_pool <path/to/attack_pool_s42.pt>

The runner reads system.cfg via the base train arg parser so all dataset/feature paths and the
FL hyperparameters (alpha, sample_rate, batch_size, lr, local_epochs) resolve exactly as in M*.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# reuse the base train.py arg parser so system.cfg + all FL args resolve identically to M*
import importlib.util


logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _load_base_train():
    train_path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("ucf101_train_base", train_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_run_id(phase, condition, seed, target, source, a_data, schedule, horizon) -> str:
    if condition == "benign":
        return f"{phase}__benign__s{seed}__H{horizon}"
    sched = "cont" if schedule == "continuous" else "pulse"
    return f"{phase}__{condition}__t{target}_src{source}__A{a_data}__{sched}__s{seed}__H{horizon}"


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--phase", default="smoke")
    p.add_argument("--condition", default="benign", choices=["benign", "g_poison"])
    p.add_argument("--seed", type=int, default=8, help="master seed (8..12)")
    p.add_argument("--target", type=int, default=-1)
    p.add_argument("--source", type=int, default=42)
    p.add_argument("--a_data", type=int, default=1, choices=[1, 2, 4])
    p.add_argument("--schedule", default="continuous", choices=["continuous", "pulse"])
    p.add_argument("--pulse_end", type=int, default=15)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n0", type=int, default=8)
    p.add_argument("--eval_points", type=str, default="0,5,10,15,20,30,45,60")
    p.add_argument("--checkpoint_points", type=str, default="0,15,30,60")
    p.add_argument("--early_dense_eval_until", type=int, default=0)
    p.add_argument("--m_star_path", type=str,
                   default="/home/xp/fedpoi/fed_multimodal/Local/results/poison_attack/M_star_alpha5_sr1.pt")
    # system.cfg points to a path that does not exist on this machine; the real feature/partition
    # root is here. Override so DataloadManager resolves alpha50/fold1 client pkls (matches how
    # M* was trained and how run_poison_batch.py runs).
    p.add_argument("--data_dir", type=str,
                   default="/home/xp/fedpoi/fed_multimodal/results")
    p.add_argument("--attack_pool", type=str, default=None,
                   help="synthetic source-class pool .pt (required for g_poison)")
    p.add_argument("--out_root", type=str,
                   default="/home/xp/fedpoi/fed_multimodal/result/fragility")
    p.add_argument("--benign_ref", type=str, default=None,
                   help="path to a completed benign result.json for the crash stop rule")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    a = parse_args()
    base = _load_base_train()
    fargs = base.parse_args()   # dataset/feature/FL hyperparameters from system.cfg + defaults

    # lock the FL setting to the M* training config (protocol §10)
    fargs.dataset = "ucf101"
    fargs.alpha = 5.0
    fargs.sample_rate = 1.0
    fargs.batch_size = 16
    fargs.local_epochs = 1
    fargs.learning_rate = 0.05
    fargs.fed_alg = "fed_avg"
    fargs.att = True
    fargs.att_name = "fuse_base"
    fargs.hid_size = 128
    fargs.modality = "multimodal"
    fargs.num_folds = 1
    fargs.data_dir = a.data_dir   # override the stale system.cfg root

    from .run_config import RunSpec
    from .driver import TrajectoryRunner
    from .attack_pool import load_attack_pool

    eval_points = tuple(int(x) for x in a.eval_points.split(",") if x != "")
    ckpt_points = tuple(int(x) for x in a.checkpoint_points.split(",") if x != "")

    run_id = make_run_id(a.phase, a.condition, a.seed, a.target, a.source,
                         a.a_data, a.schedule, a.horizon)

    spec = RunSpec(
        run_id=run_id,
        phase=a.phase,
        condition=a.condition,
        master_seed=a.seed,
        m_star_path=a.m_star_path,
        target_class=a.target,
        source_class=a.source,
        a_data=a.a_data,
        schedule=a.schedule,
        pulse_end=a.pulse_end,
        horizon=a.horizon,
        fold=a.fold,
        n0=a.n0,
        eval_points=eval_points,
        checkpoint_points=ckpt_points,
        early_dense_eval_until=a.early_dense_eval_until,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    attack_pool = None
    if a.condition != "benign":
        if a.attack_pool is None:
            raise SystemExit("--attack_pool is required for g_poison")
        # a fixed-source pool must contain only source_class content
        attack_pool = load_attack_pool(a.attack_pool, expected_condition=a.source)

    benign_acc_by_round = None
    if a.benign_ref and os.path.exists(a.benign_ref):
        import json
        prev = json.loads(Path(a.benign_ref).read_text())
        benign_acc_by_round = {}
        for k, v in prev.get("eval_history", {}).items():
            if k.startswith("round") and "acc" in v:
                benign_acc_by_round[int(k.replace("round", ""))] = v["acc"]

    result_dir = Path(a.out_root) / run_id
    ckpt_dir = result_dir / "ckpt"
    paths = {"result_dir": str(result_dir), "ckpt_dir": str(ckpt_dir)}

    logging.info("run_id=%s device=%s condition=%s target=%s source=%s A=%d schedule=%s H=%d",
                 run_id, device, a.condition, a.target, a.source, a.a_data, a.schedule, a.horizon)

    runner = TrajectoryRunner(spec, fargs, device, paths, attack_pool=attack_pool,
                              benign_acc_by_round=benign_acc_by_round)
    result = runner.run()
    logging.info("DONE run_id=%s status=%s", run_id, result.get("status"))


if __name__ == "__main__":
    main()
