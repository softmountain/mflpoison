"""Paired benign/attack trajectory driver (protocol §9, §10, §11, §14, §15).

A single call to `run_trajectory` runs ONE 0..horizon trajectory (benign or attack) from the
frozen M*, evaluating at the protocol eval points, snapshotting model state at checkpoint
points, recording per-round/per-class/per-client metrics, and writing a run manifest + result.

Design choices that make benign/attack strictly paired (protocol §10, test §16.5):
  - client processing order per round is derived from `client_sampling_seed` (shared by the
    benign/attack pair). At sr=1.0 all 10 clients participate; the order is a per-round
    permutation. FedAvg is order-invariant, but we keep the order identical anyway.
  - each client's train DataLoader gets an explicit torch.Generator seeded from
    (local_data_order_seed, round, client) -> identical shuffle permutation across the pair,
    since fixed-exposure replaces equal count and keeps dataset length constant.
  - torch global RNG is reseeded from (model_training_seed, round, client) right before each
    client's local training so dropout draws match across the pair.
  Consequently a benign run and its paired attack run are bit-identical on any CLEAN round
  (pulse rounds 16..60, or a benign trajectory throughout); they diverge only where poison
  content actually replaces clean content.

Aggregation is a self-contained sample-weighted FedAvg (no server-side optimizer, no defense),
matching FedAvg semantics; kept local so it is directly unit-testable.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fed_multimodal.constants import constants
from fed_multimodal.dataloader.dataload_manager import (
    DataloadManager,
    MMDatasetGenerator,
    collate_mm_fn_padd,
)
from fed_multimodal.model.mm_models import MMActionClassifier
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg

from . import metrics as M
from .attack_pool import AttackPool
from .poison_plan import apply_poison, build_poison_plan, PoisonPlan
from .run_config import RunManifest, RunSpec
from .seeds import SeedBundle

DEFAULT_A_SHAPE = np.array([500, constants.feature_len_dict["mfcc"]])          # [500,80]
DEFAULT_V_SHAPE = np.array([10, constants.feature_len_dict["mobilenet_v2"]])   # [10,1280] (pad target)


def load_m_star(path, device):
    """Load frozen M* into a fresh MMActionClassifier using the checkpoint's own args."""
    ckpt = torch.load(path, map_location="cpu")
    a = ckpt["args"]
    model = MMActionClassifier(
        num_classes=constants.num_class_dict[a.get("dataset", "ucf101")],
        audio_input_dim=constants.feature_len_dict["mfcc"],
        video_input_dim=constants.feature_len_dict["mobilenet_v2"],
        d_hid=a["hid_size"],
        en_att=a.get("att", False),
        att_name=a.get("att_name", "base"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device), a, ckpt


def weighted_fedavg(states: List[Dict[str, torch.Tensor]], counts: List[int]) -> Dict[str, torch.Tensor]:
    """Sample-weighted average of client state_dicts (FedAvg, no defense)."""
    total = float(sum(counts))
    out = copy.deepcopy(states[0])
    for k in out.keys():
        acc = states[0][k].float() * (counts[0] / total)
        for i in range(1, len(states)):
            acc = acc + states[i][k].float() * (counts[i] / total)
        out[k] = acc.to(states[0][k].dtype)
    return out


def _build_train_loader(audio_list, video_list, batch_size, generator):
    ds = MMDatasetGenerator(
        audio_list, video_list, DEFAULT_A_SHAPE, DEFAULT_V_SHAPE, len(audio_list),
        dataset="ucf101",
    )
    return DataLoader(
        ds, batch_size=int(batch_size), num_workers=0, shuffle=True,
        collate_fn=collate_mm_fn_padd, generator=generator,
    )


def _build_eval_loader(audio_list, video_list):
    ds = MMDatasetGenerator(
        audio_list, video_list, DEFAULT_A_SHAPE, DEFAULT_V_SHAPE, len(audio_list),
        dataset="ucf101",
    )
    return DataLoader(ds, batch_size=64, num_workers=0, shuffle=False, collate_fn=collate_mm_fn_padd)


class TrajectoryRunner:
    def __init__(self, spec: RunSpec, args, device, paths, attack_pool: Optional[AttackPool] = None,
                 benign_acc_by_round: Optional[Dict[int, float]] = None):
        self.spec = spec
        self.args = args
        self.device = device
        self.paths = paths                     # dict: result_dir, ckpt_dir
        self.attack_pool = attack_pool
        self.benign_acc_by_round = benign_acc_by_round or {}
        self.seeds = SeedBundle.from_master(spec.master_seed)
        self.log = logging.getLogger(f"frag.{spec.run_id}")

    # -- data --------------------------------------------------------------
    def _load_clients(self, dm: DataloadManager):
        clean_audio, clean_video, train_lengths = {}, {}, {}
        for cid in dm.client_ids:
            a = dm.load_audio_feat(client_id=cid, fold_idx=self.spec.fold)
            v = dm.load_video_feat(client_id=cid, fold_idx=self.spec.fold)
            clean_audio[cid] = a
            clean_video[cid] = v
            if cid not in ("dev", "test"):
                train_lengths[cid] = len(a)
        return clean_audio, clean_video, train_lengths

    # -- main --------------------------------------------------------------
    def run(self) -> dict:
        device = self.device
        result_dir = Path(self.paths["result_dir"]); result_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = Path(self.paths["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = result_dir / "manifest.json"
        result_path = result_dir / "result.json"

        dm = DataloadManager(self.args)
        dm.get_simulation_setting(alpha=self.args.alpha)
        dm.load_sim_dict(fold_idx=self.spec.fold)
        dm.get_client_ids(fold_idx=self.spec.fold)
        clean_audio, clean_video, train_lengths = self._load_clients(dm)
        client_ids = [c for c in dm.client_ids if c not in ("dev", "test")]
        num_classes = constants.num_class_dict[self.args.dataset]

        test_loader = _build_eval_loader(clean_audio["test"], clean_video["test"])

        global_model, m_args, _ = load_m_star(self.spec.m_star_path, device)

        # poison plan (attack only)
        plan: Optional[PoisonPlan] = None
        if self.spec.condition != "benign":
            if self.attack_pool is None:
                raise ValueError("attack condition requires an attack_pool")
            plan = build_poison_plan(
                source_class=self.spec.source_class,
                target_class=self.spec.target_class,
                client_train_lengths=train_lengths,
                poison_index_seed=self.seeds.poison_index_seed,
                synthetic_sample_seed=self.seeds.synthetic_sample_seed,
                attack_pool_size=self.attack_pool.size,
            )
            plan.assert_label_semantics()

        manifest = RunManifest.new(self.spec, self.seeds, plan, train_lengths)
        manifest.status = "running"
        manifest.save(manifest_path)

        # resume?
        start_round, eval_history = self._maybe_resume(manifest, global_model, ckpt_dir, result_path)
        if start_round is None:  # already completed
            return json.loads(result_path.read_text())

        criterion = nn.NLLLoss().to(device)

        # round 0 baseline eval (only when starting fresh)
        if start_round == 1 and 0 in self.spec.eval_points:
            eval_history["round0"] = self._eval(global_model, test_loader, num_classes)
            self._maybe_snapshot(0, global_model, ckpt_dir)

        stop_reason = None
        for r in range(start_round, self.spec.horizon + 1):
            attack_active = self._attack_active(r)
            order = self._client_order(len(client_ids), r)
            global_state = copy.deepcopy(global_model.state_dict())

            states, counts, per_client_norms = [], [], {}
            for idx in order:
                cid = client_ids[idx]
                is_mal = plan is not None and cid in plan.malicious_clients and attack_active
                if is_mal:
                    a_list, v_list, _rec = apply_poison(
                        clean_audio[cid], clean_video[cid], plan, cid, self.spec.a_data,
                        self.attack_pool.audio, self.attack_pool.video,
                    )
                else:
                    a_list, v_list = clean_audio[cid], clean_video[cid]

                gen = torch.Generator()
                gen.manual_seed(self.seeds.local_data_order(r, idx))
                loader = _build_train_loader(a_list, v_list, self.args.batch_size, gen)

                # deterministic dropout across the benign/attack pair
                torch.manual_seed(self.seeds.model_training(r, idx))
                torch.cuda.manual_seed_all(self.seeds.model_training(r, idx))

                client = ClientFedAvg(
                    self.args, device, criterion, loader,
                    model=copy.deepcopy(global_model), label_dict=None, num_class=num_classes,
                )
                client.update_weights()
                local_state = copy.deepcopy(client.get_parameters())
                states.append(local_state)
                counts.append(client.result["sample"])
                if is_mal or (plan is not None and cid in plan.malicious_clients):
                    per_client_norms[cid] = M.block_update_norms(local_state, global_state)
                del client

            new_state = weighted_fedavg(states, counts)
            # NaN/Inf stop rule (protocol §14.2)
            if any(not torch.isfinite(v.float()).all() for v in new_state.values()):
                stop_reason = f"NaN/Inf in aggregated weights at round {r}"
                break
            global_model.load_state_dict(new_state)

            if self._should_eval(r):
                ev = self._eval(global_model, test_loader, num_classes)
                ev["client_update_norms"] = per_client_norms
                ev["clients_per_round"] = [int(i) for i in order]
                ev["attack_active"] = bool(attack_active)
                eval_history[f"round{r}"] = ev
                # benign-relative single-round crash stop rule (protocol §14.2)
                bench = self.benign_acc_by_round.get(r)
                if bench is not None and (bench - ev["acc"]) > 10.0:
                    stop_reason = f"acc dropped {bench - ev['acc']:.1f}pt below benign at round {r}"
                    self._save_result(result_path, eval_history, manifest, plan, status="failed",
                                      stop_reason=stop_reason)
                    manifest.status = "failed"; manifest.stop_reason = stop_reason
                    manifest.save(manifest_path)
                    self.log.error(stop_reason)
                    return json.loads(result_path.read_text())

            if r in self.spec.checkpoint_points:
                self._maybe_snapshot(r, global_model, ckpt_dir)
                # incremental result save so a kill mid-run keeps evaluated rounds
                self._save_result(result_path, eval_history, manifest, plan, status="running")

        status = "failed" if stop_reason else "completed"
        self._save_result(result_path, eval_history, manifest, plan, status=status, stop_reason=stop_reason)
        manifest.status = status
        manifest.stop_reason = stop_reason
        manifest.save(manifest_path)
        self.log.info("trajectory %s %s", self.spec.run_id, status)
        return json.loads(result_path.read_text())

    # -- schedule / order --------------------------------------------------
    def _should_eval(self, r: int) -> bool:
        # protocol §9.3: always eval at the fixed points; optionally eval every round r<=dense
        return r in self.spec.eval_points or (0 < r <= self.spec.early_dense_eval_until)

    def _attack_active(self, r: int) -> bool:
        if self.spec.condition == "benign":
            return False
        if self.spec.schedule == "continuous":
            return True
        # pulse: attack rounds 1..pulse_end, clean afterwards
        return r <= self.spec.pulse_end

    def _client_order(self, n_clients: int, r: int) -> List[int]:
        rng = np.random.default_rng(self.seeds.client_sampling(r))
        k = int(round(self.args.sample_rate * n_clients))
        return [int(i) for i in rng.choice(n_clients, size=k, replace=False)]

    # -- eval / snapshot ---------------------------------------------------
    def _eval(self, model, test_loader, num_classes) -> dict:
        ev = M.evaluate_full(model, test_loader, self.device, num_classes)
        if self.spec.condition != "benign":
            ev["source_to_target"] = M.source_to_target_rate(
                model, test_loader, self.device, self.spec.source_class, self.spec.target_class,
            )
        return ev

    def _maybe_snapshot(self, r: int, model, ckpt_dir: Path):
        if r in self.spec.checkpoint_points:
            torch.save({"round": r, "model_state_dict": copy.deepcopy(model.state_dict()),
                        "run_id": self.spec.run_id},
                       ckpt_dir / f"round{r}.pt")

    def _maybe_resume(self, manifest, global_model, ckpt_dir: Path, result_path: Path):
        """Return (start_round, eval_history). start_round None -> already completed."""
        if result_path.exists():
            try:
                prev = json.loads(result_path.read_text())
            except Exception:
                prev = None
            if prev and prev.get("status") == "completed":
                return None, prev.get("eval_history", {})
            if prev and prev.get("eval_history"):
                # find latest snapshot <= last evaluated round
                snaps = sorted(int(p.stem.replace("round", "")) for p in ckpt_dir.glob("round*.pt"))
                if snaps:
                    last = max(snaps)
                    ck = torch.load(ckpt_dir / f"round{last}.pt", map_location="cpu")
                    global_model.load_state_dict(ck["model_state_dict"])
                    global_model.to(self.device)
                    self.log.info("resume from checkpoint round%d", last)
                    return last + 1, prev["eval_history"]
        return 1, {}

    def _save_result(self, result_path: Path, eval_history: dict, manifest, plan, status: str,
                     stop_reason: Optional[str] = None):
        result = {
            "run_id": self.spec.run_id,
            "phase": self.spec.phase,
            "condition": self.spec.condition,
            "target_class": self.spec.target_class,
            "source_class": self.spec.source_class,
            "a_data": self.spec.a_data,
            "schedule": self.spec.schedule,
            "horizon": self.spec.horizon,
            "master_seed": self.spec.master_seed,
            "status": status,
            "stop_reason": stop_reason,
            "is_baseline": self.spec.condition == "benign",
            "eval_history": eval_history,
            "poison_plan": plan.to_dict() if plan is not None else None,
        }
        result_path.write_text(json.dumps(result, indent=2))
