import copy
import hashlib
import random
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Mapping, Optional

import torch
import torch.nn as nn
import numpy as np

from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from mflpoison.core.types import ClientUpdate, GlobalSnapshot


class FedAvgClientTrainer:
    """Convert a legacy FedMM ClientFedAvg run into a typed model delta."""

    def __init__(
        self,
        model_factory,
        device="cpu",
        learning_rate: float = 0.05,
        local_epochs: int = 1,
        mu: float = 0.0,
        seed: int = 42,
    ):
        self.model_factory = model_factory
        self.device = torch.device(device)
        self.args = SimpleNamespace(
            dataset="ucf101",
            modality="multimodal",
            fed_alg="fed_avg",
            learning_rate=float(learning_rate),
            local_epochs=int(local_epochs),
            mu=float(mu),
        )
        self.criterion = nn.NLLLoss().to(self.device)
        self.seed = int(seed)

    def _training_seed(self, client_id: str, round_index: int) -> int:
        identity = f"{self.seed}\0{round_index}\0{client_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") % (
            2**31
        )

    @contextmanager
    def _isolated_rng(self, seed: int):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        devices = []
        if self.device.type == "cuda":
            devices = [
                torch.cuda.current_device()
                if self.device.index is None
                else int(self.device.index)
            ]
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            try:
                yield
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

    def train(
        self,
        client_id: str,
        snapshot: GlobalSnapshot,
        dataloader,
        clean_num_samples: int,
        artifact_ids=None,
    ) -> ClientUpdate:
        training_seed = self._training_seed(client_id, snapshot.round_index)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(training_seed)
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "generator"):
            sampler.generator = loader_generator
        with self._isolated_rng(training_seed):
            model = self.model_factory(snapshot.state).to(self.device)
            client = ClientFedAvg(
                self.args,
                self.device,
                self.criterion,
                dataloader,
                model=model,
            )
            client.update_weights()
        trained_state = {
            key: value.detach().cpu().clone()
            for key, value in client.get_parameters().items()
        }
        base_state = {
            key: value.detach().cpu()
            for key, value in snapshot.state.items()
        }
        delta = {}
        for key, value in trained_state.items():
            base = base_state[key].to(dtype=value.dtype)
            if value.is_floating_point() or value.is_complex():
                delta[key] = value - base
            else:
                delta[key] = torch.zeros_like(value)
        train_num_samples = len(dataloader.dataset)
        return ClientUpdate(
            client_id=str(client_id),
            delta=delta,
            round_index=int(snapshot.round_index),
            base_snapshot_hash=str(snapshot.content_hash),
            clean_num_samples=int(clean_num_samples),
            train_num_samples=int(train_num_samples),
            aggregation_weight=float(clean_num_samples),
            metrics={key: float(value) for key, value in client.result.items()
                     if isinstance(value, (int, float))},
            artifact_ids=list(artifact_ids or []),
        )
