import random
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from fed_multimodal.dtm_poison_gan import (
    DTMDiscriminator,
    DTMGANConfig,
    DTMGANTrainer,
    DTMGenerator,
)
from fed_multimodal.poison_gan.kplus1 import build_kplus1_discriminator
from fed_multimodal.temporal_adaptive_gan import (
    PoisonDiscriminator,
    TemporalAdaptiveGANConfig,
    TemporalAdaptiveGANTrainer,
    TemporalAdaptivePoisonGenerator,
)
from mflpoison.core.hashing import file_sha256
from mflpoison.core.types import GeneratorArtifact, GlobalSnapshot
from mflpoison.generators.lifecycle import (
    ClientGeneratorPartition,
    GeneratorTrainer,
    GeneratorTrainingRequest,
)


class FedMMGeneratorTrainer(GeneratorTrainer):
    """Train one legacy-compatible generator per malicious FedMM client."""

    def __init__(
        self,
        variant: str,
        output_dir,
        model_metadata: Mapping[str, object],
        modality_shapes: Mapping[str, tuple],
        num_classes: int,
        epochs: int = 1,
        max_batches: Optional[int] = None,
        log_interval: int = 0,
        device="cpu",
        config_overrides: Optional[Mapping[str, object]] = None,
        batch_size: Optional[int] = None,
    ):
        variant = str(variant).lower()
        if variant not in ("dtm", "temporal_adaptive"):
            raise ValueError("FedMM generator variant must be dtm or temporal_adaptive")
        self.variant = variant
        self.output_dir = Path(output_dir)
        self.model_metadata = dict(model_metadata)
        self.modality_shapes = dict(modality_shapes)
        self.num_classes = int(num_classes)
        self.epochs = int(epochs)
        if self.epochs < 1:
            raise ValueError("generator epochs must be positive")
        self.max_batches = max_batches
        self.log_interval = int(log_interval)
        self.device = str(device)
        self.config_overrides = dict(config_overrides or {})
        self.batch_size = None if batch_size is None else int(batch_size)
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("generator batch_size must be positive")

    def _training_dataloader(self, dataloader, seed: int):
        if self.batch_size is None or not hasattr(dataloader, "dataset"):
            return dataloader
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        num_workers = int(getattr(dataloader, "num_workers", 0))
        return DataLoader(
            dataloader.dataset,
            batch_size=self.batch_size,
            shuffle=isinstance(getattr(dataloader, "sampler", None), RandomSampler),
            num_workers=num_workers,
            collate_fn=getattr(dataloader, "collate_fn", None),
            pin_memory=bool(getattr(dataloader, "pin_memory", False)),
            drop_last=bool(getattr(dataloader, "drop_last", False)),
            worker_init_fn=getattr(dataloader, "worker_init_fn", None),
            persistent_workers=(
                num_workers > 0
                and bool(getattr(dataloader, "persistent_workers", False))
            ),
            generator=generator,
        )

    def _teacher_checkpoint(self, client_dir: Path, snapshot: GlobalSnapshot) -> Path:
        path = client_dir / "teacher_snapshot.pt"
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in snapshot.state.items()
                },
                "args": {
                    "hid_size": int(self.model_metadata["hid_size"]),
                    "att": bool(self.model_metadata["attention"]),
                    "att_name": str(self.model_metadata["attention_name"]),
                },
                "snapshot_hash": snapshot.content_hash,
                "round_index": snapshot.round_index,
            },
            path,
        )
        return path

    def _build_trainer(self, teacher_path: Path, dataloader, seed: int):
        audio_shape = self.modality_shapes["audio"]
        video_shape = self.modality_shapes["video"]
        common = dict(
            num_classes=self.num_classes,
            fake_class=self.num_classes,
            audio_seq_len=int(audio_shape[0]),
            audio_feat_dim=int(audio_shape[1]),
            video_seq_len=int(video_shape[0]),
            video_feat_dim=int(video_shape[1]),
            seed=int(seed),
        )
        common.update(self.config_overrides)
        discriminator_model, _ = build_kplus1_discriminator(
            model_path=teacher_path,
            num_classes=self.num_classes,
            audio_input_dim=int(audio_shape[1]),
            video_input_dim=int(video_shape[1]),
            hid_size=int(self.model_metadata["hid_size"]),
            att=bool(self.model_metadata["attention"]),
            att_name=str(self.model_metadata["attention_name"]),
            freeze=str(common.get("freeze_d", "backbone")),
            device=self.device,
        )
        if self.variant == "dtm":
            config = DTMGANConfig.from_dict(common)
            return DTMGANTrainer(
                DTMGenerator(config),
                DTMDiscriminator(discriminator_model),
                config,
                dataloader,
                self.device,
            )

        config = TemporalAdaptiveGANConfig.from_dict(
            {"gan_variant": "temporal_adaptive", **common}
        )
        generator = TemporalAdaptivePoisonGenerator(
            num_classes=config.num_classes,
            audio_seq_len=config.audio_seq_len,
            audio_feat_dim=config.audio_feat_dim,
            video_seq_len=config.video_seq_len,
            video_feat_dim=config.video_feat_dim,
            z_dim=config.z_dim,
            label_emb_dim=config.label_emb_dim,
            hidden_dim=config.hidden_dim,
            video_out_max=config.video_out_max,
            video_scale_max=config.video_scale_max,
            frame_noise_dim=config.frame_noise_dim,
            temporal_groups_max=config.temporal_groups_max,
            audio_stats_momentum=config.audio_stats_momentum,
        )
        return TemporalAdaptiveGANTrainer(
            generator,
            PoisonDiscriminator(discriminator_model),
            config,
            dataloader,
            self.device,
        )

    def train(
        self,
        request: GeneratorTrainingRequest,
        partition: ClientGeneratorPartition,
    ) -> GeneratorArtifact:
        if request.global_snapshot is None:
            raise ValueError("FedMM generator training requires a GlobalSnapshot")
        return self._fit(
            client_id=request.client_id,
            snapshot=request.global_snapshot,
            dataloader=partition.data,
            partition_hash=request.partition_hash,
            seed=request.seed,
            previous_artifact=request.warm_start_artifact,
            refresh_index=request.refresh_index,
            trained_round=request.round_index,
        )

    def fit(
        self,
        client_id: str,
        snapshot: GlobalSnapshot,
        dataloader,
        partition_hash: str,
        seed: int,
        previous_artifact: Optional[GeneratorArtifact] = None,
    ) -> GeneratorArtifact:
        """Legacy wrapper retained for callers that predate generator lifecycles."""

        refresh_index = (
            0 if previous_artifact is None else int(previous_artifact.refresh_index) + 1
        )
        return self._fit(
            client_id=client_id,
            snapshot=snapshot,
            dataloader=dataloader,
            partition_hash=partition_hash,
            seed=seed,
            previous_artifact=previous_artifact,
            refresh_index=refresh_index,
            trained_round=snapshot.round_index,
        )

    def _fit(
        self,
        client_id: str,
        snapshot: GlobalSnapshot,
        dataloader,
        partition_hash: str,
        seed: int,
        previous_artifact: Optional[GeneratorArtifact],
        refresh_index: int,
        trained_round: int,
    ) -> GeneratorArtifact:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        client_dir = self.output_dir / str(client_id) / snapshot.content_hash[:12]
        client_dir.mkdir(parents=True, exist_ok=True)
        teacher_path = self._teacher_checkpoint(client_dir, snapshot)
        trainer = self._build_trainer(
            teacher_path,
            self._training_dataloader(dataloader, seed),
            seed,
        )
        first_epoch = 1
        if previous_artifact is not None:
            previous_path = Path(previous_artifact.checkpoint_path)
            if not previous_path.is_file():
                raise FileNotFoundError(str(previous_path))
            if file_sha256(previous_path) != previous_artifact.checkpoint_hash:
                raise ValueError(
                    "warm-start checkpoint hash does not match its artifact"
                )
            checkpoint = trainer.load_checkpoint(
                previous_path, load_optimizers=True
            )
            if isinstance(checkpoint, Mapping):
                first_epoch = int(checkpoint.get("epoch", 0)) + 1

        metrics = {}
        last_epoch = first_epoch + self.epochs - 1
        for epoch in range(first_epoch, last_epoch + 1):
            metrics = trainer.train_epoch(
                epoch=epoch,
                max_batches=self.max_batches,
                log_interval=self.log_interval,
            )
        checkpoint_path = client_dir / (self.variant + ".pt")
        trainer.save_checkpoint(checkpoint_path, last_epoch, metrics)
        payload = torch.load(checkpoint_path, map_location="cpu")
        payload["lineage"] = {
            "client_id": str(client_id),
            "partition_hash": str(partition_hash),
            "parent_snapshot_hash": str(snapshot.content_hash),
            "parent_round": int(snapshot.round_index),
            "seed": int(seed),
            "variant": self.variant,
        }
        torch.save(payload, checkpoint_path)
        request = GeneratorTrainingRequest(
            client_id=str(client_id),
            partition_hash=str(partition_hash),
            global_snapshot_hash=str(snapshot.content_hash),
            variant=self.variant,
            round_index=int(trained_round),
            refresh_index=int(refresh_index),
            seed=int(seed),
            global_snapshot=snapshot,
            warm_start_artifact=previous_artifact,
        )
        return request.artifact(
            checkpoint_path=str(checkpoint_path),
            checkpoint_hash=file_sha256(checkpoint_path),
            metadata={"metrics": metrics},
        )
