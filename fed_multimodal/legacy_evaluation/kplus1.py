#!/usr/bin/env python3
import argparse
import json

import torch

from fed_multimodal.legacy_evaluation.checkpoint import (
    load_legacy_checkpoint,
    validate_module_state,
)
from fed_multimodal.legacy_evaluation.data import (
    add_partition_arguments,
    evaluation_loader_from_args,
)
from fed_multimodal.legacy_evaluation.reporting import (
    evaluation_metadata,
    seed_evaluation,
    write_analysis_report,
)
from fed_multimodal.poison_gan import (
    FedPoisonGANTrainer,
    PoisonDiscriminator,
    PoisonFeatureGenerator,
    PoisonGANConfig,
)
from fed_multimodal.poison_gan.kplus1 import build_kplus1_discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fed-PoisonGAN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Legacy K-class teacher checkpoint used to rebuild the discriminator",
    )
    add_partition_arguments(parser)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifact/legacy_evaluation/kplus1_legacy",
    )
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_evaluation(args.seed)
    checkpoint = load_legacy_checkpoint(
        args.checkpoint, "kplus1_legacy", map_location=args.device
    )
    config = PoisonGANConfig.from_dict(checkpoint["config"])
    data, loader = evaluation_loader_from_args(args)
    disc_model, _ = build_kplus1_discriminator(
        args.model_path,
        num_classes=config.num_classes,
        audio_input_dim=config.audio_feat_dim,
        video_input_dim=config.video_feat_dim,
        freeze=config.freeze_d,
        device=args.device,
    )
    generator = PoisonFeatureGenerator(
        num_classes=config.num_classes,
        audio_seq_len=config.audio_seq_len,
        audio_feat_dim=config.audio_feat_dim,
        video_seq_len=config.video_seq_len,
        video_feat_dim=config.video_feat_dim,
        z_dim=config.z_dim,
        label_emb_dim=config.label_emb_dim,
        hidden_dim=config.hidden_dim,
        audio_out_max=config.audio_out_max,
        video_out_max=config.video_out_max,
        video_scale_max=config.video_scale_max,
    )
    trainer = FedPoisonGANTrainer(
        generator,
        PoisonDiscriminator(disc_model),
        config,
        device=args.device,
    )
    validate_module_state(
        trainer.discriminator,
        checkpoint["discriminator_state_dict"],
        "discriminator_state_dict",
    )
    trainer.load_checkpoint(args.checkpoint, load_optimizers=False)
    metrics = trainer.evaluate(loader, num_batches=args.num_batches)

    payload = {
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "meta": evaluation_metadata(args, data),
    }
    output_path = write_analysis_report(args.output_dir, payload)
    print(json.dumps(metrics, indent=2))
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()
