#!/usr/bin/env python3
import argparse
import json

import torch

from fed_multimodal.dtm_poison_gan import (
    DTMDiscriminator,
    DTMGANConfig,
    DTMGANTrainer,
    DTMGenerator,
)
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
from fed_multimodal.poison_gan.kplus1 import build_kplus1_discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DTM-GAN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model_path",
        required=True,
        help="Legacy K-class teacher checkpoint used to rebuild the discriminator",
    )
    add_partition_arguments(parser)
    parser.add_argument(
        "--output_dir",
        default="artifact/legacy_evaluation/dtm",
    )
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_evaluation(args.seed)
    checkpoint = load_legacy_checkpoint(
        args.checkpoint, "dtm", map_location=args.device
    )
    config = DTMGANConfig.from_dict(checkpoint["config"])
    data, loader = evaluation_loader_from_args(args)
    discriminator_model, _ = build_kplus1_discriminator(
        model_path=args.model_path,
        num_classes=config.num_classes,
        audio_input_dim=config.audio_feat_dim,
        video_input_dim=config.video_feat_dim,
        freeze=config.freeze_d,
        device=args.device,
    )
    trainer = DTMGANTrainer(
        DTMGenerator(config),
        DTMDiscriminator(discriminator_model),
        config,
        device=args.device,
    )
    validate_module_state(
        trainer.discriminator,
        checkpoint["discriminator_state_dict"],
        "discriminator_state_dict",
    )
    trainer.load_checkpoint(args.checkpoint, load_optimizers=False)
    metrics = trainer.evaluate(loader, args.num_batches)
    output_path = write_analysis_report(
        args.output_dir,
        {
            "gan_type": "dtm_gan",
            "checkpoint": args.checkpoint,
            "metrics": metrics,
            "meta": evaluation_metadata(args, data),
        },
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()
